"""A diagnostic gate for the 10 EUR challenge, never a live trading permit."""

from datetime import UTC, datetime
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from enum import StrEnum
from typing import Literal

from sniper.config import BrokerCheckConfig, Model, Positive
from sniper.data.mt5_client import BrokerReader, MT5Error
from sniper.data.time import TimeDiagnostic, compare_tick_clock
from sniper.domain.broker import BrokerSnapshot
from sniper.domain.trade import Side


class Verdict(StrEnum):
    COMPATIBLE = "COMPATIBLE"
    TOO_COARSE = "TOO_COARSE"
    INSUFFICIENT_MARGIN = "INSUFFICIENT_MARGIN"
    SPREAD_TOO_HIGH = "SPREAD_TOO_HIGH"
    COST_TOO_HIGH = "COST_TOO_HIGH"
    INCOMPATIBLE_FOR_10_EUR = "INCOMPATIBLE_FOR_10_EUR"


class StopEstimate(Model):
    side: Side
    stop_pips: Decimal
    entry: Decimal
    stop_price: Decimal
    broker_stop_valid: bool
    price_loss: Decimal
    total_loss: Decimal | None
    risk_pct: Decimal | None


class BrokerCompatibilityReport(Model):
    schema_version: int = 1
    capital_eur: Positive
    snapshot: BrokerSnapshot
    policy: BrokerCheckConfig
    evaluated_at_utc: datetime
    nano_lot_compliant: bool
    spread_points: Decimal
    median_spread_points: Decimal | None = None
    margin_buy: Decimal
    margin_sell: Decimal
    spread_cost: Decimal
    round_trip_commission: Decimal | None
    slippage_cost: Decimal | None
    minimum_round_trip_cost: Decimal | None
    estimated_round_trip_cost: Decimal | None
    minimum_risk_pct: Decimal | None
    stops: tuple[StopEstimate, ...]
    verdict: Verdict
    challenge_status: Verdict
    blockers: tuple[str, ...]
    time_diagnostic: TimeDiagnostic
    warnings: tuple[str, ...] = ()
    live_trading_enabled: Literal[False] = False


def _loss(
    reader: BrokerReader,
    snapshot: BrokerSnapshot,
    side: Side,
    entry: Decimal,
    exit_price: Decimal,
) -> Decimal:
    profit = reader.profit(
        side, snapshot.symbol.name, snapshot.symbol.volumes.minimum, entry, exit_price
    )
    if not profit.is_finite() or profit > 0:
        raise MT5Error("order_calc_profit: invalid loss estimate")
    return -profit


def validate_broker_for_capital(
    reader: BrokerReader,
    capital_eur: Decimal = Decimal(10),
    symbol: str = "EURUSD",
    config: BrokerCheckConfig | None = None,
    *,
    now: datetime | None = None,
) -> BrokerCompatibilityReport:
    """Python equivalent of ValidateBrokerForCapital, using observed SDK calculations."""
    if not capital_eur.is_finite() or capital_eur <= 0:
        raise ValueError("capital must be positive and finite (EUR)")
    policy = config or BrokerCheckConfig()
    snapshot = reader.snapshot(symbol)
    instrument, account, quote = snapshot.symbol, snapshot.account, snapshot.quote
    pip_size = instrument.pip_size  # Also validates the EUR/USD-only restriction.
    evaluated_at = now or datetime.now(UTC)
    if evaluated_at.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    blockers: list[str] = []
    failures: set[Verdict] = set()

    def block(reason: str, verdict: Verdict = Verdict.INCOMPATIBLE_FOR_10_EUR) -> None:
        blockers.append(reason)
        failures.add(verdict)

    if not instrument.volumes.nano_lot_compliant:
        block("VOLUME_MIN_OR_STEP_EXCEEDS_0.0001", Verdict.TOO_COARSE)
    currency_matches = account.currency == "EUR"
    if not currency_matches:
        block("ACCOUNT_CURRENCY_NOT_EUR_NO_VERIFIED_CONVERSION")
    elif account.equity < capital_eur:
        block("ACCOUNT_EQUITY_BELOW_REQUESTED_CAPITAL", Verdict.INSUFFICIENT_MARGIN)
    if not account.trade_allowed:
        block("ACCOUNT_TRADING_NOT_ALLOWED")
    if not account.expert_allowed:
        block("EXPERT_ADVISORS_NOT_ALLOWED")
    if not instrument.full_trading_allowed:
        block("SYMBOL_NOT_FULLY_TRADABLE")
    age = Decimal(str((evaluated_at - quote.timestamp_utc).total_seconds()))
    if age > policy.max_tick_age_seconds:
        block("STALE_QUOTE")
    time_diagnostic = compare_tick_clock(
        quote.timestamp_utc, evaluated_at, float(policy.max_future_tick_seconds)
    )
    for name, value in (
        ("LEGAL_ACCESS_FRANCE", policy.legally_accessible_from_france),
        ("SCALPING", policy.scalping_allowed),
        ("HISTORICAL_TICKS", policy.historical_ticks_available),
    ):
        if value is not True:
            block(f"{name}_{'UNKNOWN' if value is None else 'NOT_ALLOWED'}")

    spread_points = (quote.ask - quote.bid) / instrument.point
    if policy.max_spread_points is None:
        block("SPREAD_LIMIT_UNCALIBRATED")
    elif spread_points > policy.max_spread_points:
        block("SPREAD_ABOVE_LIMIT", Verdict.SPREAD_TOO_HIGH)

    commission = None
    if policy.commission is None:
        block("COMMISSION_UNKNOWN")
    elif policy.commission.currency != account.currency:
        block("COMMISSION_CURRENCY_MISMATCH")
    else:
        commission = policy.commission.round_trip(instrument.volumes.minimum)
    if policy.round_trip_slippage_points is None:
        block("SLIPPAGE_UNCALIBRATED")
    if policy.max_round_trip_cost_pct is None:
        block("COST_LIMIT_UNCALIBRATED")

    margins: dict[Side, Decimal] = {}
    spread_costs: list[Decimal] = []
    slippage_costs: list[Decimal] = []
    stops: list[StopEstimate] = []
    for side in Side:
        entry = quote.ask if side == Side.BUY else quote.bid
        exit_quote = quote.bid if side == Side.BUY else quote.ask
        margin = reader.margin(side, instrument.name, instrument.volumes.minimum, entry)
        if not margin.is_finite() or margin < 0:
            raise MT5Error("order_calc_margin: invalid result")
        margins[side] = margin
        spread_costs.append(_loss(reader, snapshot, side, entry, exit_quote))
        slippage_cost = None
        if policy.round_trip_slippage_points is not None:
            distance = policy.round_trip_slippage_points * instrument.point
            adverse_price = entry - distance if side == Side.BUY else entry + distance
            if adverse_price <= 0:
                raise ValueError("slippage distance produces a nonpositive price")
            slippage_cost = _loss(reader, snapshot, side, entry, adverse_price)
            slippage_costs.append(slippage_cost)
        for pips in policy.stop_pips:
            distance = pips * pip_size
            raw_stop = exit_quote - distance if side == Side.BUY else exit_quote + distance
            rounding = ROUND_FLOOR if side == Side.BUY else ROUND_CEILING
            stop = (raw_stop / instrument.tick_size).to_integral_value(
                rounding=rounding
            ) * instrument.tick_size
            if stop <= 0:
                raise ValueError("stop distance produces a nonpositive price")
            broker_stop_valid = (
                abs(exit_quote - stop) >= instrument.stops_level_points * instrument.point
            )
            price_loss = _loss(reader, snapshot, side, entry, stop)
            if price_loss <= 0:
                raise MT5Error("order_calc_profit: stop loss must be positive")
            # Price loss already includes the Bid/Ask spread. Never add it twice.
            total = (
                price_loss + commission + slippage_cost
                if commission is not None and slippage_cost is not None
                else None
            )
            stops.append(
                StopEstimate(
                    side=side,
                    stop_pips=pips,
                    entry=entry,
                    stop_price=stop,
                    broker_stop_valid=broker_stop_valid,
                    price_loss=price_loss,
                    total_loss=total,
                    risk_pct=total / capital_eur * 100
                    if total is not None and currency_matches
                    else None,
                )
            )

    spread_cost = max(spread_costs)
    slippage = max(slippage_costs) if slippage_costs else None
    minimum_cost = spread_cost + commission if commission is not None else None
    estimated_cost = (
        minimum_cost + slippage if minimum_cost is not None and slippage is not None else None
    )
    if currency_matches:
        available_margin = min(capital_eur, account.free_margin, account.equity)
        if max(margins.values()) + policy.minimum_margin_buffer > available_margin:
            block("MINIMUM_VOLUME_MARGIN_EXCEEDS_AVAILABLE", Verdict.INSUFFICIENT_MARGIN)
        if minimum_cost is not None:
            cost_to_check = estimated_cost if estimated_cost is not None else minimum_cost
            cost_pct = cost_to_check / capital_eur * 100
            if cost_pct > policy.risk_per_trade_pct or (
                policy.max_round_trip_cost_pct is not None
                and cost_pct > policy.max_round_trip_cost_pct
            ):
                block("ROUND_TRIP_COST_EXCEEDS_BUDGET", Verdict.COST_TOO_HIGH)

    # The least risky broker-valid configured stop must work in both directions.
    valid_pips = [
        pips
        for pips in policy.stop_pips
        if all(row.broker_stop_valid for row in stops if row.stop_pips == pips)
    ]
    minimum_risk = None
    if not valid_pips:
        block("NO_CONFIGURED_STOP_MEETS_BROKER_MINIMUM")
    else:
        shortest = min(valid_pips)
        selected = [row for row in stops if row.stop_pips == shortest]
        risks = [row.risk_pct for row in selected if row.risk_pct is not None]
        minimum_risk = max(risks) if len(risks) == len(selected) else None
        if currency_matches:
            # Even if costs are unknown, a lower bound can already prove rejection.
            lower_bound = max(row.price_loss for row in selected) / capital_eur * 100
            if (minimum_risk if minimum_risk is not None else lower_bound) > min(
                policy.risk_per_trade_pct, policy.hard_max_risk_pct
            ):
                block("MINIMUM_VOLUME_STOP_RISK_EXCEEDS_BUDGET", Verdict.TOO_COARSE)

    precedence = (
        Verdict.TOO_COARSE,
        Verdict.INSUFFICIENT_MARGIN,
        Verdict.SPREAD_TOO_HIGH,
        Verdict.COST_TOO_HIGH,
        Verdict.INCOMPATIBLE_FOR_10_EUR,
    )
    verdict = next((item for item in precedence if item in failures), Verdict.COMPATIBLE)
    return BrokerCompatibilityReport(
        capital_eur=capital_eur,
        snapshot=snapshot,
        policy=policy,
        evaluated_at_utc=evaluated_at,
        nano_lot_compliant=instrument.volumes.nano_lot_compliant,
        spread_points=spread_points,
        margin_buy=margins[Side.BUY],
        margin_sell=margins[Side.SELL],
        spread_cost=spread_cost,
        round_trip_commission=commission,
        slippage_cost=slippage,
        minimum_round_trip_cost=minimum_cost,
        estimated_round_trip_cost=estimated_cost,
        minimum_risk_pct=minimum_risk,
        stops=tuple(stops),
        verdict=verdict,
        challenge_status=Verdict.INCOMPATIBLE_FOR_10_EUR if blockers else Verdict.COMPATIBLE,
        blockers=tuple(blockers),
        time_diagnostic=time_diagnostic,
        warnings=("TIME_REFERENCE_UNVERIFIED",)
        if time_diagnostic.status == "TIME_REFERENCE_UNVERIFIED"
        else (),
    )
