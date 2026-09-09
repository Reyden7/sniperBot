"""Frozen Binance Spot historical replay and qualification protocol."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal

from pydantic import Field

from sniper.binance.models import SymbolRules
from sniper.binance.opportunity import EconomicTradeFilter, OpportunityEngine
from sniper.binance.risk_engine import BinanceRiskEngine, DailyPerformanceEngine, DailyState
from sniper.binance.strategy_engine import ResearchBar, StrategySignal
from sniper.config import Model

Verdict = Literal[
    "BINANCE_STRATEGY_REJECTED",
    "BINANCE_STRATEGY_INSUFFICIENT_EVIDENCE",
    "BINANCE_STRATEGY_QUALIFIED_FOR_PAPER",
]


class CryptoTrade(Model):
    symbol: str
    setup_type: str
    entry_time_utc: datetime
    exit_time_utc: datetime
    quantity: Decimal
    entry_reference: Decimal
    entry_executed: Decimal
    exit_reference: Decimal
    exit_executed: Decimal
    stop_price: Decimal
    target_price: Decimal
    exit_reason: str
    market_pnl: Decimal
    spread_cost: Decimal = Field(ge=0)
    slippage_cost: Decimal = Field(ge=0)
    commission: Decimal = Field(ge=0)
    net_pnl: Decimal
    net_pnl_crosscheck: Decimal
    mfe_pct: Decimal
    mae_pct: Decimal


class ScenarioMetrics(Model):
    scenario: Literal["BASE", "STRESS"]
    starting_capital_eur: Decimal
    ending_capital_eur: Decimal
    net_return_pct: Decimal
    trades: int
    wins: int
    losses: int
    expectancy_eur: Decimal
    expectancy_pct: Decimal
    profit_factor: Decimal | None
    max_drawdown_pct: Decimal = Field(ge=0)
    data_weeks: int
    active_trade_weeks: int
    positive_week_ratio: Decimal
    maximum_crypto_profit_concentration: Decimal
    maximum_week_profit_concentration: Decimal
    days: int
    days_target_reached: int
    days_between_zero_and_target: int
    negative_days: int
    no_trade_days: int
    average_daily_return_pct: Decimal
    median_daily_return_pct: Decimal
    best_day_pct: Decimal
    worst_day_pct: Decimal
    gross_market_pnl: Decimal
    spread_cost: Decimal = Field(ge=0)
    slippage_cost: Decimal = Field(ge=0)
    commission: Decimal = Field(ge=0)
    exit_reasons: dict[str, int]
    setup_counts: dict[str, int]
    rejection_counts: dict[str, int]
    one_position_invariant: bool
    pnl_crosscheck_passed: bool


class BinanceQualificationReport(Model):
    protocol_id: str = "BINANCE_SPOT_V1_FROZEN_2026_09_09"
    generated_at_utc: datetime
    start_utc: datetime
    end_utc: datetime
    symbols: tuple[str, ...]
    bars_m1: int
    bars_m5: int
    signal_observations: int
    data_quality: dict[str, Any]
    account_fee_source: Literal["BINANCE_ACCOUNT_API"] = "BINANCE_ACCOUNT_API"
    maker_fee_rate: Decimal
    taker_fee_rate: Decimal
    historical_spread_source: str
    base_assumptions: dict[str, str]
    stress_assumptions: dict[str, str]
    capital_scenarios_eur: tuple[Decimal, ...]
    results: tuple[ScenarioMetrics, ...]
    qualification_checks: dict[str, bool]
    limitations: tuple[str, ...]
    verdict: Verdict
    live_trading_enabled: Literal[False] = False
    order_endpoints_present: Literal[False] = False


@dataclass
class _Position:
    signal: StrategySignal
    quantity: Decimal
    entry_time: datetime
    entry_reference: Decimal
    entry_executed: Decimal
    stop: Decimal
    target: Decimal
    entry_fee: Decimal
    spread_bps: Decimal
    maximum_reference: Decimal
    minimum_reference: Decimal


def _d(value: float | int | str) -> Decimal:
    return Decimal(str(value))


def _median(values: list[Decimal]) -> Decimal:
    if not values:
        return Decimal(0)
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def _scenario_metrics(
    *,
    scenario: Literal["BASE", "STRESS"],
    capital: Decimal,
    trades: list[CryptoTrade],
    equity_curve: list[Decimal],
    start_utc: datetime,
    end_utc: datetime,
    daily_returns: dict[date, Decimal],
    rejection_counts: Counter[str],
) -> ScenarioMetrics:
    ending = equity_curve[-1]
    wins = [trade.net_pnl for trade in trades if trade.net_pnl > 0]
    losses = [-trade.net_pnl for trade in trades if trade.net_pnl < 0]
    peak = capital
    drawdown = Decimal(0)
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            drawdown = max(drawdown, (peak - value) / peak * Decimal(100))
    week_pnl: dict[str, Decimal] = defaultdict(Decimal)
    symbol_profit: dict[str, Decimal] = defaultdict(Decimal)
    for trade in trades:
        week = trade.exit_time_utc.strftime("%G-W%V")
        week_pnl[week] += trade.net_pnl
        if trade.net_pnl > 0:
            symbol_profit[trade.symbol] += trade.net_pnl
    positive_total = sum(symbol_profit.values(), Decimal(0))
    positive_weeks = [value for value in week_pnl.values() if value > 0]
    profitable_week_total = sum(positive_weeks, Decimal(0))
    all_days = (end_utc.date() - start_utc.date()).days
    returns = [
        daily_returns.get(start_utc.date() + timedelta(days=index), Decimal(0))
        for index in range(all_days)
    ]
    traded_days = set(daily_returns)
    return ScenarioMetrics(
        scenario=scenario,
        starting_capital_eur=capital,
        ending_capital_eur=ending,
        net_return_pct=(ending - capital) / capital * Decimal(100),
        trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        expectancy_eur=sum((trade.net_pnl for trade in trades), Decimal(0)) / max(len(trades), 1),
        expectancy_pct=(ending - capital) / capital * Decimal(100) / max(len(trades), 1),
        profit_factor=(sum(wins, Decimal(0)) / sum(losses, Decimal(0))) if losses else None,
        max_drawdown_pct=drawdown,
        data_weeks=len(
            {
                (start_utc.date() + timedelta(days=index)).strftime("%G-W%V")
                for index in range(all_days)
            }
        ),
        active_trade_weeks=len(week_pnl),
        positive_week_ratio=Decimal(len(positive_weeks)) / max(len(week_pnl), 1),
        maximum_crypto_profit_concentration=(
            max(symbol_profit.values(), default=Decimal(0)) / positive_total
            if positive_total > 0
            else Decimal(1)
        ),
        maximum_week_profit_concentration=(
            max(positive_weeks, default=Decimal(0)) / profitable_week_total
            if profitable_week_total > 0
            else Decimal(1)
        ),
        days=all_days,
        days_target_reached=sum(value >= Decimal("1") for value in returns),
        days_between_zero_and_target=sum(Decimal(0) < value < Decimal("1") for value in returns),
        negative_days=sum(value < 0 for value in returns),
        no_trade_days=all_days - len(traded_days),
        average_daily_return_pct=sum(returns, Decimal(0)) / max(all_days, 1),
        median_daily_return_pct=_median(returns),
        best_day_pct=max(returns, default=Decimal(0)),
        worst_day_pct=min(returns, default=Decimal(0)),
        gross_market_pnl=sum((trade.market_pnl for trade in trades), Decimal(0)),
        spread_cost=sum((trade.spread_cost for trade in trades), Decimal(0)),
        slippage_cost=sum((trade.slippage_cost for trade in trades), Decimal(0)),
        commission=sum((trade.commission for trade in trades), Decimal(0)),
        exit_reasons=dict(Counter(trade.exit_reason for trade in trades)),
        setup_counts=dict(Counter(trade.setup_type for trade in trades)),
        rejection_counts=dict(rejection_counts),
        one_position_invariant=True,
        pnl_crosscheck_passed=all(
            abs(trade.net_pnl - trade.net_pnl_crosscheck) <= Decimal("0.00000001")
            for trade in trades
        ),
    )


def replay_spot(
    *,
    bars_by_symbol: dict[str, list[ResearchBar]],
    signals_by_symbol: dict[str, list[StrategySignal]],
    rules_by_symbol: dict[str, SymbolRules],
    spreads_bps: dict[str, Decimal],
    capital: Decimal,
    fee_rate: Decimal,
    scenario: Literal["BASE", "STRESS"],
    start_utc: datetime,
    end_utc: datetime,
) -> tuple[ScenarioMetrics, list[CryptoTrade]]:
    """Replay a single global long-only Spot position with executable sides."""
    slippage_bps = Decimal("1") if scenario == "BASE" else Decimal("3")
    spread_multiplier = Decimal(1) if scenario == "BASE" else Decimal(2)
    effective_fee = fee_rate if scenario == "BASE" else fee_rate * Decimal("1.5")
    buffer_pct = Decimal("0.02") if scenario == "BASE" else Decimal("0.05")
    bar_maps = {
        symbol: {bar.timestamp_utc: bar for bar in bars} for symbol, bars in bars_by_symbol.items()
    }
    signal_maps = {
        symbol: {signal.timestamp_utc: signal for signal in signals}
        for symbol, signals in signals_by_symbol.items()
    }
    timeline = sorted({timestamp for mapping in bar_maps.values() for timestamp in mapping})
    equity = capital
    equity_curve = [equity]
    daily = DailyPerformanceEngine()
    risk = BinanceRiskEngine()
    opportunity = OpportunityEngine(EconomicTradeFilter(buffer_pct))
    position: _Position | None = None
    trades: list[CryptoTrade] = []
    daily_start_equity: dict[date, Decimal] = {}
    daily_realized: dict[date, Decimal] = defaultdict(Decimal)
    rejections: Counter[str] = Counter()

    def close_position(timestamp: datetime, reference: Decimal, reason: str) -> None:
        nonlocal position, equity
        if position is None:
            return
        half_spread = position.spread_bps * spread_multiplier / Decimal(20000)
        slippage = slippage_bps / Decimal(10000)
        bid_reference = reference * (Decimal(1) - half_spread)
        exit_executed = bid_reference * (Decimal(1) - slippage)
        exit_fee = position.quantity * exit_executed * effective_fee
        commission = position.entry_fee + exit_fee
        net = position.quantity * (exit_executed - position.entry_executed) - commission
        market_pnl = position.quantity * (reference - position.entry_reference)
        spread_cost = position.quantity * (
            position.entry_reference * half_spread + reference * half_spread
        )
        slippage_cost = position.quantity * (
            position.entry_reference * (Decimal(1) + half_spread) * slippage
            + bid_reference * slippage
        )
        crosscheck = market_pnl - spread_cost - slippage_cost - commission
        trade = CryptoTrade(
            symbol=position.signal.symbol,
            setup_type=position.signal.setup_type.value,
            entry_time_utc=position.entry_time,
            exit_time_utc=timestamp,
            quantity=position.quantity,
            entry_reference=position.entry_reference,
            entry_executed=position.entry_executed,
            exit_reference=reference,
            exit_executed=exit_executed,
            stop_price=position.stop,
            target_price=position.target,
            exit_reason=reason,
            market_pnl=market_pnl,
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            commission=commission,
            net_pnl=net,
            net_pnl_crosscheck=crosscheck,
            mfe_pct=(position.maximum_reference - position.entry_reference)
            / position.entry_reference
            * Decimal(100),
            mae_pct=(position.minimum_reference - position.entry_reference)
            / position.entry_reference
            * Decimal(100),
        )
        trades.append(trade)
        equity += net
        equity_curve.append(equity)
        daily.record(net, commission)
        daily_realized[timestamp.date()] += net
        position = None

    for timestamp in timeline:
        daily.synchronize(timestamp.date(), equity)
        daily_start_equity.setdefault(timestamp.date(), equity)
        if position is not None:
            bar = bar_maps[position.signal.symbol].get(timestamp)
            if bar is not None:
                position.maximum_reference = max(position.maximum_reference, _d(bar.high))
                position.minimum_reference = min(position.minimum_reference, _d(bar.low))
                spread = position.spread_bps * spread_multiplier / Decimal(20000)
                bid_low = _d(bar.low) * (Decimal(1) - spread)
                bid_high = _d(bar.high) * (Decimal(1) - spread)
                if bid_low <= position.stop:
                    close_position(timestamp, position.stop, "STOP")
                elif bid_high >= position.target:
                    close_position(timestamp, position.target, "TARGET")
                elif timestamp - position.entry_time >= timedelta(minutes=60):
                    close_position(timestamp, _d(bar.close), "TIMEOUT")
        if position is not None or daily.state != DailyState.ACTIVE:
            continue
        raw_candidates: list[tuple[StrategySignal, ResearchBar, Decimal]] = []
        for symbol, mapping in signal_maps.items():
            signal = mapping.get(timestamp)
            bar = bar_maps[symbol].get(timestamp)
            if signal is None or bar is None:
                continue
            if abs(bar.open - signal.entry_reference) > signal.atr * 0.25:
                rejections["OPEN_GAP"] += 1
                continue
            raw_candidates.append((signal, bar, spreads_bps[symbol] * spread_multiplier))
        ranked = opportunity.rank(
            raw_candidates,
            fee_rate=effective_fee,
            slippage_bps_per_side=slippage_bps,
        )
        rejections["ECONOMIC_TRADE_FILTER"] += sum(not item.accepted for item in ranked)
        accepted = [item for item in ranked if item.accepted]
        if not accepted:
            continue
        signal = accepted[0].signal
        entry_bar = accepted[0].entry_bar
        spread_bps = spreads_bps[signal.symbol]
        half_spread = spread_bps * spread_multiplier / Decimal(20000)
        slippage = slippage_bps / Decimal(10000)
        entry_reference = _d(entry_bar.open)
        entry_executed = entry_reference * (Decimal(1) + half_spread) * (Decimal(1) + slippage)
        stop_distance = _d(signal.entry_reference - signal.invalid_level)
        target_distance = _d(signal.target_reference - signal.entry_reference)
        stop = entry_reference - stop_distance
        target = entry_reference + target_distance
        decision = risk.size(
            equity=equity,
            available_cash=equity / (Decimal(1) + effective_fee),
            entry_price=entry_executed,
            stop_price=stop,
            rules=rules_by_symbol[signal.symbol],
            round_trip_fee_rate=effective_fee * Decimal(2),
            round_trip_slippage_rate=slippage * Decimal(2),
        )
        if not decision.accepted:
            rejections[decision.reason] += 1
            continue
        entry_fee = decision.quantity * entry_executed * effective_fee
        position = _Position(
            signal=signal,
            quantity=decision.quantity,
            entry_time=timestamp,
            entry_reference=entry_reference,
            entry_executed=entry_executed,
            stop=stop,
            target=target,
            entry_fee=entry_fee,
            spread_bps=spread_bps,
            maximum_reference=entry_reference,
            minimum_reference=entry_reference,
        )
        # The entry M5 bar occurs after the causal signal and may immediately hit a barrier.
        position.maximum_reference = max(position.maximum_reference, _d(entry_bar.high))
        position.minimum_reference = min(position.minimum_reference, _d(entry_bar.low))
        effective_half_spread = spread_bps * spread_multiplier / Decimal(20000)
        if _d(entry_bar.low) * (Decimal(1) - effective_half_spread) <= stop:
            close_position(timestamp, stop, "STOP")
        elif _d(entry_bar.high) * (Decimal(1) - effective_half_spread) >= target:
            close_position(timestamp, target, "TARGET")
    if position is not None:
        last_bar = bar_maps[position.signal.symbol][max(bar_maps[position.signal.symbol])]
        close_position(last_bar.timestamp_utc, _d(last_bar.close), "END_OF_DATA")
    daily_returns = {
        day: pnl / daily_start_equity[day] * Decimal(100)
        for day, pnl in daily_realized.items()
        if daily_start_equity[day] > 0
    }
    return (
        _scenario_metrics(
            scenario=scenario,
            capital=capital,
            trades=trades,
            equity_curve=equity_curve,
            start_utc=start_utc,
            end_utc=end_utc,
            daily_returns=daily_returns,
            rejection_counts=rejections,
        ),
        trades,
    )


def decide_verdict(
    base: ScenarioMetrics, stress: ScenarioMetrics
) -> tuple[Verdict, dict[str, bool]]:
    """Apply frozen pre-evaluation gates without post-hoc threshold selection."""
    checks = {
        "net_expectancy_positive": base.expectancy_eur > 0,
        "net_profit_factor_gte_1_20": base.profit_factor is not None
        and base.profit_factor >= Decimal("1.20"),
        "minimum_150_trades": base.trades >= 150,
        "minimum_8_active_weeks": base.active_trade_weeks >= 8,
        "positive_weeks_gte_60pct": base.positive_week_ratio >= Decimal("0.60"),
        "crypto_profit_concentration_lte_60pct": (
            base.maximum_crypto_profit_concentration <= Decimal("0.60")
        ),
        "week_profit_concentration_lte_60pct": (
            base.maximum_week_profit_concentration <= Decimal("0.60")
        ),
        "stress_expectancy_positive": stress.expectancy_eur > 0,
        "stress_profit_factor_gte_1": stress.profit_factor is not None
        and stress.profit_factor >= Decimal(1),
        "pnl_crosscheck": base.pnl_crosscheck_passed and stress.pnl_crosscheck_passed,
    }
    if not checks["minimum_150_trades"] or not checks["minimum_8_active_weeks"]:
        return "BINANCE_STRATEGY_INSUFFICIENT_EVIDENCE", checks
    if all(checks.values()):
        return "BINANCE_STRATEGY_QUALIFIED_FOR_PAPER", checks
    return "BINANCE_STRATEGY_REJECTED", checks
