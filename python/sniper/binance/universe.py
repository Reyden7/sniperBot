"""Low-cost, liquid Binance Spot universe discovery."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import numpy as np

from sniper.binance.client import BinanceApiError, BinanceReadOnlyClient
from sniper.binance.costs import (
    BinanceCostModel,
    fallback_fee_schedule,
    fee_schedule_from_account,
    fee_schedules_from_trade_fee,
)
from sniper.binance.filters import (
    BinanceFilterError,
    minimum_order_quantity,
    operational_symbol_is_unambiguous,
    parse_symbol_rules,
)
from sniper.binance.models import FeeSchedule, MakerFillSimulation, UniverseCandidate
from sniper.binance.settings import BinanceSettings

ONE_HUNDRED = Decimal(100)
RATIO_FLOOR = Decimal("0.000000000001")


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _market_statistics(rows: list[list[Any]]) -> tuple[Decimal, Decimal]:
    closed = np.asarray([float(row[4]) for row in rows], dtype=float)
    high = np.asarray([float(row[2]) for row in rows], dtype=float)
    low = np.asarray([float(row[3]) for row in rows], dtype=float)
    if len(closed) < 3 or np.any(closed <= 0):
        return Decimal(0), Decimal(0)
    returns = np.diff(np.log(closed)) * 100
    realized_volatility = Decimal(str(float(np.std(returns, ddof=1))))
    expected_tradable_move = Decimal(str(float(np.median((high - low) / closed * 100))))
    return realized_volatility, expected_tradable_move


def _depth_quote(payload: dict[str, Any]) -> tuple[Decimal, Decimal]:
    bid_depth = sum(
        (_decimal(price) * _decimal(quantity) for price, quantity in payload["bids"]),
        Decimal(0),
    )
    ask_depth = sum(
        (_decimal(price) * _decimal(quantity) for price, quantity in payload["asks"]),
        Decimal(0),
    )
    return bid_depth, ask_depth


def _book_slippage_pct(
    payload: dict[str, Any], reference_notional: Decimal
) -> tuple[Decimal, bool]:
    """Walk both sides for a fixed quote notional and return round-trip impact."""
    asks = [(_decimal(price), _decimal(quantity)) for price, quantity in payload["asks"]]
    bids = [(_decimal(price), _decimal(quantity)) for price, quantity in payload["bids"]]
    if not asks or not bids or asks[0][0] <= 0 or bids[0][0] <= 0:
        return Decimal(0), False

    quote_left = reference_notional
    base_bought = Decimal(0)
    for price, quantity in asks:
        take_quote = min(quote_left, price * quantity)
        base_bought += take_quote / price
        quote_left -= take_quote
        if quote_left <= 0:
            break
    buy_complete = quote_left <= 0 and base_bought > 0
    buy_average = reference_notional / base_bought if buy_complete else asks[-1][0]
    buy_impact = max(buy_average / asks[0][0] - 1, Decimal(0))

    target_base = reference_notional / bids[0][0]
    base_left = target_base
    proceeds = Decimal(0)
    for price, quantity in bids:
        take_base = min(base_left, quantity)
        proceeds += take_base * price
        base_left -= take_base
        if base_left <= 0:
            break
    sell_complete = base_left <= 0 and target_base > 0
    sell_average = proceeds / target_base if sell_complete else bids[-1][0]
    sell_impact = max(1 - sell_average / bids[0][0], Decimal(0))
    return (buy_impact + sell_impact) * ONE_HUNDRED, buy_complete and sell_complete


def _account_balances(account: dict[str, Any] | None) -> dict[str, Decimal]:
    if account is None:
        return {}
    return {
        str(item["asset"]): _decimal(item["free"])
        for item in account.get("balances", [])
        if _decimal(item["free"]) > 0
    }


def resolve_fee_schedules(
    client: BinanceReadOnlyClient,
    settings: BinanceSettings,
    symbols: list[str],
    account: dict[str, Any] | None,
) -> tuple[dict[str, FeeSchedule], list[str]]:
    """Prefer current symbol/account fees and make every fallback explicit."""
    problems: list[str] = []
    schedules: dict[str, FeeSchedule] = {}
    if settings.authenticated:
        try:
            all_account_schedules = fee_schedules_from_trade_fee(client.trade_fees())
            schedules = {
                symbol: all_account_schedules[symbol]
                for symbol in symbols
                if symbol in all_account_schedules
            }
        except (BinanceApiError, KeyError, ValueError) as exc:
            problems.append(f"ACCOUNT_TRADE_FEE_UNAVAILABLE:{type(exc).__name__}")
            if account is not None and account.get("commissionRates"):
                for symbol in symbols:
                    schedules[symbol] = fee_schedule_from_account(symbol, account)
    for symbol in symbols:
        if symbol not in schedules:
            schedules[symbol] = fallback_fee_schedule(symbol, settings)
    if any(not item.is_account_specific for item in schedules.values()):
        problems.append("CONFIGURED_NONZERO_FEE_FALLBACK_USED")
    return schedules, problems


def _has_visible_special_pricing(payload: dict[str, Any]) -> bool:
    special = payload.get("specialCommission", {})
    tax = payload.get("taxCommission", {})
    discount = payload.get("discount", {})
    nonzero_special = any(_decimal(value) != 0 for value in special.values())
    nonzero_tax = any(_decimal(value) != 0 for value in tax.values())
    active_discount = bool(discount.get("enabledForAccount") and discount.get("enabledForSymbol"))
    return nonzero_special or nonzero_tax or active_discount


def _effective_fee_reduction_visible(schedule: FeeSchedule, payload: dict[str, Any]) -> bool:
    if schedule.maker_rate == 0 or schedule.taker_rate == 0:
        return True
    standard = payload.get("standardCommission", {})
    standard_maker = _decimal(standard.get("maker", schedule.maker_rate))
    standard_taker = _decimal(standard.get("taker", schedule.taker_rate))
    return schedule.maker_rate < standard_maker or schedule.taker_rate < standard_taker


def enrich_fee_schedules(
    client: BinanceReadOnlyClient,
    settings: BinanceSettings,
    schedules: dict[str, FeeSchedule],
) -> tuple[dict[str, FeeSchedule], list[str]]:
    """Attach current per-symbol special/tax/discount information when accessible."""
    if not settings.authenticated:
        return schedules, []
    commission_reader = getattr(client, "commission_rates", None)
    if commission_reader is None:
        return schedules, ["ACCOUNT_COMMISSION_DETAILS_UNAVAILABLE:CLIENT_METHOD_MISSING"]
    problems: list[str] = []
    enriched = dict(schedules)
    for symbol, schedule in schedules.items():
        if not schedule.is_account_specific:
            continue
        try:
            details = commission_reader(symbol)
        except (BinanceApiError, KeyError, ValueError) as exc:
            problems.append(f"ACCOUNT_COMMISSION_DETAILS_UNAVAILABLE:{symbol}:{type(exc).__name__}")
            continue
        enriched[symbol] = schedule.model_copy(
            update={
                "commission_details": details,
                "special_pricing_visible": _has_visible_special_pricing(details)
                or _effective_fee_reduction_visible(schedule, details),
            }
        )
    return enriched, problems


class LowCostCryptoUniverseScanner:
    """Rank liquid Spot pairs by tradable movement relative to complete costs."""

    def __init__(
        self,
        client: BinanceReadOnlyClient,
        settings: BinanceSettings,
        *,
        maker_fill_simulations: dict[str, MakerFillSimulation] | None = None,
    ):
        self.client = client
        self.settings = settings
        self.maker_fill_simulations = maker_fill_simulations or {}

    def _maker_fill_is_credible(self, symbol: str) -> bool:
        evidence = self.maker_fill_simulations.get(symbol)
        return bool(
            evidence is not None
            and evidence.observations >= self.settings.maker_fill_minimum_observations
            and evidence.fill_probability >= self.settings.maker_fill_minimum_probability
            and evidence.queue_position_modeled
            and evidence.post_only_modeled
        )

    def scan(
        self,
    ) -> tuple[list[UniverseCandidate], tuple[str, ...], dict[str, Any] | None, list[str]]:
        problems: list[str] = []
        exchange = self.client.exchange_info()
        book_map = {item["symbol"]: item for item in self.client.book_tickers()}
        ticker_map = {item["symbol"]: item for item in self.client.ticker_24h()}
        account = None
        if self.settings.authenticated:
            try:
                account = self.client.account_information()
            except BinanceApiError as exc:
                problems.append(f"ACCOUNT_UNAVAILABLE:{type(exc).__name__}")

        eligible_rules = []
        for raw_symbol in exchange.get("symbols", []):
            try:
                rules = parse_symbol_rules(raw_symbol)
            except BinanceFilterError, KeyError, ValueError:
                continue
            if not operational_symbol_is_unambiguous(rules):
                problems.append(
                    f"AMBIGUOUS_SYMBOL_EXCLUDED:{rules.symbol.encode('unicode_escape').decode()}"
                )
                continue
            if (
                rules.quote_asset in self.settings.eligible_quote_assets
                and rules.status == "TRADING"
                and rules.spot_trading_allowed
                and rules.symbol in book_map
                and rules.symbol in ticker_map
            ):
                eligible_rules.append(rules)
        quote_assets = tuple(sorted({rules.quote_asset for rules in eligible_rules}))

        liquid_rules = [
            rules
            for rules in eligible_rules
            if _decimal(ticker_map[rules.symbol].get("quoteVolume", 0))
            >= self.settings.minimum_quote_volume_24h
        ]
        liquid_rules.sort(
            key=lambda rules: _decimal(ticker_map[rules.symbol].get("quoteVolume", 0)),
            reverse=True,
        )
        priority = set(self.settings.preferred_base_assets)
        priority_rules = [rules for rules in liquid_rules if rules.base_asset in priority]
        other_rules = [rules for rules in liquid_rules if rules.base_asset not in priority]
        remaining = max(self.settings.maximum_diagnostic_pairs - len(priority_rules), 0)
        shortlisted = priority_rules + other_rules[:remaining]
        if len(shortlisted) < len(liquid_rules):
            problems.append(f"DIAGNOSTIC_PAIR_LIMIT_APPLIED:{len(shortlisted)}/{len(liquid_rules)}")
        for base_asset in self.settings.preferred_base_assets:
            if not any(rules.base_asset == base_asset for rules in shortlisted):
                problems.append(f"NO_LIQUID_TRADABLE_PAIR:{base_asset}")

        fee_schedules, fee_problems = resolve_fee_schedules(
            self.client,
            self.settings,
            [rules.symbol for rules in shortlisted],
            account,
        )
        problems.extend(fee_problems)
        fee_schedules, detail_problems = enrich_fee_schedules(
            self.client, self.settings, fee_schedules
        )
        problems.extend(detail_problems)
        balances = _account_balances(account)
        candidates: list[UniverseCandidate] = []
        for rules in shortlisted:
            book = book_map[rules.symbol]
            ticker = ticker_map[rules.symbol]
            bid = _decimal(book["bidPrice"])
            ask = _decimal(book["askPrice"])
            if bid <= 0 or ask <= 0 or ask < bid:
                problems.append(f"INVALID_BOOK:{rules.symbol}")
                continue
            try:
                klines = self.client.klines(rules.symbol, "5m", 100)
                volatility, expected_move = _market_statistics(klines)
                depth_payload = self.client.depth(rules.symbol, 20)
                bid_depth, ask_depth = _depth_quote(depth_payload)
                walked_slippage, book_can_fill_reference = _book_slippage_pct(
                    depth_payload, self.settings.slippage_reference_notional
                )
                minimum_quantity = minimum_order_quantity(rules, ask, market=True)
            except (BinanceApiError, BinanceFilterError, KeyError, ValueError) as exc:
                problems.append(f"MARKET_DIAGNOSTIC_FAILED:{rules.symbol}:{type(exc).__name__}")
                continue
            depth = min(bid_depth, ask_depth)
            if depth < self.settings.minimum_top20_depth_quote or not book_can_fill_reference:
                problems.append(f"INSUFFICIENT_LIQUIDITY:{rules.symbol}")
                continue

            slippage_floor = self.settings.slippage_bps_per_side * Decimal(2) / ONE_HUNDRED
            estimated_slippage = max(walked_slippage, slippage_floor)
            minimum_order = minimum_quantity * ask
            available = balances.get(rules.quote_asset)
            compatible = (
                None
                if account is None
                else bool(available is not None and available >= minimum_order)
            )
            fee = fee_schedules[rules.symbol]
            maker_evidence = self.maker_fill_simulations.get(rules.symbol)
            maker_credible = self._maker_fill_is_credible(rules.symbol)
            maker_adverse_selection = (
                maker_evidence.adverse_selection_pct if maker_evidence is not None else Decimal(0)
            )
            costs = BinanceCostModel(
                fee,
                slippage_bps_per_side=self.settings.slippage_bps_per_side,
                uncertainty_buffer_bps=self.settings.uncertainty_buffer_bps,
            ).compare_execution_modes(
                expected_gross_move_pct=expected_move,
                bid=bid,
                ask=ask,
                estimated_slippage_pct=estimated_slippage,
                maker_adverse_selection_pct=maker_adverse_selection,
            )
            spread_pct = (ask - bid) / ((ask + bid) / 2) * ONE_HUNDRED
            ratio_taker = expected_move / max(costs.taker_taker.estimated_cost_pct, RATIO_FLOOR)
            ratio_maker = expected_move / max(costs.maker_taker.estimated_cost_pct, RATIO_FLOOR)
            capital_compatible = compatible is not False
            maker_edge_positive = costs.maker_taker.acceptable
            taker_edge_positive = costs.taker_taker.acceptable
            maker_allowed = capital_compatible and maker_credible and maker_edge_positive
            taker_allowed = capital_compatible and taker_edge_positive
            if maker_allowed and (
                not taker_allowed
                or costs.maker_taker.expected_net_edge_pct > costs.taker_taker.expected_net_edge_pct
            ):
                preferred_mode = "MAKER_ENTRY_TAKER_EXIT"
                selected_edge = costs.maker_taker.expected_net_edge_pct
            elif taker_allowed:
                preferred_mode = "TAKER_ENTRY_TAKER_EXIT"
                selected_edge = costs.taker_taker.expected_net_edge_pct
            else:
                preferred_mode = None
                selected_edge = costs.taker_taker.expected_net_edge_pct

            quote_volume = _decimal(ticker["quoteVolume"])
            liquidity_score = Decimal(str(np.log10(float(max(quote_volume, Decimal(1))))))
            depth_score = Decimal(str(np.log10(float(max(depth, Decimal(1))))))
            spread_score = Decimal(1) / (Decimal(1) + spread_pct * Decimal(100))
            effective_fee_rate = (
                fee.maker_rate + fee.taker_rate if maker_credible else fee.taker_rate * Decimal(2)
            )
            fee_score = Decimal(1) / (Decimal(1) + effective_fee_rate * Decimal(10000))
            movement_score = min(expected_move, Decimal(5))
            effective_ratio = ratio_maker if maker_credible else ratio_taker
            ratio_score = min(effective_ratio, Decimal(20))
            ranking_score = (
                liquidity_score * Decimal(2)
                + depth_score
                + spread_score * Decimal(3)
                + fee_score * Decimal(2)
                + movement_score
                + ratio_score * Decimal(4)
            )
            reasons = [
                "SPOT_TRADING",
                "LIQUIDITY_THRESHOLDS_PASSED",
                "DYNAMIC_EXCHANGE_FILTERS",
                "DYNAMIC_DEPTH_SLIPPAGE",
                "M5_EXPECTED_TRADABLE_MOVE_PROXY",
                (
                    "POSITIVE_NET_EDGE_TAKER"
                    if taker_edge_positive
                    else "NO_TAKER_EDGE_AFTER_BUFFER"
                ),
            ]
            reasons.append(
                "MAKER_FILL_SIMULATION_CREDIBLE"
                if maker_credible
                else "MAKER_MODE_DISABLED_WITHOUT_CREDIBLE_FILL_SIMULATION"
            )
            if fee.special_pricing_visible:
                reasons.append("ACCOUNT_SPECIAL_PRICING_VISIBLE")
            if compatible is False:
                reasons.append("AVAILABLE_BALANCE_BELOW_MINIMUM_ORDER")
                reasons.append("NO_TRADE_CAPITAL_INCOMPATIBLE")

            candidates.append(
                UniverseCandidate(
                    symbol=rules.symbol,
                    base_asset=rules.base_asset,
                    quote_asset=rules.quote_asset,
                    maker_fee=fee.maker_rate,
                    taker_fee=fee.taker_rate,
                    spread_pct=spread_pct,
                    estimated_slippage_pct=estimated_slippage,
                    maker_taker_round_trip_cost_pct=costs.maker_taker.estimated_cost_pct,
                    taker_taker_round_trip_cost_pct=costs.taker_taker.estimated_cost_pct,
                    quote_volume_24h=quote_volume,
                    depth=depth,
                    expected_move_pct=expected_move,
                    move_to_cost_ratio_maker=ratio_maker,
                    move_to_cost_ratio_taker=ratio_taker,
                    expected_net_edge_taker_pct=costs.taker_taker.expected_net_edge_pct,
                    expected_net_edge_maker_pct=costs.maker_taker.expected_net_edge_pct,
                    maker_fill_simulation_credible=maker_credible,
                    preferred_execution_mode=preferred_mode,
                    trade_allowed=preferred_mode is not None,
                    status=rules.status,
                    rank=1,
                    opportunity_score=ranking_score,
                    expected_net_edge_pct=selected_edge,
                    expected_gross_move_pct=expected_move,
                    estimated_cost_pct=costs.taker_taker.estimated_cost_pct,
                    fee_source=fee.source,
                    maker_fee_rate=fee.maker_rate,
                    taker_fee_rate=fee.taker_rate,
                    commission_details=fee.commission_details,
                    special_pricing_visible=fee.special_pricing_visible,
                    bid=bid,
                    ask=ask,
                    spread_bps=spread_pct * ONE_HUNDRED,
                    trades_count_24h=int(ticker.get("count", 0)),
                    realized_volatility_m5_pct=volatility,
                    top20_bid_depth_quote=bid_depth,
                    top20_ask_depth_quote=ask_depth,
                    minimum_quantity=minimum_quantity,
                    quantity_step=rules.market_quantity_step,
                    minimum_notional=rules.minimum_notional,
                    minimum_order_at_ask_quote=minimum_order,
                    tick_size=rules.tick_size,
                    available_quote_balance=available,
                    compatible_with_available_balance=compatible,
                    ranking_components={
                        "liquidity": liquidity_score,
                        "depth": depth_score,
                        "spread": spread_score,
                        "fees": fee_score,
                        "movement": movement_score,
                        "move_to_total_cost": ratio_score,
                    },
                    reasons=tuple(reasons),
                )
            )
        candidates.sort(key=lambda item: (item.trade_allowed, item.opportunity_score), reverse=True)
        ranked = [item.model_copy(update={"rank": rank}) for rank, item in enumerate(candidates, 1)]
        if not ranked:
            problems.append("NO_ELIGIBLE_LIQUID_CRYPTO_SYMBOL")
        return ranked, quote_assets, account, problems


class CryptoUniverseScanner(LowCostCryptoUniverseScanner):
    """Backward-compatible name for the low-cost universe scanner."""
