"""Current-account low-cost market validation for Binance Spot V2."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from statistics import median
from time import sleep
from typing import Any, Literal

import numpy as np
from pydantic import Field

from sniper.binance.client import BinanceReadOnlyClient
from sniper.binance.models import UniverseCandidate
from sniper.binance.settings import BinanceSettings
from sniper.binance.universe import LowCostCryptoUniverseScanner, _book_slippage_pct
from sniper.config import Model

HORIZON_FACTORS = {"5m": 1, "15m": 3, "30m": 6, "1h": 12, "4h": 48}
STRATEGIC_HORIZONS = ("30m", "1h", "4h")

EconomicCategory = Literal["ECONOMICALLY_ATTRACTIVE", "MARGINAL", "TOO_EXPENSIVE"]


class V2PairMarketProfile(Model):
    rank: int = Field(ge=1)
    symbol: str
    base_asset: str
    quote_asset: str
    category: EconomicCategory
    maker_fee: Decimal = Field(ge=0)
    taker_fee: Decimal = Field(ge=0)
    fee_source: str
    discount: dict[str, Any] = Field(default_factory=dict)
    special_commission: dict[str, Any] = Field(default_factory=dict)
    tax_commission: dict[str, Any] = Field(default_factory=dict)
    spread_average_pct: Decimal = Field(ge=0)
    spread_p50_pct: Decimal = Field(ge=0)
    spread_p95_pct: Decimal = Field(ge=0)
    slippage_pct_by_notional: dict[str, Decimal]
    quote_volume_24h: Decimal = Field(ge=0)
    trades_count_24h: int = Field(ge=0)
    top20_bid_depth_quote: Decimal = Field(ge=0)
    top20_ask_depth_quote: Decimal = Field(ge=0)
    volatility_pct_by_horizon: dict[str, Decimal]
    expected_move_pct_by_horizon: dict[str, Decimal]
    taker_taker_round_trip_cost_pct: Decimal = Field(ge=0)
    maker_taker_round_trip_cost_pct: Decimal = Field(ge=0)
    movement_to_cost_ratio_by_horizon: dict[str, Decimal]
    expected_net_edge_pct_by_horizon: dict[str, Decimal]
    maximum_strategic_ratio: Decimal = Field(ge=0)
    liquidity_sufficient: bool
    spread_acceptable: bool
    slippage_acceptable: bool
    maker_cost_is_diagnostic_only: Literal[True] = True
    tick_size: Decimal = Field(gt=0)
    quantity_step: Decimal = Field(gt=0)
    minimum_notional: Decimal = Field(ge=0)


class V2AssetComparison(Model):
    base_asset: str
    best_symbol: str
    quote_asset: str
    movement_over_fees: Decimal = Field(ge=0)
    movement_over_total_cost: Decimal = Field(ge=0)
    spread_p50_pct: Decimal = Field(ge=0)
    quote_volume_24h: Decimal = Field(ge=0)
    slippage_500_pct: Decimal = Field(ge=0)
    category: EconomicCategory


class LowCostMarketReport(Model):
    protocol_id: Literal["BINANCE_V2_LOW_COST_UNIVERSE_2026_09_10"] = (
        "BINANCE_V2_LOW_COST_UNIVERSE_2026_09_10"
    )
    generated_at_utc: datetime
    authenticated_account_fees: Literal[True] = True
    live_trading_enabled: Literal[False] = False
    order_endpoints_present: Literal[False] = False
    ranking_method: Literal["MOVEMENT_OVER_TOTAL_TAKER_ROUND_TRIP_COST"] = (
        "MOVEMENT_OVER_TOTAL_TAKER_ROUND_TRIP_COST"
    )
    strategic_horizons: tuple[str, ...] = STRATEGIC_HORIZONS
    profiles: tuple[V2PairMarketProfile, ...]
    top20: tuple[V2PairMarketProfile, ...]
    selected_top10: tuple[V2PairMarketProfile, ...]
    asset_comparison: tuple[V2AssetComparison, ...]
    problems: tuple[str, ...] = ()


def _d(value: Any) -> Decimal:
    return Decimal(str(value))


def _percentile(values: list[Decimal], percentile: float) -> Decimal:
    if not values:
        return Decimal(0)
    return _d(float(np.percentile(np.asarray([float(item) for item in values]), percentile)))


def _spread_pct(book: dict[str, Any]) -> Decimal | None:
    bid = _d(book.get("bidPrice", 0))
    ask = _d(book.get("askPrice", 0))
    if bid <= 0 or ask <= 0 or ask < bid:
        return None
    return (ask - bid) / ((ask + bid) / Decimal(2)) * Decimal(100)


def _aggregate_ohlc(rows: list[list[Any]], factor: int) -> list[tuple[float, float, float]]:
    """Return close/high/low for complete, exchange-aligned groups only."""
    closed_rows = [row for row in rows if int(row[6]) < int(datetime.now(UTC).timestamp() * 1000)]
    if factor == 1:
        groups = [closed_rows[index : index + 1] for index in range(len(closed_rows))]
    else:
        groups_by_bucket: dict[int, list[list[Any]]] = {}
        width_ms = factor * 5 * 60 * 1000
        for row in closed_rows:
            groups_by_bucket.setdefault(int(row[0]) // width_ms, []).append(row)
        groups = [group for _, group in sorted(groups_by_bucket.items()) if len(group) == factor]
    return [
        (
            float(group[-1][4]),
            max(float(row[2]) for row in group),
            min(float(row[3]) for row in group),
        )
        for group in groups
    ]


def _horizon_statistics(rows: list[list[Any]], factor: int) -> tuple[Decimal, Decimal]:
    aggregated = _aggregate_ohlc(rows, factor)
    if len(aggregated) < 3:
        return Decimal(0), Decimal(0)
    closes = np.asarray([row[0] for row in aggregated], dtype=float)
    highs = np.asarray([row[1] for row in aggregated], dtype=float)
    lows = np.asarray([row[2] for row in aggregated], dtype=float)
    if np.any(closes <= 0):
        return Decimal(0), Decimal(0)
    volatility = np.std(np.diff(np.log(closes)) * 100, ddof=1)
    movement = np.median((highs - lows) / closes * 100)
    return _d(float(volatility)), _d(float(movement))


def _commission_parts(candidate: UniverseCandidate) -> tuple[dict[str, Any], ...]:
    details = candidate.commission_details
    return (
        dict(details.get("discount", {})),
        dict(details.get("specialCommission", {})),
        dict(details.get("taxCommission", {})),
    )


def _category(
    *,
    liquid: bool,
    spread_ok: bool,
    slippage_ok: bool,
    maximum_ratio: Decimal,
    minimum_ratio: Decimal,
) -> EconomicCategory:
    if liquid and spread_ok and slippage_ok and maximum_ratio >= minimum_ratio:
        return "ECONOMICALLY_ATTRACTIVE"
    if liquid and maximum_ratio >= Decimal("1.5"):
        return "MARGINAL"
    return "TOO_EXPENSIVE"


def build_low_cost_market_report(
    *,
    client: BinanceReadOnlyClient,
    settings: BinanceSettings,
    candidates: list[UniverseCandidate] | None = None,
    spread_samples: int = 8,
    sample_delay_seconds: float = 0.25,
    progress: Callable[[str], None] | None = None,
) -> LowCostMarketReport:
    """Validate and rank current Spot economics using authenticated account fees."""
    if not settings.authenticated:
        raise ValueError("authenticated Binance account is required")
    problems: list[str] = []
    if candidates is None:
        candidates, _, account, scan_problems = LowCostCryptoUniverseScanner(
            client, settings
        ).scan()
        problems.extend(scan_problems)
        if not account:
            raise ValueError("authenticated Binance account is unavailable")
    if not candidates:
        raise ValueError("no eligible Binance Spot candidates")
    non_account = [item.symbol for item in candidates if item.fee_source != "BINANCE_ACCOUNT_API"]
    if non_account:
        raise ValueError("account-specific fees unavailable for one or more candidates")

    wanted = {item.symbol for item in candidates}
    sampled_spreads: dict[str, list[Decimal]] = {symbol: [] for symbol in wanted}
    for sample_index in range(max(spread_samples, 1)):
        books = {item["symbol"]: item for item in client.book_tickers() if item["symbol"] in wanted}
        for symbol, book in books.items():
            value = _spread_pct(book)
            if value is not None:
                sampled_spreads[symbol].append(value)
        if sample_index + 1 < spread_samples and sample_delay_seconds > 0:
            sleep(sample_delay_seconds)

    raw_profiles: list[V2PairMarketProfile] = []
    for index, candidate in enumerate(candidates, 1):
        if progress:
            progress(f"PAIR {index}/{len(candidates)} {candidate.symbol}")
        try:
            rows = client.klines(candidate.symbol, "5m", 1000)
            depth = client.depth(candidate.symbol, 20)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            problems.append(f"V2_MARKET_DATA_FAILED:{candidate.symbol}:{type(exc).__name__}")
            continue
        spreads = sampled_spreads.get(candidate.symbol, [])
        if not spreads:
            problems.append(f"V2_SPREAD_UNAVAILABLE:{candidate.symbol}")
            continue
        bid_depth = sum((_d(p) * _d(q) for p, q in depth.get("bids", [])), Decimal(0))
        ask_depth = sum((_d(p) * _d(q) for p, q in depth.get("asks", [])), Decimal(0))
        slippages: dict[str, Decimal] = {}
        all_notionals_fill = True
        for notional in (Decimal(10), Decimal(50), Decimal(100), Decimal(500)):
            value, complete = _book_slippage_pct(depth, notional)
            slippages[str(notional)] = value
            all_notionals_fill = all_notionals_fill and complete
        volatility: dict[str, Decimal] = {}
        movements: dict[str, Decimal] = {}
        for horizon, factor in HORIZON_FACTORS.items():
            volatility[horizon], movements[horizon] = _horizon_statistics(rows, factor)

        spread_avg = sum(spreads, Decimal(0)) / Decimal(len(spreads))
        spread_p50 = _d(median(spreads))
        spread_p95 = _percentile(spreads, 95)
        slippage500 = slippages["500"]
        taker_cost = spread_p50 + slippage500 + candidate.taker_fee * Decimal(200)
        maker_cost = (
            spread_p50 / Decimal(2)
            + slippage500 / Decimal(2)
            + (candidate.maker_fee + candidate.taker_fee) * Decimal(100)
        )
        floor = Decimal("0.000000000001")
        ratios = {key: value / max(taker_cost, floor) for key, value in movements.items()}
        net_edges = {key: value - taker_cost for key, value in movements.items()}
        max_ratio = max((ratios[key] for key in STRATEGIC_HORIZONS), default=Decimal(0))
        liquid = bool(
            candidate.quote_volume_24h >= settings.minimum_quote_volume_24h
            and bid_depth >= settings.minimum_top20_depth_quote
            and ask_depth >= settings.minimum_top20_depth_quote
            and all_notionals_fill
        )
        spread_ok = spread_p95 * Decimal(100) <= settings.v2_maximum_spread_p95_bps
        slippage_ok = slippage500 * Decimal(100) <= settings.v2_maximum_slippage_500_bps
        discount, special, tax = _commission_parts(candidate)
        raw_profiles.append(
            V2PairMarketProfile(
                rank=1,
                symbol=candidate.symbol,
                base_asset=candidate.base_asset,
                quote_asset=candidate.quote_asset,
                category=_category(
                    liquid=liquid,
                    spread_ok=spread_ok,
                    slippage_ok=slippage_ok,
                    maximum_ratio=max_ratio,
                    minimum_ratio=settings.v2_minimum_move_to_cost_ratio,
                ),
                maker_fee=candidate.maker_fee,
                taker_fee=candidate.taker_fee,
                fee_source=candidate.fee_source,
                discount=discount,
                special_commission=special,
                tax_commission=tax,
                spread_average_pct=spread_avg,
                spread_p50_pct=spread_p50,
                spread_p95_pct=spread_p95,
                slippage_pct_by_notional=slippages,
                quote_volume_24h=candidate.quote_volume_24h,
                trades_count_24h=candidate.trades_count_24h,
                top20_bid_depth_quote=bid_depth,
                top20_ask_depth_quote=ask_depth,
                volatility_pct_by_horizon=volatility,
                expected_move_pct_by_horizon=movements,
                taker_taker_round_trip_cost_pct=taker_cost,
                maker_taker_round_trip_cost_pct=maker_cost,
                movement_to_cost_ratio_by_horizon=ratios,
                expected_net_edge_pct_by_horizon=net_edges,
                maximum_strategic_ratio=max_ratio,
                liquidity_sufficient=liquid,
                spread_acceptable=spread_ok,
                slippage_acceptable=slippage_ok,
                tick_size=candidate.tick_size or Decimal("0.00000001"),
                quantity_step=candidate.quantity_step,
                minimum_notional=candidate.minimum_notional,
            )
        )

    category_order = {"ECONOMICALLY_ATTRACTIVE": 0, "MARGINAL": 1, "TOO_EXPENSIVE": 2}
    raw_profiles.sort(
        key=lambda item: (
            category_order[item.category],
            -item.maximum_strategic_ratio,
            -item.quote_volume_24h,
            item.symbol,
        )
    )
    profiles = tuple(
        item.model_copy(update={"rank": rank}) for rank, item in enumerate(raw_profiles, 1)
    )
    top20 = profiles[:20]
    selected = tuple(item for item in profiles if item.category == "ECONOMICALLY_ATTRACTIVE")[:10]
    if not selected:
        problems.append("NO_ECONOMICALLY_ATTRACTIVE_PAIR")

    best_by_asset: dict[str, V2PairMarketProfile] = {}
    for item in profiles:
        best_by_asset.setdefault(item.base_asset, item)
    comparison = []
    for base_asset in settings.preferred_base_assets:
        best_profile = best_by_asset.get(base_asset)
        if best_profile is None:
            continue
        movement = max(best_profile.expected_move_pct_by_horizon[key] for key in STRATEGIC_HORIZONS)
        two_way_fees = best_profile.taker_fee * Decimal(200)
        comparison.append(
            V2AssetComparison(
                base_asset=base_asset,
                best_symbol=best_profile.symbol,
                quote_asset=best_profile.quote_asset,
                movement_over_fees=movement / max(two_way_fees, Decimal("0.000000000001")),
                movement_over_total_cost=best_profile.maximum_strategic_ratio,
                spread_p50_pct=best_profile.spread_p50_pct,
                quote_volume_24h=best_profile.quote_volume_24h,
                slippage_500_pct=best_profile.slippage_pct_by_notional["500"],
                category=best_profile.category,
            )
        )
    comparison.sort(key=lambda item: (-item.movement_over_total_cost, item.base_asset))
    return LowCostMarketReport(
        generated_at_utc=datetime.now(UTC),
        profiles=profiles,
        top20=top20,
        selected_top10=selected,
        asset_comparison=tuple(comparison),
        problems=tuple(problems),
    )
