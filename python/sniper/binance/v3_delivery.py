"""Orchestration and evidence report for frozen Binance V3 cross-sectional research."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from sniper.binance.client import BinanceReadOnlyClient
from sniper.binance.historical import collect_historical_m1, load_historical_m1
from sniper.binance.models import BinanceCheckReport, SymbolRules
from sniper.binance.qualification import CryptoTrade, ScenarioMetrics
from sniper.binance.settings import BinanceSettings
from sniper.binance.strategy_engine import bars_from_frame
from sniper.binance.v2_market import LowCostMarketReport, V2PairMarketProfile
from sniper.binance.v3_cross_sectional import (
    assemble_cross_sections,
    build_symbol_features,
    replay_relative_strength,
    theoretical_short_oracle,
)
from sniper.config import Model

V3Verdict = Literal[
    "BINANCE_V3_REJECTED",
    "BINANCE_V3_INSUFFICIENT_EVIDENCE",
    "BINANCE_V3_QUALIFIED_FOR_PAPER",
]

STABLE_BASES = frozenset({"USDT", "USDC", "FDUSD", "USD1", "RLUSD", "EUR", "EURI", "DAI", "TUSD"})


class V3Variant(Model):
    configuration: str
    leader_count: int
    timeout_hours: int
    metrics: ScenarioMetrics
    turnover: Decimal = Field(ge=0)
    average_trade_duration_minutes: Decimal = Field(ge=0)
    results_per_symbol: dict[str, dict[str, Decimal | int | None]]


class V3Benchmark(Model):
    name: str
    gross_return_pct: Decimal
    net_return_pct: Decimal
    estimated_cost_pct: Decimal = Field(ge=0)


class BinanceV3Report(Model):
    protocol_id: Literal["BINANCE_V3_CROSS_SECTIONAL_FROZEN_2026_09_10"] = (
        "BINANCE_V3_CROSS_SECTIONAL_FROZEN_2026_09_10"
    )
    generated_at_utc: datetime
    development_start_utc: datetime
    validation_start_utc: datetime
    validation_end_utc: datetime
    primary_configuration: Literal["TOP3_RS_2H"] = "TOP3_RS_2H"
    frozen_universe: tuple[str, ...]
    eligible_history_universe: tuple[str, ...]
    history_exclusions: dict[str, str]
    data_quality: dict[str, Any]
    vectorization_equivalence: dict[str, Any]
    market_regime_counts: dict[str, int]
    market_risk_off_ratio: Decimal = Field(ge=0, le=1)
    variants: tuple[V3Variant, ...]
    primary_metrics: ScenarioMetrics
    primary_stress_metrics: ScenarioMetrics
    primary_turnover: Decimal = Field(ge=0)
    weekly_returns: dict[str, Decimal]
    monthly_returns: dict[str, Decimal]
    benchmarks: tuple[V3Benchmark, ...]
    results_per_symbol: dict[str, dict[str, Decimal | int | None]]
    long_only_diagnostic: dict[str, Decimal | int | bool]
    qualification_checks: dict[str, bool]
    verdict: V3Verdict
    live_trading_enabled: Literal[False] = False
    order_endpoints_present: Literal[False] = False


def _dynamic_universe(market: LowCostMarketReport) -> tuple[V2PairMarketProfile, ...]:
    chosen: list[V2PairMarketProfile] = []
    bases = set()
    for profile in market.profiles:
        if (
            profile.category != "ECONOMICALLY_ATTRACTIVE"
            or profile.base_asset in STABLE_BASES
            or not profile.symbol.isascii()
            or not profile.symbol.isalnum()
            or profile.symbol != profile.base_asset + profile.quote_asset
            or profile.base_asset in bases
        ):
            continue
        chosen.append(profile)
        bases.add(profile.base_asset)
        if len(chosen) == 28:
            break
    for symbol in ("BTCUSDT", "ETHUSDT"):
        if symbol not in {item.symbol for item in chosen}:
            profile = next(item for item in market.profiles if item.symbol == symbol)
            chosen.append(profile)
    return tuple(chosen)


def _rules(data_root: Path, symbols: tuple[str, ...]) -> dict[str, SymbolRules]:
    wanted = set(symbols)
    check = BinanceCheckReport.model_validate_json(
        (data_root / "binance" / "reports" / "first-deliverable-binance-check.json").read_text(
            encoding="utf-8"
        )
    )
    candidates = {item.symbol: item for item in check.symbols if item.symbol in wanted}
    maximum = Decimal("1e50")
    output = {
        symbol: SymbolRules(
            symbol=symbol,
            status=candidate.status,
            base_asset=candidate.base_asset,
            quote_asset=candidate.quote_asset,
            spot_trading_allowed=True,
            permissions=("SPOT",),
            order_types=("LIMIT", "MARKET"),
            tick_size=candidate.tick_size or Decimal("0.00000001"),
            minimum_price=Decimal(0),
            maximum_price=maximum,
            minimum_quantity=candidate.minimum_quantity,
            maximum_quantity=maximum,
            quantity_step=candidate.quantity_step,
            market_minimum_quantity=candidate.minimum_quantity,
            market_maximum_quantity=maximum,
            market_quantity_step=candidate.quantity_step,
            minimum_notional=candidate.minimum_notional,
            maximum_notional=None,
            filters_raw=(),
        )
        for symbol, candidate in candidates.items()
    }
    if output.keys() != wanted:
        raise ValueError(f"missing current filters for {sorted(wanted - output.keys())}")
    return output


def _period_returns(
    trades: list[CryptoTrade], start: datetime, end: datetime
) -> tuple[dict[str, Decimal], dict[str, Decimal]]:
    weekly: dict[str, Decimal] = defaultdict(Decimal)
    monthly: dict[str, Decimal] = defaultdict(Decimal)
    for trade in trades:
        if start <= trade.exit_time_utc < end:
            weekly[trade.exit_time_utc.strftime("%G-W%V")] += trade.net_pnl / Decimal(5)
            monthly[trade.exit_time_utc.strftime("%Y-%m")] += trade.net_pnl / Decimal(5)
    return dict(weekly), dict(monthly)


def _symbol_results(trades: list[CryptoTrade]) -> dict[str, dict[str, Decimal | int | None]]:
    grouped: dict[str, list[CryptoTrade]] = defaultdict(list)
    for trade in trades:
        grouped[trade.symbol].append(trade)
    output: dict[str, dict[str, Decimal | int | None]] = {}
    for symbol, items in sorted(grouped.items()):
        wins = sum((item.net_pnl for item in items if item.net_pnl > 0), Decimal(0))
        losses = sum((-item.net_pnl for item in items if item.net_pnl < 0), Decimal(0))
        output[symbol] = {
            "trades": len(items),
            "gross_pnl": sum((item.market_pnl for item in items), Decimal(0)),
            "net_pnl": sum((item.net_pnl for item in items), Decimal(0)),
            "commission": sum((item.commission for item in items), Decimal(0)),
            "spread": sum((item.spread_cost for item in items), Decimal(0)),
            "slippage": sum((item.slippage_cost for item in items), Decimal(0)),
            "profit_factor": wins / losses if losses else None,
        }
    return output


def _benchmarks(
    bars_by_symbol: dict[str, list[Any]],
    profiles: dict[str, V2PairMarketProfile],
    variants: list[V3Variant],
    start: datetime,
    end: datetime,
) -> tuple[V3Benchmark, ...]:
    def hold(symbol: str) -> tuple[Decimal, Decimal, Decimal]:
        bars = [bar for bar in bars_by_symbol[symbol] if start <= bar.timestamp_utc < end]
        if not bars:
            return Decimal(0), Decimal(0), Decimal(0)
        gross = Decimal(str((bars[-1].close / bars[0].open - 1) * 100))
        cost = profiles[symbol].taker_taker_round_trip_cost_pct
        return gross, gross - cost, cost

    btc_gross, btc_net, btc_cost = hold("BTCUSDT")
    holds = [hold(symbol) for symbol in bars_by_symbol if symbol in profiles]
    equal_gross = sum((item[0] for item in holds), Decimal(0)) / max(len(holds), 1)
    equal_cost = sum((item[2] for item in holds), Decimal(0)) / max(len(holds), 1)
    output = [
        V3Benchmark(
            name="BUY_AND_HOLD_BTC",
            gross_return_pct=btc_gross,
            net_return_pct=btc_net,
            estimated_cost_pct=btc_cost,
        ),
        V3Benchmark(
            name="BUY_AND_HOLD_EQUAL_WEIGHT_UNIVERSE",
            gross_return_pct=equal_gross,
            net_return_pct=equal_gross - equal_cost,
            estimated_cost_pct=equal_cost,
        ),
        V3Benchmark(
            name="CASH",
            gross_return_pct=Decimal(0),
            net_return_pct=Decimal(0),
            estimated_cost_pct=Decimal(0),
        ),
    ]
    for leaders in (1, 3, 5):
        variant = next(
            item for item in variants if item.leader_count == leaders and item.timeout_hours == 2
        )
        costs = (
            (
                variant.metrics.spread_cost
                + variant.metrics.slippage_cost
                + variant.metrics.commission
            )
            / variant.metrics.starting_capital_eur
            * Decimal(100)
        )
        gross = (
            variant.metrics.gross_market_pnl / variant.metrics.starting_capital_eur * Decimal(100)
        )
        output.append(
            V3Benchmark(
                name=f"TOP{leaders}_RELATIVE_STRENGTH",
                gross_return_pct=gross,
                net_return_pct=variant.metrics.net_return_pct,
                estimated_cost_pct=costs,
            )
        )
    return tuple(output)


def run_binance_v3(
    *,
    client: BinanceReadOnlyClient,
    settings: BinanceSettings,
    data_root: Path,
    start_utc: datetime,
    validation_start_utc: datetime,
    end_utc: datetime,
    collect_history: bool,
    progress: Any | None = None,
) -> tuple[BinanceV3Report, list[CryptoTrade]]:
    if not settings.authenticated or settings.execution_permitted:
        raise ValueError("authenticated read-only Binance settings required")
    report_dir = data_root / "binance" / "reports"
    equivalence_path = report_dir / "binance-v3-vectorization-equivalence.json"
    equivalence = json.loads(equivalence_path.read_text(encoding="utf-8"))
    if not (
        equivalence.get("max_abs_feature_difference", 1) <= 1e-12
        and equivalence.get("ranking_identity") is True
        and equivalence.get("signal_identity") is True
        and equivalence.get("passed") is True
    ):
        raise ValueError("V3 vectorization equivalence audit failed")
    market = LowCostMarketReport.model_validate_json(
        (report_dir / "binance-v2-low-cost-market.json").read_text(encoding="utf-8")
    )
    frozen_profiles = _dynamic_universe(market)
    freeze_path = report_dir / "binance-v3-universe-freeze.json"
    if not freeze_path.exists():
        freeze_path.write_text(
            json.dumps(
                {
                    "protocol_id": "BINANCE_V3_CROSS_SECTIONAL_FROZEN_2026_09_10",
                    "rules": "TOP28_UNIQUE_ASCII_NON_STABLE_ATTRACTIVE_PLUS_BTCUSDT_ETHUSDT",
                    "symbols": [item.symbol for item in frozen_profiles],
                    "profiles": [item.model_dump(mode="json") for item in frozen_profiles],
                    "primary_configuration": "TOP3_RS_2H",
                    "diagnostics": [
                        f"TOP{leaders}_RS_{hours}H" for leaders in (1, 3, 5) for hours in (1, 2, 4)
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        if frozen.get("protocol_id") != "BINANCE_V3_CROSS_SECTIONAL_FROZEN_2026_09_10":
            raise ValueError("V3 universe freeze protocol mismatch")
        frozen_profiles = tuple(
            V2PairMarketProfile.model_validate(item) for item in frozen["profiles"]
        )
    frozen_symbols = tuple(item.symbol for item in frozen_profiles)
    if collect_history:
        expected_rows = int((end_utc - start_utc).total_seconds() / 60)
        try:
            existing_frame = load_historical_m1(data_root, frozen_symbols, start_utc, end_utc)
            existing_counts = {
                str(row[0]): int(row[1])
                for row in existing_frame.group_by("symbol").len().iter_rows()
            }
        except ValueError:
            existing_counts = {}
        missing_symbols = tuple(
            symbol
            for symbol in frozen_symbols
            if existing_counts.get(symbol, 0) < expected_rows * 0.98
        )
        if progress:
            progress(
                f"V3_HISTORY_START missing={len(missing_symbols)} "
                f"already_complete={len(frozen_symbols) - len(missing_symbols)}"
            )
        if missing_symbols:
            collect_historical_m1(client, data_root, missing_symbols, start_utc, end_utc)
    expected = int((end_utc - start_utc).total_seconds() / 60)
    eligible_symbols: list[str] = []
    exclusions = {}
    quality = {}
    bars = {}
    features_by_symbol = {}
    for symbol in frozen_symbols:
        symbol_frame = load_historical_m1(data_root, (symbol,), start_utc, end_utc)
        rows = bars_from_frame(symbol_frame).get(symbol, [])
        coverage = Decimal(len(rows)) / Decimal(expected)
        span_days = (rows[-1].timestamp_utc - rows[0].timestamp_utc).days if rows else 0
        quality[symbol] = {"bars_m1": len(rows), "coverage": coverage, "span_days": span_days}
        if coverage >= Decimal("0.80") and span_days >= 60:
            eligible_symbols.append(symbol)
            bars[symbol], features_by_symbol[symbol] = build_symbol_features(rows)
        else:
            exclusions[symbol] = "HISTORY_LT_80PCT_OR_60_DAYS"
    if (
        "BTCUSDT" not in eligible_symbols
        or "ETHUSDT" not in eligible_symbols
        or len(eligible_symbols) < 5
    ):
        raise ValueError("insufficient benchmark/universe history for V3")
    profiles = {item.symbol: item for item in frozen_profiles if item.symbol in eligible_symbols}
    rules = _rules(data_root, tuple(eligible_symbols))
    cross_sections, regime_counts = assemble_cross_sections(features_by_symbol)
    variants = []
    primary_trades: list[CryptoTrade] = []
    primary_turnover = Decimal(0)
    primary_metrics: ScenarioMetrics | None = None
    for leaders in (1, 3, 5):
        for hours in (1, 2, 4):
            metrics, trades, turnover = replay_relative_strength(
                bars_by_symbol=bars,
                cross_sections=cross_sections,
                profiles=profiles,
                rules=rules,
                leader_count=leaders,
                timeout_hours=hours,
                start_utc=validation_start_utc,
                end_utc=end_utc,
            )
            variants.append(
                V3Variant(
                    configuration=f"TOP{leaders}_RS_{hours}H",
                    leader_count=leaders,
                    timeout_hours=hours,
                    metrics=metrics,
                    turnover=turnover,
                    average_trade_duration_minutes=(
                        sum(
                            (
                                Decimal(
                                    str(
                                        (trade.exit_time_utc - trade.entry_time_utc).total_seconds()
                                    )
                                )
                                / Decimal(60)
                                for trade in trades
                            ),
                            Decimal(0),
                        )
                        / Decimal(max(len(trades), 1))
                    ),
                    results_per_symbol=_symbol_results(trades),
                )
            )
            if leaders == 3 and hours == 2:
                primary_metrics = metrics
                primary_trades = trades
                primary_turnover = turnover
    if primary_metrics is None:
        raise RuntimeError("frozen primary V3 configuration missing")
    stress_metrics, _, _ = replay_relative_strength(
        bars_by_symbol=bars,
        cross_sections=cross_sections,
        profiles=profiles,
        rules=rules,
        leader_count=3,
        timeout_hours=2,
        start_utc=validation_start_utc,
        end_utc=end_utc,
        stress=True,
    )
    checks = {
        "gross_expectancy_positive": primary_metrics.gross_market_pnl > 0,
        "net_expectancy_positive": primary_metrics.expectancy_eur > 0,
        "profit_factor_gte_1_20": primary_metrics.profit_factor is not None
        and primary_metrics.profit_factor >= Decimal("1.20"),
        "drawdown_lte_10pct": primary_metrics.max_drawdown_pct <= Decimal(10),
        "positive_weeks_gte_60pct": primary_metrics.positive_week_ratio >= Decimal("0.60"),
        "stress_expectancy_nonnegative": stress_metrics.expectancy_eur >= 0,
    }
    if primary_metrics.gross_market_pnl <= 0:
        verdict: V3Verdict = "BINANCE_V3_REJECTED"
    elif primary_metrics.trades < 30 or primary_metrics.active_trade_weeks < 4:
        verdict = "BINANCE_V3_INSUFFICIENT_EVIDENCE"
    elif all(checks.values()):
        verdict = "BINANCE_V3_QUALIFIED_FOR_PAPER"
    else:
        verdict = "BINANCE_V3_REJECTED"
    weekly, monthly = _period_returns(primary_trades, validation_start_utc, end_utc)
    risk_off = regime_counts.get("MARKET_RISK_OFF", 0)
    total_regimes = sum(regime_counts.values())
    v3 = BinanceV3Report(
        generated_at_utc=datetime.now(UTC),
        development_start_utc=start_utc,
        validation_start_utc=validation_start_utc,
        validation_end_utc=end_utc,
        frozen_universe=frozen_symbols,
        eligible_history_universe=tuple(eligible_symbols),
        history_exclusions=exclusions,
        data_quality=quality,
        vectorization_equivalence=equivalence,
        market_regime_counts=regime_counts,
        market_risk_off_ratio=Decimal(risk_off) / Decimal(max(total_regimes, 1)),
        variants=tuple(variants),
        primary_metrics=primary_metrics,
        primary_stress_metrics=stress_metrics,
        primary_turnover=primary_turnover,
        weekly_returns=weekly,
        monthly_returns=monthly,
        benchmarks=_benchmarks(bars, profiles, variants, validation_start_utc, end_utc),
        results_per_symbol=_symbol_results(primary_trades),
        long_only_diagnostic=theoretical_short_oracle(cross_sections, bars),
        qualification_checks=checks,
        verdict=verdict,
    )
    return v3, primary_trades
