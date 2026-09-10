"""Authenticated low-cost universe freeze and deterministic Binance V2 replay."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from sniper.binance.client import BinanceReadOnlyClient
from sniper.binance.filters import parse_symbol_rules
from sniper.binance.historical import collect_historical_m1, load_historical_m1
from sniper.binance.models import BinanceCheckReport, SymbolRules
from sniper.binance.qualification import CryptoTrade, ScenarioMetrics, replay_spot
from sniper.binance.settings import BinanceSettings
from sniper.binance.strategy_engine import BinanceV2StrategyEngine, bars_from_frame
from sniper.binance.v2_market import (
    LowCostMarketReport,
    build_low_cost_market_report,
)
from sniper.config import Model

V2Verdict = Literal[
    "BINANCE_V2_REJECTED",
    "BINANCE_V2_INSUFFICIENT_EVIDENCE",
    "BINANCE_V2_QUALIFIED_FOR_PAPER",
]


class V2Breakdown(Model):
    key: str
    trades: int = Field(ge=0)
    wins: int = Field(ge=0)
    gross_market_pnl: Decimal
    net_pnl: Decimal
    commissions: Decimal = Field(ge=0)
    total_costs: Decimal = Field(ge=0)
    expectancy: Decimal
    profit_factor: Decimal | None


class BinanceV2Report(Model):
    protocol_id: Literal["BINANCE_V2_FROZEN_2026_09_10"] = "BINANCE_V2_FROZEN_2026_09_10"
    generated_at_utc: datetime
    start_utc: datetime
    end_utc: datetime
    market: LowCostMarketReport
    symbols: tuple[str, ...]
    bars_m1: int
    bars_m5: int
    signal_observations: int
    signal_rejections: dict[str, int]
    data_quality: dict[str, Any]
    base_metrics: ScenarioMetrics
    stress_metrics: ScenarioMetrics
    results_by_crypto: tuple[V2Breakdown, ...]
    results_by_setup: tuple[V2Breakdown, ...]
    diagnostics: dict[str, Any]
    qualification_checks: dict[str, bool]
    limitations: tuple[str, ...]
    verdict: V2Verdict
    live_trading_enabled: Literal[False] = False
    order_endpoints_present: Literal[False] = False


def _load_authenticated_check(data_root: Path) -> BinanceCheckReport:
    path = data_root / "binance" / "reports" / "first-deliverable-binance-check.json"
    if not path.exists():
        raise ValueError("authenticated binance-check report is required")
    report = BinanceCheckReport.model_validate_json(path.read_text(encoding="utf-8"))
    if not report.account.get("authenticated"):
        raise ValueError("authenticated=false")
    if report.account.get("account_type") != "SPOT":
        raise ValueError("account_type is not SPOT")
    if any(item.fee_source != "BINANCE_ACCOUNT_API" for item in report.symbols):
        raise ValueError("account-specific Binance fees are incomplete")
    return report


def _rules(client: BinanceReadOnlyClient, symbols: tuple[str, ...]) -> dict[str, SymbolRules]:
    wanted = set(symbols)
    output = {
        parsed.symbol: parsed
        for raw in client.exchange_info().get("symbols", [])
        if (parsed := parse_symbol_rules(raw)).symbol in wanted
    }
    if output.keys() != wanted:
        raise ValueError(f"missing current Binance filters: {sorted(wanted - output.keys())}")
    return output


def _breakdowns(trades: list[CryptoTrade], attribute: str) -> tuple[V2Breakdown, ...]:
    grouped: dict[str, list[CryptoTrade]] = defaultdict(list)
    for trade in trades:
        grouped[str(getattr(trade, attribute))].append(trade)
    output = []
    for key, items in sorted(grouped.items()):
        wins = [item.net_pnl for item in items if item.net_pnl > 0]
        losses = [-item.net_pnl for item in items if item.net_pnl < 0]
        costs = [item.spread_cost + item.slippage_cost + item.commission for item in items]
        output.append(
            V2Breakdown(
                key=key,
                trades=len(items),
                wins=len(wins),
                gross_market_pnl=sum((item.market_pnl for item in items), Decimal(0)),
                net_pnl=sum((item.net_pnl for item in items), Decimal(0)),
                commissions=sum((item.commission for item in items), Decimal(0)),
                total_costs=sum(costs, Decimal(0)),
                expectancy=sum((item.net_pnl for item in items), Decimal(0)) / Decimal(len(items)),
                profit_factor=(sum(wins, Decimal(0)) / sum(losses, Decimal(0))) if losses else None,
            )
        )
    return tuple(output)


def _verdict(
    base: ScenarioMetrics, stress: ScenarioMetrics
) -> tuple[V2Verdict, dict[str, bool], dict[str, Any]]:
    total_cost = base.spread_cost + base.slippage_cost + base.commission
    gross = base.gross_market_pnl
    commission_share = base.commission / gross if gross > 0 else None
    cost_share = total_cost / gross if gross > 0 else None
    diagnosis = "EDGE_SURVIVES_COSTS"
    if gross <= 0:
        diagnosis = "NO_GROSS_EDGE"
    elif base.commission > gross * Decimal("0.50"):
        diagnosis = "TURNOVER_TOO_EXPENSIVE"
    elif base.net_return_pct <= 0:
        diagnosis = "EDGE_DESTROYED_BY_COSTS"
    checks = {
        "positive_net_return": base.net_return_pct > 0,
        "profit_factor_gte_1_20": base.profit_factor is not None
        and base.profit_factor >= Decimal("1.20"),
        "minimum_100_trades": base.trades >= 100,
        "minimum_8_active_weeks": base.active_trade_weeks >= 8,
        "positive_weeks_gte_60pct": base.positive_week_ratio >= Decimal("0.60"),
        "stress_positive": stress.net_return_pct > 0,
        "costs_below_gross_profit": gross > total_cost,
        "pnl_crosscheck": base.pnl_crosscheck_passed and stress.pnl_crosscheck_passed,
        "historical_bid_ask_coverage": False,
    }
    diagnostics = {
        "primary_diagnosis": diagnosis,
        "commission_share_of_gross_market_pnl": commission_share,
        "total_cost_share_of_gross_market_pnl": cost_share,
        "gross_market_pnl": gross,
        "total_costs": total_cost,
        "net_pnl": base.ending_capital_eur - base.starting_capital_eur,
    }
    if diagnosis == "NO_GROSS_EDGE" or (
        base.trades >= 100
        and base.active_trade_weeks >= 8
        and not all(value for key, value in checks.items() if key != "historical_bid_ask_coverage")
    ):
        return "BINANCE_V2_REJECTED", checks, diagnostics
    if not checks["minimum_100_trades"] or not checks["minimum_8_active_weeks"]:
        return "BINANCE_V2_INSUFFICIENT_EVIDENCE", checks, diagnostics
    if all(checks.values()):
        return "BINANCE_V2_QUALIFIED_FOR_PAPER", checks, diagnostics
    return "BINANCE_V2_INSUFFICIENT_EVIDENCE", checks, diagnostics


def run_binance_v2(
    *,
    client: BinanceReadOnlyClient,
    settings: BinanceSettings,
    data_root: Path,
    start_utc: datetime,
    end_utc: datetime,
    collect_history: bool,
    progress: Any | None = None,
) -> tuple[BinanceV2Report, list[CryptoTrade]]:
    check = _load_authenticated_check(data_root)
    reports = data_root / "binance" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    market_path = reports / "binance-v2-low-cost-market.json"
    freeze_path = reports / "binance-v2-universe-freeze.json"
    if market_path.exists() and freeze_path.exists():
        market = LowCostMarketReport.model_validate_json(market_path.read_text(encoding="utf-8"))
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        if frozen.get("protocol_id") != "BINANCE_V2_FROZEN_2026_09_10":
            raise ValueError("V2 universe freeze protocol mismatch")
        frozen_symbols = tuple(item["symbol"] for item in frozen.get("selected_top10", []))
        selected = tuple(item for item in market.selected_top10 if item.symbol in frozen_symbols)
        if tuple(item.symbol for item in selected) != frozen_symbols:
            raise ValueError("V2 market report and universe freeze disagree")
    else:
        market = build_low_cost_market_report(
            client=client,
            settings=settings,
            candidates=list(check.symbols),
            progress=progress,
        )
        market_path.write_text(market.model_dump_json(indent=2), encoding="utf-8")
        selected = market.selected_top10
        if not selected:
            raise ValueError("no V2 pair passed the frozen economic filters")
        freeze_path.write_text(
            json.dumps(
                {
                    "protocol_id": "BINANCE_V2_FROZEN_2026_09_10",
                    "frozen_at_utc": datetime.now(UTC).isoformat(),
                    "ranking_method": market.ranking_method,
                    "selected_top10": [item.model_dump(mode="json") for item in selected],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    symbols = tuple(item.symbol for item in selected)
    if collect_history:
        if progress:
            progress(f"HISTORY_START {len(symbols)} symbols")
        collect_historical_m1(client, data_root, symbols, start_utc, end_utc)
    frame = load_historical_m1(data_root, symbols, start_utc, end_utc)
    m1_by_symbol = bars_from_frame(frame)
    quality: dict[str, Any] = {}
    m5_by_symbol = {}
    signals_by_symbol = {}
    strategy_rejections: Counter[str] = Counter()
    profile_by_symbol = {item.symbol: item for item in selected}
    engine = BinanceV2StrategyEngine()
    for symbol in symbols:
        rows = m1_by_symbol.get(symbol, [])
        gaps = sum(
            current.timestamp_utc - previous.timestamp_utc != timedelta(minutes=1)
            for previous, current in zip(rows, rows[1:], strict=False)
        )
        quality[symbol] = {
            "m1_bars": len(rows),
            "first_timestamp_utc": rows[0].timestamp_utc.isoformat() if rows else None,
            "last_timestamp_utc": rows[-1].timestamp_utc.isoformat() if rows else None,
            "non_one_minute_gaps": gaps,
        }
        profile = profile_by_symbol[symbol]
        m5, signals, rejected = engine.generate(
            rows,
            total_round_trip_cost_pct=float(profile.taker_taker_round_trip_cost_pct),
        )
        m5_by_symbol[symbol] = m5
        signals_by_symbol[symbol] = signals
        strategy_rejections.update(rejected)
        if progress:
            progress(f"REPLAY_INPUT {symbol} bars={len(rows)} signals={len(signals)}")
    rules = _rules(client, symbols)
    spreads = {
        symbol: profile_by_symbol[symbol].spread_p50_pct * Decimal(100) for symbol in symbols
    }
    fees = {symbol: profile_by_symbol[symbol].taker_fee for symbol in symbols}
    slippage = {
        symbol: profile_by_symbol[symbol].slippage_pct_by_notional["500"] for symbol in symbols
    }
    base, trades = replay_spot(
        bars_by_symbol=m5_by_symbol,
        signals_by_symbol=signals_by_symbol,
        rules_by_symbol=rules,
        spreads_bps=spreads,
        capital=Decimal("500"),
        fee_rate=Decimal(0),
        scenario="BASE",
        start_utc=start_utc,
        end_utc=end_utc,
        fee_rates_by_symbol=fees,
        slippage_pct_by_symbol=slippage,
        maximum_trades_per_day=3,
        cooldown_minutes=30,
    )
    stress, _ = replay_spot(
        bars_by_symbol=m5_by_symbol,
        signals_by_symbol=signals_by_symbol,
        rules_by_symbol=rules,
        spreads_bps=spreads,
        capital=Decimal("500"),
        fee_rate=Decimal(0),
        scenario="STRESS",
        start_utc=start_utc,
        end_utc=end_utc,
        fee_rates_by_symbol=fees,
        slippage_pct_by_symbol=slippage,
        maximum_trades_per_day=3,
        cooldown_minutes=30,
    )
    verdict, checks, diagnostics = _verdict(base, stress)
    report = BinanceV2Report(
        generated_at_utc=datetime.now(UTC),
        start_utc=start_utc,
        end_utc=end_utc,
        market=market,
        symbols=symbols,
        bars_m1=frame.height,
        bars_m5=sum(len(rows) for rows in m5_by_symbol.values()),
        signal_observations=sum(len(rows) for rows in signals_by_symbol.values()),
        signal_rejections=dict(strategy_rejections),
        data_quality={"canonical_utc": True, "per_symbol": quality},
        base_metrics=base,
        stress_metrics=stress,
        results_by_crypto=_breakdowns(trades, "symbol"),
        results_by_setup=_breakdowns(trades, "setup_type"),
        diagnostics=diagnostics,
        qualification_checks=checks,
        limitations=(
            "NO_HISTORICAL_BID_ASK_OR_ORDER_BOOK_COVERAGE",
            "CURRENT_SPREAD_AND_DEPTH_SNAPSHOTS_USED_AS_EXECUTION_PROXY",
            "CURRENT_ACCOUNT_FEES_APPLIED_TO_HISTORICAL_REPLAY",
            "MIXED_QUOTES_REPORTED_AS_NORMALIZED_QUOTE_CAPITAL_UNITS",
            "OHLC_SAME_BAR_AMBIGUITY_RESOLVED_STOP_FIRST",
            "SPOT_LONG_ONLY_NO_SHORT_SELLING",
            "MAKER_MODE_NOT_USED_WITHOUT_CREDIBLE_FILL_SIMULATION",
        ),
        verdict=verdict,
    )
    return report, trades


def render_binance_v2(report: BinanceV2Report) -> str:
    base = report.base_metrics
    total_costs = base.spread_cost + base.slippage_cost + base.commission
    lines = [
        "# SNIPER Binance V2 — Low-cost universe + replay",
        "",
        f"- Période UTC: {report.start_utc.isoformat()} → {report.end_utc.isoformat()}",
        "- Frais: réels compte Binance (`BINANCE_ACCOUNT_API`), par paire",
        "- Exécution: TAKER_ENTRY_TAKER_EXIT (maker désactivé faute de simulation crédible)",
        "- LIVE / endpoints d'ordre: désactivé / absent",
        "",
        "## TOP 20 low-cost pairs",
        "",
        "| # | Paire | Catégorie | Coût TT % | Ratio 30m | Ratio 1h | "
        "Ratio 4h | Spread p50 % | Slip 500 % |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report.market.top20:
        ratio = item.movement_to_cost_ratio_by_horizon
        lines.append(
            f"| {item.rank} | {item.symbol} | {item.category} | "
            f"{item.taker_taker_round_trip_cost_pct:.6f} | {ratio['30m']:.3f} | "
            f"{ratio['1h']:.3f} | {ratio['4h']:.3f} | {item.spread_p50_pct:.6f} | "
            f"{item.slippage_pct_by_notional['500']:.6f} |"
        )
    lines.extend(["", "## TOP 10 gelé", "", ", ".join(report.symbols), "", "## Replay BASE", ""])
    for label, value in (
        ("Trades", base.trades),
        ("Net return %", base.net_return_pct),
        ("Profit factor", base.profit_factor),
        ("Expectancy", base.expectancy_eur),
        ("Max drawdown %", base.max_drawdown_pct),
        ("Commissions", base.commission),
        ("Coûts totaux", total_costs),
        (
            "Semaines positives",
            f"{round(float(base.positive_week_ratio) * base.active_trade_weeks)}"
            f"/{base.active_trade_weeks}",
        ),
        ("Jours >= +1 %", base.days_target_reached),
    ):
        lines.append(f"- {label}: {value}")
    sections = (
        ("Résultats par crypto", report.results_by_crypto),
        ("Résultats par setup", report.results_by_setup),
    )
    for title, values in sections:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Clé | Trades | Net | PF | Expectancy | Coûts |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for breakdown in values:
            lines.append(
                f"| {breakdown.key} | {breakdown.trades} | {breakdown.net_pnl:.6f} | "
                f"{breakdown.profit_factor} | {breakdown.expectancy:.6f} | "
                f"{breakdown.total_costs:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Diagnostic",
            "",
            f"- {report.diagnostics['primary_diagnosis']}",
            f"- Verdict: `{report.verdict}`",
            "",
        ]
    )
    return "\n".join(lines)
