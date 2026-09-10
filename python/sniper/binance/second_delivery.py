"""Orchestration and reporting for the frozen second Binance deliverable."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from sniper.binance.client import BinanceReadOnlyClient
from sniper.binance.filters import parse_symbol_rules
from sniper.binance.historical import collect_historical_m1, load_historical_m1
from sniper.binance.models import SymbolRules, UniverseCandidate
from sniper.binance.qualification import (
    BinanceQualificationReport,
    CryptoTrade,
    decide_verdict,
    replay_spot,
)
from sniper.binance.settings import BinanceSettings
from sniper.binance.strategy_engine import BinanceStrategyEngine, bars_from_frame
from sniper.binance.universe import LowCostCryptoUniverseScanner


def _account_fee_snapshot(data_root: Path) -> tuple[Decimal, Decimal]:
    path = data_root / "binance" / "reports" / "first-deliverable-binance-check.json"
    if not path.exists():
        raise ValueError("authenticated binance-check report is required")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("account", {}).get("authenticated"):
        raise ValueError("binance-check report is not authenticated")
    actual = [
        item
        for item in report.get("symbols", [])
        if item.get("fee_source") == "BINANCE_ACCOUNT_API"
    ]
    if not actual:
        raise ValueError("no BINANCE_ACCOUNT_API fee snapshot is available")
    maker = {Decimal(str(item["maker_fee_rate"])) for item in actual}
    taker = {Decimal(str(item["taker_fee_rate"])) for item in actual}
    if len(maker) != 1 or len(taker) != 1:
        raise ValueError("account fee snapshot is not uniform across observed priority symbols")
    return maker.pop(), taker.pop()


def _qualification_candidates(
    candidates: list[UniverseCandidate],
) -> list[UniverseCandidate]:
    return [
        item
        for item in candidates
        if item.quote_asset == "EUR"
        and item.quote_volume_24h >= Decimal("100000")
        and item.spread_bps <= Decimal("20")
        and item.top20_bid_depth_quote >= Decimal("5000")
        and item.top20_ask_depth_quote >= Decimal("5000")
    ]


def _current_rules(
    client: BinanceReadOnlyClient, symbols: tuple[str, ...]
) -> dict[str, SymbolRules]:
    wanted = set(symbols)
    rules = {}
    for payload in client.exchange_info().get("symbols", []):
        if payload.get("symbol") in wanted:
            parsed = parse_symbol_rules(payload)
            rules[parsed.symbol] = parsed
    missing = wanted - rules.keys()
    if missing:
        raise ValueError(f"missing current Binance filters for: {sorted(missing)}")
    return rules


def run_second_delivery(
    *,
    client: BinanceReadOnlyClient,
    settings: BinanceSettings,
    data_root: Path,
    start_utc: datetime,
    end_utc: datetime,
    collect_history: bool,
) -> tuple[BinanceQualificationReport, list[CryptoTrade]]:
    """Collect/replay the preregistered V1 without enabling any execution path."""
    maker_fee, taker_fee = _account_fee_snapshot(data_root)
    freeze_path = data_root / "binance" / "reports" / "qualification-universe-freeze.json"
    if freeze_path.exists():
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        if frozen.get("protocol_id") != "BINANCE_SPOT_V1_FROZEN_2026_09_09":
            raise ValueError("qualification universe freeze protocol mismatch")
        selected = [UniverseCandidate.model_validate(item) for item in frozen.get("candidates", [])]
    else:
        universe, _, _, _ = LowCostCryptoUniverseScanner(client, settings).scan()
        selected = _qualification_candidates(universe)
    symbols = tuple(sorted(item.symbol for item in selected))
    if not symbols:
        raise ValueError("no EUR research pair passed the frozen liquidity snapshot filters")
    if collect_history:
        freeze_path.parent.mkdir(parents=True, exist_ok=True)
        freeze_path.write_text(
            json.dumps(
                {
                    "protocol_id": "BINANCE_SPOT_V1_FROZEN_2026_09_09",
                    "generated_at_utc": datetime.now(UTC).isoformat(),
                    "candidates": [item.model_dump(mode="json") for item in selected],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        collect_historical_m1(client, data_root, symbols, start_utc, end_utc)
    frame = load_historical_m1(data_root, symbols, start_utc, end_utc)
    m1_by_symbol = bars_from_frame(frame)
    if not freeze_path.exists():
        # Recovery for datasets created before the freeze manifest existed: only symbols
        # that were actually collected in the initial run can enter the immutable universe.
        selected = [item for item in selected if m1_by_symbol.get(item.symbol)]
        symbols = tuple(sorted(item.symbol for item in selected))
        freeze_path.parent.mkdir(parents=True, exist_ok=True)
        freeze_path.write_text(
            json.dumps(
                {
                    "protocol_id": "BINANCE_SPOT_V1_FROZEN_2026_09_09",
                    "generated_at_utc": datetime.now(UTC).isoformat(),
                    "recovered_from_existing_history": True,
                    "candidates": [item.model_dump(mode="json") for item in selected],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    per_symbol_quality = {}
    for symbol in symbols:
        rows = m1_by_symbol.get(symbol, [])
        gaps = sum(
            current.timestamp_utc - previous.timestamp_utc != timedelta(minutes=1)
            for previous, current in zip(rows, rows[1:], strict=False)
        )
        per_symbol_quality[symbol] = {
            "m1_bars": len(rows),
            "first_timestamp_utc": rows[0].timestamp_utc.isoformat() if rows else None,
            "last_timestamp_utc": rows[-1].timestamp_utc.isoformat() if rows else None,
            "non_one_minute_gaps": gaps,
        }
    m5_by_symbol = {}
    signals_by_symbol = {}
    strategy = BinanceStrategyEngine()
    for symbol in symbols:
        m5, signals = strategy.generate(m1_by_symbol.get(symbol, []))
        m5_by_symbol[symbol] = m5
        signals_by_symbol[symbol] = signals
    rules_by_symbol = _current_rules(client, symbols)
    spreads = {item.symbol: item.spread_bps for item in selected}
    results = []
    primary_trades: list[CryptoTrade] = []
    for capital in (Decimal("100"), Decimal("500"), Decimal("1000")):
        for scenario in ("BASE", "STRESS"):
            metrics, trades = replay_spot(
                bars_by_symbol=m5_by_symbol,
                signals_by_symbol=signals_by_symbol,
                rules_by_symbol=rules_by_symbol,
                spreads_bps=spreads,
                capital=capital,
                fee_rate=taker_fee,
                scenario=scenario,
                start_utc=start_utc,
                end_utc=end_utc,
            )
            results.append(metrics)
            if capital == Decimal("500") and scenario == "BASE":
                primary_trades = trades
    primary_base = next(
        item
        for item in results
        if item.starting_capital_eur == Decimal("500") and item.scenario == "BASE"
    )
    primary_stress = next(
        item
        for item in results
        if item.starting_capital_eur == Decimal("500") and item.scenario == "STRESS"
    )
    verdict, checks = decide_verdict(primary_base, primary_stress)
    checks["historical_bid_ask_coverage"] = False
    # Public Spot REST does not reconstruct historical executable books. This fail-closed
    # evidence rule was preregistered before replay and caps the verdict.
    verdict = "BINANCE_STRATEGY_INSUFFICIENT_EVIDENCE"
    return (
        BinanceQualificationReport(
            generated_at_utc=datetime.now(UTC),
            start_utc=start_utc,
            end_utc=end_utc,
            symbols=symbols,
            bars_m1=frame.height,
            bars_m5=sum(len(items) for items in m5_by_symbol.values()),
            signal_observations=sum(len(items) for items in signals_by_symbol.values()),
            data_quality={
                "canonical_utc": True,
                "duplicate_symbol_timestamps": 0,
                "per_symbol": per_symbol_quality,
            },
            maker_fee_rate=maker_fee,
            taker_fee_rate=taker_fee,
            historical_spread_source="QUALIFICATION_START_BOOK_TICKER_SNAPSHOT_NOT_HISTORICAL",
            base_assumptions={
                "fee": str(taker_fee),
                "fee_source": "BINANCE_ACCOUNT_API",
                "slippage_bps_per_side": "1",
                "spread_multiplier": "1",
                "uncertainty_buffer_bps": "2",
            },
            stress_assumptions={
                "fee_multiplier": "1.5",
                "slippage_bps_per_side": "3",
                "spread_multiplier": "2",
                "uncertainty_buffer_bps": "5",
            },
            capital_scenarios_eur=(Decimal("100"), Decimal("500"), Decimal("1000")),
            results=tuple(results),
            qualification_checks=checks,
            limitations=(
                "NO_HISTORICAL_BID_ASK_OR_ORDER_BOOK_COVERAGE",
                "CURRENT_SPREAD_SNAPSHOT_USED_AS_EXECUTION_PROXY",
                "CURRENT_EXCHANGE_FILTERS_APPLIED_TO_HISTORICAL_REPLAY",
                "OHLC_SAME_BAR_AMBIGUITY_RESOLVED_STOP_FIRST",
                "SPOT_LONG_ONLY_NO_SHORT_SELLING",
                "REAL_ACCOUNT_BALANCE_NOT_USED_FOR_RESEARCH_SIZING",
            ),
            verdict=verdict,
        ),
        primary_trades,
    )


def render_second_delivery(report: BinanceQualificationReport) -> str:
    """Render a concise human report without hiding failed gates or no-trade days."""
    lines = [
        "# SNIPER — Deuxième livrable Binance Spot",
        "",
        f"- Verdict: `{report.verdict}`",
        f"- Période UTC: {report.start_utc.isoformat()} → {report.end_utc.isoformat()}",
        f"- Symboles: {', '.join(report.symbols)}",
        f"- M1 / M5 / signaux: {report.bars_m1:,} / {report.bars_m5:,} / "
        f"{report.signal_observations:,}",
        f"- Frais maker/taker: {report.maker_fee_rate} / {report.taker_fee_rate} "
        "(`BINANCE_ACCOUNT_API`)",
        "- LIVE / ordres: DISABLED / ABSENT",
        "",
        "## Résultats",
        "",
        "| Capital EUR | Scénario | Trades | Expectancy EUR | PF net | Return % | "
        "DD % | Semaines actives | Semaines + | Jours >=1% | Jours négatifs | Sans trade |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report.results:
        lines.append(
            f"| {item.starting_capital_eur} | {item.scenario} | {item.trades} | "
            f"{item.expectancy_eur:.8f} | {item.profit_factor} | "
            f"{item.net_return_pct:.6f} | {item.max_drawdown_pct:.6f} | "
            f"{item.active_trade_weeks} | {item.positive_week_ratio:.2%} | "
            f"{item.days_target_reached} | {item.negative_days} | {item.no_trade_days} |"
        )
    lines.extend(["", "## Qualification", ""])
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — `{name}`"
        for name, passed in report.qualification_checks.items()
    )
    lines.extend(["", "## Limites", ""])
    lines.extend(f"- `{item}`" for item in report.limitations)
    lines.extend(
        [
            "",
            "Le solde réel n'a servi ni au calibrage ni au sizing du replay. Aucune "
            "conversion EUR/USDT, aucun ordre, aucune route LIVE et aucune optimisation "
            "post-résultats n'ont été effectués.",
        ]
    )
    return "\n".join(lines) + "\n"
