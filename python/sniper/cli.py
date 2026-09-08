"""Read-only MT5 diagnostics and historical EUR/USD data collection."""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from sniper.backtest.engine import BacktestEngine, BacktestResult, DummyStrategy
from sniper.backtest.execution_model import (
    BrokerSimulationConfig,
    CommissionModel,
    MinimumCommission,
    NoCommission,
    PerLotCommission,
)
from sniper.backtest.slippage import FixedSlippage, NoSlippage, RandomSlippage, SlippageModel
from sniper.config import BrokerCheckConfig, Settings
from sniper.data.backtest_source import load_parquet_ticks
from sniper.data.collector import DataQualityReport, collect_ticks
from sniper.data.feature_source import load_parquet_bars
from sniper.data.history import HistoryCollectionReport, collect_history
from sniper.data.mt5_client import MT5Client, MT5Error
from sniper.data.parquet_store import ParquetStore
from sniper.data.time import CollectionConfig, TimePolicy, epoch_ms, parse_instant
from sniper.domain.broker import VolumeConstraints
from sniper.domain.edge_validation import EdgeValidationReport
from sniper.domain.evaluation import SignalEvaluationReport
from sniper.domain.event_discovery import EventDiscoveryReport
from sniper.domain.signal import SignalAnalysisReport
from sniper.domain.trade import Side
from sniper.evaluation.edge_validation import (
    EdgeValidationConfig,
    EdgeValidator,
    render_edge_report,
)
from sniper.evaluation.engine import SignalEvaluationConfig, SignalEvaluator
from sniper.evaluation.event_discovery import (
    EventCandidateDiscoverer,
    EventDiscoveryConfig,
    render_event_discovery_report,
)
from sniper.features.engine import FeatureEngine
from sniper.risk.broker_compatibility import (
    BrokerCompatibilityReport,
    Verdict,
    validate_broker_for_capital,
)
from sniper.strategy.filters import EvaluationContext
from sniper.strategy.scorer import SignalEngine

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@app.callback()
def main() -> None:
    """SNIPER: diagnostics MT5 et donnees EUR/USD en lecture seule."""


def safe_error(exc: Exception) -> str:
    """Avoid echoing terminal identity, paths or invalid secret-like values."""
    if isinstance(exc, ValidationError):
        fields = sorted({".".join(map(str, error["loc"])) for error in exc.errors()})
        return "Invalid configuration or terminal data: " + ", ".join(fields)
    if isinstance(exc, MT5Error):
        return str(exc)
    if isinstance(exc, FileNotFoundError):
        return str(exc)
    if isinstance(exc, (OSError, AttributeError, OverflowError, KeyError, TypeError)):
        return "Unable to read configuration or terminal data"
    return str(exc)


def print_report(report: BrokerCompatibilityReport) -> None:
    console = Console(markup=False)
    snapshot = report.snapshot
    currency = snapshot.account.currency

    def money(value: Decimal | None) -> str:
        return "UNKNOWN" if value is None else f"{value:.6f} {currency}"

    def tri(value: bool | None) -> str:
        return "UNKNOWN" if value is None else ("YES" if value else "NO")

    table = Table(title="SNIPER Broker Compatibility", show_header=False)
    table.add_column("Field")
    table.add_column("Value")
    rows = [
        ("Capital", f"{report.capital_eur:.2f} EUR"),
        ("Account currency", currency),
        (
            "Account equity / free margin",
            f"{money(snapshot.account.equity)} / {money(snapshot.account.free_margin)}",
        ),
        ("Symbol", snapshot.symbol.name),
        ("Quote UTC", snapshot.quote.timestamp_utc.isoformat()),
        (
            "Quote server representation",
            snapshot.quote.timestamp_server.isoformat()
            if snapshot.quote.timestamp_server
            else "UNKNOWN (server timezone not provided)",
        ),
        (
            "Quote local representation",
            snapshot.quote.timestamp_local.isoformat()
            if snapshot.quote.timestamp_local
            else "UNKNOWN",
        ),
        ("Quote time basis", snapshot.quote.time_basis),
        ("Bid / Ask", f"{snapshot.quote.bid} / {snapshot.quote.ask}"),
        ("Min volume", str(snapshot.symbol.volumes.minimum)),
        ("Volume step", str(snapshot.symbol.volumes.step)),
        ("Max volume", str(snapshot.symbol.volumes.maximum)),
        ("Nano-lot compliant", tri(report.nano_lot_compliant)),
        ("Tick size / value", f"{snapshot.symbol.tick_size} / {money(snapshot.symbol.tick_value)}"),
        ("Contract size", str(snapshot.symbol.contract_size)),
        ("Current spread", f"{report.spread_points} points"),
        ("Median spread", "UNKNOWN (run collect-ticks for a measured range)"),
        (
            "Commission model",
            report.policy.commission.model_dump_json() if report.policy.commission else "UNKNOWN",
        ),
        ("Round-trip commission at min volume", money(report.round_trip_commission)),
        ("Min round-trip cost (spread + commission)", money(report.minimum_round_trip_cost)),
        ("Estimated round-trip cost (+ slippage)", money(report.estimated_round_trip_cost)),
        (
            "Margin at min volume BUY / SELL",
            f"{money(report.margin_buy)} / {money(report.margin_sell)}",
        ),
        ("EA allowed (account)", tri(snapshot.account.expert_allowed)),
        ("Scalping allowed (verified terms)", tri(report.policy.scalping_allowed)),
        (
            "Legal access from France (verified terms)",
            tri(report.policy.legally_accessible_from_france),
        ),
        ("Historical ticks (verified)", tri(report.policy.historical_ticks_available)),
    ]
    for label, value in rows:
        table.add_row(label, value)
    for stop in report.stops:
        pct = "UNKNOWN" if stop.risk_pct is None else f"{stop.risk_pct:.4f}%"
        table.add_row(
            f"Risk at {stop.stop_pips} pip stop {stop.side}",
            f"{money(stop.total_loss)} ({pct}); price loss: {money(stop.price_loss)}; "
            f"SL={stop.stop_price}; broker valid={tri(stop.broker_stop_valid)}",
        )
    table.add_row("Verdict", report.verdict.value)
    table.add_row("Challenge status", report.challenge_status.value)
    table.add_row("Live trading enabled", "NO")
    console.print(table)
    for blocker in report.blockers:
        console.print(f"- {blocker}")


@app.command("broker-check")
def broker_check(
    capital: Annotated[str, typer.Option(help="Capital en EUR.")] = "10",
    symbol: Annotated[str, typer.Option(help="Nom EUR/USD exact dans le terminal.")] = "EURUSD",
    stop_pips: Annotated[list[str] | None, typer.Option(help="Repetable: --stop-pips 3.")] = None,
    config: Annotated[
        Path | None, typer.Option(help="Politique BrokerCheckConfig au format JSON.")
    ] = None,
    terminal_path: Annotated[str | None, typer.Option(help="Chemin du terminal MT5.")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Rapport JSON auditable.")] = False,
) -> None:
    """Lire MT5 et verifier les contraintes broker sans envoyer d'ordre."""
    try:
        settings = Settings()
        policy = (
            BrokerCheckConfig.model_validate_json(config.read_text(encoding="utf-8"))
            if config
            else settings.broker
        )
        if stop_pips is not None:
            policy = BrokerCheckConfig.model_validate(
                {**policy.model_dump(), "stop_pips": stop_pips}
            )
        amount = Decimal(capital)
        if not amount.is_finite() or amount <= 0:
            raise ValueError("capital must be positive and finite (EUR)")
        with MT5Client(
            path=terminal_path or settings.mt5_path, timeout_ms=settings.mt5_timeout_ms
        ) as client:
            report = validate_broker_for_capital(client, amount, symbol, policy)
    except (MT5Error, ValueError, InvalidOperation, OSError, AttributeError, OverflowError) as exc:
        message = safe_error(exc)
        if json_output:
            typer.echo(json.dumps({"error": message, "live_trading_enabled": False}))
        else:
            typer.echo(f"SNIPER broker-check: {message}", err=True)
        raise typer.Exit(2) from None
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        print_report(report)
    raise typer.Exit(0 if report.verdict == Verdict.COMPATIBLE else 1)


def print_data_report(report: DataQualityReport) -> None:
    console = Console(markup=False)
    spread = report.spread_points
    table = Table(title="SNIPER EURUSD Data Quality", show_header=False)
    table.add_column("Field")
    table.add_column("Value")
    table.add_row(
        "Requested UTC",
        f"{report.requested_start_utc.isoformat()} -> {report.requested_end_utc.isoformat()}",
    )
    table.add_row(
        "Covered UTC",
        f"{report.covered_start_utc or 'EMPTY'} -> {report.covered_end_utc or 'EMPTY'}",
    )
    table.add_row("Source / accepted ticks", f"{report.source_rows} / {report.accepted_ticks}")
    table.add_row(
        "Duplicates / invalid / outside range",
        f"{report.duplicate_ticks} / {report.invalid_ticks} / {report.out_of_range_ticks}",
    )
    table.add_row("Detected gaps", f"{len(report.gaps)} (> {report.gap_threshold_seconds:g}s)")
    table.add_row(
        "Spread points min / median / p95 / max",
        "EMPTY"
        if spread is None
        else f"{spread.minimum} / {spread.median} / {spread.p95} / {spread.maximum}",
    )
    table.add_row(
        "Bars M1 / M5 / M15", f"{report.bars['M1']} / {report.bars['M5']} / {report.bars['M15']}"
    )
    table.add_row(
        "Parquet partitions ticks / bars",
        f"{report.tick_partitions_written} / {report.bar_partitions_written}",
    )
    table.add_row("Time basis", report.time_basis)
    table.add_row("Output", report.output_root)
    console.print(table)


@app.command("collect-ticks")
def collect_ticks_command(
    start: Annotated[str, typer.Option(help="Debut ISO 8601 avec fuseau, inclus.")],
    end: Annotated[str, typer.Option(help="Fin ISO 8601 avec fuseau, exclue.")],
    output: Annotated[Path, typer.Option(help="Racine des partitions Parquet.")] = Path("data"),
    symbol: Annotated[str, typer.Option(help="EURUSD uniquement en V1.")] = "EURUSD",
    chunk_minutes: Annotated[int, typer.Option(help="Taille des requetes MT5.")] = 60,
    gap_seconds: Annotated[float, typer.Option(help="Seuil de trou entre deux ticks.")] = 60.0,
    terminal_path: Annotated[str | None, typer.Option(help="Chemin du terminal MT5.")] = None,
    server_timezone: Annotated[
        str | None, typer.Option(help="Fuseau IANA du serveur, pour affichage seulement.")
    ] = None,
    local_timezone: Annotated[
        str, typer.Option(help="Fuseau IANA local d'affichage.")
    ] = "Europe/Paris",
    json_output: Annotated[bool, typer.Option("--json", help="Rapport JSON auditable.")] = False,
) -> None:
    """Telecharger, valider et partitionner des ticks MT5 EURUSD."""
    try:
        settings = Settings()
        start_utc, end_utc = parse_instant(start), parse_instant(end)
        time_policy = TimePolicy(
            source_basis="UTC",
            server_timezone=server_timezone,
            local_timezone=local_timezone,
        )
        config = CollectionConfig(
            time=time_policy,
            chunk_minutes=chunk_minutes,
            gap_threshold_seconds=gap_seconds,
        )
        with MT5Client(
            path=terminal_path or settings.mt5_path,
            timeout_ms=settings.mt5_timeout_ms,
            time_policy=time_policy,
        ) as client:
            report = collect_ticks(
                client, ParquetStore(output), start_utc, end_utc, config, symbol=symbol
            )
        report_dir = output / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"collection-{epoch_ms(start_utc)}-{epoch_ms(end_utc)}.json"
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    except (
        MT5Error,
        ValidationError,
        ValueError,
        InvalidOperation,
        OSError,
        AttributeError,
        OverflowError,
        KeyError,
        TypeError,
    ) as exc:
        message = safe_error(exc)
        if json_output:
            typer.echo(json.dumps({"error": message, "data_fabricated": False}))
        else:
            typer.echo(f"SNIPER collect-ticks: {message}", err=True)
        raise typer.Exit(2) from None
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        print_data_report(report)


@app.command("collect-history")
def collect_history_command(
    days: Annotated[int, typer.Option(help="Nombre de jours UTC complets a collecter.")] = 90,
    output: Annotated[Path, typer.Option(help="Racine des partitions Parquet.")] = Path("data"),
    symbol: Annotated[str, typer.Option(help="EURUSD uniquement en V1.")] = "EURUSD",
    end: Annotated[
        str | None,
        typer.Option(help="Fin UTC optionnelle; par defaut minuit UTC courant."),
    ] = None,
    chunk_minutes: Annotated[int, typer.Option(help="Taille des requetes MT5.")] = 60,
    gap_seconds: Annotated[float, typer.Option(help="Seuil de trou entre deux ticks.")] = 60.0,
    terminal_path: Annotated[str | None, typer.Option(help="Chemin du terminal MT5.")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Rapport JSON auditable.")] = False,
) -> None:
    """Collecter l'historique par jour avec reprise sur manifestes verifies."""
    try:
        if days < 1:
            raise ValueError("days must be positive")
        end_utc = (
            parse_instant(end)
            if end is not None
            else datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        )
        start_utc = end_utc - timedelta(days=days)
        settings = Settings()
        time_policy = TimePolicy(source_basis="UTC", local_timezone="Europe/Paris")
        config = CollectionConfig(
            time=time_policy,
            chunk_minutes=chunk_minutes,
            gap_threshold_seconds=gap_seconds,
        )
        with MT5Client(
            path=terminal_path or settings.mt5_path,
            timeout_ms=settings.mt5_timeout_ms,
            time_policy=time_policy,
        ) as client:
            report = collect_history(
                client,
                ParquetStore(output),
                start_utc,
                end_utc,
                config,
                symbol=symbol,
            )
        report_dir = output / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / f"history-{epoch_ms(start_utc)}-{epoch_ms(end_utc)}.json"
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    except (
        MT5Error,
        ValidationError,
        ValueError,
        InvalidOperation,
        OSError,
        AttributeError,
        OverflowError,
        KeyError,
        TypeError,
    ) as exc:
        message = safe_error(exc)
        if json_output:
            typer.echo(json.dumps({"error": message, "data_fabricated": False}))
        else:
            typer.echo(f"SNIPER collect-history: {message}", err=True)
        raise typer.Exit(2) from None
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        print_history_report(report)


def print_history_report(report: HistoryCollectionReport) -> None:
    table = Table(title="SNIPER Historical Collection — DAILY RESUMABLE", show_header=False)
    table.add_column("Field")
    table.add_column("Value")
    for label, value in (
        ("Requested UTC", f"{report.requested_start_utc} -> {report.requested_end_utc}"),
        ("Downloaded / resumed days", f"{report.downloaded_chunks} / {report.resumed_chunks}"),
        ("New ticks", report.accepted_ticks_downloaded),
        ("Resume policy", report.resume_policy),
        ("Output", report.output_root),
        ("Live trading", "DISABLED"),
    ):
        table.add_row(label, str(value))
    Console(markup=False).print(table)


def print_backtest_report(result: BacktestResult) -> None:
    metrics = result.metrics
    profile = result.broker_profile
    table = Table(title="SNIPER Backtest — DummyStrategy(TEST_ONLY)", show_header=False)
    table.add_column("Field")
    table.add_column("Value")
    for label, value in (
        ("Ticks", result.ticks_processed),
        ("Broker profile", f"{profile.name} ({profile.kind})"),
        ("Real broker capability", "NO — simulated research assumptions only"),
        ("Effective leverage", f"1:{profile.effective_leverage}"),
        ("Volume min / step", f"{profile.volume_minimum} / {profile.volume_step} lot"),
        ("Starting / ending equity", f"{metrics.starting_equity} / {metrics.ending_equity} EUR"),
        ("Net return", f"{metrics.net_return_pct}%"),
        ("Gross / net PnL", f"{metrics.gross_pnl} / {metrics.net_pnl} EUR"),
        (
            "Spread / slippage / commission",
            f"{metrics.total_spread_cost} / {metrics.total_slippage} / "
            f"{metrics.total_commission} EUR",
        ),
        (
            "Trades / wins / losses",
            f"{metrics.trades_count} / {metrics.winning_trades} / {metrics.losing_trades}",
        ),
        ("Max drawdown", f"{metrics.max_drawdown_value} EUR ({metrics.max_drawdown_pct}%)"),
        ("Rejected orders", len(result.rejections)),
        ("Live trading", "DISABLED"),
    ):
        table.add_row(label, str(value))
    Console(markup=False).print(table)


@app.command("backtest")
def backtest_command(
    start: Annotated[str, typer.Option(help="Debut UTC ISO 8601 inclus.")],
    end: Annotated[str, typer.Option(help="Fin UTC ISO 8601 exclue.")],
    symbol: Annotated[str, typer.Option(help="EURUSD uniquement en V1.")] = "EURUSD",
    capital: Annotated[str, typer.Option(help="Capital initial EUR.")] = "10",
    data: Annotated[Path, typer.Option(help="Racine des Parquet Phase B.")] = Path("data"),
    execution_latency_ms: Annotated[int, typer.Option(help="0, 25, 50, 100 ou 250.")] = 0,
    side: Annotated[Side, typer.Option(help="Sens du DummyStrategy de test.")] = Side.BUY,
    volume: Annotated[str, typer.Option(help="Volume du trade de test.")] = "0.0001",
    hold_ticks: Annotated[int, typer.Option(help="Ticks avant fermeture manuelle.")] = 10,
    stop_points: Annotated[str, typer.Option(help="Stop du test en points.")] = "50",
    take_profit_points: Annotated[str, typer.Option(help="Target du test en points.")] = "50",
    slippage_model: Annotated[str, typer.Option(help="fixed, random ou none.")] = "fixed",
    slippage_points: Annotated[int, typer.Option(help="Points fixes ou maximum aleatoire.")] = 1,
    seed: Annotated[int, typer.Option(help="Seed du slippage aleatoire.")] = 42,
    commission_per_lot_side: Annotated[str, typer.Option(help="Commission EUR/lot/cote.")] = "0",
    minimum_commission_side: Annotated[str, typer.Option(help="Minimum EUR par cote.")] = "0",
    volume_minimum: Annotated[str, typer.Option(help="Volume minimum simule.")] = "0.0001",
    volume_step: Annotated[str, typer.Option(help="Pas de volume simule.")] = "0.0001",
    margin_per_lot: Annotated[
        str, typer.Option(help="Marge EUR simulee par lot.")
    ] = "3333.333333333333",
    max_risk_pct: Annotated[str, typer.Option(help="Risque maximum par trade.")] = "0.50",
    json_output: Annotated[bool, typer.Option("--json", help="Rapport et journal JSON.")] = False,
) -> None:
    """Rejouer les ticks avec DummyStrategy(TEST_ONLY), jamais une strategie SNIPER."""
    try:
        start_utc, end_utc = parse_instant(start), parse_instant(end)
        amount = Decimal(capital)
        requested_volume = Decimal(volume)
        minimum = Decimal(volume_minimum)
        step = Decimal(volume_step)
        commission_rate = Decimal(commission_per_lot_side)
        commission_minimum = Decimal(minimum_commission_side)
        config = BrokerSimulationConfig(
            initial_capital=amount,
            volumes=VolumeConstraints(minimum=minimum, step=step, maximum=Decimal(1)),
            execution_latency_ms=execution_latency_ms,
            margin_per_lot=Decimal(margin_per_lot),
            max_risk_pct=Decimal(max_risk_pct),
        )
        config.validate_latency()
        if slippage_points < 0 and slippage_model != "fixed":
            raise ValueError("negative slippage points are only valid for fixed favorable stress")
        if slippage_model == "fixed":
            slippage: SlippageModel = FixedSlippage(Decimal(slippage_points))
        elif slippage_model == "random":
            slippage = RandomSlippage(slippage_points, seed)
        elif slippage_model == "none":
            slippage = NoSlippage()
        else:
            raise ValueError("slippage-model must be fixed, random or none")
        if commission_minimum > 0:
            commission: CommissionModel = MinimumCommission(commission_rate, commission_minimum)
        elif commission_rate > 0:
            commission = PerLotCommission(commission_rate)
        else:
            commission = NoCommission()
        ticks = load_parquet_ticks(data, symbol, start_utc, end_utc)
        if not ticks:
            raise ValueError("the requested Phase B range contains no ticks")
        strategy = DummyStrategy(
            side=side,
            volume=requested_volume,
            hold_ticks=hold_ticks,
            stop_points=Decimal(stop_points),
            take_profit_points=Decimal(take_profit_points),
        )
        result = BacktestEngine(
            config,
            slippage=slippage,
            commission=commission,
        ).run(ticks, strategy)
        report_dir = data / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"backtest-{epoch_ms(start_utc)}-{epoch_ms(end_utc)}.json"
        report_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    except (
        ValidationError,
        ValueError,
        InvalidOperation,
        OSError,
        AttributeError,
        OverflowError,
        KeyError,
        TypeError,
    ) as exc:
        message = safe_error(exc)
        if json_output:
            typer.echo(json.dumps({"error": message, "live_trading_enabled": False}))
        else:
            typer.echo(f"SNIPER backtest: {message}", err=True)
        raise typer.Exit(2) from None
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        print_backtest_report(result)


def print_signal_report(report: SignalAnalysisReport) -> None:
    decision = report.decision
    table = Table(title="SNIPER Signal Analysis — FIXED_NOT_OPTIMIZED", show_header=False)
    table.add_column("Field")
    table.add_column("Value")
    for label, value in (
        ("As of UTC", decision.as_of_utc.isoformat()),
        (
            "Decision / market bias / trigger",
            f"{decision.side.value} / {decision.market_bias.value} / "
            f"{decision.trigger_direction.value}",
        ),
        ("Score / tier", f"{decision.score} / {decision.tier.value}"),
        ("Reasons", ", ".join(decision.reasons) or "NONE"),
        ("Blockers", ", ".join(decision.blockers) or "NONE"),
        (
            "Proposed stop / target points",
            f"{decision.proposed_stop_distance_points} / "
            f"{decision.proposed_target_distance_points}",
        ),
        ("Sizing authority", decision.sizing_authority),
        ("Live trading", "DISABLED"),
    ):
        table.add_row(label, str(value))
    Console(markup=False).print(table)


@app.command("signal")
def signal_command(
    as_of: Annotated[str, typer.Option(help="Instant UTC ISO 8601 de l'analyse.")],
    symbol: Annotated[str, typer.Option(help="EURUSD uniquement en V1.")] = "EURUSD",
    data: Annotated[Path, typer.Option(help="Racine des Parquet Phase B.")] = Path("data"),
    max_spread_points: Annotated[
        str | None, typer.Option(help="Seuil mesure/configure; absent = blocage fail-closed.")
    ] = None,
    max_tick_age_seconds: Annotated[
        str, typer.Option(help="Age maximal du dernier tick visible.")
    ] = "2",
    session_status: Annotated[str, typer.Option(help="allowed, blocked ou unknown.")] = "unknown",
    news_status: Annotated[str, typer.Option(help="clear, blocked ou unknown.")] = "unknown",
    lookback_days: Annotated[int, typer.Option(help="Historique maximum des bougies.")] = 30,
    json_output: Annotated[bool, typer.Option("--json", help="Features et decision JSON.")] = False,
) -> None:
    """Analyser EURUSD sans sizing, ordre, optimisation ni trading live."""
    try:
        instant = parse_instant(as_of)
        session_map = {"allowed": True, "blocked": False, "unknown": None}
        news_map = {"clear": True, "blocked": False, "unknown": None}
        if session_status not in session_map or news_status not in news_map:
            raise ValueError("session-status/news-status value is invalid")
        context = EvaluationContext(
            max_spread_points=Decimal(max_spread_points) if max_spread_points is not None else None,
            max_tick_age_seconds=Decimal(max_tick_age_seconds),
            session_allowed=session_map[session_status],
            news_clear=news_map[news_status],
        )
        bars = load_parquet_bars(data, symbol, instant, lookback_days=lookback_days)
        ticks = load_parquet_ticks(
            data,
            symbol,
            instant - timedelta(seconds=60),
            instant + timedelta(microseconds=1),
        )
        features = FeatureEngine().compute(
            bars=bars,
            ticks=ticks,
            as_of_utc=instant,
            symbol=symbol,
        )
        decision = SignalEngine().evaluate(features, context)
        report = SignalAnalysisReport(features=features, decision=decision)
        report_dir = data / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"signal-{epoch_ms(instant)}.json"
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    except (
        ValidationError,
        ValueError,
        InvalidOperation,
        OSError,
        AttributeError,
        OverflowError,
        KeyError,
        TypeError,
    ) as exc:
        message = safe_error(exc)
        if json_output:
            typer.echo(json.dumps({"error": message, "live_trading_enabled": False}))
        else:
            typer.echo(f"SNIPER signal: {message}", err=True)
        raise typer.Exit(2) from None
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        print_signal_report(report)


def print_evaluation_report(report: SignalEvaluationReport) -> None:
    table = Table(title="SNIPER Phase D.5 — SIGNAL EVALUATION ONLY")
    table.add_column("Score")
    table.add_column("Obs.", justify="right")
    table.add_column("Win 30s")
    table.add_column("MFE")
    table.add_column("MAE")
    table.add_column("Net potentiel")
    table.add_column("Cout/target")
    for bucket in report.score_buckets:
        table.add_row(
            bucket.label,
            str(bucket.observations),
            "N/A"
            if bucket.win_direction_30s_pct is None
            else f"{bucket.win_direction_30s_pct:.2f}%",
            "N/A"
            if bucket.average_mfe_30s_points is None
            else f"{bucket.average_mfe_30s_points:.2f}",
            "N/A"
            if bucket.average_mae_30s_points is None
            else f"{bucket.average_mae_30s_points:.2f}",
            "N/A"
            if bucket.average_potential_net_return_30s_points is None
            else f"{bucket.average_potential_net_return_30s_points:.2f}",
            "N/A"
            if bucket.target_cost_share_pct is None
            else f"{bucket.target_cost_share_pct:.2f}%",
        )
    console = Console(markup=False)
    console.print(table)
    console.print(f"Diagnostic: {report.monotonic_diagnostics.conclusion}")
    console.print("Optimization: NOT PERFORMED | Live trading: DISABLED")


@app.command("evaluate-signals")
def evaluate_signals_command(
    start: Annotated[str, typer.Option(help="Debut UTC ISO 8601 inclus.")],
    end: Annotated[str, typer.Option(help="Fin UTC ISO 8601 exclue.")],
    data: Annotated[Path, typer.Option(help="Racine des Parquet Phase B.")] = Path("data"),
    symbol: Annotated[str, typer.Option(help="EURUSD uniquement en V1.")] = "EURUSD",
    interval_seconds: Annotated[int, typer.Option(help="Pas regulier d'observation.")] = 60,
    max_horizon_tick_delay_seconds: Annotated[
        str, typer.Option(help="Tolerance max du premier tick apres chaque horizon.")
    ] = "2",
    max_spread_points: Annotated[
        str | None, typer.Option(help="Seuil mesure/configure; absent = blocage fail-closed.")
    ] = None,
    max_tick_age_seconds: Annotated[
        str, typer.Option(help="Age maximal du dernier tick visible.")
    ] = "2",
    session_status: Annotated[str, typer.Option(help="allowed, blocked ou unknown.")] = "unknown",
    news_status: Annotated[str, typer.Option(help="clear, blocked ou unknown.")] = "unknown",
    slippage_points_per_side: Annotated[
        str, typer.Option(help="Slippage attendu par cote, en points.")
    ] = "1",
    commission_points_round_trip: Annotated[
        str, typer.Option(help="Commission aller-retour convertie en points.")
    ] = "0",
    minimum_target_cost_ratio: Annotated[
        str, typer.Option(help="Ratio cible/cout de reference; mesure seulement.")
    ] = "3",
    json_output: Annotated[bool, typer.Option("--json", help="Rapport JSON auditable.")] = False,
) -> None:
    """Mesurer hors echantillon la valeur predictive du score, sans optimisation."""
    try:
        start_utc, end_utc = parse_instant(start), parse_instant(end)
        session_map = {"allowed": True, "blocked": False, "unknown": None}
        news_map = {"clear": True, "blocked": False, "unknown": None}
        if session_status not in session_map or news_status not in news_map:
            raise ValueError("session-status/news-status value is invalid")
        context = EvaluationContext(
            max_spread_points=Decimal(max_spread_points) if max_spread_points is not None else None,
            max_tick_age_seconds=Decimal(max_tick_age_seconds),
            session_allowed=session_map[session_status],
            news_clear=news_map[news_status],
        )
        config = SignalEvaluationConfig(
            interval_seconds=interval_seconds,
            max_horizon_tick_delay_seconds=Decimal(max_horizon_tick_delay_seconds),
            slippage_points_per_side=Decimal(slippage_points_per_side),
            commission_points_round_trip=Decimal(commission_points_round_trip),
            minimum_target_cost_ratio=Decimal(minimum_target_cost_ratio),
        )
        duration_days = max(1, (end_utc - start_utc).days + 2)
        bars = load_parquet_bars(data, symbol, end_utc, lookback_days=duration_days + 30)
        ticks = load_parquet_ticks(
            data,
            symbol,
            start_utc - timedelta(seconds=60),
            end_utc,
        )
        report = SignalEvaluator(config).run(
            bars=bars,
            ticks=ticks,
            start_utc=start_utc,
            end_utc=end_utc,
            context=context,
            symbol=symbol,
        )
        report_dir = data / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / f"signal-evaluation-{epoch_ms(start_utc)}-{epoch_ms(end_utc)}.json"
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    except (
        ValidationError,
        ValueError,
        InvalidOperation,
        OSError,
        AttributeError,
        OverflowError,
        KeyError,
        TypeError,
    ) as exc:
        message = safe_error(exc)
        if json_output:
            typer.echo(json.dumps({"error": message, "live_trading_enabled": False}))
        else:
            typer.echo(f"SNIPER evaluate-signals: {message}", err=True)
        raise typer.Exit(2) from None
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        print_evaluation_report(report)


def print_edge_validation_report(report: EdgeValidationReport, human_path: Path) -> None:
    summary = Table(title="SNIPER Phase D.6 — EDGE VALIDATION ONLY", show_header=False)
    summary.add_column("Field")
    summary.add_column("Value")
    for label, value in (
        ("Diagnostic", report.diagnosis),
        ("Ticks analyzed", f"{report.performance.ticks_analyzed:,}"),
        ("Observations", f"{report.performance.observations_produced:,}"),
        ("Usable candidates", f"{report.candidate_overall.usable:,}"),
        (
            "Candidate net expectancy",
            report.candidate_overall.expectancy_after_commission_points,
        ),
        ("Elapsed seconds", f"{report.performance.elapsed_seconds:.2f}"),
        ("Ticks / second", f"{report.performance.ticks_per_second:,.0f}"),
        ("Human report", human_path.resolve()),
        ("Signal parameters modified", "NO"),
        ("Live trading", "DISABLED"),
    ):
        summary.add_row(label, str(value))
    Console(markup=False).print(summary)


@app.command("validate-edge")
def validate_edge_command(
    start: Annotated[str, typer.Option(help="Debut UTC ISO 8601 inclus.")],
    end: Annotated[str, typer.Option(help="Fin UTC ISO 8601 exclue.")],
    data: Annotated[Path, typer.Option(help="Racine des Parquet EURUSD.")] = Path("data"),
    symbol: Annotated[str, typer.Option(help="EURUSD uniquement en V1.")] = "EURUSD",
    max_spread_points: Annotated[
        str, typer.Option(help="Seuil de spread du contexte Signal Engine.")
    ] = "2",
    max_tick_age_seconds: Annotated[
        str, typer.Option(help="Age maximal du tick du contexte Signal Engine.")
    ] = "2",
    slippage_points_per_side: Annotated[
        str, typer.Option(help="Hypothese simulee de slippage par cote.")
    ] = "1",
    commission_eur_per_lot_per_side: Annotated[
        str, typer.Option(help="Hypothese simulee de commission EUR/lot/cote.")
    ] = "2",
    commission_minimum_eur_per_side: Annotated[
        str, typer.Option(help="Minimum simule de commission EUR/cote.")
    ] = "0",
    bootstrap_seed: Annotated[
        int, typer.Option(help="Seed fixe du bootstrap par jour.")
    ] = 20260908,
    bootstrap_resamples: Annotated[int, typer.Option(help="Nombre de reechantillonnages.")] = 5000,
    json_output: Annotated[bool, typer.Option("--json", help="Afficher aussi le JSON.")] = False,
) -> None:
    """Valider l'edge du moteur gele en streaming, sans optimisation ni trading."""
    try:
        if symbol != "EURUSD":
            raise ValueError("Phase D.6 supports EURUSD only")
        start_utc, end_utc = parse_instant(start), parse_instant(end)
        config = EdgeValidationConfig(
            max_spread_points=Decimal(max_spread_points),
            max_tick_age_seconds=Decimal(max_tick_age_seconds),
            slippage_points_per_side=Decimal(slippage_points_per_side),
            commission_eur_per_lot_per_side=Decimal(commission_eur_per_lot_per_side),
            commission_minimum_eur_per_side=Decimal(commission_minimum_eur_per_side),
            bootstrap_seed=bootstrap_seed,
            bootstrap_resamples=bootstrap_resamples,
        )
        report = EdgeValidator(config).run(
            data_root=data,
            start_utc=start_utc,
            end_utc=end_utc,
        )
        report_dir = data / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        stem = f"edge-validation-{epoch_ms(start_utc)}-{epoch_ms(end_utc)}"
        json_path = report_dir / f"{stem}.json"
        human_path = report_dir / f"{stem}.md"
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        human_path.write_text(render_edge_report(report), encoding="utf-8")
    except (
        ValidationError,
        ValueError,
        RuntimeError,
        InvalidOperation,
        OSError,
        AttributeError,
        OverflowError,
        KeyError,
        TypeError,
    ) as exc:
        message = safe_error(exc)
        if json_output:
            typer.echo(json.dumps({"error": message, "live_trading_enabled": False}))
        else:
            typer.echo(f"SNIPER validate-edge: {message}", err=True)
        raise typer.Exit(2) from None
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        print_edge_validation_report(report, human_path)


def print_event_discovery_report(report: EventDiscoveryReport, human_path: Path) -> None:
    summary = Table(title="SNIPER Phase D.7 — EVENT DISCOVERY ONLY", show_header=False)
    summary.add_column("Field")
    summary.add_column("Value")
    first_original = report.direction_diagnostics[0]
    first_inverse = report.direction_diagnostics[1]
    for label, value in (
        ("Conclusion", report.conclusion),
        ("Ticks analyzed", f"{report.performance.ticks_analyzed:,}"),
        ("Active seconds scanned", f"{report.performance.active_seconds_scanned:,}"),
        ("Signal evaluations", f"{report.performance.signal_engine_evaluations:,}"),
        ("Episodes", f"{report.episode_summary.episodes:,}"),
        ("Usable first crossings", f"{first_original.usable:,}"),
        ("Original gross 30s", first_original.gross_expectancy_points.seconds_30),
        ("Inverse gross 30s", first_inverse.gross_expectancy_points.seconds_30),
        ("Elapsed seconds", f"{report.performance.elapsed_seconds:.2f}"),
        ("Human report", human_path.resolve()),
        ("Signal parameters modified", "NO"),
        ("Live trading", "DISABLED"),
    ):
        summary.add_row(label, str(value))
    Console(markup=False).print(summary)


@app.command("discover-candidate-events")
def discover_candidate_events_command(
    start: Annotated[str, typer.Option(help="Debut UTC ISO 8601 inclus.")],
    end: Annotated[str, typer.Option(help="Fin UTC ISO 8601 exclue.")],
    data: Annotated[Path, typer.Option(help="Racine des Parquet EURUSD.")] = Path("data"),
    symbol: Annotated[str, typer.Option(help="EURUSD uniquement en V1.")] = "EURUSD",
    max_spread_points: Annotated[
        str, typer.Option(help="Seuil de spread gele du contexte Signal Engine.")
    ] = "2",
    max_tick_age_seconds: Annotated[
        str, typer.Option(help="Age maximal gele du tick Signal Engine.")
    ] = "2",
    slippage_points_per_side: Annotated[
        str, typer.Option(help="Hypothese simulee de slippage par cote.")
    ] = "1",
    commission_eur_per_lot_per_side: Annotated[
        str, typer.Option(help="Hypothese simulee de commission EUR/lot/cote.")
    ] = "2",
    commission_minimum_eur_per_side: Annotated[
        str, typer.Option(help="Minimum simule de commission EUR/cote.")
    ] = "0",
    json_output: Annotated[bool, typer.Option("--json", help="Afficher aussi le JSON.")] = False,
) -> None:
    """Découvrir les épisodes >=90 du moteur gelé, sans optimisation ni trading."""
    try:
        if symbol != "EURUSD":
            raise ValueError("Phase D.7 supports EURUSD only")
        start_utc, end_utc = parse_instant(start), parse_instant(end)
        config = EventDiscoveryConfig(
            max_spread_points=Decimal(max_spread_points),
            max_tick_age_seconds=Decimal(max_tick_age_seconds),
            slippage_points_per_side=Decimal(slippage_points_per_side),
            commission_eur_per_lot_per_side=Decimal(commission_eur_per_lot_per_side),
            commission_minimum_eur_per_side=Decimal(commission_minimum_eur_per_side),
        )
        report = EventCandidateDiscoverer(config).run(
            data_root=data,
            start_utc=start_utc,
            end_utc=end_utc,
        )
        report_dir = data / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        stem = f"event-discovery-{epoch_ms(start_utc)}-{epoch_ms(end_utc)}"
        json_path = report_dir / f"{stem}.json"
        human_path = report_dir / f"{stem}.md"
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        human_path.write_text(render_event_discovery_report(report), encoding="utf-8")
    except (
        ValidationError,
        ValueError,
        RuntimeError,
        InvalidOperation,
        OSError,
        AttributeError,
        OverflowError,
        KeyError,
        TypeError,
    ) as exc:
        message = safe_error(exc)
        if json_output:
            typer.echo(json.dumps({"error": message, "live_trading_enabled": False}))
        else:
            typer.echo(f"SNIPER discover-candidate-events: {message}", err=True)
        raise typer.Exit(2) from None
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        print_event_discovery_report(report, human_path)
