import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from conftest import NOW
from sniper import cli
from sniper.data.mt5_client import MT5Client
from sniper.data.parquet_store import ParquetStore
from typer.testing import CliRunner

from tests.unit.test_features import BASE, full_inputs
from tests.unit.test_features import ticks as feature_ticks
from tests.unit.test_signal_evaluation import evaluation_ticks

runner = CliRunner()


@pytest.fixture
def connected_cli(api, policy, monkeypatch, tmp_path):
    api.tick.time_msc = int(datetime.now(UTC).timestamp() * 1000)
    monkeypatch.setattr(cli, "MT5Client", lambda **kwargs: MT5Client(api=api, **kwargs))
    path = tmp_path / "verified-synthetic-policy.json"
    path.write_text(policy.model_dump_json(), encoding="utf-8")
    return ["broker-check", "--capital", "10", "--symbol", "EURUSD", "--config", str(path)]


def test_cli_desired_command_and_json(connected_cli, api):
    result = runner.invoke(cli.app, connected_cli + ["--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["capital_eur"] == "10"
    assert report["verdict"] == "COMPATIBLE"
    assert report["snapshot"]["symbol"]["name"] == "EURUSD"
    assert report["live_trading_enabled"] is False
    assert api.calls[-1] == ("shutdown",)


def test_cli_human_report_fields(connected_cli):
    result = runner.invoke(cli.app, connected_cli, env={"COLUMNS": "220"})
    assert result.exit_code == 0, result.output
    for label in [
        "SNIPER Broker Compatibility",
        "Capital",
        "10.00 EUR",
        "Nano-lot compliant",
        "Commission model",
        "Current spread",
        "Min round-trip cost",
        "Risk at 3 pip stop",
        "Risk at 5 pip stop",
        "Risk at 10 pip stop",
        "COMPATIBLE",
    ]:
        assert label in result.output


def test_cli_without_policy_reports_unknown(connected_cli):
    result = runner.invoke(
        cli.app, ["broker-check", "--capital", "10", "--symbol", "EURUSD", "--json"]
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout)["verdict"] == "INCOMPATIBLE_FOR_10_EUR"


def test_custom_stop_distances(connected_cli):
    result = runner.invoke(cli.app, connected_cli + ["--stop-pips", "7", "--json"])
    assert result.exit_code == 0
    assert {row["stop_pips"] for row in json.loads(result.stdout)["stops"]} == {"7"}


def test_cli_refusal_code_and_diagnostics(connected_cli, api):
    api.info.volume_min = 0.01
    result = runner.invoke(cli.app, connected_cli + ["--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["verdict"] == "TOO_COARSE"


@pytest.mark.parametrize("failure", ["initialize", "account_info", "order_calc_profit"])
def test_cli_terminal_error_is_safe_json(connected_cli, api, failure):
    api.fail = failure
    result = runner.invoke(cli.app, connected_cli + ["--json"])
    assert result.exit_code == 2
    assert failure in json.loads(result.stdout)["error"]
    assert "PRIVATE" not in result.stdout


@pytest.mark.parametrize(
    "extra", [["--capital", "NaN"], ["--capital", "0"], ["--capital", "no"], ["--stop-pips", "-1"]]
)
def test_cli_invalid_arguments_fail_before_terminal(connected_cli, api, extra):
    result = runner.invoke(cli.app, connected_cli + extra + ["--json"])
    assert result.exit_code == 2
    assert "error" in json.loads(result.stdout)
    assert api.calls == []


def test_live_environment_cannot_trigger_connection(connected_cli, api, monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    result = runner.invoke(cli.app, connected_cli + ["--json"])
    assert result.exit_code == 2
    assert api.calls == []


def test_bad_config_values_not_echoed(connected_cli):
    path = Path(connected_cli[-1])
    path.write_text('{"commission": "PRIVATE_SECRET"}', encoding="utf-8")
    result = runner.invoke(cli.app, connected_cli + ["--json"])
    assert result.exit_code == 2
    assert "PRIVATE_SECRET" not in result.stdout


def test_help_does_not_connect(monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("help must not initialize MT5")

    monkeypatch.setattr(cli, "MT5Client", forbidden)
    assert runner.invoke(cli.app, ["--help"]).exit_code == 0
    assert runner.invoke(cli.app, ["broker-check", "--help"]).exit_code == 0
    assert runner.invoke(cli.app, ["collect-ticks", "--help"]).exit_code == 0
    assert runner.invoke(cli.app, ["collect-history", "--help"]).exit_code == 0
    assert runner.invoke(cli.app, ["collect-holdout", "--help"]).exit_code == 0
    assert runner.invoke(cli.app, ["evaluate-signals", "--help"]).exit_code == 0
    assert runner.invoke(cli.app, ["validate-edge", "--help"]).exit_code == 0
    assert runner.invoke(cli.app, ["discover-candidate-events", "--help"]).exit_code == 0
    assert runner.invoke(cli.app, ["research-v2", "--help"]).exit_code == 0


def test_collect_ticks_cli_writes_parquet_and_json_report(connected_cli, api, tmp_path):
    base = int(NOW.timestamp() * 1000)
    api.tick_rows = [
        {
            "time": base // 1000,
            "time_msc": base,
            "bid": 1.1,
            "ask": 1.10002,
            "last": 0,
            "volume": 1,
            "volume_real": 1,
            "flags": 6,
        }
    ]
    output = tmp_path / "market-data"
    result = runner.invoke(
        cli.app,
        [
            "collect-ticks",
            "--start",
            "2026-09-08T10:00:00Z",
            "--end",
            "2026-09-08T10:01:00Z",
            "--output",
            str(output),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["accepted_ticks"] == 1
    assert len(list(output.glob("ticks/**/*.parquet"))) == 1
    assert len(list(output.glob("bars/**/*.parquet"))) == 3
    assert len(list(output.glob("reports/*.json"))) == 1


def test_collect_ticks_rejects_naive_time_before_mt5(connected_cli, api, tmp_path):
    result = runner.invoke(
        cli.app,
        [
            "collect-ticks",
            "--start",
            "2026-09-08T10:00:00",
            "--end",
            "2026-09-08T10:01:00Z",
            "--output",
            str(tmp_path),
            "--json",
        ],
    )
    assert result.exit_code == 2
    assert json.loads(result.stdout)["data_fabricated"] is False
    assert api.calls == []


def test_collect_history_rejects_non_positive_days_before_mt5(connected_cli, api, tmp_path):
    result = runner.invoke(
        cli.app,
        ["collect-history", "--days", "0", "--output", str(tmp_path), "--json"],
    )
    assert result.exit_code == 2
    assert json.loads(result.stdout)["data_fabricated"] is False
    assert api.calls == []


def test_collect_history_mt5_failure_creates_no_completion_manifest(connected_cli, api, tmp_path):
    api.fail = "initialize"
    result = runner.invoke(
        cli.app,
        [
            "collect-history",
            "--days",
            "1",
            "--end",
            "2026-09-08T00:00:00Z",
            "--output",
            str(tmp_path),
            "--json",
        ],
    )
    assert result.exit_code == 2
    assert json.loads(result.stdout)["data_fabricated"] is False
    assert not list(tmp_path.glob("manifests/**/*.json"))


def test_collect_ticks_mt5_failure_is_explicit(connected_cli, api, tmp_path):
    api.fail = "initialize"
    result = runner.invoke(
        cli.app,
        [
            "collect-ticks",
            "--start",
            "2026-09-08T10:00:00Z",
            "--end",
            "2026-09-08T10:01:00Z",
            "--output",
            str(tmp_path),
            "--json",
        ],
    )
    assert result.exit_code == 2
    body = json.loads(result.stdout)
    assert body == {"error": "initialize failed (MT5 code -6)", "data_fabricated": False}
    assert not list(tmp_path.glob("**/*.parquet"))


def test_backtest_cli_uses_phase_b_parquet_and_declares_dummy_strategy(tmp_path):
    partition = (
        tmp_path
        / "market-data"
        / "ticks"
        / "symbol=EURUSD"
        / "date=2026-09-08"
        / "part-000.parquet"
    )
    partition.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp_utc": [
                datetime(2026, 9, 8, 10, 0, tzinfo=UTC),
                datetime(2026, 9, 8, 10, 0, 1, tzinfo=UTC),
            ],
            "source_time_msc": [1, 2],
            "bid": [1.10000, 1.10003],
            "ask": [1.10002, 1.10005],
        }
    ).write_parquet(partition)
    result = runner.invoke(
        cli.app,
        [
            "backtest",
            "--symbol",
            "EURUSD",
            "--start",
            "2026-09-08T10:00:00Z",
            "--end",
            "2026-09-08T10:01:00Z",
            "--capital",
            "10",
            "--data",
            str(tmp_path / "market-data"),
            "--slippage-model",
            "none",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert body["mode"] == "BACKTEST"
    assert body["live_trading_enabled"] is False
    assert body["strategy"] == "DummyStrategy(TEST_ONLY)"
    assert body["broker_profile"]["kind"] == "SIMULATED"
    assert body["broker_profile"]["is_real_broker_capability"] is False
    assert body["broker_profile"]["volume_minimum"] == "0.0001"
    assert body["broker_profile"]["volume_step"] == "0.0001"
    assert "MetaQuotes-Demo" in body["broker_profile"]["disclaimer"]
    assert body["price_semantics"]["execution_quote_price"].startswith("ASK for BUY")
    assert body["ticks_processed"] == 2
    assert body["metrics"]["trades_count"] == 1
    assert len(body["trades"]) == 1
    assert len(list((tmp_path / "market-data" / "reports").glob("backtest-*.json"))) == 1


def test_backtest_cli_rejects_unsupported_latency(tmp_path):
    result = runner.invoke(
        cli.app,
        [
            "backtest",
            "--start",
            "2026-09-08T10:00:00Z",
            "--end",
            "2026-09-08T10:01:00Z",
            "--data",
            str(tmp_path),
            "--execution-latency-ms",
            "7",
            "--json",
        ],
    )
    assert result.exit_code == 2
    assert "0, 25, 50, 100, 250" in json.loads(result.stdout)["error"]


def test_signal_cli_reports_features_reasons_blockers_and_no_sizing(tmp_path):
    data = tmp_path / "market-data"
    store = ParquetStore(data)
    for bars in full_inputs().values():
        store.write_bars(bars)
    tick_path = data / "ticks" / "symbol=EURUSD" / "date=2026-09-08" / "part-000.parquet"
    tick_path.parent.mkdir(parents=True)
    generated = feature_ticks()
    pl.DataFrame(
        {
            "timestamp_utc": [tick.timestamp_utc for tick in generated],
            "source_time_msc": list(range(len(generated))),
            "bid": [float(tick.bid) for tick in generated],
            "ask": [float(tick.ask) for tick in generated],
        }
    ).write_parquet(tick_path)
    result = runner.invoke(
        cli.app,
        [
            "signal",
            "--as-of",
            (BASE + timedelta(hours=2)).isoformat(),
            "--data",
            str(data),
            "--max-spread-points",
            "2",
            "--session-status",
            "allowed",
            "--news-status",
            "clear",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert body["mode"] == "ANALYSIS_ONLY"
    assert body["thresholds_status"] == "FIXED_NOT_OPTIMIZED"
    assert body["decision"]["side"] == "BUY"
    assert body["decision"]["market_bias"] == "BUY"
    assert body["decision"]["trigger_direction"] == "BUY"
    assert all(state["complete"] for state in body["features"]["warmup"].values())
    assert body["decision"]["sizing_authority"] == "RISK_ENGINE_ONLY"
    assert "volume" not in body["decision"]
    assert body["live_trading_enabled"] is False


def test_evaluate_signals_cli_writes_fixed_out_of_sample_report(tmp_path):
    data = tmp_path / "market-data"
    store = ParquetStore(data)
    for bars in full_inputs().values():
        store.write_bars(bars)
    generated = evaluation_ticks()
    tick_path = data / "ticks" / "symbol=EURUSD" / "date=2026-09-08" / "part-000.parquet"
    tick_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp_utc": [tick.timestamp_utc for tick in generated],
            "source_time_msc": list(range(len(generated))),
            "bid": [float(tick.bid) for tick in generated],
            "ask": [float(tick.ask) for tick in generated],
        }
    ).write_parquet(tick_path)
    start = BASE + timedelta(hours=2)
    result = runner.invoke(
        cli.app,
        [
            "evaluate-signals",
            "--start",
            start.isoformat(),
            "--end",
            (start + timedelta(seconds=240)).isoformat(),
            "--data",
            str(data),
            "--max-spread-points",
            "2",
            "--session-status",
            "allowed",
            "--news-status",
            "clear",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert body["mode"] == "SIGNAL_EVALUATION_ONLY"
    assert body["optimization_performed"] is False
    assert body["complete_outcome_observations"] > 0
    assert body["observations"][0]["outcomes"]["long"]["returns_points"]["return_after_30s"] == "2"
    assert len(list((data / "reports").glob("signal-evaluation-*.json"))) == 1
