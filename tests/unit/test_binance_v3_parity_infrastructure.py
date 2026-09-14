from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from sniper.binance.forward_paper import ForwardPaperRunner, ForwardPaperState
from sniper.binance.m1_continuity import repair_m1_continuity
from sniper.binance.strategy_engine import ResearchBar
from sniper.binance.v2_market import V2PairMarketProfile
from sniper.binance.v3_cross_sectional import RankedAsset, RelativeStrengthFeature
from sniper.binance.v3_parity import audit_replay_forward_parity

START = datetime(2026, 9, 10, 10, tzinfo=UTC)


class FakeKlineClient:
    def __init__(self, blocked: set[datetime] | None = None):
        self.blocked = blocked or set()
        self.calls = 0

    def klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        *,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> list[list[object]]:
        assert interval == "1m"
        assert start_time_ms is not None and end_time_ms is not None
        self.calls += 1
        rows = []
        cursor = start_time_ms
        while cursor <= end_time_ms and len(rows) < limit:
            timestamp = datetime.fromtimestamp(cursor / 1000, tz=UTC)
            if timestamp not in self.blocked:
                rows.append(
                    [
                        cursor,
                        "100",
                        "101",
                        "99",
                        "100.5",
                        "10",
                        cursor + 59_999,
                        "1000",
                        20,
                        "5",
                        "500",
                        "0",
                    ]
                )
            cursor += 60_000
        return rows


def test_m1_interruption_backfill_is_complete_idempotent_and_crosses_m5(tmp_path: Path):
    missing = START + timedelta(minutes=4)
    client = FakeKlineClient({missing})
    interrupted = repair_m1_continuity(
        client,  # type: ignore[arg-type]
        tmp_path,
        ("TESTUSDT",),
        START,
        START + timedelta(minutes=10),
        observed_at_utc=START + timedelta(hours=1),
    )
    assert not interrupted.complete
    assert interrupted.missing_per_symbol["TESTUSDT"] == (missing,)

    client.blocked.clear()
    resumed = repair_m1_continuity(
        client,  # type: ignore[arg-type]
        tmp_path,
        ("TESTUSDT",),
        START,
        START + timedelta(minutes=10),
        observed_at_utc=START + timedelta(hours=1),
    )
    assert resumed.complete
    assert resumed.coverage_pct == 100.0
    calls_after_repair = client.calls
    parts_after_repair = tuple(tmp_path.rglob("part-*.parquet"))

    duplicate = repair_m1_continuity(
        client,  # type: ignore[arg-type]
        tmp_path,
        ("TESTUSDT",),
        START,
        START + timedelta(minutes=10),
        observed_at_utc=START + timedelta(hours=1),
    )
    assert duplicate.complete
    assert client.calls == calls_after_repair
    assert tuple(tmp_path.rglob("part-*.parquet")) == parts_after_repair


def test_interruption_timestamp_is_skipped_once_and_checkpointed(tmp_path: Path):
    runner = object.__new__(ForwardPaperRunner)
    runner.events_path = tmp_path / "events.jsonl"
    runner.state_path = tmp_path / "state.json"
    runner.status_path = tmp_path / "status.json"
    runner.data_root = tmp_path
    runner.current_stage = "TEST"
    state = ForwardPaperState(
        epoch="FORWARD_PAPER_V3_PARITY_VALIDATED",
        started_at_utc=START,
        paper_validation_start_utc=START,
        updated_at_utc=START,
        fee_snapshot_at_utc=datetime.now(UTC),
    )
    delayed = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(minutes=10)
    delayed = delayed.replace(minute=delayed.minute - delayed.minute % 5)
    runner.run_cycle(state, signal_time=delayed)
    assert state.last_processed_signal_time == delayed
    reloaded = ForwardPaperState.model_validate_json(runner.state_path.read_text(encoding="utf-8"))
    assert reloaded.last_processed_signal_time == delayed
    first_events = runner.events_path.read_text(encoding="utf-8").splitlines()
    assert len(first_events) == 1
    assert "RUNNER_INTERRUPTION_NO_RETROACTIVE_CATCHUP" in first_events[0]

    runner.run_cycle(state, signal_time=delayed)
    assert runner.events_path.read_text(encoding="utf-8").splitlines() == first_events


def _profile(symbol: str) -> V2PairMarketProfile:
    return V2PairMarketProfile(
        rank=1,
        symbol=symbol,
        base_asset=symbol.removesuffix("USDT"),
        quote_asset="USDT",
        category="ECONOMICALLY_ATTRACTIVE",
        maker_fee=Decimal("0.001"),
        taker_fee=Decimal("0.001"),
        fee_source="BINANCE_ACCOUNT_API",
        spread_average_pct=Decimal("0.02"),
        spread_p50_pct=Decimal("0.02"),
        spread_p95_pct=Decimal("0.03"),
        slippage_pct_by_notional={"500": Decimal("0.02")},
        quote_volume_24h=Decimal("10000000"),
        trades_count_24h=100000,
        top20_bid_depth_quote=Decimal("100000"),
        top20_ask_depth_quote=Decimal("100000"),
        volatility_pct_by_horizon={"1h": Decimal("1")},
        expected_move_pct_by_horizon={"1h": Decimal("1")},
        taker_taker_round_trip_cost_pct=Decimal("0.24"),
        maker_taker_round_trip_cost_pct=Decimal("0.14"),
        movement_to_cost_ratio_by_horizon={"1h": Decimal("4")},
        expected_net_edge_pct_by_horizon={"1h": Decimal("0.76")},
        maximum_strategic_ratio=Decimal("4"),
        liquidity_sufficient=True,
        spread_acceptable=True,
        slippage_acceptable=True,
        tick_size=Decimal("0.01"),
        quantity_step=Decimal("0.001"),
        minimum_notional=Decimal("10"),
    )


def test_replay_forward_emulator_share_every_decision_output():
    symbols = ("BTCUSDT", "ETHUSDT", "AAAUSDT")
    profiles = {symbol: _profile(symbol) for symbol in symbols}
    bars = {}
    cross_sections = {}
    for offset in range(4):
        timestamp = START + timedelta(minutes=5 * offset)
        ranked = []
        for rank, symbol in enumerate(symbols, 1):
            feature = RelativeStrengthFeature(
                symbol=symbol,
                timestamp_utc=timestamp,
                close=100,
                atr=1,
                return_15m=0.01,
                return_30m=0.01,
                return_1h=0.02,
                return_4h=0.03,
                return_24h=0.04,
                volume_relative=1.5,
                trade_activity_relative=1.5,
                atr_percentile=0.8,
                trend_efficiency=0.8,
                expected_move_pct=1.0,
            )
            ranked.append(
                RankedAsset(
                    feature=feature,
                    relative_return_vs_btc=0.01,
                    relative_return_vs_eth=0.01,
                    relative_return_vs_universe=0.01,
                    score=1 / rank,
                    rank=rank,
                )
            )
            bars.setdefault(symbol, []).append(
                ResearchBar(
                    symbol=symbol,
                    timestamp_utc=timestamp,
                    open=100,
                    high=102,
                    low=99,
                    close=101,
                    volume=100,
                    quote_volume=10000,
                    trades_count=100,
                    taker_buy_quote_volume=5000,
                )
            )
        cross_sections[timestamp] = ("MARKET_RISK_ON", ranked)
    report = audit_replay_forward_parity(
        bars_by_symbol=bars,
        cross_sections=cross_sections,
        profiles=profiles,
    )
    assert report.timestamps_tested == 4
    assert report.passed
