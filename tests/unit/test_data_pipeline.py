from datetime import UTC, datetime, timedelta
from decimal import Decimal as D

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from sniper.data.collector import build_bars, collect_ticks
from sniper.data.mt5_client import MT5Client
from sniper.data.parquet_store import ParquetStore
from sniper.data.time import CollectionConfig, TimePolicy
from sniper.domain.tick import Tick


def tick(second: int, bid_points: int, *, source_msc: int | None = None) -> Tick:
    instant = datetime(2026, 9, 8, 10, tzinfo=UTC) + timedelta(seconds=second)
    bid = D("1.10000") + D(bid_points) * D("0.00001")
    policy = TimePolicy()
    _, local = policy.representations(instant)
    return Tick(
        timestamp_utc=instant,
        timestamp_local=local,
        source_time_msc=source_msc if source_msc is not None else int(instant.timestamp() * 1000),
        bid=bid,
        ask=bid + D("0.00002"),
        last=0,
        volume=1,
        volume_real=1,
        flags=6,
        point=D("0.00001"),
        time_basis="test UTC",
    )


def test_incremental_m1_then_m5_then_m15_aggregation():
    bars = build_bars([tick(0, 0), tick(20, 4), tick(60, -2), tick(301, 8), tick(901, 1)])
    assert [len(bars[key]) for key in ("M1", "M5", "M15")] == [4, 3, 2]
    first = bars["M1"][0]
    assert first.open == D("1.10001")
    assert first.high == D("1.10005")
    assert first.low == D("1.10001")
    assert first.close == D("1.10005")
    assert first.tick_volume == 2
    assert first.is_complete is True
    assert bars["M1"][-1].is_complete is False
    assert bars["M5"][0].tick_volume == 3
    assert bars["M15"][0].tick_volume == 4


@given(st.lists(st.integers(-30, 30), min_size=1, max_size=60))
def test_bar_ohlc_and_volume_invariants(prices):
    ticks = [tick(index % 60, price, source_msc=index) for index, price in enumerate(prices)]
    bar = build_bars(ticks)["M1"][0]
    mids = [item.mid for item in ticks]
    assert bar.open == mids[0]
    assert bar.close == mids[-1]
    assert bar.high == max(mids)
    assert bar.low == min(mids)
    assert bar.tick_volume == len(ticks)


def test_parquet_partitions_are_idempotent_and_preserve_same_millisecond(tmp_path):
    store = ParquetStore(tmp_path)
    first = tick(0, 0, source_msc=123)
    distinct = tick(0, 1, source_msc=123)
    store.write_ticks([first, distinct, first])
    store.write_ticks([first, distinct])
    path = tmp_path / "ticks" / "symbol=EURUSD" / "date=2026-09-08" / "part-000.parquet"
    frame = pl.read_parquet(path)
    assert frame.height == 2
    assert frame["source_time_msc"].to_list() == [123, 123]
    assert set(frame.columns) >= {
        "timestamp_utc",
        "timestamp_server",
        "timestamp_local",
        "bid",
        "ask",
        "spread",
        "spread_points",
    }


def test_collection_validates_deduplicates_reports_gaps_and_writes_bars(api, tmp_path):
    start = datetime(2026, 9, 8, 10, tzinfo=UTC)

    def row(ms, bid, ask, flags=6):
        return {
            "time": ms // 1000,
            "time_msc": ms,
            "bid": bid,
            "ask": ask,
            "last": 0.0,
            "volume": 1,
            "volume_real": 1.0,
            "flags": flags,
        }

    base = int(start.timestamp() * 1000)
    one = row(base, 1.10000, 1.10002)
    api.tick_rows = [
        one,
        dict(one),
        row(base, 1.10001, 1.10004),
        row(base + 60_000, 1.10002, 1.10004),
        row(base + 61_000, 1.10005, 1.10004),
    ]
    config = CollectionConfig(chunk_minutes=2, gap_threshold_seconds=30)
    with MT5Client(api=api, time_policy=config.time) as client:
        report = collect_ticks(
            client, ParquetStore(tmp_path), start, start + timedelta(minutes=2), config
        )
    assert report.source_rows == 5
    assert report.accepted_ticks == 3
    assert report.duplicate_ticks == 1
    assert report.invalid_ticks == 1
    assert report.out_of_range_ticks == 0
    assert len(report.gaps) == 1
    assert report.spread_points is not None
    assert report.spread_points.minimum == 2
    assert report.spread_points.maximum == 3
    assert report.bars == {"M1": 2, "M5": 1, "M15": 1}
    assert report.tick_partitions_written == 1
    assert report.bar_partitions_written == 3

    api.tick_rows = [row(base + 30_000, 1.10003, 1.10005)]
    with MT5Client(api=api, time_policy=config.time) as client:
        collect_ticks(
            client,
            ParquetStore(tmp_path),
            start + timedelta(seconds=30),
            start + timedelta(seconds=60),
            config,
        )
    bar_path = (
        tmp_path
        / "bars"
        / "timeframe=M1"
        / "symbol=EURUSD"
        / "date=2026-09-08"
        / "part-000.parquet"
    )
    stored = pl.read_parquet(bar_path)
    assert stored.height == 2
    assert stored.sort("open_time_utc")["tick_volume"].to_list() == [3, 1]
