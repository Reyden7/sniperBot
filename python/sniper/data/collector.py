"""Chunked historical collection, validation and incremental bar reconstruction."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from math import ceil
from typing import Any, Literal

from pydantic import Field

from sniper.config import Model
from sniper.data.mt5_client import MT5Client
from sniper.data.parquet_store import ParquetStore, WriteResult
from sniper.data.time import CollectionConfig, as_utc
from sniper.domain.bar import Bar, Timeframe
from sniper.domain.tick import Tick


class Gap(Model):
    after_utc: datetime
    before_utc: datetime
    duration_seconds: float = Field(gt=0, allow_inf_nan=False)


class SpreadStatistics(Model):
    minimum: Decimal
    median: Decimal
    p95: Decimal
    maximum: Decimal


class DataQualityReport(Model):
    symbol: Literal["EURUSD"] = "EURUSD"
    requested_start_utc: datetime
    requested_end_utc: datetime
    covered_start_utc: datetime | None
    covered_end_utc: datetime | None
    source_rows: int = Field(ge=0)
    accepted_ticks: int = Field(ge=0)
    duplicate_ticks: int = Field(ge=0)
    invalid_ticks: int = Field(ge=0)
    out_of_range_ticks: int = Field(ge=0)
    gap_threshold_seconds: float = Field(gt=0)
    gaps: tuple[Gap, ...]
    spread_points: SpreadStatistics | None
    bars: dict[Timeframe, int]
    tick_partitions_written: int = Field(ge=0)
    bar_partitions_written: int = Field(ge=0)
    output_root: str
    time_basis: str


@dataclass
class _BarState:
    bucket: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    tick_volume: int
    real_volume: Decimal
    spread_weighted: Decimal


def floor_time(value: datetime, minutes: int) -> datetime:
    utc = as_utc(value)
    return utc.replace(minute=(utc.minute // minutes) * minutes, second=0, microsecond=0)


class IncrementalBarBuilder:
    """Single-pass builder. It never rescans prior ticks or bars."""

    def __init__(self, timeframe: Timeframe) -> None:
        self.timeframe = timeframe
        self.minutes = {"M1": 1, "M5": 5, "M15": 15}[timeframe]
        self._state: _BarState | None = None

    def _finish(self, *, is_complete: bool) -> Bar:
        state = self._state
        if state is None:
            raise RuntimeError("no bar to finish")
        return Bar(
            timeframe=self.timeframe,
            open_time_utc=state.bucket,
            open=state.open,
            high=state.high,
            low=state.low,
            close=state.close,
            tick_volume=state.tick_volume,
            real_volume=state.real_volume,
            spread=state.spread_weighted / state.tick_volume,
            is_complete=is_complete,
        )

    def _add(
        self,
        *,
        timestamp: datetime,
        open_price: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        tick_volume: int,
        real_volume: Decimal,
        spread: Decimal,
    ) -> Bar | None:
        bucket = floor_time(timestamp, self.minutes)
        closed = None
        if self._state is not None and bucket < self._state.bucket:
            raise ValueError("input must be ordered by UTC timestamp")
        if self._state is None or bucket != self._state.bucket:
            if self._state is not None:
                closed = self._finish(is_complete=True)
            self._state = _BarState(
                bucket=bucket,
                open=open_price,
                high=high,
                low=low,
                close=close,
                tick_volume=tick_volume,
                real_volume=real_volume,
                spread_weighted=spread * tick_volume,
            )
        else:
            self._state.high = max(self._state.high, high)
            self._state.low = min(self._state.low, low)
            self._state.close = close
            self._state.tick_volume += tick_volume
            self._state.real_volume += real_volume
            self._state.spread_weighted += spread * tick_volume
        return closed

    def add_tick(self, tick: Tick) -> Bar | None:
        if self.timeframe != "M1":
            raise ValueError("ticks can only feed the M1 builder")
        return self._add(
            timestamp=tick.timestamp_utc,
            open_price=tick.mid,
            high=tick.mid,
            low=tick.mid,
            close=tick.mid,
            tick_volume=1,
            real_volume=tick.volume_real,
            spread=tick.spread,
        )

    def add_bar(self, bar: Bar) -> Bar | None:
        if self.timeframe == "M1" or bar.timeframe != "M1":
            raise ValueError("only M1 bars can feed M5/M15 builders")
        return self._add(
            timestamp=bar.open_time_utc,
            open_price=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            tick_volume=bar.tick_volume,
            real_volume=bar.real_volume,
            spread=bar.spread,
        )

    def flush(self) -> Bar | None:
        if self._state is None:
            return None
        result = self._finish(is_complete=False)
        self._state = None
        return result


def build_bars(ticks: Iterable[Tick]) -> dict[Timeframe, list[Bar]]:
    m1_builder = IncrementalBarBuilder("M1")
    m5_builder = IncrementalBarBuilder("M5")
    m15_builder = IncrementalBarBuilder("M15")
    result: dict[Timeframe, list[Bar]] = {"M1": [], "M5": [], "M15": []}

    def accept_m1(bar: Bar) -> None:
        result["M1"].append(bar)
        closed_m5 = m5_builder.add_bar(bar)
        if closed_m5 is not None:
            result["M5"].append(closed_m5)
        closed_m15 = m15_builder.add_bar(bar)
        if closed_m15 is not None:
            result["M15"].append(closed_m15)

    for tick in ticks:
        closed_m1 = m1_builder.add_tick(tick)
        if closed_m1 is not None:
            accept_m1(closed_m1)
    final_m1 = m1_builder.flush()
    if final_m1 is not None:
        accept_m1(final_m1)
    final_m5 = m5_builder.flush()
    if final_m5 is not None:
        result["M5"].append(final_m5)
    final_m15 = m15_builder.flush()
    if final_m15 is not None:
        result["M15"].append(final_m15)
    return result


def _tick_from_row(row: dict[str, Any], symbol: str, point: Decimal, client: MT5Client) -> Tick:
    raw_msc = int(row["time_msc"])
    if raw_msc <= 0:
        raw_msc = int(row["time"]) * 1000
    timestamp_utc = client.time_policy.decode(raw_msc)
    server, local = client.time_policy.representations(timestamp_utc)
    return Tick(
        symbol=symbol,
        timestamp_utc=timestamp_utc,
        timestamp_server=server,
        timestamp_local=local,
        source_time_msc=raw_msc,
        bid=Decimal(str(row["bid"])),
        ask=Decimal(str(row["ask"])),
        last=Decimal(str(row["last"])),
        volume=Decimal(str(row["volume"])),
        volume_real=Decimal(str(row["volume_real"])),
        flags=int(row["flags"]),
        point=point,
        time_basis=client.time_policy.basis_evidence,
    )


def _spread_statistics(ticks: list[Tick]) -> SpreadStatistics | None:
    if not ticks:
        return None
    values = sorted(tick.spread_points for tick in ticks)
    size = len(values)
    middle = size // 2
    median = values[middle] if size % 2 else (values[middle - 1] + values[middle]) / 2
    p95 = values[max(0, ceil(size * 0.95) - 1)]
    return SpreadStatistics(minimum=values[0], median=median, p95=p95, maximum=values[-1])


def collect_ticks(
    client: MT5Client,
    store: ParquetStore,
    start_utc: datetime,
    end_utc: datetime,
    config: CollectionConfig,
    symbol: str = "EURUSD",
) -> DataQualityReport:
    start, end = as_utc(start_utc), as_utc(end_utc)
    if symbol != "EURUSD" or start >= end:
        raise ValueError("expected EURUSD and a nonempty UTC range")
    if client.time_policy != config.time:
        raise ValueError("client and collection time policies must match")
    point = client.tick_metadata(symbol)
    chunk = timedelta(minutes=config.chunk_minutes)
    cursor = start
    rows_seen = 0
    invalid = 0
    out_of_range = 0
    candidates: list[Tick] = []
    while cursor < end:
        boundary = min(cursor + chunk, end)
        for row in client.copy_ticks_range(
            symbol, cursor, boundary, max_ticks=config.max_ticks_per_chunk
        ):
            rows_seen += 1
            try:
                tick = _tick_from_row(row, symbol, point, client)
                if start <= tick.timestamp_utc < end:
                    candidates.append(tick)
                else:
                    out_of_range += 1
            except ValueError, ArithmeticError, OverflowError, TypeError:
                invalid += 1
        cursor = boundary

    candidates.sort(key=lambda tick: tick.source_time_msc)
    unique: list[Tick] = []
    identities: set[tuple[Any, ...]] = set()
    for tick in candidates:
        if tick.identity in identities:
            continue
        identities.add(tick.identity)
        unique.append(tick)
    duplicate_count = len(candidates) - len(unique)

    gaps = tuple(
        Gap(
            after_utc=left.timestamp_utc,
            before_utc=right.timestamp_utc,
            duration_seconds=(right.timestamp_utc - left.timestamp_utc).total_seconds(),
        )
        for left, right in zip(unique, unique[1:], strict=False)
        if (right.timestamp_utc - left.timestamp_utc).total_seconds() > config.gap_threshold_seconds
    )
    bars = build_bars(unique)
    tick_write = store.write_ticks(unique)
    affected_days = {tick.timestamp_utc.date() for tick in unique}
    stored_bars: dict[Timeframe, list[Bar]] = {"M1": [], "M5": [], "M15": []}
    for day in sorted(affected_days):
        day_bars = build_bars(store.read_ticks([day], point))
        for timeframe, values in day_bars.items():
            stored_bars[timeframe].extend(values)
    bar_writes: list[WriteResult] = [
        store.write_bars(values, replace=True) for values in stored_bars.values()
    ]
    return DataQualityReport(
        requested_start_utc=start,
        requested_end_utc=end,
        covered_start_utc=unique[0].timestamp_utc if unique else None,
        covered_end_utc=unique[-1].timestamp_utc if unique else None,
        source_rows=rows_seen,
        accepted_ticks=len(unique),
        duplicate_ticks=duplicate_count,
        invalid_ticks=invalid,
        out_of_range_ticks=out_of_range,
        gap_threshold_seconds=config.gap_threshold_seconds,
        gaps=gaps,
        spread_points=_spread_statistics(unique),
        bars={timeframe: len(values) for timeframe, values in bars.items()},
        tick_partitions_written=tick_write.partitions,
        bar_partitions_written=sum(write.partitions for write in bar_writes),
        output_root=str(store.root.resolve()),
        time_basis=config.time.basis_evidence,
    )
