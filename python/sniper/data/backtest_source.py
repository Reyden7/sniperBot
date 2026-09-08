"""Chronological, half-open reads from Phase B tick Parquet partitions."""

from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import polars as pl

from sniper.data.time import as_utc
from sniper.domain.trade import MarketTick


def _days(start: date, end: date) -> list[date]:
    result = []
    cursor = start
    while cursor <= end:
        result.append(cursor)
        cursor += timedelta(days=1)
    return result


def load_parquet_ticks(
    root: Path, symbol: str, start_utc: datetime, end_utc: datetime
) -> list[MarketTick]:
    start, end = as_utc(start_utc), as_utc(end_utc)
    if symbol != "EURUSD" or start >= end:
        raise ValueError("expected EURUSD and a nonempty UTC range")
    paths = [
        root / "ticks" / "symbol=EURUSD" / f"date={day.isoformat()}" / "part-000.parquet"
        for day in _days(start.date(), end.date())
    ]
    existing = [path for path in paths if path.exists()]
    if not existing:
        raise FileNotFoundError("no Phase B tick partitions cover the requested range")
    frames = []
    for file_order, path in enumerate(existing):
        frames.append(
            pl.read_parquet(path)
            .with_row_index("_row_order")
            .with_columns(pl.lit(file_order).alias("_file_order"))
        )
    frame = (
        pl.concat(frames, how="diagonal_relaxed")
        .filter((pl.col("timestamp_utc") >= start) & (pl.col("timestamp_utc") < end))
        .sort(["timestamp_utc", "source_time_msc", "_file_order", "_row_order"])
    )
    return [
        MarketTick(
            timestamp_utc=row["timestamp_utc"],
            bid=Decimal(str(row["bid"])),
            ask=Decimal(str(row["ask"])),
        )
        for row in frame.iter_rows(named=True)
    ]
