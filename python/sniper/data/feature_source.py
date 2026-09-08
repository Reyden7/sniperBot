"""Read Phase B bar partitions for point-in-time feature evaluation."""

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import polars as pl

from sniper.data.time import as_utc
from sniper.domain.bar import Bar, Timeframe


def load_parquet_bars(
    root: Path, symbol: str, as_of_utc: datetime, *, lookback_days: int = 30
) -> dict[Timeframe, list[Bar]]:
    as_of = as_utc(as_of_utc)
    if symbol != "EURUSD" or lookback_days < 1:
        raise ValueError("expected EURUSD and a positive lookback")
    earliest = as_of - timedelta(days=lookback_days)
    result: dict[Timeframe, list[Bar]] = {"M1": [], "M5": [], "M15": []}
    for timeframe in ("M1", "M5", "M15"):
        paths = sorted(
            (root / "bars" / f"timeframe={timeframe}" / "symbol=EURUSD").glob(
                "date=*/part-*.parquet"
            )
        )
        for path in paths:
            frame = pl.read_parquet(path)
            has_completion = "is_complete" in frame.columns
            for row in frame.iter_rows(named=True):
                opened = row["open_time_utc"]
                if not earliest <= opened <= as_of:
                    continue
                result[timeframe].append(
                    Bar(
                        symbol=str(row["symbol"]),
                        timeframe=timeframe,
                        open_time_utc=opened,
                        open=Decimal(str(row["open"])),
                        high=Decimal(str(row["high"])),
                        low=Decimal(str(row["low"])),
                        close=Decimal(str(row["close"])),
                        tick_volume=int(row["tick_volume"]),
                        real_volume=Decimal(str(row["real_volume"])),
                        spread=Decimal(str(row["spread"])),
                        is_complete=bool(row["is_complete"]) if has_completion else False,
                    )
                )
        result[timeframe].sort(key=lambda bar: bar.open_time_utc)
    return result
