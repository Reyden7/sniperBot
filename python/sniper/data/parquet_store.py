"""Idempotent daily Parquet partitions for validated ticks and bars."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl

from sniper.domain.bar import Bar
from sniper.domain.tick import Tick

TICK_IDENTITY = [
    "source_time_msc",
    "bid",
    "ask",
    "last",
    "volume",
    "volume_real",
    "flags",
]


@dataclass(frozen=True)
class WriteResult:
    partitions: int
    rows_received: int
    rows_stored: int


class ParquetStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def _atomic_write(frame: pl.DataFrame, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".parquet.tmp")
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        temporary.replace(target)

    def _merge_partition(
        self, target: Path, rows: list[dict[str, Any]], unique_on: list[str], sort_on: str
    ) -> int:
        incoming = pl.DataFrame(rows)
        if target.exists():
            incoming = pl.concat([pl.read_parquet(target), incoming], how="diagonal_relaxed")
        merged = incoming.unique(subset=unique_on, keep="first", maintain_order=True).sort(
            sort_on, maintain_order=True
        )
        self._atomic_write(merged, target)
        return merged.height

    def write_ticks(self, ticks: Iterable[Tick]) -> WriteResult:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        received = 0
        for tick in ticks:
            received += 1
            day = tick.timestamp_utc.date().isoformat()
            groups[day].append(tick.parquet_row())
        stored = 0
        for day, rows in groups.items():
            target = self.root / "ticks" / "symbol=EURUSD" / f"date={day}" / "part-000.parquet"
            stored += self._merge_partition(target, rows, TICK_IDENTITY, "timestamp_utc")
        return WriteResult(partitions=len(groups), rows_received=received, rows_stored=stored)

    def read_ticks(self, days: Iterable[date], point: Decimal) -> list[Tick]:
        result: list[Tick] = []
        for day in sorted(set(days)):
            target = (
                self.root
                / "ticks"
                / "symbol=EURUSD"
                / f"date={day.isoformat()}"
                / "part-000.parquet"
            )
            if not target.exists():
                continue
            for row in pl.read_parquet(target).iter_rows(named=True):
                result.append(
                    Tick(
                        symbol=str(row["symbol"]),
                        timestamp_utc=row["timestamp_utc"],
                        timestamp_server=row["timestamp_server"],
                        timestamp_local=row["timestamp_local"],
                        source_time_msc=int(row["source_time_msc"]),
                        bid=Decimal(str(row["bid"])),
                        ask=Decimal(str(row["ask"])),
                        last=Decimal(str(row["last"])),
                        volume=Decimal(str(row["volume"])),
                        volume_real=Decimal(str(row["volume_real"])),
                        flags=int(row["flags"]),
                        point=point,
                        time_basis=str(row["time_basis"]),
                    )
                )
        return sorted(result, key=lambda tick: tick.source_time_msc)

    def write_bars(self, bars: Iterable[Bar], *, replace: bool = False) -> WriteResult:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        received = 0
        for bar in bars:
            received += 1
            day = bar.open_time_utc.date().isoformat()
            groups[(bar.timeframe, day)].append(
                {
                    "symbol": bar.symbol,
                    "timeframe": bar.timeframe,
                    "open_time_utc": bar.open_time_utc,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "tick_volume": bar.tick_volume,
                    "real_volume": float(bar.real_volume),
                    "spread": float(bar.spread),
                    "is_complete": bar.is_complete,
                }
            )
        stored = 0
        for (timeframe, day), rows in groups.items():
            target = (
                self.root
                / "bars"
                / f"timeframe={timeframe}"
                / "symbol=EURUSD"
                / f"date={day}"
                / "part-000.parquet"
            )
            if replace:
                frame = (
                    pl.DataFrame(rows)
                    .unique(subset=["open_time_utc"], maintain_order=True)
                    .sort("open_time_utc", maintain_order=True)
                )
                self._atomic_write(frame, target)
                stored += frame.height
            else:
                stored += self._merge_partition(target, rows, ["open_time_utc"], "open_time_utc")
        return WriteResult(partitions=len(groups), rows_received=received, rows_stored=stored)
