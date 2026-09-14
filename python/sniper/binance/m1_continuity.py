"""Sequential, idempotent Binance M1 archive repair for forward paper."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from sniper.binance.client import BinanceReadOnlyClient
from sniper.binance.market_data import normalize_rest_kline
from sniper.binance.storage import BinanceParquetStore

ONE_MINUTE = timedelta(minutes=1)


@dataclass(frozen=True)
class M1ContinuityReport:
    start_utc: datetime
    end_utc: datetime
    expected_per_symbol: int
    present_per_symbol: dict[str, int]
    missing_per_symbol: dict[str, tuple[datetime, ...]]
    repaired_per_symbol: dict[str, int]
    duplicates_skipped: int

    @property
    def complete(self) -> bool:
        return all(not missing for missing in self.missing_per_symbol.values())

    @property
    def coverage_pct(self) -> float:
        denominator = self.expected_per_symbol * max(len(self.present_per_symbol), 1)
        return 100.0 * sum(self.present_per_symbol.values()) / denominator if denominator else 100.0


def closed_minute_boundary(value: datetime) -> datetime:
    """Exclusive end boundary containing only candles closed before ``value``."""
    aware = value.astimezone(UTC)
    return aware.replace(second=0, microsecond=0)


def _minute_range(start_utc: datetime, end_utc: datetime) -> tuple[datetime, ...]:
    count = max(int((end_utc - start_utc).total_seconds() // 60), 0)
    return tuple(start_utc + index * ONE_MINUTE for index in range(count))


def _archive_paths(data_root: Path) -> list[str]:
    paths: list[Path] = []
    for dataset in ("historical_m1", "klines"):
        root = data_root / "binance" / "normalized" / dataset
        paths.extend(root.glob("date=*/part-*.parquet"))
    return [str(path) for path in sorted(paths)]


def archived_m1_times(
    data_root: Path,
    symbols: tuple[str, ...],
    start_utc: datetime,
    end_utc: datetime,
) -> dict[str, set[datetime]]:
    output: dict[str, set[datetime]] = {symbol: set() for symbol in symbols}
    paths = _archive_paths(data_root)
    if not paths:
        return output
    rows = (
        pl.scan_parquet(paths)
        .filter(
            pl.col("symbol").is_in(symbols)
            & (pl.col("interval") == "1m")
            & pl.col("closed")
            & (pl.col("timestamp_utc") >= start_utc)
            & (pl.col("timestamp_utc") < end_utc)
        )
        .select("symbol", "timestamp_utc")
        .unique()
        .collect()
    )
    for symbol, timestamp in rows.iter_rows():
        canonical = timestamp.replace(tzinfo=UTC) if timestamp.tzinfo is None else timestamp
        output[str(symbol)].add(canonical)
    return output


def inspect_m1_continuity(
    data_root: Path,
    symbols: tuple[str, ...],
    start_utc: datetime,
    end_utc: datetime,
) -> M1ContinuityReport:
    start = closed_minute_boundary(start_utc)
    end = closed_minute_boundary(end_utc)
    expected = set(_minute_range(start, end))
    present = archived_m1_times(data_root, symbols, start, end)
    missing = {symbol: tuple(sorted(expected - present[symbol])) for symbol in symbols}
    return M1ContinuityReport(
        start_utc=start,
        end_utc=end,
        expected_per_symbol=len(expected),
        present_per_symbol={symbol: len(expected & present[symbol]) for symbol in symbols},
        missing_per_symbol=missing,
        repaired_per_symbol={symbol: 0 for symbol in symbols},
        duplicates_skipped=0,
    )


def _contiguous_ranges(times: tuple[datetime, ...]) -> list[tuple[datetime, datetime]]:
    if not times:
        return []
    ranges: list[tuple[datetime, datetime]] = []
    start = previous = times[0]
    for current in times[1:]:
        if current != previous + ONE_MINUTE:
            ranges.append((start, previous + ONE_MINUTE))
            start = current
        previous = current
    ranges.append((start, previous + ONE_MINUTE))
    return ranges


def _bounded_fetch_ranges(times: tuple[datetime, ...]) -> list[tuple[datetime, datetime]]:
    """Coalesce sparse repairs into Binance's 1000-candle pages."""
    if not times:
        return []
    output = []
    cursor = times[0]
    final = times[-1] + ONE_MINUTE
    while cursor < final:
        end = min(cursor + timedelta(minutes=1000), final)
        output.append((cursor, end))
        cursor = end
    return output


def repair_m1_continuity(
    client: BinanceReadOnlyClient,
    data_root: Path,
    symbols: tuple[str, ...],
    start_utc: datetime,
    end_utc: datetime,
    *,
    observed_at_utc: datetime | None = None,
) -> M1ContinuityReport:
    """Backfill only missing, already-closed M1 candles and verify the exact grid."""
    observed = (observed_at_utc or datetime.now(UTC)).astimezone(UTC)
    end = min(closed_minute_boundary(end_utc), closed_minute_boundary(observed))
    before = inspect_m1_continuity(data_root, symbols, start_utc, end)

    def repair_symbol(symbol: str) -> tuple[str, int, int]:
        store = BinanceParquetStore(data_root)
        repaired = 0
        duplicates = 0
        for range_start, range_end in _bounded_fetch_ranges(before.missing_per_symbol[symbol]):
            cursor = int(range_start.timestamp() * 1000)
            end_ms = int(range_end.timestamp() * 1000)
            while cursor < end_ms:
                payload = client.klines(
                    symbol,
                    "1m",
                    min(1000, max((end_ms - cursor) // 60_000, 1)),
                    start_time_ms=cursor,
                    end_time_ms=end_ms - 1,
                )
                if not payload:
                    break
                raw_rows: list[dict[str, Any]] = []
                normalized_rows: list[dict[str, Any]] = []
                last_open = cursor - 60_000
                wanted = set(before.missing_per_symbol[symbol])
                for row in payload:
                    open_ms = int(row[0])
                    close_ms = int(row[6])
                    last_open = max(last_open, open_ms)
                    timestamp = datetime.fromtimestamp(open_ms / 1000, tz=UTC)
                    if timestamp not in wanted or close_ms >= int(observed.timestamp() * 1000):
                        continue
                    raw_rows.append(
                        {
                            "timestamp_utc": timestamp,
                            "symbol": symbol,
                            "interval": "1m",
                            "exchange_open_time_ms": open_ms,
                            "payload_json": str(list(row)),
                        }
                    )
                    event = normalize_rest_kline(symbol, "1m", row)
                    if event.closed:
                        normalized_rows.append(event.model_dump(mode="python"))
                store.write(
                    "raw",
                    "historical_klines",
                    raw_rows,
                    key_fields=("symbol", "interval", "exchange_open_time_ms"),
                )
                write = store.write(
                    "normalized",
                    "historical_m1",
                    normalized_rows,
                    key_fields=("symbol", "interval", "timestamp_utc"),
                )
                repaired += write.written
                duplicates += write.duplicates_skipped
                next_cursor = last_open + 60_000
                if next_cursor <= cursor:
                    raise RuntimeError(f"M1 repair pagination did not advance for {symbol}")
                cursor = next_cursor
        return symbol, repaired, duplicates

    with ThreadPoolExecutor(max_workers=min(8, max(len(symbols), 1))) as executor:
        results = list(executor.map(repair_symbol, symbols))
    after = inspect_m1_continuity(data_root, symbols, before.start_utc, before.end_utc)
    return M1ContinuityReport(
        start_utc=after.start_utc,
        end_utc=after.end_utc,
        expected_per_symbol=after.expected_per_symbol,
        present_per_symbol=after.present_per_symbol,
        missing_per_symbol=after.missing_per_symbol,
        repaired_per_symbol={symbol: repaired for symbol, repaired, _ in results},
        duplicates_skipped=sum(duplicates for _, _, duplicates in results),
    )
