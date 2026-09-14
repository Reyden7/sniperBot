"""Idempotent public Binance kline history collection for causal replay."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from sniper.binance.client import BinanceReadOnlyClient
from sniper.binance.market_data import normalize_rest_kline
from sniper.binance.storage import BinanceParquetStore


def _epoch_ms(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("historical boundaries must be timezone-aware UTC")
    return int(value.astimezone(UTC).timestamp() * 1000)


def collect_historical_m1(
    client: BinanceReadOnlyClient,
    data_root: Path,
    symbols: tuple[str, ...],
    start_utc: datetime,
    end_utc: datetime,
) -> dict[str, int]:
    """Download closed M1 bars page-by-page without overwriting raw data."""
    start_ms = _epoch_ms(start_utc)
    end_ms = _epoch_ms(end_utc)
    if end_ms <= start_ms:
        raise ValueError("historical end must be after start")

    def collect_symbol(symbol: str) -> tuple[str, int]:
        store = BinanceParquetStore(data_root)
        cursor = start_ms
        accepted = 0
        while cursor < end_ms:
            payload = client.klines(
                symbol,
                "1m",
                1000,
                start_time_ms=cursor,
                end_time_ms=end_ms - 1,
            )
            if not payload:
                break
            raw_rows: list[dict[str, Any]] = []
            normalized_rows: list[dict[str, Any]] = []
            last_open = cursor
            for row in payload:
                last_open = max(last_open, int(row[0]))
                raw_rows.append(
                    {
                        "timestamp_utc": datetime.fromtimestamp(int(row[0]) / 1000, tz=UTC),
                        "symbol": symbol,
                        "interval": "1m",
                        "exchange_open_time_ms": int(row[0]),
                        "payload_json": str(list(row)),
                    }
                )
                event = normalize_rest_kline(symbol, "1m", row)
                if event.closed and event.timestamp_utc < end_utc:
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
            accepted += write.written
            next_cursor = last_open + 60_000
            if next_cursor <= cursor:
                raise RuntimeError("Binance historical pagination did not advance")
            cursor = next_cursor
            if len(payload) < 1000:
                break
        return symbol, accepted

    # Public klines are independent per symbol. Eight bounded workers keep collection
    # practical while remaining far below Binance Spot request-weight limits.
    with ThreadPoolExecutor(max_workers=min(8, max(len(symbols), 1))) as executor:
        results = executor.map(collect_symbol, symbols)
        return dict(results)


def load_historical_m1(
    data_root: Path,
    symbols: tuple[str, ...],
    start_utc: datetime,
    end_utc: datetime,
) -> pl.DataFrame:
    """Load only requested normalized history and deduplicate immutable parts."""
    root = data_root / "binance" / "normalized" / "historical_m1"
    paths = sorted(root.glob("date=*/part-*.parquet"))
    if not paths:
        raise ValueError("no Binance historical M1 Parquet data found")
    decimal_columns = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    )
    paths_by_decimal_schema: dict[tuple[str, ...], list[Path]] = {}
    for path in paths:
        schema = pl.read_parquet_schema(path)
        signature = tuple(str(schema[column]) for column in decimal_columns)
        paths_by_decimal_schema.setdefault(signature, []).append(path)
    scans = [
        pl.scan_parquet(schema_paths)
        .filter(
            pl.col("symbol").is_in(symbols)
            & (pl.col("timestamp_utc") >= start_utc)
            & (pl.col("timestamp_utc") < end_utc)
            & pl.col("closed")
        )
        .with_columns(pl.col(column).cast(pl.Float64) for column in decimal_columns)
        for schema_paths in paths_by_decimal_schema.values()
    ]
    return (
        pl.concat(scans, how="vertical_relaxed")
        .unique(subset=["symbol", "timestamp_utc"], keep="last")
        .sort(["timestamp_utc", "symbol"])
        .collect()
    )
