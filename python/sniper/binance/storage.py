"""Idempotent immutable-part Parquet storage for Binance raw and normalized data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl


@dataclass(frozen=True)
class StorageWrite:
    path: Path | None
    written: int
    duplicates_skipped: int


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"unsupported storage value: {type(value).__name__}")


def _key(record: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(record[field]) for field in fields)


class BinanceParquetStore:
    """Write immutable, hash-named parts and deduplicate canonical event keys."""

    def __init__(self, data_root: Path):
        self.root = data_root / "binance"

    def write(
        self,
        layer: str,
        dataset: str,
        records: list[dict[str, Any]],
        *,
        key_fields: tuple[str, ...],
    ) -> StorageWrite:
        if layer not in {"raw", "normalized", "features", "backtests", "reports"}:
            raise ValueError("invalid Binance storage layer")
        if not records:
            return StorageWrite(None, 0, 0)
        unique: dict[tuple[str, ...], dict[str, Any]] = {}
        for record in records:
            unique.setdefault(_key(record, key_fields), record)
        duplicates = len(records) - len(unique)
        first_timestamp = records[0].get("timestamp_utc", datetime.now(UTC))
        if isinstance(first_timestamp, str):
            day = first_timestamp[:10]
        elif isinstance(first_timestamp, datetime):
            day = first_timestamp.astimezone(UTC).date().isoformat()
        else:
            day = datetime.now(UTC).date().isoformat()
        directory = self.root / layer / dataset / f"date={day}"
        directory.mkdir(parents=True, exist_ok=True)
        part_files = sorted(directory.glob("part-*.parquet"))
        existing: set[tuple[str, ...]] = set()
        if part_files:
            existing_frame = (
                pl.scan_parquet([str(path) for path in part_files])
                .select(list(key_fields))
                .collect()
            )
            existing = {
                tuple(str(value) for value in row) for row in existing_frame.iter_rows(named=False)
            }
        fresh = [record for key, record in unique.items() if key not in existing]
        duplicates += len(unique) - len(fresh)
        if not fresh:
            return StorageWrite(None, 0, duplicates)
        canonical = json.dumps(
            fresh, sort_keys=True, separators=(",", ":"), default=_json_default
        ).encode()
        digest = hashlib.sha256(canonical).hexdigest()[:20]
        target = directory / f"part-{digest}.parquet"
        if target.exists():
            return StorageWrite(None, 0, duplicates + len(fresh))
        temporary = directory / f".{target.name}.tmp"
        pl.DataFrame(fresh, infer_schema_length=None).write_parquet(temporary, compression="zstd")
        temporary.replace(target)
        return StorageWrite(target, len(fresh), duplicates)
