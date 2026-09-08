"""Daily resumable MT5 history collection for EURUSD."""

from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Literal

from pydantic import Field

from sniper.config import Model
from sniper.data.collector import collect_ticks
from sniper.data.mt5_client import MT5Client
from sniper.data.parquet_store import ParquetStore
from sniper.data.time import CollectionConfig, as_utc, epoch_ms


class HistoryChunk(Model):
    start_utc: datetime
    end_utc: datetime
    status: Literal["DOWNLOADED", "RESUMED"]
    accepted_ticks: int = Field(ge=0)


class HistoryCollectionReport(Model):
    mode: Literal["HISTORICAL_DATA_COLLECTION"] = "HISTORICAL_DATA_COLLECTION"
    live_trading_enabled: Literal[False] = False
    symbol: Literal["EURUSD"] = "EURUSD"
    requested_start_utc: datetime
    requested_end_utc: datetime
    pagination: Literal["UTC_DAY"] = "UTC_DAY"
    resume_policy: Literal["VERIFIED_COMPLETION_MANIFEST"] = "VERIFIED_COMPLETION_MANIFEST"
    chunks: tuple[HistoryChunk, ...]
    downloaded_chunks: int = Field(ge=0)
    resumed_chunks: int = Field(ge=0)
    accepted_ticks_downloaded: int = Field(ge=0)
    output_root: str


class HistoryChunkManifest(Model):
    complete: Literal[True] = True
    symbol: Literal["EURUSD"] = "EURUSD"
    start_utc: datetime
    end_utc: datetime
    accepted_ticks: int = Field(ge=0)


def _daily_chunks(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    chunks = []
    cursor = start
    while cursor < end:
        next_midnight = datetime.combine(cursor.date() + timedelta(days=1), time(), cursor.tzinfo)
        boundary = min(next_midnight, end)
        chunks.append((cursor, boundary))
        cursor = boundary
    return chunks


def _manifest_path(root: Path, start: datetime, end: datetime) -> Path:
    return (
        root
        / "manifests"
        / "history"
        / "symbol=EURUSD"
        / f"chunk-{epoch_ms(start)}-{epoch_ms(end)}.json"
    )


def collect_history(
    client: MT5Client,
    store: ParquetStore,
    start_utc: datetime,
    end_utc: datetime,
    config: CollectionConfig,
    *,
    symbol: str = "EURUSD",
) -> HistoryCollectionReport:
    start, end = as_utc(start_utc), as_utc(end_utc)
    if symbol != "EURUSD" or start >= end:
        raise ValueError("expected EURUSD and a nonempty UTC range")
    chunks: list[HistoryChunk] = []
    downloaded = 0
    resumed = 0
    accepted = 0
    for chunk_start, chunk_end in _daily_chunks(start, end):
        manifest_path = _manifest_path(store.root, chunk_start, chunk_end)
        if manifest_path.exists():
            manifest = HistoryChunkManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            if manifest.start_utc == chunk_start and manifest.end_utc == chunk_end:
                chunks.append(
                    HistoryChunk(
                        start_utc=chunk_start,
                        end_utc=chunk_end,
                        status="RESUMED",
                        accepted_ticks=manifest.accepted_ticks,
                    )
                )
                resumed += 1
                continue
        report = collect_ticks(
            client,
            store,
            chunk_start,
            chunk_end,
            config,
            symbol=symbol,
        )
        manifest = HistoryChunkManifest(
            start_utc=chunk_start,
            end_utc=chunk_end,
            accepted_ticks=report.accepted_ticks,
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        chunks.append(
            HistoryChunk(
                start_utc=chunk_start,
                end_utc=chunk_end,
                status="DOWNLOADED",
                accepted_ticks=report.accepted_ticks,
            )
        )
        downloaded += 1
        accepted += report.accepted_ticks
    return HistoryCollectionReport(
        requested_start_utc=start,
        requested_end_utc=end,
        chunks=tuple(chunks),
        downloaded_chunks=downloaded,
        resumed_chunks=resumed,
        accepted_ticks_downloaded=accepted,
        output_root=str(store.root.resolve()),
    )
