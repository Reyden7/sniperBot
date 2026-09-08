"""Collection-only integrity manifest for the sealed V2 HOLDOUT dataset."""

from datetime import datetime
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import Field

from sniper.config import Model
from sniper.data.history import HistoryCollectionReport
from sniper.data.time import as_utc


class SealedHoldoutReport(Model):
    mode: Literal["HOLDOUT_COLLECTION_AND_INTEGRITY_ONLY"] = "HOLDOUT_COLLECTION_AND_INTEGRITY_ONLY"
    role: Literal["HOLDOUT"] = "HOLDOUT"
    state: Literal["SEALED"] = "SEALED"
    live_trading_enabled: Literal[False] = False
    model_metrics_computed: Literal[False] = False
    manual_exploration_permitted: Literal[False] = False
    requested_start_utc: datetime
    requested_end_utc_exclusive: datetime
    available_start_utc: datetime | None
    available_end_utc_inclusive: datetime | None
    ticks_collected: int = Field(ge=0)
    calendar_chunks: int = Field(ge=0)
    active_market_chunks: int = Field(ge=0)
    empty_chunks: int = Field(ge=0)
    integrity_status: Literal["VALID", "NO_DATA"]
    output_root: str
    seal_file: str
    disclaimer: str


def validate_and_seal_holdout(root: Path, history: HistoryCollectionReport) -> SealedHoldoutReport:
    """Read only timestamps/counts required for integrity; never derive market/model metrics."""
    start = as_utc(history.requested_start_utc)
    end = as_utc(history.requested_end_utc)
    tick_files = sorted((root / "ticks" / "symbol=EURUSD").glob("date=*/part-*.parquet"))
    available_start: datetime | None = None
    available_end: datetime | None = None
    ticks = 0
    if tick_files:
        summary = (
            pl.scan_parquet(tick_files)
            .select(
                pl.len().alias("ticks"),
                pl.col("timestamp_utc").min().alias("start"),
                pl.col("timestamp_utc").max().alias("end"),
            )
            .collect()
        )
        ticks = int(summary["ticks"][0])
        available_start = summary["start"][0]
        available_end = summary["end"][0]
        if available_start is not None and not (start <= as_utc(available_start) < end):
            raise ValueError("HOLDOUT contains a timestamp outside its declared range")
        if available_end is not None and not (start <= as_utc(available_end) < end):
            raise ValueError("HOLDOUT contains a timestamp outside its declared range")
    active_chunks = sum(chunk.accepted_ticks > 0 for chunk in history.chunks)
    seal_path = root / ".holdout-sealed.json"
    report = SealedHoldoutReport(
        requested_start_utc=start,
        requested_end_utc_exclusive=end,
        available_start_utc=available_start,
        available_end_utc_inclusive=available_end,
        ticks_collected=ticks,
        calendar_chunks=len(history.chunks),
        active_market_chunks=active_chunks,
        empty_chunks=len(history.chunks) - active_chunks,
        integrity_status="VALID" if ticks else "NO_DATA",
        output_root=str(root.resolve()),
        seal_file=str(seal_path.resolve()),
        disclaimer=(
            "Collection and timestamp/count integrity only. No feature, label, prediction, "
            "performance metric or manual market exploration was computed."
        ),
    )
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    seal_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report
