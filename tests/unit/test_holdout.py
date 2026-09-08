from datetime import UTC, datetime

import polars as pl
from sniper.data.history import HistoryChunk, HistoryCollectionReport
from sniper.data.holdout import validate_and_seal_holdout


def test_holdout_integrity_report_contains_no_model_metrics(tmp_path):
    start = datetime(2026, 2, 1, tzinfo=UTC)
    end = datetime(2026, 2, 2, tzinfo=UTC)
    tick_path = tmp_path / "ticks" / "symbol=EURUSD" / "date=2026-02-01"
    tick_path.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp_utc": [
                datetime(2026, 2, 1, 1, tzinfo=UTC),
                datetime(2026, 2, 1, 2, tzinfo=UTC),
            ]
        }
    ).write_parquet(tick_path / "part-000.parquet", statistics=True)
    history = HistoryCollectionReport(
        requested_start_utc=start,
        requested_end_utc=end,
        chunks=(
            HistoryChunk(
                start_utc=start,
                end_utc=end,
                status="DOWNLOADED",
                accepted_ticks=2,
            ),
        ),
        downloaded_chunks=1,
        resumed_chunks=0,
        accepted_ticks_downloaded=2,
        output_root=str(tmp_path),
    )

    report = validate_and_seal_holdout(tmp_path, history)

    assert report.state == "SEALED"
    assert report.integrity_status == "VALID"
    assert report.ticks_collected == 2
    assert report.model_metrics_computed is False
    assert (tmp_path / ".holdout-sealed.json").exists()
