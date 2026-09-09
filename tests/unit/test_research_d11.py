from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from sniper.evaluation.research_d11 import (
    ResearchD11Engine,
    _candidate_anatomy,
    _economic_calibration,
    _ev_distribution,
    _numeric_shift,
    _ranking_curves,
)


def _ranking_frame(rows: int = 100) -> pl.DataFrame:
    start = datetime(2026, 7, 1, tzinfo=UTC)
    predicted = [index / rows for index in range(rows)]
    return pl.DataFrame(
        {
            "timestamp_utc": [start + timedelta(minutes=index) for index in range(rows)],
            "day": ["2026-07-01"] * rows,
            "fold": [1] * rows,
            "role": ["MODEL_DEVELOPMENT"] * rows,
            "week": ["2026-W27"] * rows,
            "session": ["LONDON"] * rows,
            "side": ["LONG"] * rows,
            "predicted_EV": predicted,
            "safety_buffer": [0.1] * rows,
            "conservative_EV": [value - 0.1 for value in predicted],
            "actual_BASE_net_pnl": [value * 2 - 1 for value in predicted],
            "actual_executable_pnl": [value * 2 for value in predicted],
            "actual_STRESS_net_pnl": [value * 2 - 2 for value in predicted],
            "actual_outcome": ["NEITHER"] * rows,
            "MFE": [2.0] * rows,
            "MAE": [1.0] * rows,
            "is_D10_candidate": [False] * rows,
            "P_TARGET": [0.2] * rows,
            "P_STOP": [0.4] * rows,
            "P_NEITHER": [0.4] * rows,
        }
    )


def test_d11_ev_thresholds_are_density_counts_only():
    report = _ev_distribution(_ranking_frame())

    assert len(report) == 1
    long = next(item for item in report if item["side"] == "LONG")
    assert long["density_counts"]["conservative_EV_gt_0"] == 89
    assert long["density_counts"]["conservative_EV_gt_1"] == 0


def test_d11_ranking_curve_uses_predeclared_top_five_percent():
    curves = _ranking_curves(_ranking_frame())

    top_five = next(
        item
        for item in curves
        if item["scope"] == "FOLD_1" and item["side"] == "LONG" and item["top_fraction"] == 0.05
    )
    assert top_five["N"] == 5
    assert top_five["BASE_expectancy"] > top_five["unconditional_BASE_expectancy"]


def test_d11_economic_calibration_has_ten_equal_count_groups():
    calibration = _economic_calibration(_ranking_frame())
    row = next(item for item in calibration if item["scope"] == "FOLD_1" and item["side"] == "LONG")

    assert len(row["groups"]) == 10
    assert {group["N"] for group in row["groups"]} == {10}
    assert row["groups"][0]["mean_predicted_EV"] < row["groups"][-1]["mean_predicted_EV"]


def test_d11_candidate_anatomy_is_always_post_hoc():
    frame = _ranking_frame().with_columns(
        pl.when(pl.int_range(pl.len()) == 99)
        .then(True)
        .otherwise(pl.col("is_D10_candidate"))
        .alias("is_D10_candidate")
    )

    rows, summary = _candidate_anatomy(frame)

    assert len(rows) == 1
    assert summary["interpretation"] == "EXPLORATORY_POST_HOC"


def test_d11_regime_shift_metrics_are_interpretable():
    shifted = _numeric_shift(
        pl.Series([0.0, 1.0, 2.0]).to_numpy(),
        pl.Series([1.0, 2.0, 3.0]).to_numpy(),
    )

    assert shifted["standardized_mean_difference"] > 0
    assert shifted["PSI"] >= 0
    assert shifted["normalized_wasserstein"] > 0


def test_d11_refuses_holdout_before_hash_or_model_access(tmp_path: Path):
    with pytest.raises(PermissionError, match="refuses any HOLDOUT"):
        ResearchD11Engine().run(
            data_root=tmp_path / "data" / "holdout-v2",
            start_utc=datetime(2026, 6, 10, tzinfo=UTC),
            end_utc=datetime(2026, 9, 8, tzinfo=UTC),
            protocol_path=tmp_path / "protocol.yaml",
            repository_root=tmp_path,
        )
