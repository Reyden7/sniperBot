from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sniper.evaluation.research_d10 import (
    _block_bootstrap,
    _encoded_outcomes,
    _multiclass_metrics,
    _payoff_means,
    _validate_research_scope,
    build_d10_freeze_manifest,
    d10_temporal_folds,
)


def test_d10_outer_folds_are_purged_and_freeze_check_has_forty_percent():
    start = datetime(2026, 6, 10, tzinfo=UTC)
    end = datetime(2026, 9, 8, tzinfo=UTC)
    minutes = int((end - start).total_seconds() // 60)
    frame = pl.DataFrame(
        {
            "timestamp_utc": [start + timedelta(minutes=value) for value in range(minutes)],
            "b02_primary_long_complete": [True] * minutes,
            "b02_primary_short_complete": [True] * minutes,
        }
    )

    folds, audit = d10_temporal_folds(frame, start, end)

    assert [item["role"] for item in folds] == [
        "MODEL_DEVELOPMENT",
        "MODEL_DEVELOPMENT",
        "MODEL_DEVELOPMENT",
        "INTERNAL_FREEZE_CHECK",
    ]
    assert [item["purged_observations"] for item in audit] == [4, 4, 4, 4]
    assert all(item["invariant_passed"] for item in audit)
    assert folds[-1]["validation_end_utc"] - folds[-1]["validation_start_utc"] == timedelta(days=36)


def test_conditional_payoffs_use_real_neither_base_pnl():
    frame = pl.DataFrame(
        {
            "b02_primary_long_outcome": ["TARGET_FIRST", "STOP_FIRST", "NEITHER", "NEITHER"],
            "b02_primary_long_base_net_points": [21.0, -11.0, 3.5, -2.5],
        }
    )

    values, report = _payoff_means(frame, np.array([0, 1, 2, 3]), "b02_primary_long")

    assert values.tolist() == [21.0, -11.0, 0.5]
    assert report["NEITHER"]["mean_BASE_net_pnl_points"] == 0.5
    assert not report["NEITHER"]["used_missing_class_fallback"]


def test_incomplete_outcomes_are_encoded_as_unusable_not_as_a_class():
    frame = pl.DataFrame({"b02_primary_long_outcome": ["TARGET_FIRST", None]})

    encoded = _encoded_outcomes(frame, "b02_primary_long")

    assert encoded.tolist() == [0, -1]


def test_multiclass_probabilities_are_scored_as_three_exclusive_outcomes():
    y = np.array([0, 1, 2])
    probabilities = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])

    metrics = _multiclass_metrics(y, probabilities)

    assert metrics["probability_sum_max_abs_error"] == pytest.approx(0)
    assert metrics["class_distribution"]["NEITHER"]["count"] == 1
    assert set(metrics["calibration"]) == {"TARGET_FIRST", "STOP_FIRST", "NEITHER"}


def test_d10_explicitly_refuses_holdout_root(tmp_path: Path):
    with pytest.raises(PermissionError, match="refuses any HOLDOUT"):
        _validate_research_scope(
            tmp_path / "data" / "holdout-v2",
            datetime(2026, 6, 10, tzinfo=UTC),
            datetime(2026, 9, 8, tzinfo=UTC),
        )


def test_freeze_manifest_is_reproducible_and_has_no_probability_gate():
    protocol = Path(__file__).parents[2] / "docs" / "research-protocol-d10.yaml"

    first = build_d10_freeze_manifest(protocol)
    second = build_d10_freeze_manifest(protocol)

    assert first == second
    assert first["freeze_manifest_sha256"] == second["freeze_manifest_sha256"]
    assert "0.60" not in first["candidate_rule"]
    assert not first["holdout_access_permitted"]


def test_daily_block_bootstrap_is_reproducible():
    frame = pl.DataFrame(
        {
            "day": ["2026-08-01", "2026-08-01", "2026-08-02", "2026-08-02"],
            "base": [2.0, -1.0, 4.0, -2.0],
            "stress": [1.0, -2.0, 3.0, -3.0],
        }
    )

    first = _block_bootstrap(frame, 17)
    second = _block_bootstrap(frame, 17)

    assert first == second
    assert first["BASE_expectancy"]["block"] == "UTC_DAY"
    assert first["BASE_expectancy"]["replications"] == 2000
