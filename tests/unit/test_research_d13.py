from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sniper.evaluation.research_d13 import (
    ResearchD13Engine,
    TrainPreprocessor,
    _freeze_hash,
    economic_arrays,
    nested_folds,
    reduce_redundant_features,
)


def test_d13_redundancy_is_train_only_ordered_and_deterministic():
    frame = pl.DataFrame(
        {
            "simple": np.arange(100, dtype=float),
            "duplicate": np.arange(100, dtype=float) * 2,
            "distinct": np.sin(np.arange(100, dtype=float)),
        }
    )
    train = np.arange(80)

    first = reduce_redundant_features(frame, train, ("simple", "duplicate", "distinct"))
    second = reduce_redundant_features(frame, train, ("simple", "duplicate", "distinct"))

    assert first == second
    assert first["retained"] == ["simple", "distinct"]
    assert first["removed"][0]["feature"] == "duplicate"
    assert first["removed"][0]["retained_representative"] == "simple"


def test_d13_preprocessor_bounds_scaling_and_drops_constant_feature():
    train = np.column_stack((np.arange(100, dtype=float), np.ones(100)))
    validation = np.asarray([[1e12, 1.0], [np.nan, 1.0]])
    preprocessor = TrainPreprocessor.fit(train)

    raw, scaled = preprocessor.transform(validation)

    assert preprocessor.active.tolist() == [True, False]
    assert np.all(np.isfinite(raw))
    assert np.all(np.isfinite(scaled))
    assert np.max(np.abs(scaled)) <= 10.0


def test_d13_economic_accounting_reconstructs_bid_ask_base_pnl():
    predictions = {
        "indices": np.asarray([0, 1]),
        "regime": {"LOGISTIC_L2": np.asarray([0.8, 0.9])},
        "direction": {"RIDGE": np.asarray([6.0, -7.0])},
    }
    targets = {
        "regime": np.asarray([1, 1]),
        "movement": np.asarray([3.0, 3.0]),
        "spread": np.asarray([1.0, 1.0]),
        "base_cost": np.asarray([3.4, 3.4]),
        "commission_base": np.asarray([0.4, 0.4]),
        "direction": np.asarray([5.0, -6.0]),
        "long_exec": np.asarray([4.0, -7.0]),
        "short_exec": np.asarray([-6.0, 5.0]),
        "long_base": np.asarray([1.6, -9.4]),
        "short_base": np.asarray([-8.4, 2.6]),
        "long_stress": np.asarray([-0.8, -11.8]),
        "short_stress": np.asarray([-10.8, 0.2]),
    }
    config = {
        "regime_model": "LOGISTIC_L2",
        "direction_model": "RIDGE",
        "regime_threshold": 0.6,
        "uncertainty_buffer_points": 0.0,
    }

    result = economic_arrays(predictions, targets, config)

    assert result["candidate"].tolist() == [True, True]
    assert result["long_side"].tolist() == [True, False]
    np.testing.assert_allclose(result["actual_base"], result["reconstructed_base"])


def test_d13_nested_folds_enforce_label_end_before_validation():
    start = datetime(2026, 6, 10, tzinfo=UTC)
    outer_start = start + timedelta(days=30)
    timestamps = np.arange(30 * 24 * 60, dtype=np.int64) * 60 * 1_000_000 + int(
        start.timestamp() * 1_000_000
    )
    eligible = np.arange(len(timestamps))

    _, audits = nested_folds(timestamps, start, outer_start, eligible)

    assert len(audits) == 3
    assert all(audit["invariant_passed"] for audit in audits)
    assert all(
        audit["max_label_end_time_train_utc"] <= audit["validation_start_utc"] for audit in audits
    )


def test_d13_freeze_hash_is_stable_and_excludes_its_own_field():
    manifest = {"name": "V4_FREEZE_MANIFEST", "threshold": 0.6}
    first = _freeze_hash(manifest)
    manifest["freeze_manifest_sha256"] = first

    assert _freeze_hash(manifest) == first


def test_d13_refuses_holdout_before_protocol_access(tmp_path: Path):
    with pytest.raises(PermissionError, match="refuses any HOLDOUT"):
        ResearchD13Engine().run(
            data_root=tmp_path / "data" / "holdout-v2",
            start_utc=datetime(2026, 6, 10, tzinfo=UTC),
            end_utc=datetime(2026, 9, 8, tzinfo=UTC),
            protocol_path=tmp_path / "missing.yaml",
        )
