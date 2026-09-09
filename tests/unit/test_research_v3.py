from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sniper.domain.research_v3 import BarrierConfiguration
from sniper.evaluation.research_v3 import (
    _barrier_side_labels,
    _causal_micro_features,
    _combined_result,
    purge_v3_training_labels,
    v3_temporal_folds,
)


def test_v3_causal_features_are_unchanged_when_future_prices_change():
    timestamps = np.array([-60, -30, -5, 0, 5], dtype=np.int64) * 1_000_000
    bids = np.array([1.10000, 1.10005, 1.10010, 1.10012, 1.10013])
    asks = bids + 0.00002

    before = _causal_micro_features(timestamps, bids, asks, 3, m1_atr_points=10)
    bids[4], asks[4] = 1.20000, 1.20002
    after = _causal_micro_features(timestamps, bids, asks, 3, m1_atr_points=10)

    assert before == after


def test_long_barrier_uses_executable_bid_and_target_first_path():
    configuration = BarrierConfiguration(
        id="B01",
        stop_atr=0.75,
        target_atr=1.25,
        timeout_seconds=180,
        minimum_target_to_base_cost=1.5,
    )
    timestamps = np.array([0, 10, 20, 180], dtype=np.int64) * 1_000_000
    bids = np.array([1.10000, 1.10015, 1.09990, 1.10000])
    asks = bids + 0.00002

    labels = _barrier_side_labels(
        timestamps,
        bids,
        asks,
        0,
        configuration,
        "LONG",
        atr_points=10,
        base_cost_points=3,
        costs={"optimistic": 2, "base": 3, "stress": 5},
    )

    assert labels["b01_long_outcome"] == "TARGET_FIRST"
    assert labels["b01_long_executable_return_points"] == pytest.approx(13)
    assert labels["b01_long_base_net_points"] == pytest.approx(12)
    assert labels["b01_long_time_to_target_seconds"] == 10


def test_v3_walk_forward_has_strict_expanding_train_only_folds():
    start = datetime(2026, 6, 10, tzinfo=UTC)
    end = datetime(2026, 9, 8, tzinfo=UTC)
    frame = pl.DataFrame({"timestamp_utc": [start + timedelta(days=day) for day in range(90)]})

    folds, reports = v3_temporal_folds(frame, start, end)

    assert len(folds) == 4
    assert all(report.evaluation_role == "RESEARCH_WALK_FORWARD" for report in reports)
    assert all(report.preprocessing_fit_scope == "TRAIN_ONLY" for report in reports)
    for train, validation in folds:
        assert train.max() < validation.min()
    assert [len(train) for train, _ in folds] == sorted(len(train) for train, _ in folds)


@pytest.mark.parametrize(
    ("configuration_id", "timeout_seconds", "expected_purged"),
    (("B01", 180, 2), ("B02_PRIMARY", 300, 4), ("B03", 600, 9), ("B04", 900, 14)),
)
def test_purged_walk_forward_requires_complete_train_labels_before_validation(
    configuration_id: str, timeout_seconds: int, expected_purged: int
):
    start = datetime(2026, 6, 10, tzinfo=UTC)
    end = datetime(2026, 9, 8, tzinfo=UTC)
    minutes = int((end - start).total_seconds() // 60)
    frame = pl.DataFrame(
        {"timestamp_utc": [start + timedelta(minutes=minute) for minute in range(minutes)]}
    )
    folds, reports = v3_temporal_folds(frame, start, end)
    configuration = BarrierConfiguration(
        id=configuration_id,
        stop_atr=1,
        target_atr=2,
        timeout_seconds=timeout_seconds,
        minimum_target_to_base_cost=2,
    )

    purged_folds, audit = purge_v3_training_labels(frame, folds, reports, configuration)

    assert [item.purged_observations for item in audit] == [expected_purged] * 4
    assert all(item.invariant_passed for item in audit)
    timestamps_us = frame["timestamp_utc"].cast(pl.Int64).to_numpy()
    for (train, _), report in zip(purged_folds, reports, strict=True):
        max_label_end_us = int(timestamps_us[train].max()) + timeout_seconds * 1_000_000
        assert max_label_end_us <= int(report.validation_start_utc.timestamp() * 1_000_000)


def test_meta_gate_uses_separate_long_short_probabilities_and_can_skip_short():
    configuration = BarrierConfiguration(
        id="B02_PRIMARY",
        stop_atr=1,
        target_atr=2,
        timeout_seconds=300,
        minimum_target_to_base_cost=2,
    )
    row = {
        "week": "2026-W30",
        "session": "ASIA",
        "cost_optimistic_points": 2.0,
        "cost_base_points": 4.0,
    }
    for side, outcome, gross, executable in (
        ("long", "TARGET_FIRST", 21.0, 20.0),
        ("short", "STOP_FIRST", -11.0, -12.0),
    ):
        row.update(
            {
                f"b02_primary_{side}_target_points": 20.0,
                f"b02_primary_{side}_stop_points": 10.0,
                f"b02_primary_{side}_outcome": outcome,
                f"b02_primary_{side}_gross_return_points": gross,
                f"b02_primary_{side}_executable_return_points": executable,
                f"b02_primary_{side}_optimistic_net_points": executable,
                f"b02_primary_{side}_base_net_points": executable - 2,
                f"b02_primary_{side}_stress_net_points": executable - 4,
                f"b02_primary_{side}_mfe_points": max(executable, 0),
                f"b02_primary_{side}_mae_points": max(-executable, 0),
            }
        )
    frame = pl.DataFrame([row])
    long_predictions = {
        "indices": np.array([0]),
        "logistic_probability": np.array([0.8]),
    }
    short_predictions = {
        "indices": np.array([0]),
        "logistic_probability": np.array([0.2]),
    }

    report, selected = _combined_result(frame, configuration, long_predictions, short_predictions)

    assert report.long_candidates == 1
    assert report.short_candidates == 0
    assert selected["selected_side"].to_list() == ["LONG"]
