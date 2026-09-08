from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sniper.backtest.execution_model import BrokerSimulationConfig
from sniper.domain.dataset import (
    DatasetDeclaration,
    DatasetRegistry,
    DatasetRole,
    DatasetState,
)
from sniper.evaluation.research_v2 import (
    SCENARIOS,
    _future_labels,
    _gate_result,
    temporal_folds,
)

START = datetime(2026, 6, 10, tzinfo=UTC)
END = datetime(2026, 9, 8, tzinfo=UTC)


def test_temporal_folds_are_strict_and_last_is_internal_freeze_check():
    timestamps = [START + timedelta(days=day) for day in range(90)]
    frame = pl.DataFrame({"timestamp_utc": timestamps})

    folds, reports = temporal_folds(frame, START, END)

    assert [report.evaluation_role for report in reports] == [
        "MODEL_COMPARISON",
        "MODEL_COMPARISON",
        "INTERNAL_FREEZE_CHECK",
    ]
    for (train, validation), report in zip(folds, reports, strict=True):
        assert train.max() < validation.min()
        assert timestamps[int(train.max())] < report.validation_start_utc
        assert report.preprocessing_fit_scope == "TRAIN_ONLY"


def test_future_label_uses_first_executable_quote_at_exact_horizon():
    seconds = np.array([0, 30, 60, 180, 300, 600, 900], dtype=np.int64)
    timestamps_us = seconds * 1_000_000
    bids = np.array([1.10000, 1.10010, 1.10020, 1.10030, 1.10040, 1.10050, 1.10060])
    asks = bids + 0.00002

    labels = _future_labels(timestamps_us, bids, asks, 0, BrokerSimulationConfig())

    assert labels["h30_market_return_points"] == pytest.approx(10.0)
    assert labels["h30_long_executable_return_points"] == pytest.approx(8.0)
    assert labels["h900_market_return_points"] == pytest.approx(60.0)


def test_internal_freeze_gate_cannot_include_model_comparison_rows():
    frame = pl.DataFrame(
        {
            "cost_base_points": [2.0, 2.0, 2.0],
            "spread_points": [1.0, 1.0, 1.0],
            "h300_market_return_points": [100.0, 100.0, 5.0],
            "h300_long_executable_return_points": [99.0, 99.0, 4.0],
            "h300_short_executable_return_points": [-101.0, -101.0, -6.0],
            "h300_long_mfe_points": [100.0, 100.0, 6.0],
            "h300_short_mfe_points": [0.0, 0.0, 0.0],
            "h300_long_mae_points": [1.0, 1.0, 2.0],
            "h300_short_mae_points": [100.0, 100.0, 6.0],
        }
    )
    predictions = {
        "indices": np.array([0, 1, 2]),
        "fold_number": np.array([1, 2, 3]),
        "opportunity_probability": np.array([0.9, 0.9, 0.9]),
        "opportunity_mfe": np.array([100.0, 100.0, 6.0]),
        "direction_probability": np.array([0.9, 0.9, 0.9]),
        "signed_return": np.array([100.0, 100.0, 5.0]),
    }

    result, selected = _gate_result(
        frame, 300, predictions, SCENARIOS[1], 1.5, "INTERNAL_FREEZE_CHECK"
    )

    assert result.evaluation_scope == "INTERNAL_FREEZE_CHECK"
    assert result.candidates_count == 1
    assert selected["actual_gross_points"].to_list() == [5.0]


def test_holdout_and_forward_roles_are_never_research_accessible():
    registry = DatasetRegistry(
        datasets=tuple(
            DatasetDeclaration(
                dataset_id=role.value,
                role=role,
                state=DatasetState.SEALED,
                permitted_uses=("integrity_validation",),
                forbidden_uses=("model_evaluation",),
            )
            for role in (DatasetRole.HOLDOUT, DatasetRole.FORWARD)
        )
    )

    for role in (DatasetRole.HOLDOUT, DatasetRole.FORWARD):
        with pytest.raises(PermissionError, match="sealed for model research"):
            registry.require_research_access(role.value)
