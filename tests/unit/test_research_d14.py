from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from sniper.evaluation.research_d14 import (
    ResearchD14Engine,
    cost_frontier,
    direction_bins,
    oracle_decomposition,
)


def test_d14_direction_bins_are_equal_count_and_cover_every_oof_row():
    predicted = np.linspace(-2.0, 2.0, 103)
    realized = predicted * 3.0
    long_exec = realized - 0.5
    short_exec = -realized - 0.5
    long_base = long_exec - 2.0
    short_base = short_exec - 2.0

    bins = direction_bins(
        predicted,
        realized,
        long_exec,
        short_exec,
        long_base,
        short_base,
        absolute_sort=True,
    )

    assert len(bins) == 10
    assert sum(item["N"] for item in bins) == 103
    assert max(item["N"] for item in bins) - min(item["N"] for item in bins) <= 1


def test_d14_oracle_direction_chooses_best_realized_side():
    probability = np.asarray([0.8, 0.8])
    predicted = np.asarray([1.0, 1.0])
    oracle_regime = np.asarray([True, True])
    market = np.asarray([5.0, -6.0])
    long_exec = np.asarray([4.0, -7.0])
    short_exec = np.asarray([-6.0, 5.0])
    long_base = long_exec - 2.0
    short_base = short_exec - 2.0
    long_stress = long_exec - 4.0
    short_stress = short_exec - 4.0

    result = oracle_decomposition(
        probability,
        predicted,
        oracle_regime,
        market,
        long_exec,
        short_exec,
        long_base,
        short_base,
        long_stress,
        short_stress,
        np.asarray(["W1", "W1"]),
        np.asarray(["LONDON", "NEW_YORK"]),
    )

    current = result["CURRENT_REGIME_CURRENT_DIRECTION"]
    oracle = result["CURRENT_REGIME_ORACLE_DIRECTION"]
    assert current["base_expectancy_points"] == pytest.approx(-3.5)
    assert oracle["base_expectancy_points"] == pytest.approx(2.5)


def test_d14_cost_frontier_keeps_observed_spread_in_candidate_rule():
    result = cost_frontier(
        probability=np.asarray([0.8, 0.8]),
        predicted=np.asarray([2.0, 0.4]),
        signal_spread=np.asarray([1.0, 0.5]),
        selected_executable=np.asarray([1.0, 1.0]),
        base_commission=np.asarray([0.4, 0.4]),
        weeks=np.asarray(["W1", "W1"]),
    )

    assert result["observed_spread_only"]["candidate_count"] == 1
    assert result["base_assumptions_unchanged"] is True


def test_d14_refuses_holdout_before_source_access(tmp_path: Path):
    with pytest.raises(PermissionError, match="refuses any HOLDOUT"):
        ResearchD14Engine().run(
            data_root=tmp_path / "holdout-v2",
            start_utc=datetime(2026, 6, 10, tzinfo=UTC),
            end_utc=datetime(2026, 9, 8, tzinfo=UTC),
            protocol_path=tmp_path / "missing.yaml",
        )
