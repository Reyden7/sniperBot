from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sniper.backtest.execution_model import BrokerSimulationConfig
from sniper.evaluation.research_d12 import (
    FEATURE_GROUPS,
    ResearchD12Engine,
    _apply_cooldown,
    _benjamini_hochberg,
    _event_labels,
    _event_positions,
    _second_context,
    _with_folds,
)


def _raw(seconds: int) -> pl.DataFrame:
    start = datetime(2026, 6, 10, tzinfo=UTC)
    timestamps = [start + timedelta(seconds=index) for index in range(seconds)]
    bids = np.asarray([1.10 + index * 0.000001 for index in range(seconds)])
    asks = bids + 0.00001
    return pl.DataFrame(
        {
            "timestamp_utc": timestamps,
            "bid": bids,
            "ask": asks,
            "spread_points": np.ones(seconds),
        }
    )


def test_d12_causal_second_features_ignore_appended_future():
    prefix = _second_context(_raw(1_000))
    extended = _second_context(_raw(1_100))

    for name in ("rv_300", "range_900", "return_60", "rate_15"):
        np.testing.assert_allclose(
            prefix[name],
            extended[name][:1_000],
            equal_nan=True,
        )


def test_d12_cooldown_keeps_first_eligible_timestamp():
    times = np.arange(0, 10) * 1_000_000
    positions = np.arange(10)

    selected = _apply_cooldown(positions, np.ones(10, dtype=bool), times, 3)

    assert selected.tolist() == [0, 3, 6, 9]


def test_d12_compression_release_uses_current_60_second_range():
    size = 20_000
    times = np.arange(size, dtype=np.int64) * 1_000_000
    context = {
        "times": times,
        "mids": np.zeros(size),
        "rv_300": np.ones(size),
        "rv_900": np.ones(size),
        "rv_60": np.ones(size),
        "return_60": np.zeros(size),
        "return_5": np.zeros(size),
        "rate_5": np.ones(size),
        "rate5_median_4h": np.full(size, 2.0),
        "range_300": np.full(size, 5.0),
        "range_60": np.ones(size),
        "range_p20_4h": np.full(size, 5.0),
    }
    day_start = 15_000 * 1_000_000
    day_end = 16_000 * 1_000_000

    _, quiet_counts, _ = _event_positions(context, day_start, day_end)
    context["range_60"][15_000:16_000] = 10.0
    _, released_counts, _ = _event_positions(context, day_start, day_end)

    assert quiet_counts["COMPRESSION_RELEASE_EVENT"] == 0
    assert released_counts["COMPRESSION_RELEASE_EVENT"] == 2


def test_d12_executable_labels_use_bid_and_ask():
    raw = _raw(902)
    times = raw["timestamp_utc"].cast(pl.Int64).to_numpy().astype(np.int64)
    bids = raw["bid"].to_numpy().astype(float)
    asks = raw["ask"].to_numpy().astype(float)

    labels = _event_labels(times, bids, asks, int(times[0]), BrokerSimulationConfig())

    assert labels is not None
    assert labels["h30_long_executable_return_points"] == pytest.approx(2.0)
    assert labels["h30_short_executable_return_points"] == pytest.approx(-4.0)
    assert labels["h30_signed_executable_return_points"] == pytest.approx(3.0)


def test_d12_benjamini_hochberg_is_monotone_in_rank():
    adjusted = _benjamini_hochberg([0.001, 0.01, 0.03, None])

    assert adjusted[:3] == pytest.approx([0.003, 0.015, 0.03])
    assert adjusted[3] is None


def test_d12_assigns_four_declared_validation_folds():
    start = datetime(2026, 6, 10, tzinfo=UTC)
    end = datetime(2026, 9, 8, tzinfo=UTC)
    duration = end - start
    frame = pl.DataFrame(
        {
            "timestamp_utc": [
                start + duration * fraction for fraction in (0.1, 0.31, 0.41, 0.51, 0.61)
            ]
        }
    )

    assert _with_folds(frame, start, end)["fold"].to_list() == [0, 1, 2, 3, 4]


def test_d12_feature_registry_contains_all_predeclared_interactions():
    interactions = [name for name, family in FEATURE_GROUPS.items() if family == "regime"]

    assert len([name for name in interactions if "_x_" in name]) == 8


def test_d12_refuses_holdout_before_protocol_or_model_access(tmp_path: Path):
    with pytest.raises(PermissionError, match="refuses any HOLDOUT"):
        ResearchD12Engine().run(
            data_root=tmp_path / "data" / "holdout-v2",
            start_utc=datetime(2026, 6, 10, tzinfo=UTC),
            end_utc=datetime(2026, 9, 8, tzinfo=UTC),
            protocol_path=tmp_path / "missing.yaml",
        )
