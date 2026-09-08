from datetime import UTC, datetime, timedelta
from decimal import Decimal as D

import numpy as np
import polars as pl
from sniper.data.parquet_store import ParquetStore
from sniper.evaluation.edge_validation import (
    EdgeValidationConfig,
    EdgeValidator,
    _session,
    _streaming_tick_features,
)
from sniper.features.engine import FeatureEngine

from tests.unit.test_features import BASE, full_inputs, ticks


def write_ticks(root, start, seconds):
    timestamps = [start + timedelta(seconds=index) for index in range(seconds)]
    bids = [1.10000 + index * 0.000001 for index in range(seconds)]
    path = root / "ticks" / "symbol=EURUSD" / "date=2026-09-08" / "part-000.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp_utc": timestamps,
            "bid": bids,
            "ask": [bid + 0.00001 for bid in bids],
        }
    ).write_parquet(path)


def test_edge_validator_streams_daily_observations_and_keeps_engine_frozen(tmp_path):
    for bars in full_inputs().values():
        ParquetStore(tmp_path).write_bars(bars)
    start = BASE + timedelta(hours=2)
    write_ticks(tmp_path, start, 600)
    report = EdgeValidator(
        EdgeValidationConfig(
            bootstrap_resamples=1000,
            minimum_usable_observations=1000,
            minimum_candidate_observations=30,
            minimum_active_weeks=4,
        )
    ).run(data_root=tmp_path, start_utc=start, end_utc=start + timedelta(minutes=10))
    assert report.performance.ticks_analyzed == 600
    assert report.performance.observations_produced == 10
    assert len(report.score_buckets) == 7
    assert {bucket.key for bucket in report.score_buckets} == {
        "0-49",
        "50-59",
        "60-69",
        "70-79",
        "80-89",
        "90-94",
        "95-100",
    }
    assert report.signal_engine_parameters_modified is False
    assert report.optimization_performed is False
    assert report.live_trading_enabled is False
    assert report.diagnosis == "INSUFFICIENT_DATA"
    observation_files = list(tmp_path.glob("evaluations/**/*.parquet"))
    assert len(observation_files) == 1
    observations = pl.read_parquet(observation_files[0])
    assert observations.height == 10
    usable = observations.filter(pl.col("usable"))
    assert usable.height > 0
    first = usable.row(0, named=True)
    assert first["expectancy_after_spread_points"] < first["expectancy_before_costs_points"]
    assert first["expectancy_after_slippage_points"] == (
        first["expectancy_after_spread_points"] - 2
    )
    assert first["expectancy_after_commission_points"] < first["expectancy_after_slippage_points"]
    assert first["mfe_30s_eur"] >= 0
    assert first["mae_30s_eur"] >= 0


def test_session_classification_uses_utc_instants_and_dst_aware_timezones():
    assert _session(datetime(2026, 6, 10, 12, tzinfo=UTC)) == "LONDON_NEW_YORK"
    assert _session(datetime(2026, 6, 10, 7, tzinfo=UTC)) == "LONDON"
    assert _session(datetime(2026, 6, 10, 0, tzinfo=UTC)) == "ASIA"
    assert _session(datetime(2026, 12, 10, 13, tzinfo=UTC)) == "LONDON_NEW_YORK"


def test_streaming_tick_adapter_matches_frozen_feature_engine():
    generated = ticks()
    expected = (
        FeatureEngine()
        .compute(bars=full_inputs(), ticks=generated, as_of_utc=generated[-1].timestamp_utc)
        .ticks
    )
    timestamps_us = np.array(
        [int(tick.timestamp_utc.timestamp() * 1_000_000) for tick in generated],
        dtype=np.int64,
    )
    bids = np.array([float(tick.bid) for tick in generated])
    asks = np.array([float(tick.ask) for tick in generated])
    spreads = np.array([float(tick.spread / D("0.00001")) for tick in generated])
    observed = _streaming_tick_features(
        timestamps_us,
        bids,
        asks,
        spreads,
        0,
        len(generated) - 1,
        {},
    )
    assert observed == expected


def test_simulated_broker_assumptions_are_explicit(tmp_path):
    for bars in full_inputs().values():
        ParquetStore(tmp_path).write_bars(bars)
    start = BASE + timedelta(hours=2)
    write_ticks(tmp_path, start, 600)
    report = EdgeValidator(EdgeValidationConfig(bootstrap_resamples=1000)).run(
        data_root=tmp_path,
        start_utc=start,
        end_utc=start + timedelta(minutes=10),
    )
    assert report.simulation.broker_profile.kind == "SIMULATED"
    assert report.simulation.broker_profile.is_real_broker_capability is False
    assert report.simulation.evaluation_volume_lots == 0.0001
    assert report.simulation.slippage_points_per_side == 1
    assert report.simulation.commission_eur_per_lot_per_side == 2
    assert "MetaQuotes-Demo" in report.simulation.disclaimer
    assert report.confidence_intervals is None or report.confidence_intervals.seed == 20260908
