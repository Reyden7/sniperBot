from datetime import timedelta
from decimal import Decimal as D

import polars as pl
import sniper.evaluation.event_discovery as event_discovery
from sniper.data.parquet_store import ParquetStore
from sniper.domain.signal import Bias, ScoreComponents, SignalDecision, SignalSide
from sniper.evaluation.event_discovery import (
    EventCandidateDiscoverer,
    EventDiscoveryConfig,
    _minute_score_upper_bounds,
)
from sniper.features.engine import FeatureEngine
from sniper.strategy.filters import EvaluationContext
from sniper.strategy.scorer import SignalEngine, score_tier

from tests.unit.test_edge_validation import write_ticks
from tests.unit.test_features import BASE, full_inputs, ticks


def _decision(snapshot, score, *, blocked=False):
    side = SignalSide.BUY if score >= 90 and not blocked else SignalSide.WAIT
    return SignalDecision(
        as_of_utc=snapshot.as_of_utc,
        side=side,
        market_bias=Bias.BUY,
        trigger_direction=Bias.BUY,
        score=score,
        tier=score_tier(score),
        components=ScoreComponents(
            m15_regime_context=0,
            m5_trend_alignment=0,
            m1_momentum=0,
            tick_confirmation=0,
            market_structure=0,
            volatility_quality=0,
            spread_execution_quality=0,
            session_news_quality=0,
        ),
        reasons=(),
        blockers=("TEST_BLOCKER",) if blocked else (),
        proposed_stop_distance_points=20,
        proposed_target_distance_points=30,
    )


def test_event_discovery_uses_hysteresis_and_keeps_peak_non_tradable(tmp_path, monkeypatch):
    for bars in full_inputs().values():
        ParquetStore(tmp_path).write_bars(bars)
    start = BASE + timedelta(hours=2)
    write_ticks(tmp_path, start, 300)
    discoverer = EventCandidateDiscoverer(EventDiscoveryConfig())
    scores = {0: 80, 1: 90, 2: 92, 3: 89, 4: 91, 5: 84, 6: 90, 7: 95, 8: 95}

    def evaluate(snapshot, context, *, point):
        second = int((snapshot.as_of_utc - start).total_seconds())
        return _decision(snapshot, scores.get(second, 80), blocked=second == 8)

    monkeypatch.setattr(discoverer.signal_engine, "evaluate", evaluate)
    monkeypatch.setattr(
        event_discovery,
        "_minute_score_upper_bounds",
        lambda snapshot, context, signal_engine, point: {Bias.BUY: 100, Bias.SELL: 100},
    )
    report = discoverer.run(
        data_root=tmp_path,
        start_utc=start,
        end_utc=start + timedelta(seconds=300),
    )

    assert report.episode_summary.episodes == 2
    assert report.direction_diagnostics[0].moment == "FIRST_CROSSING"
    assert report.direction_diagnostics[0].direction_mode == "ORIGINAL"
    assert report.direction_diagnostics[0].tradable is True
    assert report.direction_diagnostics[1].direction_mode == "INVERSE_DIAGNOSTIC"
    assert report.direction_diagnostics[1].tradable is False
    assert report.direction_diagnostics[2].moment == "PEAK_SCORE"
    assert report.direction_diagnostics[2].tradable is False
    assert report.signal_engine_parameters_modified is False
    assert report.strategy_direction_modified is False
    assert report.live_trading_enabled is False

    events = pl.read_parquet(report.event_parquet_path)
    assert events.height == 4
    first = events.filter(pl.col("moment") == "FIRST_CROSSING").sort("episode_id")
    assert first["score"].to_list() == [90, 90]
    assert first["episode_peak_score"].to_list() == [92, 95]
    assert first["episode_end_reason"].to_list() == ["SCORE_BELOW_90", "BLOCKER"]


def test_event_discovery_methodology_thresholds_are_not_configurable():
    for config in (
        EventDiscoveryConfig(sampling_interval_seconds=2),
        EventDiscoveryConfig(activation_score=91),
        EventDiscoveryConfig(rearm_score=84),
    ):
        try:
            config.validate_methodology()
        except ValueError:
            pass
        else:
            raise AssertionError("fixed D.7 methodology unexpectedly accepted a changed threshold")


def test_minute_bound_is_not_below_the_unchanged_signal_score():
    generated = ticks()
    snapshot = FeatureEngine().compute(
        bars=full_inputs(),
        ticks=generated,
        as_of_utc=generated[-1].timestamp_utc,
    )
    context = EvaluationContext(
        max_spread_points=D("2"),
        max_tick_age_seconds=D("2"),
        session_allowed=True,
        news_clear=True,
    )
    engine = SignalEngine()
    actual = engine.evaluate(snapshot, context)
    bounds = _minute_score_upper_bounds(snapshot, context, engine, D("0.00001"))
    assert actual.score <= max(bounds.values(), default=0)
