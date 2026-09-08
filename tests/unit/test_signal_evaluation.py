from datetime import timedelta
from decimal import Decimal as D

from sniper.domain.trade import MarketTick
from sniper.evaluation.engine import SignalEvaluationConfig, SignalEvaluator
from sniper.strategy.filters import EvaluationContext

from tests.unit.test_features import BASE, full_inputs


def evaluation_ticks(*, future_jump=D(0)):
    start = BASE + timedelta(hours=2) - timedelta(seconds=10)
    result = []
    for index in range(251):
        timestamp = start + timedelta(seconds=index)
        price = D("1.10000") + D(index) * D("0.000001")
        if index > 70:
            price += future_jump
        result.append(MarketTick(timestamp_utc=timestamp, bid=price, ask=price + D("0.00001")))
    return result


def run_evaluation(ticks):
    start = BASE + timedelta(hours=2)
    return SignalEvaluator(SignalEvaluationConfig(interval_seconds=60)).run(
        bars=full_inputs(),
        ticks=ticks,
        start_utc=start,
        end_utc=start + timedelta(seconds=240),
        context=EvaluationContext(
            max_spread_points=D(2),
            max_tick_age_seconds=D(2),
            session_allowed=True,
            news_clear=True,
        ),
    )


def test_evaluator_records_executable_long_short_returns_mfe_mae_and_costs():
    report = run_evaluation(evaluation_ticks())
    first = report.observations[0]
    assert first.blockers == ()
    assert first.outcomes is not None
    assert first.outcomes.long.returns_points.return_after_30s == D(2)
    assert first.outcomes.short.returns_points.return_after_30s == D(-4)
    assert first.outcomes.long.mfe_30s_points == D(2)
    assert first.outcomes.long.mae_30s_points == D("0.9")
    assert first.expected_total_execution_cost_points == D(3)
    assert first.target_cost_ratio is not None
    assert first.meets_minimum_target_cost_ratio is True
    assert first.outcome_status == "COMPLETE"
    bucket = next(bucket for bucket in report.score_buckets if bucket.observations)
    assert bucket.label == "80-89"
    assert bucket.win_direction_30s_pct == D(100)
    assert report.optimization_performed is False
    assert report.live_trading_enabled is False
    assert report.complete_outcome_observations == report.unblocked_observations


def test_future_prices_change_outcomes_but_not_signal_features_or_score():
    baseline = run_evaluation(evaluation_ticks())
    changed = run_evaluation(evaluation_ticks(future_jump=D("0.001")))
    left, right = baseline.observations[0], changed.observations[0]
    assert left.score == right.score
    assert left.components == right.components
    assert left.market_bias == right.market_bias
    assert left.trigger_direction == right.trigger_direction
    assert left.outcomes != right.outcomes


def test_blocked_observation_never_receives_future_outcomes():
    start = BASE + timedelta(hours=2)
    report = SignalEvaluator(SignalEvaluationConfig(interval_seconds=60)).run(
        bars=full_inputs(),
        ticks=evaluation_ticks(),
        start_utc=start,
        end_utc=start + timedelta(seconds=240),
        context=EvaluationContext(),
    )
    assert report.observations
    assert all(
        row.blockers and row.outcomes is None and row.outcome_status == "SIGNAL_BLOCKED"
        for row in report.observations
    )


def test_missing_precise_future_horizon_is_reported_not_fabricated():
    start = BASE + timedelta(hours=2)
    sparse = [
        tick
        for tick in evaluation_ticks()
        if tick.timestamp_utc <= start or tick.timestamp_utc >= start + timedelta(seconds=20)
    ]
    report = SignalEvaluator(
        SignalEvaluationConfig(interval_seconds=60, max_horizon_tick_delay_seconds=D(2))
    ).run(
        bars=full_inputs(),
        ticks=sparse,
        start_utc=start,
        end_utc=start + timedelta(seconds=240),
        context=EvaluationContext(
            max_spread_points=D(2),
            max_tick_age_seconds=D(2),
            session_allowed=True,
            news_clear=True,
        ),
    )
    first = report.observations[0]
    assert first.blockers == ()
    assert first.outcome_status == "FUTURE_DATA_INCOMPLETE"
    assert first.outcomes is None
