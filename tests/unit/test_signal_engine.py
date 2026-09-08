from datetime import timedelta
from decimal import Decimal as D

import pytest
from sniper.domain.signal import TickFeatures
from sniper.features.engine import FeatureEngine
from sniper.strategy.filters import EvaluationContext
from sniper.strategy.scorer import SignalEngine, score_tier

from tests.unit.test_features import BASE, full_inputs, ticks, trending_bars


def decision(*, down=False, context=None):
    snapshot = FeatureEngine().compute(
        bars=full_inputs(down=down),
        ticks=ticks(down=down),
        as_of_utc=BASE + timedelta(hours=2),
    )
    return SignalEngine().evaluate(
        snapshot,
        context or EvaluationContext(max_spread_points=D(2), session_allowed=True, news_clear=True),
    )


def test_bullish_features_produce_explainable_buy_candidate():
    result = decision()
    assert result.side == "BUY"
    assert result.market_bias == "BUY"
    assert result.trigger_direction == "BUY"
    assert result.score >= 90
    assert result.tier in ("CANDIDATE", "PREMIUM_CANDIDATE")
    assert result.blockers == ()
    assert "M15_BUY_TREND" in result.reasons
    assert "M5_MOMENTUM_ALIGNED" in result.reasons
    assert "SPREAD_ACCEPTABLE" in result.reasons
    assert result.components.total == result.score
    assert result.sizing_authority == "RISK_ENGINE_ONLY"
    assert "volume" not in result.model_dump()


def test_bearish_features_produce_explainable_sell_candidate():
    result = decision(down=True)
    assert result.side == "SELL"
    assert result.market_bias == "SELL"
    assert result.trigger_direction == "SELL"
    assert result.score >= 90


def test_unknown_session_news_and_spread_fail_closed_to_wait():
    result = decision(context=EvaluationContext())
    assert result.side == "WAIT"
    assert set(result.blockers) >= {
        "SPREAD_LIMIT_UNCALIBRATED",
        "SESSION_UNVERIFIED",
        "NEWS_STATUS_UNKNOWN",
    }


@pytest.mark.parametrize(
    "score,tier",
    [
        (0, "WAIT"),
        (79, "WAIT"),
        (80, "WATCH"),
        (89, "WATCH"),
        (90, "CANDIDATE"),
        (94, "CANDIDATE"),
        (95, "PREMIUM_CANDIDATE"),
        (100, "PREMIUM_CANDIDATE"),
    ],
)
def test_fixed_unoptimized_score_thresholds(score, tier):
    assert score_tier(score) == tier


def test_spread_above_limit_blocks_even_a_high_score():
    result = decision(
        context=EvaluationContext(max_spread_points=D("0.5"), session_allowed=True, news_clear=True)
    )
    assert result.side == "WAIT"
    assert "SPREAD_TOO_HIGH" in result.blockers


def test_identical_inputs_are_deterministic():
    assert decision().model_dump() == decision().model_dump()


def test_stale_market_data_zeroes_tick_score_and_blocks_entry():
    snapshot = FeatureEngine().compute(
        bars=full_inputs(), ticks=ticks(), as_of_utc=BASE + timedelta(hours=2)
    )
    stale_ticks = TickFeatures.model_validate(
        {
            **snapshot.ticks.model_dump(),
            "last_tick_age_seconds": D(3),
        }
    )
    stale = snapshot.model_copy(update={"ticks": stale_ticks})
    result = SignalEngine().evaluate(
        stale,
        EvaluationContext(
            max_spread_points=D(2),
            max_tick_age_seconds=D(2),
            session_allowed=True,
            news_clear=True,
        ),
    )
    assert result.side == "WAIT"
    assert result.components.tick_confirmation == 0
    assert "MARKET_DATA_STALE" in result.blockers
    assert result.trigger_direction == "NEUTRAL"


def test_no_recent_tick_is_not_positive_tick_rate_evidence():
    old_ticks = [
        tick.model_copy(update={"timestamp_utc": tick.timestamp_utc - timedelta(seconds=10)})
        for tick in ticks()
    ]
    snapshot = FeatureEngine().compute(
        bars=full_inputs(), ticks=old_ticks, as_of_utc=BASE + timedelta(hours=2)
    )
    result = SignalEngine().evaluate(
        snapshot,
        EvaluationContext(
            max_spread_points=D(2),
            max_tick_age_seconds=D(20),
            session_allowed=True,
            news_clear=True,
        ),
    )
    assert snapshot.ticks.rate_1s == 0
    assert "TICK_RATE_STABLE_OR_ACCELERATING" not in result.reasons


def test_incomplete_feature_warmup_forces_wait():
    bars = full_inputs()
    bars["M15"] = bars["M15"][1:]
    snapshot = FeatureEngine().compute(
        bars=bars, ticks=ticks(), as_of_utc=BASE + timedelta(hours=2)
    )
    result = SignalEngine().evaluate(
        snapshot,
        EvaluationContext(max_spread_points=D(2), session_allowed=True, news_clear=True),
    )
    assert snapshot.warmup["M15"].complete is False
    assert result.side == "WAIT"
    assert "FEATURE_WARMUP_INCOMPLETE" in result.blockers


def test_trigger_against_m5_trend_is_explicitly_blocked_in_v1():
    bars = full_inputs()
    bars["M1"] = trending_bars("M1", 120, 1, down=True)
    snapshot = FeatureEngine().compute(
        bars=bars, ticks=ticks(down=True), as_of_utc=BASE + timedelta(hours=2)
    )
    result = SignalEngine().evaluate(
        snapshot,
        EvaluationContext(max_spread_points=D(2), session_allowed=True, news_clear=True),
    )
    assert result.market_bias == "BUY"
    assert result.trigger_direction == "SELL"
    assert result.side == "WAIT"
    assert "MARKET_TRIGGER_DIVERGENCE" in result.blockers
    assert "TRIGGER_AGAINST_M5_TREND" in result.blockers
