"""Fixed, explainable Phase D scoring. No optimized thresholds or sizing."""

from decimal import ROUND_CEILING, Decimal

from sniper.domain.signal import (
    Bias,
    FeatureSnapshot,
    MarketRegime,
    ScoreComponents,
    SignalDecision,
    SignalSide,
    SignalTier,
    TrendDirection,
)
from sniper.strategy.filters import EvaluationContext, blockers


def _aligned(value: Decimal, bias: Bias) -> bool:
    return (bias == Bias.BUY and value > 0) or (bias == Bias.SELL and value < 0)


def _market_bias(snapshot: FeatureSnapshot) -> Bias:
    votes = 0
    votes += (
        1
        if snapshot.m15.direction == TrendDirection.UP
        else -1
        if snapshot.m15.direction == TrendDirection.DOWN
        else 0
    )
    votes += (
        1 if snapshot.m15.structure == "HH_HL" else -1 if snapshot.m15.structure == "LH_LL" else 0
    )
    votes += (
        1
        if snapshot.m5.direction == TrendDirection.UP
        else -1
        if snapshot.m5.direction == TrendDirection.DOWN
        else 0
    )
    votes += 1 if snapshot.m5.momentum > 0 else -1 if snapshot.m5.momentum < 0 else 0
    return Bias.BUY if votes > 0 else Bias.SELL if votes < 0 else Bias.NEUTRAL


def _trigger_direction(snapshot: FeatureSnapshot, *, ticks_stale: bool) -> Bias:
    if ticks_stale:
        return Bias.NEUTRAL
    votes = 0
    for value in (
        snapshot.m1.momentum_1,
        snapshot.m1.momentum_3,
        snapshot.m1.momentum_5,
        snapshot.ticks.price_acceleration,
    ):
        votes += 1 if value > 0 else -1 if value < 0 else 0
    if snapshot.ticks.uptick_ratio > Decimal("0.55"):
        votes += 1
    elif snapshot.ticks.downtick_ratio > Decimal("0.55"):
        votes -= 1
    return Bias.BUY if votes > 0 else Bias.SELL if votes < 0 else Bias.NEUTRAL


def score_tier(score: int) -> SignalTier:
    if not 0 <= score <= 100:
        raise ValueError("score must be between 0 and 100")
    if score < 80:
        return SignalTier.WAIT
    if score < 90:
        return SignalTier.WATCH
    if score < 95:
        return SignalTier.CANDIDATE
    return SignalTier.PREMIUM_CANDIDATE


class SignalEngine:
    def evaluate(
        self,
        snapshot: FeatureSnapshot,
        context: EvaluationContext | None = None,
        *,
        point: Decimal = Decimal("0.00001"),
    ) -> SignalDecision:
        if snapshot.symbol != "EURUSD" or point <= 0:
            raise ValueError("Signal Engine V1 supports EURUSD with a positive point")
        context = context or EvaluationContext()
        ticks_stale = snapshot.ticks.last_tick_age_seconds > context.max_tick_age_seconds
        market_bias = _market_bias(snapshot)
        trigger_direction = _trigger_direction(snapshot, ticks_stale=ticks_stale)
        reasons: list[str] = []
        m15 = 0
        if snapshot.m15.regime == MarketRegime.TREND and (
            (market_bias == Bias.BUY and snapshot.m15.direction == TrendDirection.UP)
            or (market_bias == Bias.SELL and snapshot.m15.direction == TrendDirection.DOWN)
        ):
            m15 = 15
            reasons.append(f"M15_{market_bias.value}_TREND")
        elif snapshot.m15.regime == MarketRegime.RANGE:
            m15 = 5
            reasons.append("M15_RANGE")

        m5 = 0
        if (market_bias == Bias.BUY and snapshot.m5.direction == TrendDirection.UP) or (
            market_bias == Bias.SELL and snapshot.m5.direction == TrendDirection.DOWN
        ):
            m5 += 8
            reasons.append("M5_TREND_ALIGNED")
        if _aligned(snapshot.m5.momentum, market_bias):
            m5 += 7
            reasons.append("M5_MOMENTUM_ALIGNED")

        m1 = sum(
            5
            for value in (
                snapshot.m1.momentum_1,
                snapshot.m1.momentum_3,
                snapshot.m1.momentum_5,
            )
            if _aligned(value, trigger_direction)
        )
        if m1:
            reasons.append(f"M1_MOMENTUM_{trigger_direction.value}")

        tick_score = 0
        if not ticks_stale and trigger_direction != Bias.NEUTRAL:
            directional_ratio = (
                snapshot.ticks.uptick_ratio
                if trigger_direction == Bias.BUY
                else snapshot.ticks.downtick_ratio
            )
            if directional_ratio >= Decimal("0.60"):
                tick_score += 8
                reasons.append(f"TICK_RATIO_{trigger_direction.value}")
            if _aligned(snapshot.ticks.price_acceleration, trigger_direction):
                tick_score += 4
                reasons.append(f"TICK_ACCELERATION_{trigger_direction.value}")
            if snapshot.ticks.rate_1s > 0 and snapshot.ticks.tick_rate_acceleration >= 0:
                tick_score += 3
                reasons.append("TICK_RATE_STABLE_OR_ACCELERATING")

        structure_score = 0
        expected_structure = "HH_HL" if market_bias == Bias.BUY else "LH_LL"
        if snapshot.m15.structure == expected_structure:
            structure_score += 8
            reasons.append(f"M15_STRUCTURE_{expected_structure}")
        expected_breakout = "BULLISH" if market_bias == Bias.BUY else "BEARISH"
        if snapshot.m5.breakout_retest == expected_breakout:
            structure_score += 7
            reasons.append(f"M5_BREAKOUT_RETEST_{expected_breakout}")

        if snapshot.m1.atr > 0 and Decimal("0.60") <= snapshot.m1.range_atr_ratio <= Decimal(
            "1.50"
        ):
            volatility = 10
            reasons.append("VOLATILITY_BALANCED")
        elif snapshot.m1.atr > 0:
            volatility = 6
            reasons.append("VOLATILITY_EXTREME_OR_COMPRESSED")
        else:
            volatility = 0

        spread_score = 0
        if (
            context.max_spread_points is not None
            and snapshot.ticks.current_spread_points <= context.max_spread_points
        ):
            spread_score = 6
            if snapshot.ticks.current_spread_points <= (
                snapshot.ticks.median_spread_points * Decimal("1.25")
            ):
                spread_score += 4
            reasons.append("SPREAD_ACCEPTABLE")

        session_news = (2 if context.session_allowed is True else 0) + (
            3 if context.news_clear is True else 0
        )
        if session_news == 5:
            reasons.append("SESSION_AND_NEWS_CLEAR")
        components = ScoreComponents(
            m15_regime_context=m15,
            m5_trend_alignment=m5,
            m1_momentum=m1,
            tick_confirmation=tick_score,
            market_structure=structure_score,
            volatility_quality=volatility,
            spread_execution_quality=spread_score,
            session_news_quality=session_news,
        )
        score = components.total
        found_blockers = blockers(snapshot, context)
        if market_bias == Bias.NEUTRAL or trigger_direction == Bias.NEUTRAL:
            found_blockers.append("DIRECTION_CONFLICT")
        elif market_bias != trigger_direction:
            found_blockers.append("MARKET_TRIGGER_DIVERGENCE")
        expected_m5 = TrendDirection.UP if trigger_direction == Bias.BUY else TrendDirection.DOWN
        if trigger_direction != Bias.NEUTRAL:
            if snapshot.m5.direction == TrendDirection.FLAT:
                found_blockers.append("M5_DIRECTION_UNCONFIRMED")
            elif snapshot.m5.direction != expected_m5:
                found_blockers.append("TRIGGER_AGAINST_M5_TREND")
        tier = score_tier(score)
        if tier in (SignalTier.WAIT, SignalTier.WATCH):
            reasons.append("SCORE_BELOW_CANDIDATE_THRESHOLD")
        side = (
            SignalSide(trigger_direction.value)
            if not found_blockers and tier in (SignalTier.CANDIDATE, SignalTier.PREMIUM_CANDIDATE)
            else SignalSide.WAIT
        )
        stop_points = None
        target_points = None
        if snapshot.m1.atr > 0:
            stop_points = max(
                1,
                int(
                    (snapshot.m1.atr / point * Decimal("1.20")).to_integral_value(
                        rounding=ROUND_CEILING
                    )
                ),
            )
            target_points = max(
                stop_points + 1,
                int(
                    (Decimal(stop_points) * Decimal("1.50")).to_integral_value(
                        rounding=ROUND_CEILING
                    )
                ),
            )
        return SignalDecision(
            as_of_utc=snapshot.as_of_utc,
            side=side,
            market_bias=market_bias,
            trigger_direction=trigger_direction,
            score=score,
            tier=tier,
            components=components,
            reasons=tuple(reasons),
            blockers=tuple(found_blockers),
            proposed_stop_distance_points=stop_points,
            proposed_target_distance_points=target_points,
        )
