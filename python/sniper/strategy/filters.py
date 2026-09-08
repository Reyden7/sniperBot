"""Fail-closed Phase D data, spread, session and news filters."""

from decimal import Decimal

from pydantic import Field

from sniper.config import Model
from sniper.domain.signal import FeatureSnapshot


class EvaluationContext(Model):
    max_spread_points: Decimal | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_tick_age_seconds: Decimal = Field(default=Decimal(2), gt=0, allow_inf_nan=False)
    session_allowed: bool | None = None
    news_clear: bool | None = None


def blockers(snapshot: FeatureSnapshot, context: EvaluationContext) -> list[str]:
    result = []
    incomplete = [timeframe for timeframe, state in snapshot.warmup.items() if not state.complete]
    if incomplete:
        result.append("FEATURE_WARMUP_INCOMPLETE")
        result.extend(f"INSUFFICIENT_{timeframe}_HISTORY" for timeframe in incomplete)
    if snapshot.visible_ticks < 3:
        result.append("INSUFFICIENT_TICK_HISTORY")
    if snapshot.ticks.last_tick_age_seconds > context.max_tick_age_seconds:
        result.append("MARKET_DATA_STALE")
    if context.max_spread_points is None:
        result.append("SPREAD_LIMIT_UNCALIBRATED")
    elif snapshot.ticks.current_spread_points > context.max_spread_points:
        result.append("SPREAD_TOO_HIGH")
    if context.session_allowed is not True:
        result.append(
            "SESSION_UNVERIFIED" if context.session_allowed is None else "SESSION_BLOCKED"
        )
    if context.news_clear is not True:
        result.append("NEWS_STATUS_UNKNOWN" if context.news_clear is None else "HIGH_IMPACT_NEWS")
    return result
