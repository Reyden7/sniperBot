"""Explainable, sizing-free outputs of the V1 Signal Engine."""

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import Field

from sniper.config import Model, NonNegative


class SignalSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"


class Bias(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


class SignalTier(StrEnum):
    WAIT = "WAIT"
    WATCH = "WATCH"
    CANDIDATE = "CANDIDATE"
    PREMIUM_CANDIDATE = "PREMIUM_CANDIDATE"


class TrendDirection(StrEnum):
    UP = "UP"
    DOWN = "DOWN"
    FLAT = "FLAT"


class MarketRegime(StrEnum):
    TREND = "TREND"
    RANGE = "RANGE"


class PatternFeatures(Model):
    engulfing: Literal["BULLISH", "BEARISH"] | None = None
    hammer: bool = False
    shooting_star: bool = False


class M15Features(Model):
    ema: Decimal
    ema_slope: Decimal
    atr: NonNegative
    structure: Literal["HH_HL", "LH_LL", "MIXED"]
    direction: TrendDirection
    regime: MarketRegime


class M5Features(Model):
    ema_short: Decimal
    ema_long: Decimal
    momentum: Decimal
    atr: NonNegative
    direction: TrendDirection
    support: Decimal
    resistance: Decimal
    support_distance_atr: NonNegative
    resistance_distance_atr: NonNegative
    breakout_retest: Literal["BULLISH", "BEARISH"] | None = None


class M1Features(Model):
    momentum_1: Decimal
    momentum_3: Decimal
    momentum_5: Decimal
    atr: NonNegative
    range_atr_ratio: NonNegative
    body_ratio: NonNegative
    upper_wick_ratio: NonNegative
    lower_wick_ratio: NonNegative
    pattern: PatternFeatures
    compression: bool
    expansion: bool


class TickFeatures(Model):
    rate_1s: int = Field(ge=0)
    rate_5s: int = Field(ge=0)
    rate_15s: int = Field(ge=0)
    uptick_ratio: Decimal = Field(ge=0, le=1)
    downtick_ratio: Decimal = Field(ge=0, le=1)
    price_acceleration: Decimal
    tick_rate_acceleration: Decimal
    last_tick_timestamp_utc: datetime
    last_tick_age_seconds: NonNegative
    current_spread_points: NonNegative
    median_spread_points: NonNegative


class TimeframeWarmup(Model):
    required_bars: int = Field(gt=0)
    available_bars: int = Field(ge=0)
    complete: bool
    feature_families: tuple[str, ...]


class FeatureSnapshot(Model):
    symbol: Literal["EURUSD"] = "EURUSD"
    as_of_utc: datetime
    m15: M15Features
    m5: M5Features
    m1: M1Features
    ticks: TickFeatures
    completed_bars: dict[Literal["M15", "M5", "M1"], int]
    warmup: dict[Literal["M15", "M5", "M1"], TimeframeWarmup]
    visible_ticks: int = Field(ge=0)


class ScoreComponents(Model):
    m15_regime_context: int = Field(ge=0, le=15)
    m5_trend_alignment: int = Field(ge=0, le=15)
    m1_momentum: int = Field(ge=0, le=15)
    tick_confirmation: int = Field(ge=0, le=15)
    market_structure: int = Field(ge=0, le=15)
    volatility_quality: int = Field(ge=0, le=10)
    spread_execution_quality: int = Field(ge=0, le=10)
    session_news_quality: int = Field(ge=0, le=5)

    @property
    def total(self) -> int:
        return sum(self.model_dump().values())


class SignalDecision(Model):
    symbol: Literal["EURUSD"] = "EURUSD"
    as_of_utc: datetime
    side: SignalSide
    market_bias: Bias
    trigger_direction: Bias
    score: int = Field(ge=0, le=100)
    tier: SignalTier
    components: ScoreComponents
    reasons: tuple[str, ...]
    blockers: tuple[str, ...]
    proposed_stop_distance_points: int | None = Field(default=None, gt=0)
    proposed_target_distance_points: int | None = Field(default=None, gt=0)
    sizing_authority: Literal["RISK_ENGINE_ONLY"] = "RISK_ENGINE_ONLY"


class SignalAnalysisReport(Model):
    mode: Literal["ANALYSIS_ONLY"] = "ANALYSIS_ONLY"
    live_trading_enabled: Literal[False] = False
    thresholds_status: Literal["FIXED_NOT_OPTIMIZED"] = "FIXED_NOT_OPTIMIZED"
    features: FeatureSnapshot
    decision: SignalDecision
