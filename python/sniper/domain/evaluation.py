"""Out-of-sample Phase D.5 signal evaluation records."""

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import Field

from sniper.config import Model, NonNegative, Positive
from sniper.domain.signal import Bias, ScoreComponents, SignalSide, SignalTier


class HorizonReturns(Model):
    return_after_5s: Decimal
    return_after_10s: Decimal
    return_after_30s: Decimal
    return_after_60s: Decimal
    return_after_180s: Decimal


class DirectionOutcome(Model):
    returns_points: HorizonReturns
    mfe_30s_points: NonNegative
    mae_30s_points: NonNegative


class EvaluationOutcomes(Model):
    long: DirectionOutcome
    short: DirectionOutcome


class SignalObservation(Model):
    timestamp_utc: datetime
    score: int = Field(ge=0, le=100)
    tier: SignalTier
    side: SignalSide
    market_bias: Bias
    trigger_direction: Bias
    components: ScoreComponents
    blockers: tuple[str, ...]
    current_quote_timestamp_utc: datetime
    current_quote_age_seconds: NonNegative
    current_bid: Positive
    current_ask: Positive
    spread_points: NonNegative
    m1_atr_points: NonNegative
    session: Literal["ASIA", "LONDON", "LONDON_NEW_YORK", "NEW_YORK", "OFF_HOURS"]
    expected_total_execution_cost_points: NonNegative
    target_cost_ratio: NonNegative | None
    meets_minimum_target_cost_ratio: bool | None
    outcome_status: Literal["SIGNAL_BLOCKED", "COMPLETE", "FUTURE_DATA_INCOMPLETE"]
    outcomes: EvaluationOutcomes | None


class ScoreBucket(Model):
    label: Literal["50-59", "60-69", "70-79", "80-89", "90-94", "95-100"]
    score_min: int = Field(ge=0, le=100)
    score_max: int = Field(ge=0, le=100)
    observations: int = Field(ge=0)
    win_direction_30s_pct: NonNegative | None
    average_mfe_30s_points: NonNegative | None
    average_mae_30s_points: NonNegative | None
    average_potential_net_return_30s_points: Decimal | None
    average_total_execution_cost_points: NonNegative | None
    average_target_cost_ratio: NonNegative | None
    target_cost_share_pct: NonNegative | None


class MonotonicDiagnostics(Model):
    eligible_buckets: int = Field(ge=0)
    win_rate_non_decreasing: bool | None
    mfe_non_decreasing: bool | None
    mae_non_increasing: bool | None
    expectancy_non_decreasing: bool | None
    conclusion: Literal[
        "INSUFFICIENT_BUCKETS",
        "MONOTONIC_RELATIONSHIP_OBSERVED",
        "MONOTONIC_RELATIONSHIP_NOT_OBSERVED",
    ]


class SignalEvaluationReport(Model):
    mode: Literal["SIGNAL_EVALUATION_ONLY"] = "SIGNAL_EVALUATION_ONLY"
    live_trading_enabled: Literal[False] = False
    optimization_performed: Literal[False] = False
    thresholds_status: Literal["FIXED_NOT_OPTIMIZED"] = "FIXED_NOT_OPTIMIZED"
    symbol: Literal["EURUSD"] = "EURUSD"
    requested_start_utc: datetime
    requested_end_utc: datetime
    interval_seconds: int = Field(gt=0)
    max_horizon_tick_delay_seconds: Positive
    point: Positive
    slippage_points_per_side: NonNegative
    commission_points_round_trip: NonNegative
    minimum_target_cost_ratio: Positive
    executable_price_semantics: str
    session_definition: str
    recorded_observations: int = Field(ge=0)
    unblocked_observations: int = Field(ge=0)
    complete_outcome_observations: int = Field(ge=0)
    observations: tuple[SignalObservation, ...]
    score_buckets: tuple[ScoreBucket, ...]
    monotonic_diagnostics: MonotonicDiagnostics
