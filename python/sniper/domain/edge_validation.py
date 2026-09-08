"""Auditable Phase D.6 aggregate models; no strategy calibration or live trading."""

from datetime import datetime
from typing import Literal

from pydantic import Field

from sniper.backtest.execution_model import SimulatedBrokerProfile
from sniper.config import Model

EdgeDiagnosis = Literal["INSUFFICIENT_DATA", "EDGE_NOT_DETECTED", "EDGE_WEAK", "EDGE_CANDIDATE"]


class ConfidenceInterval(Model):
    estimate: float
    lower_95: float
    upper_95: float


class BootstrapConfidenceIntervals(Model):
    method: Literal["UTC_DAY_BLOCK_BOOTSTRAP"] = "UTC_DAY_BLOCK_BOOTSTRAP"
    scope: Literal["UNBLOCKED_BUY_SELL_CANDIDATES"] = "UNBLOCKED_BUY_SELL_CANDIDATES"
    seed: int
    resamples: int = Field(gt=0)
    active_days: int = Field(gt=0)
    directional_accuracy_pct: ConfidenceInterval
    net_expectancy_points: ConfidenceInterval
    mfe_30s_points: ConfidenceInterval
    mae_30s_points: ConfidenceInterval


class AggregateSlice(Model):
    key: str
    observations: int = Field(ge=0)
    blocked: int = Field(ge=0)
    usable: int = Field(ge=0)
    buy: int = Field(ge=0)
    sell: int = Field(ge=0)
    wait: int = Field(ge=0)
    sample_sufficient: bool
    directional_accuracy_pct: float | None
    average_return_after_5s_points: float | None
    average_return_after_10s_points: float | None
    average_return_after_30s_points: float | None
    average_return_after_60s_points: float | None
    average_return_after_180s_points: float | None
    expectancy_before_costs_points: float | None
    expectancy_after_spread_points: float | None
    expectancy_after_slippage_points: float | None
    expectancy_after_commission_points: float | None
    expectancy_after_commission_eur: float | None
    average_mfe_30s_points: float | None
    average_mae_30s_points: float | None
    average_mfe_30s_eur: float | None
    average_mae_30s_eur: float | None
    target_hit_first: int = Field(ge=0)
    stop_hit_first: int = Field(ge=0)
    neither_hit: int = Field(ge=0)


class SpearmanDiagnostics(Model):
    observations: int = Field(ge=0)
    score_vs_directional_accuracy: float | None
    score_vs_net_expectancy: float | None
    score_vs_mfe: float | None
    score_vs_mae: float | None
    desired_relationships_met: int = Field(ge=0, le=4)


class EdgeSimulationAssumptions(Model):
    broker_profile: SimulatedBrokerProfile
    evaluation_volume_lots: float
    slippage_points_per_side: float
    commission_eur_per_lot_per_side: float
    commission_minimum_eur_per_side: float
    disclaimer: str


class EdgePerformance(Model):
    elapsed_seconds: float = Field(ge=0)
    ticks_analyzed: int = Field(ge=0)
    observations_produced: int = Field(ge=0)
    samples_skipped_no_feature_base: int = Field(ge=0)
    ticks_per_second: float = Field(ge=0)
    peak_python_memory_mb: float = Field(ge=0)
    peak_loaded_tick_frame_mb: float = Field(ge=0)
    memory_measurement_note: str


class DiagnosticEvidence(Model):
    minimum_usable_observations: int
    minimum_candidate_observations: int
    minimum_active_weeks: int
    usable_observations: int
    candidate_observations: int
    active_weeks: int
    positive_net_weeks: int
    positive_net_week_fraction: float | None
    net_expectancy_positive: bool
    net_expectancy_ci_lower_positive: bool
    stable_across_weeks: bool
    resists_simulated_costs: bool
    score_quality_coherent: bool


class EdgeValidationReport(Model):
    mode: Literal["EDGE_VALIDATION_ONLY"] = "EDGE_VALIDATION_ONLY"
    live_trading_enabled: Literal[False] = False
    optimization_performed: Literal[False] = False
    signal_engine_parameters_modified: Literal[False] = False
    symbol: Literal["EURUSD"] = "EURUSD"
    requested_start_utc: datetime
    requested_end_utc: datetime
    sampling_policy: str
    session_policy: str
    outcome_horizon_seconds: int = 180
    stop_target_horizon_seconds: int = 180
    max_horizon_tick_delay_seconds: float
    signal_engine_source_sha256: dict[str, str]
    evaluation_context_assumptions: dict[str, str]
    simulation: EdgeSimulationAssumptions
    observation_parquet_root: str
    performance: EdgePerformance
    score_distribution: dict[str, int]
    overall: AggregateSlice
    candidate_overall: AggregateSlice
    score_buckets: tuple[AggregateSlice, ...]
    sessions: tuple[AggregateSlice, ...]
    candidate_sessions: tuple[AggregateSlice, ...]
    weeks: tuple[AggregateSlice, ...]
    candidate_weeks: tuple[AggregateSlice, ...]
    weekdays: tuple[AggregateSlice, ...]
    confidence_intervals: BootstrapConfidenceIntervals | None
    spearman: SpearmanDiagnostics
    diagnostic_evidence: DiagnosticEvidence
    diagnosis: EdgeDiagnosis
    limitations: tuple[str, ...]
