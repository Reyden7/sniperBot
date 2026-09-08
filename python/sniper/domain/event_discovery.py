"""Auditable Phase D.7 event-discovery report models."""

from datetime import datetime
from typing import Literal

from pydantic import Field

from sniper.config import Model
from sniper.domain.edge_validation import EdgeSimulationAssumptions

EpisodeMoment = Literal["FIRST_CROSSING", "PEAK_SCORE"]
DirectionMode = Literal["ORIGINAL", "INVERSE_DIAGNOSTIC"]


class HorizonMeans(Model):
    seconds_5: float | None
    seconds_10: float | None
    seconds_30: float | None
    seconds_60: float | None
    seconds_180: float | None


class DirectionDiagnostic(Model):
    moment: EpisodeMoment
    direction_mode: DirectionMode
    tradable: bool
    observations: int = Field(ge=0)
    usable: int = Field(ge=0)
    executable_accuracy_30s_pct: float | None
    gross_accuracy_30s_pct: float | None
    gross_expectancy_points: HorizonMeans
    executable_expectancy_points: HorizonMeans
    simulated_net_expectancy_30s_points: float | None
    average_mfe_30s_points: float | None
    average_mae_30s_points: float | None


class NumericFeatureDiagnostic(Model):
    feature: str
    transformation: str
    observations: int = Field(ge=0)
    spearman_vs_return: HorizonMeans
    spearman_vs_mfe_30s: float | None
    spearman_vs_mae_30s: float | None


class GroupDiagnostic(Model):
    analysis: str
    value: str
    observations: int = Field(ge=0)
    sample_sufficient: bool
    executable_accuracy_30s_pct: float | None
    gross_expectancy_points: HorizonMeans
    average_mfe_30s_points: float | None
    average_mae_30s_points: float | None


class EpisodeSummary(Model):
    episodes: int = Field(ge=0)
    completed_episodes: int = Field(ge=0)
    open_at_range_end: int = Field(ge=0)
    average_duration_seconds: float | None
    median_duration_seconds: float | None
    maximum_duration_seconds: float | None
    average_first_score: float | None
    average_peak_score: float | None


class CostFeasibility(Model):
    observations: int = Field(ge=0)
    average_total_round_trip_cost_points: float | None
    average_mfe_30s_points: float | None
    mfe_to_cost_ratio: float | None
    fraction_mfe_exceeding_cost_pct: float | None
    note: str


class EventDiscoveryPerformance(Model):
    elapsed_seconds: float = Field(ge=0)
    ticks_analyzed: int = Field(ge=0)
    active_seconds_scanned: int = Field(ge=0)
    signal_engine_evaluations: int = Field(ge=0)
    bar_feature_snapshots: int = Field(ge=0)
    ticks_per_second: float = Field(ge=0)
    scanned_seconds_per_second: float = Field(ge=0)
    peak_python_memory_mb: float = Field(ge=0)
    peak_loaded_tick_frame_mb: float = Field(ge=0)
    memory_measurement_note: str


class EventDiscoveryReport(Model):
    mode: Literal["EVENT_BASED_CANDIDATE_DISCOVERY"] = "EVENT_BASED_CANDIDATE_DISCOVERY"
    live_trading_enabled: Literal[False] = False
    optimization_performed: Literal[False] = False
    signal_engine_parameters_modified: Literal[False] = False
    strategy_direction_modified: Literal[False] = False
    symbol: Literal["EURUSD"] = "EURUSD"
    requested_start_utc: datetime
    requested_end_utc: datetime
    sampling_policy: str
    episode_policy: str
    first_crossing_policy: str
    peak_score_policy: str
    analysis_policy: str
    session_policy: str
    signal_engine_source_sha256: dict[str, str]
    evaluation_context_assumptions: dict[str, str]
    simulation: EdgeSimulationAssumptions
    event_parquet_path: str
    performance: EventDiscoveryPerformance
    episode_summary: EpisodeSummary
    direction_diagnostics: tuple[DirectionDiagnostic, ...]
    numeric_feature_diagnostics: tuple[NumericFeatureDiagnostic, ...]
    categorical_feature_diagnostics: tuple[GroupDiagnostic, ...]
    interaction_diagnostics: tuple[GroupDiagnostic, ...]
    cost_feasibility: CostFeasibility
    conclusion: str
    limitations: tuple[str, ...]
