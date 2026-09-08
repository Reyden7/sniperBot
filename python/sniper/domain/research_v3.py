"""Serializable Phase D.9 triple-barrier research report models."""

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from sniper.config import Model
from sniper.domain.edge_validation import EdgeSimulationAssumptions
from sniper.domain.research_v2 import CostScenario, TemporalFold

V3Conclusion = Literal[
    "V3_RESEARCH_REJECTED",
    "V3_NEEDS_MORE_RESEARCH",
    "V3_FREEZE_CANDIDATE",
]


class BarrierConfiguration(Model):
    id: Literal["B01", "B02_PRIMARY", "B03", "B04"]
    stop_atr: float = Field(gt=0)
    target_atr: float = Field(gt=0)
    timeout_seconds: Literal[180, 300, 600, 900]
    minimum_target_to_base_cost: float = Field(gt=0)


class SideOutcomeReport(Model):
    configuration_id: str
    side: Literal["LONG", "SHORT"]
    observations: int = Field(ge=0)
    target_before_stop: int = Field(ge=0)
    stop_before_target: int = Field(ge=0)
    neither_before_timeout: int = Field(ge=0)
    target_before_stop_rate: float = Field(ge=0, le=1)
    average_mfe_points: float
    average_mae_points: float
    average_mfe_mae_quality: float
    average_time_to_target_seconds: float | None
    average_time_to_stop_seconds: float | None
    models: dict[str, Any]
    feature_importance: dict[str, Any]


class CombinedBarrierReport(Model):
    configuration_id: str
    timeout_seconds: int
    candidates_count: int = Field(ge=0)
    long_candidates: int = Field(ge=0)
    short_candidates: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    target_before_stop_rate: float | None
    gross_expectancy_points: float | None
    executable_expectancy_points: float | None
    optimistic_net_expectancy_points: float | None
    base_net_expectancy_points: float | None
    stress_net_expectancy_points: float | None
    profit_factor_base: float | None
    average_mfe_points: float | None
    average_mae_points: float | None
    average_mfe_mae_quality: float | None
    weekly_distribution: tuple[dict[str, Any], ...]
    session_distribution: tuple[dict[str, Any], ...]


class V3Performance(Model):
    elapsed_seconds: float = Field(ge=0)
    ticks_analyzed: int = Field(ge=0)
    observations: int = Field(ge=0)
    labeled_side_events: int = Field(ge=0)
    peak_python_memory_mb: float = Field(ge=0)
    peak_loaded_tick_frame_mb: float = Field(ge=0)


class PurgedFoldAudit(Model):
    configuration_id: str
    fold: int = Field(ge=1, le=4)
    timeout_seconds: int = Field(gt=0)
    validation_start_utc: datetime
    original_train_observations: int = Field(ge=0)
    purged_observations: int = Field(ge=0)
    retained_train_observations: int = Field(ge=0)
    max_label_end_time_train_utc: datetime | None
    invariant_passed: bool


class ResearchV3Report(Model):
    mode: Literal["SNIPER_V3_DIRECTION_RESEARCH"] = "SNIPER_V3_DIRECTION_RESEARCH"
    live_trading_enabled: Literal[False] = False
    phase_e_started: Literal[False] = False
    sizing_authority: Literal["RISK_ENGINE_ONLY"] = "RISK_ENGINE_ONLY"
    v1_status: Literal["RESEARCH_REJECTED"] = "RESEARCH_REJECTED"
    v2_status: Literal["V2_RESEARCH_REJECTED"] = "V2_RESEARCH_REJECTED"
    dataset_role: Literal["RESEARCH"] = "RESEARCH"
    holdout_state: Literal["SEALED"] = "SEALED"
    holdout_opened: Literal[False] = False
    holdout_evaluated: Literal[False] = False
    symbol: Literal["EURUSD"] = "EURUSD"
    requested_start_utc: datetime
    requested_end_utc: datetime
    sampling_policy: str
    label_policy: str
    split_policy: str
    purge_policy: Literal["NONE_ORIGINAL", "LABEL_END_BEFORE_VALIDATION"] = "NONE_ORIGINAL"
    purge_audit: tuple[PurgedFoldAudit, ...] = ()
    feature_definitions: dict[str, str]
    barrier_configurations: tuple[BarrierConfiguration, ...]
    primary_configuration_id: Literal["B02_PRIMARY"] = "B02_PRIMARY"
    primary_model: Literal["REGULARIZED_LOGISTIC_REGRESSION"] = "REGULARIZED_LOGISTIC_REGRESSION"
    meta_gate_policy: str
    cost_scenarios: tuple[CostScenario, ...]
    simulation_profile: EdgeSimulationAssumptions
    folds: tuple[TemporalFold, ...]
    opportunity_quality: dict[str, Any]
    long_results: tuple[SideOutcomeReport, ...]
    short_results: tuple[SideOutcomeReport, ...]
    combined_results: tuple[CombinedBarrierReport, ...]
    registered_hypotheses: tuple[dict[str, Any], ...]
    freeze_criteria_predeclared: dict[str, Any]
    performance: V3Performance
    dataset_path: str
    conclusion: V3Conclusion
    limitations: tuple[str, ...]
