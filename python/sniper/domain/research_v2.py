"""Serializable Phase D.8 research report models."""

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from sniper.config import Model
from sniper.domain.edge_validation import EdgeSimulationAssumptions

V2Conclusion = Literal[
    "V2_RESEARCH_REJECTED",
    "V2_NEEDS_MORE_RESEARCH",
    "V2_VALIDATION_CANDIDATE",
]


class CostScenario(Model):
    name: Literal["OPTIMISTIC", "BASE", "STRESS"]
    slippage_points_per_side: float = Field(ge=0)
    commission_eur_per_lot_per_side: float = Field(ge=0)
    commission_minimum_eur_per_side: float = Field(ge=0)
    disclaimer: str


class TemporalFold(Model):
    fold: int = Field(gt=0)
    train_start_utc: datetime
    train_end_utc_exclusive: datetime
    validation_start_utc: datetime
    validation_end_utc_exclusive: datetime
    train_observations: int = Field(ge=0)
    validation_observations: int = Field(ge=0)
    preprocessing_fit_scope: Literal["TRAIN_ONLY"] = "TRAIN_ONLY"
    evaluation_role: Literal["MODEL_COMPARISON", "INTERNAL_FREEZE_CHECK"]


class HorizonModelReport(Model):
    horizon_seconds: int
    opportunity: dict[str, Any]
    direction: dict[str, Any]
    explainability: dict[str, Any]
    evaluation_windows: dict[str, Any]


class GateResult(Model):
    horizon_seconds: int
    scenario: Literal["OPTIMISTIC", "BASE", "STRESS"]
    move_to_cost_threshold: float
    evaluation_scope: Literal["MODEL_COMPARISON", "INTERNAL_FREEZE_CHECK", "ALL_VALIDATION"]
    candidates_count: int = Field(ge=0)
    gross_expectancy_points: float | None
    executable_expectancy_points: float | None
    simulated_net_expectancy_points: float | None
    average_mfe_points: float | None
    average_mae_points: float | None
    average_move_to_cost_ratio: float | None
    percentage_profitable: float | None


class V2Performance(Model):
    elapsed_seconds: float = Field(ge=0)
    ticks_analyzed: int = Field(ge=0)
    observations: int = Field(ge=0)
    peak_python_memory_mb: float = Field(ge=0)
    peak_loaded_tick_frame_mb: float = Field(ge=0)


class ResearchV2Report(Model):
    mode: Literal["SNIPER_V2_RESEARCH"] = "SNIPER_V2_RESEARCH"
    live_trading_enabled: Literal[False] = False
    phase_e_started: Literal[False] = False
    sizing_authority: Literal["RISK_ENGINE_ONLY"] = "RISK_ENGINE_ONLY"
    v1_status: Literal["RESEARCH_REJECTED"] = "RESEARCH_REJECTED"
    research_dataset_role: Literal["RESEARCH"] = "RESEARCH"
    holdout_opened: Literal[False] = False
    holdout_evaluated: Literal[False] = False
    symbol: Literal["EURUSD"] = "EURUSD"
    requested_start_utc: datetime
    requested_end_utc: datetime
    sampling_policy: str
    label_policy: str
    split_policy: str
    feature_policy: str
    model_policy: str
    execution_gate_policy: str
    freeze_criteria_predeclared: dict[str, Any]
    source_hashes: dict[str, str]
    simulation_profile: EdgeSimulationAssumptions
    cost_scenarios: tuple[CostScenario, ...]
    dataset_path: str
    folds: tuple[TemporalFold, ...]
    horizons: tuple[HorizonModelReport, ...]
    combined_validation: tuple[GateResult, ...]
    model_comparison_validation: tuple[GateResult, ...]
    internal_freeze_check: tuple[GateResult, ...]
    selected_base_gate: GateResult
    weekly_distribution: tuple[dict[str, Any], ...]
    session_distribution: tuple[dict[str, Any], ...]
    performance: V2Performance
    conclusion: V2Conclusion
    limitations: tuple[str, ...]
