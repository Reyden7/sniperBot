"""Serializable Phase D.13 hierarchical V4 research report."""

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from sniper.config import Model

D13Conclusion = Literal["V4_NOT_READY_TO_FREEZE", "V4_FREEZE_READY"]


class D13Performance(Model):
    elapsed_seconds: float = Field(ge=0)
    source_observations: int = Field(ge=0)
    primary_universe_observations: int = Field(ge=0)
    outer_oof_observations: int = Field(ge=0)
    peak_python_memory_mb: float = Field(ge=0)


class ResearchD13Report(Model):
    mode: Literal["SNIPER_V4_HIERARCHICAL_RESEARCH"] = "SNIPER_V4_HIERARCHICAL_RESEARCH"
    research_performance_only: Literal[True] = True
    final_independent_validation: Literal[False] = False
    actionable_buy_sell_outputs_created: Literal[False] = False
    live_trading_enabled: Literal[False] = False
    phase_e_started: Literal[False] = False
    sizing_authority: Literal["RISK_ENGINE_ONLY_NOT_CALLED"] = "RISK_ENGINE_ONLY_NOT_CALLED"
    martingale_enabled: Literal[False] = False
    grid_enabled: Literal[False] = False
    averaging_down_enabled: Literal[False] = False
    dataset_role: Literal["RESEARCH_DEVELOPMENT"] = "RESEARCH_DEVELOPMENT"
    holdout_state: Literal["SEALED"] = "SEALED"
    holdout_opened: Literal[False] = False
    holdout_evaluated: Literal[False] = False
    symbol: Literal["EURUSD"] = "EURUSD"
    requested_start_utc: datetime
    requested_end_utc: datetime
    protocol_path: str
    protocol_sha256: str
    source_hashes: dict[str, str]
    architecture: dict[str, Any]
    event_universe: dict[str, Any]
    feature_reduction: dict[str, Any]
    model_policy: dict[str, Any]
    preprocessing_audit: dict[str, Any]
    probability_calibration: dict[str, Any]
    outer_purge_audit: tuple[dict[str, Any], ...]
    nested_purge_audit: tuple[dict[str, Any], ...]
    outer_fold_results: tuple[dict[str, Any], ...]
    nested_research_metrics: dict[str, Any]
    selected_configuration: dict[str, Any]
    model_diagnostics: dict[str, Any]
    baselines: dict[str, Any]
    research_economics: dict[str, Any]
    internal_research_check: dict[str, Any]
    acceptance: dict[str, Any]
    freeze_manifest_path: str
    freeze_manifest_sha256: str
    freeze_manifest: dict[str, Any]
    future_forward_dataset: dict[str, Any]
    artifacts: dict[str, str]
    performance: D13Performance
    conclusion: D13Conclusion
    limitations: tuple[str, ...]
