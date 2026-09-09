"""Serializable Phase D.10 three-outcome research report."""

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from sniper.config import Model
from sniper.domain.edge_validation import EdgeSimulationAssumptions

D10Conclusion = Literal[
    "D10_RESEARCH_REJECTED",
    "D10_INSUFFICIENT_EVIDENCE",
    "D10_NEEDS_MORE_RESEARCH",
    "D10_FREEZE_CANDIDATE",
]


class D10Performance(Model):
    elapsed_seconds: float = Field(ge=0)
    source_ticks: int = Field(ge=0)
    observations: int = Field(ge=0)
    eligible_labeled_observations: int = Field(ge=0)
    peak_python_memory_mb: float = Field(ge=0)


class ResearchD10Report(Model):
    mode: Literal["SNIPER_D10_THREE_OUTCOME_RESEARCH"] = "SNIPER_D10_THREE_OUTCOME_RESEARCH"
    live_trading_enabled: Literal[False] = False
    phase_e_started: Literal[False] = False
    sizing_authority: Literal["RISK_ENGINE_ONLY"] = "RISK_ENGINE_ONLY"
    v3_status: Literal["V3_RESEARCH_REJECTED"] = "V3_RESEARCH_REJECTED"
    dataset_role: Literal["RESEARCH"] = "RESEARCH"
    holdout_state: Literal["SEALED"] = "SEALED"
    holdout_opened: Literal[False] = False
    holdout_evaluated: Literal[False] = False
    symbol: Literal["EURUSD"] = "EURUSD"
    requested_start_utc: datetime
    requested_end_utc: datetime
    protocol_path: str
    protocol_sha256: str
    freeze_manifest_path: str
    freeze_manifest_sha256: str
    source_hashes: dict[str, str]
    primary_configuration: dict[str, Any]
    model_policy: dict[str, Any]
    ev_policy: dict[str, Any]
    safety_buffer_policy: dict[str, Any]
    split_policy: dict[str, Any]
    simulation_profile: EdgeSimulationAssumptions
    folds: tuple[dict[str, Any], ...]
    outer_purge_audit: tuple[dict[str, Any], ...]
    nested_purge_audit: tuple[dict[str, Any], ...]
    fold_results: tuple[dict[str, Any], ...]
    model_development: dict[str, Any]
    internal_freeze_check: dict[str, Any]
    freeze_acceptance: dict[str, Any]
    performance: D10Performance
    dataset_path: str
    conclusion: D10Conclusion
    limitations: tuple[str, ...]
