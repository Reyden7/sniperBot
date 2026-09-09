"""Serializable Phase D.11 predictive-ranking audit report."""

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from sniper.config import Model

D11Conclusion = Literal[
    "D11_NO_RANKING_SIGNAL",
    "D11_UNSTABLE_RANKING_SIGNAL",
    "D11_RANKING_SIGNAL_WORTH_FURTHER_RESEARCH",
]


class D11Performance(Model):
    elapsed_seconds: float = Field(ge=0)
    source_ticks: int = Field(ge=0)
    source_observations: int = Field(ge=0)
    ranking_side_observations: int = Field(ge=0)
    d10_candidates_reproduced: int = Field(ge=0)
    peak_python_memory_mb: float = Field(ge=0)


class ResearchD11Report(Model):
    mode: Literal["SNIPER_D11_PREDICTIVE_RANKING_AUDIT"] = "SNIPER_D11_PREDICTIVE_RANKING_AUDIT"
    diagnostic_only: Literal[True] = True
    live_trading_enabled: Literal[False] = False
    phase_e_started: Literal[False] = False
    d12_created: Literal[False] = False
    sizing_authority: Literal["NONE_DIAGNOSTIC_ONLY"] = "NONE_DIAGNOSTIC_ONLY"
    d10_official_verdict: Literal["D10_INSUFFICIENT_EVIDENCE"] = "D10_INSUFFICIENT_EVIDENCE"
    dataset_role: Literal["RESEARCH"] = "RESEARCH"
    holdout_state: Literal["SEALED"] = "SEALED"
    holdout_opened: Literal[False] = False
    holdout_evaluated: Literal[False] = False
    symbol: Literal["EURUSD"] = "EURUSD"
    requested_start_utc: datetime
    requested_end_utc: datetime
    hashes_verified_before_reproduction: dict[str, Any]
    source_hashes: dict[str, str]
    locked_elements_unchanged: dict[str, bool]
    ev_distribution: tuple[dict[str, Any], ...]
    ranking_correlations: tuple[dict[str, Any], ...]
    ranking_curves: tuple[dict[str, Any], ...]
    economic_calibration: tuple[dict[str, Any], ...]
    fold_stability: dict[str, Any]
    regime_shift: dict[str, Any]
    safety_buffer_audit: tuple[dict[str, Any], ...]
    candidate_anatomy: tuple[dict[str, Any], ...]
    candidate_anatomy_summary: dict[str, Any]
    verdict_evidence: dict[str, Any]
    artifacts: dict[str, str]
    performance: D11Performance
    conclusion: D11Conclusion
    limitations: tuple[str, ...]
