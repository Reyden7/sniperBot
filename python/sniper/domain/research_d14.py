"""Serializable Phase D.14 economic bottleneck diagnostic report."""

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from sniper.config import Model

D14Conclusion = Literal[
    "D14_DIRECTION_ECONOMICALLY_TOO_WEAK",
    "D14_COST_DOMINATED_SIGNAL",
    "D14_DIRECTION_CALIBRATION_WORTH_RESEARCH",
    "D14_REGIME_GATE_INSUFFICIENT",
]


class D14Performance(Model):
    elapsed_seconds: float = Field(ge=0)
    oof_observations: int = Field(ge=0)
    peak_python_memory_mb: float = Field(ge=0)


class ResearchD14Report(Model):
    mode: Literal["SNIPER_D14_DIAGNOSTIC_ONLY"] = "SNIPER_D14_DIAGNOSTIC_ONLY"
    diagnostic_only: Literal[True] = True
    models_retrained: Literal[False] = False
    d13_parameters_modified: Literal[False] = False
    final_independent_validation: Literal[False] = False
    holdout_state: Literal["SEALED"] = "SEALED"
    holdout_opened: Literal[False] = False
    holdout_evaluated: Literal[False] = False
    phase_e_started: Literal[False] = False
    sizing_enabled: Literal[False] = False
    live_trading_enabled: Literal[False] = False
    symbol: Literal["EURUSD"] = "EURUSD"
    requested_start_utc: datetime
    requested_end_utc: datetime
    protocol_path: str
    protocol_sha256: str
    source_hashes: dict[str, str]
    hash_verification: dict[str, Any]
    accounting_audit: dict[str, Any]
    edge_decomposition: dict[str, Any]
    direction_calibration: dict[str, Any]
    oracle_decomposition: dict[str, Any]
    required_directional_skill: dict[str, Any]
    cost_frontier: dict[str, Any]
    gross_edge_first: dict[str, Any]
    hgb_diagnostic: dict[str, Any]
    verdict_checks: dict[str, Any]
    quality_assurance: dict[str, Any]
    artifacts: dict[str, str]
    performance: D14Performance
    conclusion: D14Conclusion
    limitations: tuple[str, ...]
