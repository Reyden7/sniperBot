"""Serializable Phase D.12 feature and regime discovery report."""

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from sniper.config import Model

D12Conclusion = Literal[
    "D12_NO_USEFUL_FEATURE_SIGNAL",
    "D12_UNSTABLE_FEATURE_SIGNAL",
    "D12_FEATURES_WORTH_V4",
]


class D12Performance(Model):
    elapsed_seconds: float = Field(ge=0)
    ticks_analyzed: int = Field(ge=0)
    detector_seconds: int = Field(ge=0)
    events_produced: int = Field(ge=0)
    complete_events: int = Field(ge=0)
    ticks_per_second: float = Field(ge=0)
    peak_python_memory_mb: float = Field(ge=0)
    peak_loaded_tick_frame_mb: float = Field(ge=0)


class ResearchD12Report(Model):
    mode: Literal["SNIPER_D12_FEATURE_REGIME_DISCOVERY"] = "SNIPER_D12_FEATURE_REGIME_DISCOVERY"
    discovery_only: Literal[True] = True
    creates_v4: Literal[False] = False
    buy_sell_outputs_created: Literal[False] = False
    trade_skip_model_created: Literal[False] = False
    live_trading_enabled: Literal[False] = False
    phase_e_started: Literal[False] = False
    sizing_authority: Literal["RISK_ENGINE_ONLY_NOT_CALLED"] = "RISK_ENGINE_ONLY_NOT_CALLED"
    martingale_enabled: Literal[False] = False
    grid_enabled: Literal[False] = False
    averaging_down_enabled: Literal[False] = False
    dataset_role: Literal["RESEARCH"] = "RESEARCH"
    holdout_state: Literal["SEALED"] = "SEALED"
    holdout_opened: Literal[False] = False
    holdout_evaluated: Literal[False] = False
    symbol: Literal["EURUSD"] = "EURUSD"
    requested_start_utc: datetime
    requested_end_utc: datetime
    protocol_sha256: str
    source_hashes: dict[str, str]
    causal_feature_policy: dict[str, Any]
    sampling_summary: tuple[dict[str, Any], ...]
    label_summary: tuple[dict[str, Any], ...]
    feature_matrix_summary: dict[str, Any]
    feature_status_summary: dict[str, Any]
    family_summary: tuple[dict[str, Any], ...]
    multivariate_diagnostics: dict[str, Any]
    early_day_hypothesis: dict[str, Any]
    multiple_testing: dict[str, Any]
    verdict_evidence: dict[str, Any]
    artifacts: dict[str, str]
    performance: D12Performance
    conclusion: D12Conclusion
    limitations: tuple[str, ...]
