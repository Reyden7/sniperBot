"""Serializable Phase D.9A methodology-audit report models."""

from datetime import datetime
from typing import Any, Literal

from sniper.config import Model
from sniper.domain.research_v3 import PurgedFoldAudit, V3Conclusion


class MethodologyAuditReport(Model):
    mode: Literal["SNIPER_D9A_METHODOLOGY_AUDIT"] = "SNIPER_D9A_METHODOLOGY_AUDIT"
    requested_start_utc: datetime
    requested_end_utc: datetime
    dataset_role: Literal["RESEARCH"] = "RESEARCH"
    holdout_state: Literal["SEALED"] = "SEALED"
    holdout_opened: Literal[False] = False
    holdout_evaluated: Literal[False] = False
    live_trading_enabled: Literal[False] = False
    phase_e_started: Literal[False] = False
    d10_executed: Literal[False] = False
    original_report_path: str
    original_report_sha256: str
    purged_report_path: str
    purge_rule: str
    purge_audit: tuple[PurgedFoldAudit, ...]
    anti_leakage_tests: dict[str, Any]
    protocol_invariants: dict[str, bool]
    comparison: dict[str, Any]
    outcome_diagnostic: dict[str, Any]
    original_verdict: V3Conclusion
    purged_verdict: V3Conclusion
    final_v3_verdict: V3Conclusion
    d10_specification_path: str
    limitations: tuple[str, ...]
