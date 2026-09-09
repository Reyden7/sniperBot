"""Phase D.9A purged replay and locked V3 comparison."""

import hashlib
from pathlib import Path
from typing import Any, cast

from sniper.domain.methodology_audit import MethodologyAuditReport
from sniper.domain.research_v3 import CombinedBarrierReport, ResearchV3Report, SideOutcomeReport


def _primary_side(report: ResearchV3Report, side: str) -> SideOutcomeReport:
    results = report.long_results if side == "LONG" else report.short_results
    return next(item for item in results if item.configuration_id == "B02_PRIMARY")


def _primary_combined(report: ResearchV3Report) -> CombinedBarrierReport:
    return next(item for item in report.combined_results if item.configuration_id == "B02_PRIMARY")


def _pair(original: object, purged: object) -> dict[str, object]:
    delta = None
    if original is not None and purged is not None:
        delta = float(cast(float, purged)) - float(cast(float, original))
    return {"original": original, "purged": purged, "delta": delta}


def _outcomes(side: SideOutcomeReport) -> dict[str, object]:
    total = side.observations
    return {
        "observations": total,
        "TARGET_FIRST": side.target_before_stop,
        "STOP_FIRST": side.stop_before_target,
        "NEITHER": side.neither_before_timeout,
        "TARGET_FIRST_proportion": side.target_before_stop / total if total else None,
        "STOP_FIRST_proportion": side.stop_before_target / total if total else None,
        "NEITHER_proportion": side.neither_before_timeout / total if total else None,
    }


def build_methodology_audit(
    *,
    original: ResearchV3Report,
    purged: ResearchV3Report,
    original_report_path: Path,
    purged_report_path: Path,
    d10_specification_path: Path,
) -> MethodologyAuditReport:
    if purged.purge_policy != "LABEL_END_BEFORE_VALIDATION" or not purged.purge_audit:
        raise ValueError("D.9A requires a purged V3 replay")
    if not all(item.invariant_passed for item in purged.purge_audit):
        raise RuntimeError("At least one purged fold violates the label boundary")
    original_long, purged_long = _primary_side(original, "LONG"), _primary_side(purged, "LONG")
    original_short, purged_short = _primary_side(original, "SHORT"), _primary_side(purged, "SHORT")
    original_combined, purged_combined = _primary_combined(original), _primary_combined(purged)
    original_bytes = original_report_path.read_bytes()
    comparison: dict[str, Any] = {
        "B02_PRIMARY": {
            "LONG": {
                "logistic_pr_auc": _pair(
                    original_long.models["logistic_regression"]["pr_auc"],
                    purged_long.models["logistic_regression"]["pr_auc"],
                ),
                "logistic_brier": _pair(
                    original_long.models["logistic_regression"]["brier"],
                    purged_long.models["logistic_regression"]["brier"],
                ),
            },
            "SHORT": {
                "logistic_pr_auc": _pair(
                    original_short.models["logistic_regression"]["pr_auc"],
                    purged_short.models["logistic_regression"]["pr_auc"],
                ),
                "logistic_brier": _pair(
                    original_short.models["logistic_regression"]["brier"],
                    purged_short.models["logistic_regression"]["brier"],
                ),
            },
            "candidates": _pair(
                original_combined.candidates_count, purged_combined.candidates_count
            ),
            "executable_expectancy_points": _pair(
                original_combined.executable_expectancy_points,
                purged_combined.executable_expectancy_points,
            ),
            "base_net_expectancy_points": _pair(
                original_combined.base_net_expectancy_points,
                purged_combined.base_net_expectancy_points,
            ),
            "stress_net_expectancy_points": _pair(
                original_combined.stress_net_expectancy_points,
                purged_combined.stress_net_expectancy_points,
            ),
        }
    }
    same_barriers = original.barrier_configurations == purged.barrier_configurations
    same_features = original.feature_definitions == purged.feature_definitions
    same_costs = original.cost_scenarios == purged.cost_scenarios
    combined_target = purged_long.target_before_stop + purged_short.target_before_stop
    combined_stop = purged_long.stop_before_target + purged_short.stop_before_target
    combined_neither = purged_long.neither_before_timeout + purged_short.neither_before_timeout
    combined_total = purged_long.observations + purged_short.observations
    return MethodologyAuditReport(
        requested_start_utc=purged.requested_start_utc,
        requested_end_utc=purged.requested_end_utc,
        original_report_path=str(original_report_path.resolve()),
        original_report_sha256=hashlib.sha256(original_bytes).hexdigest(),
        purged_report_path=str(purged_report_path.resolve()),
        purge_rule="observation_timestamp + timeout_seconds <= validation_start",
        purge_audit=purged.purge_audit,
        anti_leakage_tests={
            "runtime_invariant": "max(label_end_time TRAIN) <= validation_start",
            "all_configuration_folds_passed": True,
            "configuration_fold_checks": len(purged.purge_audit),
            "terminal_quote_tolerance_seconds": 2,
            "tolerance_extends_label_end": False,
        },
        protocol_invariants={
            "same_features": same_features,
            "same_barriers": same_barriers,
            "same_primary_model": original.primary_model == purged.primary_model,
            "same_probability_threshold_and_meta_gate": (
                original.meta_gate_policy == purged.meta_gate_policy
            ),
            "same_cost_scenarios": same_costs,
            "same_simulation_profile": original.simulation_profile == purged.simulation_profile,
            "all_locked_elements_unchanged": all(
                (
                    same_features,
                    same_barriers,
                    original.primary_model == purged.primary_model,
                    original.meta_gate_policy == purged.meta_gate_policy,
                    same_costs,
                    original.simulation_profile == purged.simulation_profile,
                )
            ),
        },
        comparison=comparison,
        outcome_diagnostic={
            "LONG": _outcomes(purged_long),
            "SHORT": _outcomes(purged_short),
            "COMBINED_SIDE_EVENTS": {
                "observations": combined_total,
                "TARGET_FIRST": combined_target,
                "STOP_FIRST": combined_stop,
                "NEITHER": combined_neither,
                "TARGET_FIRST_proportion": combined_target / combined_total,
                "STOP_FIRST_proportion": combined_stop / combined_total,
                "NEITHER_proportion": combined_neither / combined_total,
            },
            "v3_formula": "p_target * target - (1 - p_target) * stop - costs",
            "methodological_note": (
                "The preregistered V3 MetaGate assimilates STOP_FIRST and NEITHER through "
                "the single (1-p_target) loss term. D.9A diagnoses but does not alter it."
            ),
        },
        original_verdict=original.conclusion,
        purged_verdict=purged.conclusion,
        final_v3_verdict=purged.conclusion,
        d10_specification_path=str(d10_specification_path.resolve()),
        limitations=(
            "D.9A is a methodology audit on RESEARCH, not new strategy selection.",
            "D.10 was specified but not executed.",
            "The HOLDOUT remained sealed and was neither opened nor evaluated.",
            "No Phase E, sizing, execution or live-trading implementation was started.",
        ),
    )


def render_methodology_audit(report: MethodologyAuditReport) -> str:
    primary = report.comparison["B02_PRIMARY"]

    def values(metric: dict[str, object]) -> str:
        return f"{metric['original']} | {metric['purged']} | {metric['delta']}"

    lines = [
        "# SNIPER Phase D.9A — Methodology Audit & Purged Replay",
        "",
        f"Verdict V3 final : **{report.final_v3_verdict}**",
        "",
        "## Purge par configuration et fold",
        "",
        "| Config | Fold | Timeout | TRAIN original | Purgées | TRAIN retenu | "
        "Max label end | Validation start | Test |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for item in report.purge_audit:
        lines.append(
            f"| {item.configuration_id} | {item.fold} | {item.timeout_seconds}s | "
            f"{item.original_train_observations} | {item.purged_observations} | "
            f"{item.retained_train_observations} | {item.max_label_end_time_train_utc} | "
            f"{item.validation_start_utc} | {'PASS' if item.invariant_passed else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## V3_ORIGINAL vs V3_PURGED — B02_PRIMARY",
            "",
            "| Mesure | Original | Purged | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for side in ("LONG", "SHORT"):
        for name in ("logistic_pr_auc", "logistic_brier"):
            metric = primary[side][name]
            lines.append(f"| {side} {name} | {values(metric)} |")
    for name in (
        "candidates",
        "executable_expectancy_points",
        "base_net_expectancy_points",
        "stress_net_expectancy_points",
    ):
        lines.append(f"| {name} | {values(primary[name])} |")
    lines.extend(
        [
            "",
            "## Diagnostic des trois issues — B02_PRIMARY purgé",
            "",
            "| Scope | Observations | TARGET_FIRST | STOP_FIRST | NEITHER |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for scope in ("LONG", "SHORT", "COMBINED_SIDE_EVENTS"):
        item = report.outcome_diagnostic[scope]
        lines.append(
            f"| {scope} | {item['observations']} | "
            f"{item['TARGET_FIRST']} ({item['TARGET_FIRST_proportion']:.6f}) | "
            f"{item['STOP_FIRST']} ({item['STOP_FIRST_proportion']:.6f}) | "
            f"{item['NEITHER']} ({item['NEITHER_proportion']:.6f}) |"
        )
    lines.extend(
        [
            "",
            "La formule V3 préenregistrée `p_target * target - (1-p_target) * stop - costs` "
            "assimile STOP_FIRST et NEITHER. D.9A ne la modifie pas.",
            "",
            "## Garde-fous",
            "",
            "- Dataset : RESEARCH uniquement",
            "- HOLDOUT : SEALED, non ouvert, non évalué",
            "- Paramètres V3 : inchangés, seule la frontière TRAIN est purgée",
            "- D.10 : spécifiée, non exécutée",
            "- Phase E et trading live : non commencés",
            "",
        ]
    )
    return "\n".join(lines)
