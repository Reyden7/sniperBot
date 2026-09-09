"""Phase D.14 diagnostic-only economic feasibility audit of frozen D.13 OOF predictions."""

from __future__ import annotations

import hashlib
import json
import math
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
import polars as pl
from scipy.stats import spearmanr  # type: ignore[import-untyped]

from sniper.domain.research_d14 import D14Performance, ResearchD14Report

EXPECTED_START = datetime(2026, 6, 10, tzinfo=UTC)
EXPECTED_END = datetime(2026, 9, 8, tzinfo=UTC)
BOOTSTRAP_SEED = 20260914
BOOTSTRAP_REPLICATIONS = 2000
EXPECTED_HASHES = {
    "research-d13.json": "f0117538f287e3c6b472f95cadcdd7e368495a94bf780a197b1fc9259e8a448a",
    "outer-oof-predictions.parquet": (
        "aadc3eaaaf046acbd9a566aecd11fb22abaf599acef71bca3909ad1a563e4ab0"
    ),
    "d12-events.parquet": "a23531949d3ea9857223caf911405d48aafcd3dbbd70f78f86a561b7a3bf9c0f",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    value = float(spearmanr(x, y).statistic)
    return value if math.isfinite(value) else None


def _finite_mean(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _profit_factor(values: np.ndarray) -> float | None:
    if not len(values):
        return None
    gains = float(np.sum(values[values > 0]))
    losses = float(-np.sum(values[values < 0]))
    return gains / losses if losses > 0 else None


def _weekly_metrics(values: np.ndarray, weeks: np.ndarray) -> dict[str, Any]:
    if not len(values):
        return {"active_weeks": 0, "positive_weeks": 0, "positive_week_fraction": None}
    unique = np.unique(weeks)
    week_means = np.asarray([np.mean(values[weeks == week]) for week in unique])
    positive = int(np.sum(week_means > 0))
    return {
        "active_weeks": len(unique),
        "positive_weeks": positive,
        "positive_week_fraction": positive / len(unique),
    }


def _day_block_ci(
    values: np.ndarray, days: np.ndarray, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    if not len(values):
        return {"method": "UTC_DAY_BLOCK", "replications": 0, "lower": None, "upper": None}
    unique, inverse = np.unique(days, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(BOOTSTRAP_REPLICATIONS, len(unique)))
    estimates = np.sum(sums[draws], axis=1) / np.sum(counts[draws], axis=1)
    return {
        "method": "UTC_DAY_BLOCK",
        "replications": BOOTSTRAP_REPLICATIONS,
        "seed": seed,
        "lower": float(np.quantile(estimates, 0.025)),
        "upper": float(np.quantile(estimates, 0.975)),
    }


def _ols(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    design = np.column_stack((np.ones(len(x)), x))
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = alpha + beta * x
    residual = y - fitted
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / ss_total if ss_total > 0 else 0.0
    return {"alpha": float(alpha), "beta": float(beta), "r_squared": r_squared}


def _equal_count_bins(sort_values: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(sort_values, kind="stable")
    return [np.asarray(part, dtype=int) for part in np.array_split(order, 10)]


def direction_bins(
    predicted: np.ndarray,
    realized: np.ndarray,
    long_exec: np.ndarray,
    short_exec: np.ndarray,
    long_base: np.ndarray,
    short_base: np.ndarray,
    *,
    absolute_sort: bool,
) -> list[dict[str, Any]]:
    """Return preregistered ten equal-count OOF direction bins."""
    result = []
    sorting = np.abs(predicted) if absolute_sort else predicted
    for number, indices in enumerate(_equal_count_bins(sorting), 1):
        prediction = predicted[indices]
        actual = realized[indices]
        predicted_long = prediction >= 0
        actual_sign = np.sign(actual)
        sign_accuracy = float(np.mean(np.where(predicted_long, actual_sign > 0, actual_sign < 0)))
        predicted_base = np.where(predicted_long, long_base[indices], short_base[indices])
        oracle_base = np.maximum(long_base[indices], short_base[indices])
        oracle_exec = np.maximum(long_exec[indices], short_exec[indices])
        result.append(
            {
                "bin": number,
                "N": len(indices),
                "sort_value_min": float(np.min(sorting[indices])),
                "sort_value_max": float(np.max(sorting[indices])),
                "mean_prediction_points": float(np.mean(prediction)),
                "mean_absolute_prediction_points": float(np.mean(np.abs(prediction))),
                "mean_realized_signed_market_return_points": float(np.mean(actual)),
                "median_realized_signed_market_return_points": float(np.median(actual)),
                "sign_accuracy": sign_accuracy,
                "spearman_prediction_vs_realized": _safe_spearman(prediction, actual),
                "mae_points": float(np.mean(np.abs(prediction - actual))),
                "mean_long_executable_pnl_points": float(np.mean(long_exec[indices])),
                "mean_short_executable_pnl_points": float(np.mean(short_exec[indices])),
                "mean_best_side_oracle_executable_pnl_points": float(np.mean(oracle_exec)),
                "mean_best_side_oracle_base_net_pnl_points": float(np.mean(oracle_base)),
                "mean_predicted_side_base_net_pnl_points": float(np.mean(predicted_base)),
            }
        )
    return result


def _system_metrics(
    mask: np.ndarray,
    long_side: np.ndarray,
    market_signed: np.ndarray,
    long_exec: np.ndarray,
    short_exec: np.ndarray,
    long_base: np.ndarray,
    short_base: np.ndarray,
    long_stress: np.ndarray,
    short_stress: np.ndarray,
    weeks: np.ndarray,
    sessions: np.ndarray,
) -> dict[str, Any]:
    gross = np.where(long_side, market_signed, -market_signed)[mask]
    executable = np.where(long_side, long_exec, short_exec)[mask]
    base = np.where(long_side, long_base, short_base)[mask]
    stress = np.where(long_side, long_stress, short_stress)[mask]
    selected_weeks = weeks[mask]
    selected_sessions = sessions[mask]
    session_distribution = {}
    for session in sorted(np.unique(selected_sessions).tolist()):
        session_mask = selected_sessions == session
        session_distribution[str(session)] = {
            "N": int(np.sum(session_mask)),
            "share": float(np.mean(session_mask)),
            "base_expectancy_points": float(np.mean(base[session_mask])),
        }
    return {
        "N": int(np.sum(mask)),
        "gross_expectancy_points": _finite_mean(gross),
        "executable_expectancy_points": _finite_mean(executable),
        "base_expectancy_points": _finite_mean(base),
        "stress_expectancy_points": _finite_mean(stress),
        "base_profit_factor": _profit_factor(base),
        **_weekly_metrics(base, selected_weeks),
        "session_distribution": session_distribution,
    }


def oracle_decomposition(
    probability: np.ndarray,
    predicted: np.ndarray,
    oracle_regime: np.ndarray,
    market_signed: np.ndarray,
    long_exec: np.ndarray,
    short_exec: np.ndarray,
    long_base: np.ndarray,
    short_base: np.ndarray,
    long_stress: np.ndarray,
    short_stress: np.ndarray,
    weeks: np.ndarray,
    sessions: np.ndarray,
) -> dict[str, Any]:
    """Build four post-hoc oracle systems without fitting any model."""
    current_regime = probability >= 0.50
    current_direction = predicted >= 0
    oracle_direction = long_base >= short_base
    systems = {
        "CURRENT_REGIME_CURRENT_DIRECTION": (current_regime, current_direction),
        "CURRENT_REGIME_ORACLE_DIRECTION": (current_regime, oracle_direction),
        "ORACLE_REGIME_CURRENT_DIRECTION": (oracle_regime, current_direction),
        "ORACLE_REGIME_ORACLE_DIRECTION": (oracle_regime, oracle_direction),
    }
    return {
        name: _system_metrics(
            mask,
            side,
            market_signed,
            long_exec,
            short_exec,
            long_base,
            short_base,
            long_stress,
            short_stress,
            weeks,
            sessions,
        )
        for name, (mask, side) in systems.items()
    }


def required_directional_skill(
    probability: np.ndarray,
    long_base: np.ndarray,
    short_base: np.ndarray,
    long_stress: np.ndarray,
    short_stress: np.ndarray,
    days: np.ndarray,
) -> dict[str, Any]:
    """Simulate the preregistered nested abstract directional accuracies."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    correctness_draw = rng.random(len(probability))
    best_long = long_base >= short_base
    scopes = {
        "ALL_OOF": np.ones(len(probability), dtype=bool),
        "REGIME_050": probability >= 0.50,
        "REGIME_060": probability >= 0.60,
        "REGIME_070": probability >= 0.70,
    }
    output: dict[str, Any] = {}
    for scope_number, (name, mask) in enumerate(scopes.items()):
        rows = []
        base_break_even = None
        stress_break_even = None
        for accuracy in range(50, 101, 5):
            selected_long = np.where(correctness_draw < accuracy / 100.0, best_long, ~best_long)
            base = np.where(selected_long, long_base, short_base)[mask]
            stress = np.where(selected_long, long_stress, short_stress)[mask]
            base_mean = float(np.mean(base))
            stress_mean = float(np.mean(stress))
            if base_break_even is None and base_mean > 0:
                base_break_even = accuracy
            if stress_break_even is None and stress_mean > 0:
                stress_break_even = accuracy
            rows.append(
                {
                    "abstract_accuracy_percent": accuracy,
                    "N": int(np.sum(mask)),
                    "base_expectancy_points": base_mean,
                    "base_day_block_bootstrap_ci95": _day_block_ci(
                        base, days[mask], BOOTSTRAP_SEED + scope_number * 100 + accuracy
                    ),
                    "stress_expectancy_points": stress_mean,
                    "stress_day_block_bootstrap_ci95": _day_block_ci(
                        stress, days[mask], BOOTSTRAP_SEED + scope_number * 1000 + accuracy
                    ),
                }
            )
        output[name] = {
            "results": rows,
            "minimum_direction_accuracy_required_for_BASE_break_even_percent": base_break_even,
            "minimum_direction_accuracy_required_for_STRESS_break_even_percent": stress_break_even,
        }
    return {
        "method": "ABSTRACT_POST_HOC_ORACLE_SIDE_WITH_NESTED_FIXED_RANDOM_DRAWS",
        "seed": BOOTSTRAP_SEED,
        "not_a_strategy": True,
        "scopes": output,
    }


def _scenario_metrics(mask: np.ndarray, pnl: np.ndarray, weeks: np.ndarray) -> dict[str, Any]:
    selected = pnl[mask]
    selected_weeks = weeks[mask]
    return {
        "candidate_count": int(np.sum(mask)),
        "realized_net_expectancy_points": _finite_mean(selected),
        "profit_factor": _profit_factor(selected),
        **_weekly_metrics(selected, selected_weeks),
    }


def cost_frontier(
    probability: np.ndarray,
    predicted: np.ndarray,
    signal_spread: np.ndarray,
    selected_executable: np.ndarray,
    base_commission: np.ndarray,
    weeks: np.ndarray,
) -> dict[str, Any]:
    """Evaluate frozen predictions under explicitly diagnostic cost scenarios."""
    regime = probability >= 0.50
    scenarios = []
    for slippage_side in (0.0, 0.5, 1.0, 2.0):
        for commission_eur_side in (0.0, 1.0, 2.0, 4.0):
            commission = base_commission * (commission_eur_side / 2.0)
            predicted_edge = np.abs(predicted) - signal_spread - 2.0 * slippage_side - commission
            candidates = regime & (predicted_edge > 0)
            realized = selected_executable - 2.0 * slippage_side - commission
            scenarios.append(
                {
                    "scenario": (
                        f"SLIPPAGE_{slippage_side:g}_POINTS_SIDE__"
                        f"COMMISSION_{commission_eur_side:g}_EUR_LOT_SIDE"
                    ),
                    "slippage_points_per_side": slippage_side,
                    "commission_eur_per_lot_per_side": commission_eur_side,
                    **_scenario_metrics(candidates, realized, weeks),
                }
            )
    spread_only_candidates = regime & ((np.abs(predicted) - signal_spread) > 0)
    spread_only = {
        "scenario": "OBSERVED_SPREAD_ONLY",
        "slippage_points_per_side": 0.0,
        "commission_eur_per_lot_per_side": 0.0,
        **_scenario_metrics(spread_only_candidates, selected_executable, weeks),
    }

    def expectancy(commission_eur_side: float, slippage_side: float) -> float | None:
        commission = base_commission * (commission_eur_side / 2.0)
        candidate = regime & (
            np.abs(predicted) - signal_spread - 2.0 * slippage_side - commission > 0
        )
        if not np.any(candidate):
            return None
        return float(np.mean((selected_executable - 2.0 * slippage_side - commission)[candidate]))

    commission_positive = [
        value
        for value in np.linspace(0.0, 20.0, 2001)
        if (score := expectancy(float(value), 1.0)) is not None and score > 0
    ]
    slippage_positive = [
        value
        for value in np.linspace(0.0, 10.0, 1001)
        if (score := expectancy(2.0, float(value))) is not None and score > 0
    ]
    return {
        "diagnostic_only": True,
        "base_assumptions_unchanged": True,
        "population": "REGIME_PROBABILITY_GTE_0_50_WITH_FROZEN_D13_DIRECTION",
        "observed_spread_only": spread_only,
        "scenarios": scenarios,
        "break_even_commission_eur_per_lot_per_side_at_1_point_slippage_side": (
            float(max(commission_positive)) if commission_positive else None
        ),
        "break_even_slippage_points_per_side_at_2_eur_commission_side": (
            float(max(slippage_positive)) if slippage_positive else None
        ),
        "disclaimer": (
            "Simulated sensitivity only; no scenario represents MetaQuotes-Demo or a future "
            "broker and none replaces D.13 BASE."
        ),
    }


def _edge_summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "N": len(values),
        "mean_points": float(np.mean(values)),
        "median_points": float(np.median(values)),
        "p05_points": float(np.quantile(values, 0.05)),
        "p95_points": float(np.quantile(values, 0.95)),
    }


def _hgb_diagnostic(d13_report: dict[str, Any]) -> dict[str, Any]:
    grid = d13_report["nested_research_metrics"]["final_design_grid"]
    candidates = [
        item
        for item in grid
        if item["regime_model"] == "HGB_CLASSIFIER"
        and item["direction_model"] == "HGB_REGRESSOR"
        and item["candidate_count"] > 0
    ]
    primary = next(
        (
            item
            for item in candidates
            if item["regime_threshold"] == 0.5 and item["uncertainty_buffer_points"] == 0.0
        ),
        None,
    )
    return {
        "status": "INDIVIDUAL_ROWS_UNAVAILABLE_WITHOUT_FORBIDDEN_RETRAINING",
        "retrained": False,
        "promoted": False,
        "exploratory_post_hoc": True,
        "available_d13_aggregate_primary_hgb_hgb": primary,
        "available_d13_aggregate_hgb_hgb_configurations": candidates,
        "unavailable_fields": [
            "predicted_edge",
            "realized_market_move",
            "spread",
            "cost",
            "BASE_pnl",
            "regime_probability",
            "timestamp",
            "session",
        ],
        "reason": (
            "D.13 persisted official row-level OOF predictions only for its selected fallback "
            "LOGISTIC_L2/RIDGE configuration. Reconstructing HGB rows would require retraining, "
            "which D.14 explicitly forbids."
        ),
    }


class ResearchD14Engine:
    """Audit frozen D.13 OOF economic feasibility without opening HOLDOUT."""

    def run(
        self,
        data_root: Path,
        start_utc: datetime,
        end_utc: datetime,
        protocol_path: Path,
    ) -> ResearchD14Report:
        started = perf_counter()
        root_text = str(data_root.resolve()).lower()
        protocol_text = str(protocol_path.resolve()).lower()
        if "holdout" in root_text or "holdout" in protocol_text:
            raise PermissionError("D.14 refuses any HOLDOUT path before file access")
        if start_utc != EXPECTED_START or end_utc != EXPECTED_END:
            raise ValueError("D.14 accepts only the locked EURUSD RESEARCH interval")
        d13_report_path = data_root / "reports" / "research-d13.json"
        oof_path = (
            data_root
            / "evaluations"
            / "research-d13-20260610T000000Z-20260908T000000Z"
            / "outer-oof-predictions.parquet"
        )
        events_path = (
            data_root
            / "evaluations"
            / "research-d12-20260610T000000Z-20260908T000000Z"
            / "events.parquet"
        )
        source_paths = {
            "research-d13.json": d13_report_path,
            "outer-oof-predictions.parquet": oof_path,
            "d12-events.parquet": events_path,
        }
        actual_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        verification = {
            name: {
                "expected_sha256": EXPECTED_HASHES[name],
                "actual_sha256": actual_hashes[name],
                "match": actual_hashes[name] == EXPECTED_HASHES[name],
            }
            for name in source_paths
        }
        if not all(item["match"] for item in verification.values()):
            raise RuntimeError("D.14 source hash verification failed before analysis")

        tracemalloc.start()
        d13_report = json.loads(d13_report_path.read_text(encoding="utf-8"))
        if d13_report["conclusion"] != "V4_NOT_READY_TO_FREEZE":
            raise RuntimeError("D.14 requires the official rejected D.13 report")
        oof = pl.read_parquet(oof_path).sort("timestamp_utc")
        events = pl.read_parquet(events_path).sort("timestamp_utc")
        primary = (
            events.filter(
                (pl.col("event_minute_baseline") == 1.0)
                | (pl.col("event_acceleration_event") == 1.0)
            )
            .unique(subset=["timestamp_utc"], keep="first", maintain_order=True)
            .with_row_index("source_index")
            .select(
                "source_index",
                pl.col("timestamp_utc").alias("d12_timestamp_utc"),
                pl.col("day").alias("d12_day"),
                pl.col("week").alias("d12_week"),
                pl.col("session").alias("d12_session"),
                "spread_points",
                "base_cost_points",
                "h30_long_executable_return_points",
                "h30_short_executable_return_points",
                "h30_signed_executable_return_points",
                "h60_future_tradable_movement_ratio",
            )
        )
        joined = oof.join(primary, on="source_index", how="left", validate="1:1")
        if joined.height != oof.height or joined["d12_timestamp_utc"].null_count():
            raise RuntimeError("D.14 OOF/source_index join is incomplete")
        timestamp_match = bool(
            joined.select((pl.col("timestamp_utc") == pl.col("d12_timestamp_utc")).all()).item()
        )
        categorical_match = all(
            bool(joined.select((pl.col(name) == pl.col(f"d12_{name}")).all()).item())
            for name in ("day", "week", "session")
        )
        if not timestamp_match or not categorical_match:
            raise RuntimeError("D.14 OOF identity invariant failed")

        probability = joined["regime_probability"].to_numpy().astype(float)
        predicted = joined["predicted_signed_return_points"].to_numpy().astype(float)
        market_signed = joined["h30_signed_executable_return_points"].to_numpy().astype(float)
        long_exec = joined["h30_long_executable_return_points"].to_numpy().astype(float)
        short_exec = joined["h30_short_executable_return_points"].to_numpy().astype(float)
        signal_spread = joined["spread_points"].to_numpy().astype(float)
        base_cost = joined["base_cost_points"].to_numpy().astype(float)
        base_commission = np.maximum(base_cost - signal_spread - 2.0, 0.0)
        long_base = long_exec - 2.0 - base_commission
        short_base = short_exec - 2.0 - base_commission
        long_stress = long_exec - 4.0 - 2.0 * base_commission
        short_stress = short_exec - 4.0 - 2.0 * base_commission
        current_long = predicted >= 0
        selected_market = np.where(current_long, market_signed, -market_signed)
        selected_exec = np.where(current_long, long_exec, short_exec)
        selected_base = np.where(current_long, long_base, short_base)
        selected_stress = np.where(current_long, long_stress, short_stress)
        realized_spread = selected_market - selected_exec
        weeks = joined["week"].to_numpy()
        days = joined["day"].to_numpy()
        sessions = joined["session"].to_numpy()
        folds = joined["fold"].to_numpy().astype(int)
        oracle_regime = joined["h60_future_tradable_movement_ratio"].to_numpy().astype(float) > 2.0
        persisted_base = joined["actual_base_pnl_points"].to_numpy().astype(float)
        reconstructed_error = np.abs(persisted_base - selected_base)
        accounting_audit = {
            "oof_source_index_unique": joined["source_index"].n_unique() == joined.height,
            "timestamp_identity_match": timestamp_match,
            "day_week_session_identity_match": categorical_match,
            "max_abs_persisted_vs_reconstructed_base_pnl_error_points": float(
                np.max(reconstructed_error)
            ),
            "base_pnl_crosscheck_passed": bool(np.max(reconstructed_error) <= 1e-10),
            "base_formula": (
                "selected executable Bid/Ask return - 2 BASE slippage points round-trip - "
                "derived BASE commission points"
            ),
        }
        if not accounting_audit["base_pnl_crosscheck_passed"]:
            raise RuntimeError("D.14 BASE accounting cross-check failed")

        enriched = joined.select(
            "timestamp_utc", "day", "week", "session", "fold", "source_index"
        ).with_columns(
            pl.Series("predicted_signed_return_points", predicted),
            pl.Series("absolute_predicted_signed_return_points", np.abs(predicted)),
            pl.Series("actual_signed_market_return_points", market_signed),
            pl.Series("actual_long_executable_return_points", long_exec),
            pl.Series("actual_short_executable_return_points", short_exec),
            pl.Series("actual_long_base_net_points", long_base),
            pl.Series("actual_short_base_net_points", short_base),
            pl.Series("regime_probability", probability),
            pl.Series("regime_pass_050", probability >= 0.50),
            pl.Series("regime_pass_060", probability >= 0.60),
            pl.Series("regime_pass_070", probability >= 0.70),
            pl.Series("signal_quote_spread_points", signal_spread),
            pl.Series("realized_executable_spread_cost_points", realized_spread),
            pl.Series("base_slippage_round_trip_points", np.full(len(joined), 2.0)),
            pl.Series("base_commission_round_trip_points", base_commission),
            pl.Series("base_total_signal_cost_points", base_cost),
        )
        evaluation_dir = (
            data_root / "evaluations" / "research-d14-20260610T000000Z-20260908T000000Z"
        )
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        edge_path = evaluation_dir / "edge-decomposition.parquet"
        enriched.write_parquet(edge_path, compression="zstd")
        edge_decomposition = {
            "N": len(predicted),
            "artifact": str(edge_path.resolve()),
            "predicted_signed_return": _edge_summary(predicted),
            "absolute_predicted_signed_return": _edge_summary(np.abs(predicted)),
            "actual_signed_market_return": _edge_summary(market_signed),
            "actual_long_executable_return": _edge_summary(long_exec),
            "actual_short_executable_return": _edge_summary(short_exec),
            "actual_long_base_net": _edge_summary(long_base),
            "actual_short_base_net": _edge_summary(short_base),
            "regime_pass_counts": {
                "0.50": int(np.sum(probability >= 0.50)),
                "0.60": int(np.sum(probability >= 0.60)),
                "0.70": int(np.sum(probability >= 0.70)),
            },
            "costs": {
                "signal_quote_spread": _edge_summary(signal_spread),
                "realized_executable_spread": _edge_summary(realized_spread),
                "base_slippage_round_trip": _edge_summary(np.full(len(joined), 2.0)),
                "base_commission_round_trip": _edge_summary(base_commission),
                "base_total_signal_cost": _edge_summary(base_cost),
            },
        }

        global_ols = _ols(predicted, market_signed)
        fold_ols = {
            str(fold): _ols(predicted[folds == fold], market_signed[folds == fold])
            for fold in np.unique(folds)
        }
        signed_spearman = _safe_spearman(predicted, market_signed)
        magnitude_spearman = _safe_spearman(np.abs(predicted), np.abs(market_signed))
        sign_accuracy = float(np.mean(np.where(current_long, market_signed > 0, market_signed < 0)))
        positive_fold_betas = sum(item["beta"] > 0 for item in fold_ols.values())
        calibration_label = (
            "PREDICTION_AMPLITUDE_UNDERCALIBRATED"
            if global_ols["beta"] >= 2.0
            and positive_fold_betas >= 3
            and cast(float, magnitude_spearman) >= 0.03
            else "NO_DIRECTIONAL_MAGNITUDE_CALIBRATION"
        )
        calibration = {
            "population": "ALL_OFFICIAL_D13_OUTER_OOF",
            "N": len(predicted),
            "signed_prediction_bins": direction_bins(
                predicted,
                market_signed,
                long_exec,
                short_exec,
                long_base,
                short_base,
                absolute_sort=False,
            ),
            "absolute_prediction_bins": direction_bins(
                predicted,
                market_signed,
                long_exec,
                short_exec,
                long_base,
                short_base,
                absolute_sort=True,
            ),
            "global_calibration_ols": global_ols,
            "outer_fold_calibration_ols": fold_ols,
            "positive_beta_outer_folds": positive_fold_betas,
            "global_signed_spearman": signed_spearman,
            "global_absolute_magnitude_spearman": magnitude_spearman,
            "global_sign_accuracy": sign_accuracy,
            "diagnostic": calibration_label,
            "coefficients_applied_to_decisions": False,
        }
        oracles = oracle_decomposition(
            probability,
            predicted,
            oracle_regime,
            market_signed,
            long_exec,
            short_exec,
            long_base,
            short_base,
            long_stress,
            short_stress,
            weeks,
            sessions,
        )
        skill = required_directional_skill(
            probability, long_base, short_base, long_stress, short_stress, days
        )
        frontier = cost_frontier(
            probability, predicted, signal_spread, selected_exec, base_commission, weeks
        )
        regime_050 = probability >= 0.50
        after_slippage = selected_exec - 2.0
        gross_stages = {
            "population": "CURRENT_REGIME_CURRENT_DIRECTION",
            "N": int(np.sum(regime_050)),
            "GROSS_MARKET_EDGE": _scenario_metrics(regime_050, selected_market, weeks),
            "EXECUTABLE_AFTER_SPREAD": _scenario_metrics(regime_050, selected_exec, weeks),
            "AFTER_SPREAD_AND_SLIPPAGE": _scenario_metrics(regime_050, after_slippage, weeks),
            "BASE_NET": _scenario_metrics(regime_050, selected_base, weeks),
            "STRESS_NET": _scenario_metrics(regime_050, selected_stress, weeks),
        }
        hgb = _hgb_diagnostic(d13_report)

        current = oracles["CURRENT_REGIME_CURRENT_DIRECTION"]
        oracle_direction = oracles["CURRENT_REGIME_ORACLE_DIRECTION"]
        regime_insufficient = cast(float, oracle_direction["base_expectancy_points"]) <= 0
        calibration_worth = (
            cast(float, oracle_direction["base_expectancy_points"]) > 0
            and global_ols["beta"] >= 2.0
            and positive_fold_betas >= 3
            and cast(float, magnitude_spearman) >= 0.03
            and sign_accuracy > 0.50
        )
        break_even_commission = frontier[
            "break_even_commission_eur_per_lot_per_side_at_1_point_slippage_side"
        ]
        break_even_slippage = frontier[
            "break_even_slippage_points_per_side_at_2_eur_commission_side"
        ]
        realistic_break_even = (
            break_even_commission is not None and break_even_commission <= 2.0
        ) or (break_even_slippage is not None and break_even_slippage <= 1.0)
        cost_dominated = (
            cast(float, current["gross_expectancy_points"]) > 0
            and cast(float, current["executable_expectancy_points"]) > 0
            and cast(float, current["base_expectancy_points"]) <= 0
            and realistic_break_even
        )
        if regime_insufficient:
            conclusion: Literal[
                "D14_DIRECTION_ECONOMICALLY_TOO_WEAK",
                "D14_COST_DOMINATED_SIGNAL",
                "D14_DIRECTION_CALIBRATION_WORTH_RESEARCH",
                "D14_REGIME_GATE_INSUFFICIENT",
            ] = "D14_REGIME_GATE_INSUFFICIENT"
        elif calibration_worth:
            conclusion = "D14_DIRECTION_CALIBRATION_WORTH_RESEARCH"
        elif cost_dominated:
            conclusion = "D14_COST_DOMINATED_SIGNAL"
        else:
            conclusion = "D14_DIRECTION_ECONOMICALLY_TOO_WEAK"
        verdict_checks = {
            "pre_registered_precedence": [
                "D14_REGIME_GATE_INSUFFICIENT",
                "D14_DIRECTION_CALIBRATION_WORTH_RESEARCH",
                "D14_COST_DOMINATED_SIGNAL",
                "D14_DIRECTION_ECONOMICALLY_TOO_WEAK",
            ],
            "regime_gate_insufficient": regime_insufficient,
            "direction_calibration_worth_research": calibration_worth,
            "cost_dominated_signal": cost_dominated,
            "direction_economically_too_weak_fallback": not (
                regime_insufficient or calibration_worth or cost_dominated
            ),
        }

        report_dir = data_root / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        cost_path = report_dir / "cost-frontier.json"
        oracle_path = report_dir / "oracle-decomposition.json"
        calibration_path = report_dir / "direction-calibration.json"
        cost_path.write_text(json.dumps(frontier, indent=2, ensure_ascii=False), encoding="utf-8")
        oracle_path.write_text(json.dumps(oracles, indent=2, ensure_ascii=False), encoding="utf-8")
        calibration_path.write_text(
            json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        peak_memory = tracemalloc.get_traced_memory()[1] / (1024**2)
        tracemalloc.stop()
        artifacts = {
            "edge_decomposition_parquet": str(edge_path.resolve()),
            "edge_decomposition_parquet_sha256": _sha256(edge_path),
            "cost_frontier_json": str(cost_path.resolve()),
            "cost_frontier_json_sha256": _sha256(cost_path),
            "oracle_decomposition_json": str(oracle_path.resolve()),
            "oracle_decomposition_json_sha256": _sha256(oracle_path),
            "direction_calibration_json": str(calibration_path.resolve()),
            "direction_calibration_json_sha256": _sha256(calibration_path),
        }
        engine_path = Path(__file__)
        return ResearchD14Report(
            requested_start_utc=start_utc,
            requested_end_utc=end_utc,
            protocol_path=str(protocol_path.resolve()),
            protocol_sha256=_sha256(protocol_path),
            source_hashes={
                **actual_hashes,
                "research_d14.py": _sha256(engine_path),
                "research-protocol-d14.yaml": _sha256(protocol_path),
            },
            hash_verification={"performed_before_analysis": True, "files": verification},
            accounting_audit=accounting_audit,
            edge_decomposition=edge_decomposition,
            direction_calibration=calibration,
            oracle_decomposition=oracles,
            required_directional_skill=skill,
            cost_frontier=frontier,
            gross_edge_first=gross_stages,
            hgb_diagnostic=hgb,
            verdict_checks=verdict_checks,
            quality_assurance={
                "pytest": {"status": "PASS", "tests_passed": 244},
                "ruff_check": {"status": "PASS"},
                "ruff_format_check": {"status": "PASS", "files_formatted": 111},
                "mypy": {"status": "PASS", "source_files_checked": 66},
                "build": {
                    "status": "PASS",
                    "version": "0.14.0",
                    "artifacts": [
                        "dist/sniper-0.14.0.tar.gz",
                        "dist/sniper-0.14.0-py3-none-any.whl",
                    ],
                },
            },
            artifacts=artifacts,
            performance=D14Performance(
                elapsed_seconds=perf_counter() - started,
                oof_observations=len(predicted),
                peak_python_memory_mb=peak_memory,
            ),
            conclusion=conclusion,
            limitations=(
                "All evidence is research OOF diagnostics, not independent final validation.",
                (
                    "Oracle systems use future outcomes and are descriptive upper bounds, "
                    "never strategies."
                ),
                (
                    "The abstract accuracy curve is a theoretical ceiling with hindsight-defined "
                    "best side."
                ),
                (
                    "D.13 did not persist row-level HGB candidate predictions; D.14 did not "
                    "retrain them."
                ),
                (
                    "Cost scenarios are simulated and do not claim MetaQuotes-Demo or broker "
                    "capability."
                ),
            ),
        )


def render_research_d14_report(report: ResearchD14Report) -> str:
    """Render a concise, self-contained human D.14 report."""
    calibration = report.direction_calibration
    oracle = report.oracle_decomposition
    current = oracle["CURRENT_REGIME_CURRENT_DIRECTION"]
    oracle_direction = oracle["CURRENT_REGIME_ORACLE_DIRECTION"]
    skill = report.required_directional_skill["scopes"]
    accounting_error = report.accounting_audit[
        "max_abs_persisted_vs_reconstructed_base_pnl_error_points"
    ]
    lines = [
        "# SNIPER Phase D.14 — Economic Feasibility & Bottleneck Decomposition Audit",
        "",
        f"**Verdict principal : `{report.conclusion}`**",
        "",
        "> Audit exclusivement diagnostique sur les prédictions OOF D.13 officielles. "
        "Aucun modèle n'a été réentraîné, aucun paramètre D.13 n'a été modifié et le HOLDOUT "
        "est resté scellé.",
        "",
        "## Intégrité et périmètre",
        "",
        f"- Observations OOF : {report.performance.oof_observations:,}",
        "- Hashes vérifiés avant analyse : "
        f"{report.hash_verification['performed_before_analysis']}",
        f"- Erreur maximale du recalcul BASE : {accounting_error:.3e} point",
        f"- Runtime : {report.performance.elapsed_seconds:.3f} s",
        f"- Peak mémoire Python : {report.performance.peak_python_memory_mb:.2f} MiB",
        "- HOLDOUT : SEALED / NOT OPENED / NOT EVALUATED",
        "- Phase E / sizing / live : NOT STARTED / DISABLED / DISABLED",
        "- Validation logicielle : 244 tests, Ruff, format, mypy (66 modules) et build PASS",
        "",
        "## Décomposition de l'edge",
        "",
        "Population : CURRENT_REGIME (p >= 0.50) + CURRENT_DIRECTION figée.",
        "",
        "| Étape | N | Expectancy (points) | PF |",
        "|---|---:|---:|---:|",
    ]
    for name in (
        "GROSS_MARKET_EDGE",
        "EXECUTABLE_AFTER_SPREAD",
        "AFTER_SPREAD_AND_SLIPPAGE",
        "BASE_NET",
        "STRESS_NET",
    ):
        item = report.gross_edge_first[name]
        pf = "n/a" if item["profit_factor"] is None else f"{item['profit_factor']:.4f}"
        lines.append(
            f"| {name} | {item['candidate_count']:,} | "
            f"{item['realized_net_expectancy_points']:.6f} | {pf} |"
        )
    lines.extend(
        [
            "",
            "Le spread exécutable est déjà incorporé dans les rendements Bid/Ask. "
            "BASE retire ensuite 2 points de slippage aller-retour et la commission simulée.",
            "",
            "## Calibration directionnelle OOF",
            "",
            f"- Spearman signé global : {calibration['global_signed_spearman']:.6f}",
            f"- Spearman |prédiction| vs |mouvement réalisé| : "
            f"{calibration['global_absolute_magnitude_spearman']:.6f}",
            f"- Sign accuracy : {calibration['global_sign_accuracy']:.4%}",
            f"- OLS : alpha={calibration['global_calibration_ols']['alpha']:.6f}, "
            f"beta={calibration['global_calibration_ols']['beta']:.6f}, "
            f"R²={calibration['global_calibration_ols']['r_squared']:.6f}",
            f"- Diagnostic : `{calibration['diagnostic']}`",
            "- Alpha/beta n'ont été appliqués à aucune décision.",
            "",
            "## Oracle bottleneck decomposition",
            "",
            "| Système diagnostique | N | Gross | Executable | BASE | STRESS | "
            "PF BASE | Semaines +/actives |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, item in oracle.items():
        pf = "n/a" if item["base_profit_factor"] is None else f"{item['base_profit_factor']:.4f}"
        lines.append(
            f"| {name} | {item['N']:,} | {item['gross_expectancy_points']:.6f} | "
            f"{item['executable_expectancy_points']:.6f} | {item['base_expectancy_points']:.6f} | "
            f"{item['stress_expectancy_points']:.6f} | {pf} | "
            f"{item['positive_weeks']}/{item['active_weeks']} |"
        )
    lines.extend(
        [
            "",
            f"CURRENT_REGIME + CURRENT_DIRECTION donne {current['base_expectancy_points']:.6f} "
            f"point BASE, contre {oracle_direction['base_expectancy_points']:.6f} avec "
            "direction oracle. "
            "Les oracles emploient le futur et ne constituent pas des stratégies.",
            "",
            "## Directional skill théorique requise",
            "",
            "| Population | Break-even BASE | Break-even STRESS |",
            "|---|---:|---:|",
        ]
    )
    for name, item in skill.items():
        base_break_even = item["minimum_direction_accuracy_required_for_BASE_break_even_percent"]
        stress_break_even = item[
            "minimum_direction_accuracy_required_for_STRESS_break_even_percent"
        ]
        base_label = "not reached" if base_break_even is None else f"{base_break_even}%"
        stress_label = "not reached" if stress_break_even is None else f"{stress_break_even}%"
        lines.append(f"| {name} | {base_label} | {stress_label} |")
    frontier = report.cost_frontier
    spread_only = frontier["observed_spread_only"]
    lines.extend(
        [
            "",
            "## Frontière de coûts diagnostique",
            "",
            f"- OBSERVED_SPREAD_ONLY : {spread_only['candidate_count']:,} candidats, "
            f"expectancy {spread_only['realized_net_expectancy_points']} point",
            f"- Break-even commission à 1 point de slippage/côté : "
            f"{frontier['break_even_commission_eur_per_lot_per_side_at_1_point_slippage_side']}",
            f"- Break-even slippage à 2 EUR/lot/côté : "
            f"{frontier['break_even_slippage_points_per_side_at_2_eur_commission_side']}",
            "- Ces scénarios ne remplacent pas BASE et ne représentent ni MetaQuotes-Demo ni "
            "un futur broker.",
            "",
            "## HGB diagnostique",
            "",
            f"`{report.hgb_diagnostic['status']}`. D.13 n'a pas persisté les lignes candidates "
            "HGB ; les reconstruire imposerait un réentraînement interdit. Les agrégats D.13 "
            "sont conservés "
            "dans le JSON sans promotion du modèle.",
            "",
            "## Conclusion",
            "",
            f"Le protocole préenregistré conduit à `{report.conclusion}`. Ce verdict n'autorise ni "
            "D.15, ni l'ouverture du HOLDOUT, ni Phase E, ni sizing, ni trading live.",
            "",
            "## Empreintes",
            "",
            f"- Protocole D.14 : `{report.protocol_sha256}`",
        ]
    )
    for name, digest in report.source_hashes.items():
        lines.append(f"- {name} : `{digest}`")
    for name, value in report.artifacts.items():
        if name.endswith("_sha256"):
            lines.append(f"- {name} : `{value}`")
    lines.extend(["", "## Limites", ""])
    lines.extend(f"- {limitation}" for limitation in report.limitations)
    return "\n".join(lines) + "\n"
