"""Phase D.13 hierarchical V4 research development on the sealed RESEARCH dataset."""

from __future__ import annotations

import hashlib
import json
import math
import tracemalloc
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
import polars as pl
from scipy.stats import spearmanr  # type: ignore[import-untyped]
from sklearn.ensemble import (  # type: ignore[import-untyped]
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mutual_info_score,
)

from sniper.domain.research_d13 import D13Performance, ResearchD13Report
from sniper.evaluation.edge_validation import MICROSECONDS

RANDOM_SEED_D13 = 20260913
PRIMARY_REGIME_HORIZON = 60
PRIMARY_DIRECTION_HORIZON = 30
PURGE_SECONDS = 60
REGIME_FEATURE_PRIORITY = (
    "realized_volatility_300s_points",
    "volatility_percentile_300s",
    "realized_move_to_base_cost_300s",
    "micro_pullback_amplitude_60s_points",
    "return_autocorrelation_lag1_900s",
    "variance_ratio_5_900s",
    "spread_vs_median_60s",
)
DIRECTION_FEATURE_PRIORITY = (
    "current_signed_run_length",
    "return_5s_points",
    "return_15s_points",
    "signed_tick_imbalance_1s",
    "signed_tick_imbalance_5s",
    "displacement_per_tick_15s_points",
    "breakout_velocity_15s_points_per_second",
)
REGIME_MODELS = ("LOGISTIC_L2", "HGB_CLASSIFIER")
DIRECTION_MODELS = ("RIDGE", "HGB_REGRESSOR")
REGIME_THRESHOLDS = (0.50, 0.60, 0.70)
UNCERTAINTY_BUFFERS = (0.0, 1.0, 2.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze_hash(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "freeze_manifest_sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(canonical).hexdigest()


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return None
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else None


def _quantile_bins(values: np.ndarray, bins: int = 10) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=int)
    result[order] = np.minimum(np.arange(len(values)) * bins // max(len(values), 1), bins - 1)
    return result


def _pairwise_mi(x: np.ndarray, y: np.ndarray) -> float | None:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 10 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return None
    return float(mutual_info_score(_quantile_bins(x), _quantile_bins(y)))


def reduce_redundant_features(
    frame: pl.DataFrame,
    indices: np.ndarray,
    priority: tuple[str, ...],
    threshold: float = 0.90,
) -> dict[str, Any]:
    """Apply the pre-registered, ordered train-only redundancy rule."""
    values = {name: frame[name].to_numpy().astype(float)[indices] for name in priority}
    pairwise = []
    retained: list[str] = []
    removed = []
    for position, feature in enumerate(priority):
        for other in priority[position + 1 :]:
            rho = _safe_spearman(values[feature], values[other])
            pairwise.append(
                {
                    "feature_a": feature,
                    "feature_b": other,
                    "spearman": rho,
                    "absolute_spearman": abs(rho) if rho is not None else None,
                    "mutual_information_10x10": _pairwise_mi(values[feature], values[other]),
                }
            )
        redundant_with = None
        redundant_rho = None
        for representative in retained:
            rho = _safe_spearman(values[feature], values[representative])
            if rho is not None and abs(rho) >= threshold:
                redundant_with, redundant_rho = representative, rho
                break
        if redundant_with is None:
            retained.append(feature)
        else:
            removed.append(
                {
                    "feature": feature,
                    "retained_representative": redundant_with,
                    "spearman": redundant_rho,
                    "reason": "LATER_PRIORITY_MEMBER_WITH_ABSOLUTE_SPEARMAN_GTE_0_90",
                }
            )
    return {
        "fit_observations": len(indices),
        "threshold": threshold,
        "priority": list(priority),
        "retained": retained,
        "removed": removed,
        "pairwise": pairwise,
    }


@dataclass
class TrainPreprocessor:
    medians: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    active: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> TrainPreprocessor:
        medians = np.nanmedian(values, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        imputed = np.where(np.isfinite(values), values, medians)
        lower = np.quantile(imputed, 0.005, axis=0)
        upper = np.quantile(imputed, 0.995, axis=0)
        clipped = np.clip(imputed, lower, upper)
        means = np.mean(clipped, axis=0)
        scales = np.std(clipped, axis=0)
        active = np.isfinite(scales) & (scales >= 1e-8)
        if not np.any(active):
            raise RuntimeError("D.13 preprocessing removed every feature as quasi-singular")
        return cls(medians, lower, upper, means, scales, active)

    def transform(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        imputed = np.where(np.isfinite(values), values, self.medians)
        raw = np.clip(imputed, self.lower, self.upper)[:, self.active]
        scaled = (raw - self.means[self.active]) / self.scales[self.active]
        scaled = np.clip(scaled, -10.0, 10.0)
        if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(scaled)):
            raise RuntimeError("D.13 preprocessing produced a non-finite value")
        return raw, scaled


def _dataset_path(data_root: Path, protocol_path: Path) -> Path:
    if "holdout" in str(data_root.resolve()).lower():
        raise PermissionError("D.13 refuses any HOLDOUT path before protocol or model access")
    if not protocol_path.exists():
        raise ValueError("D.13 locked protocol is missing")
    dataset = (
        data_root
        / "evaluations"
        / "research-d12-20260610T000000Z-20260908T000000Z"
        / "events.parquet"
    ).resolve()
    allowed = (data_root / "evaluations").resolve()
    if allowed not in dataset.parents or "holdout" in str(dataset).lower():
        raise PermissionError("D.13 dataset escaped the RESEARCH evaluations root")
    if not dataset.exists():
        raise ValueError("D.12 RESEARCH event artifact is required before D.13")
    expected = "a23531949d3ea9857223caf911405d48aafcd3dbbd70f78f86a561b7a3bf9c0f"
    if _sha256(dataset) != expected:
        raise RuntimeError("D.13 D.12 source artifact hash differs from the locked protocol")
    return dataset


def _fold_boundaries(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    span = end - start
    points = [start + span * value for value in (0.30, 0.40, 0.50, 0.60, 1.00)]
    return list(zip(points[:-1], points[1:], strict=True))


def _purged_indices(
    timestamps: np.ndarray,
    validation_start: datetime,
    validation_end: datetime,
    eligible_train: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    start_us = int(validation_start.timestamp() * MICROSECONDS)
    end_us = int(validation_end.timestamp() * MICROSECONDS)
    raw_train = np.flatnonzero(timestamps < start_us)
    if eligible_train is not None:
        raw_train = np.intersect1d(raw_train, eligible_train, assume_unique=True)
    train = raw_train[timestamps[raw_train] + PURGE_SECONDS * MICROSECONDS <= start_us]
    validation = np.flatnonzero((timestamps >= start_us) & (timestamps < end_us))
    max_end = int(np.max(timestamps[train])) + PURGE_SECONDS * MICROSECONDS if len(train) else None
    invariant = max_end is None or max_end <= start_us
    if not invariant:
        raise RuntimeError("D.13 purge invariant failed")
    audit = {
        "validation_start_utc": validation_start,
        "validation_end_utc": validation_end,
        "raw_train_observations": len(raw_train),
        "purged_observations": len(raw_train) - len(train),
        "retained_train_observations": len(train),
        "validation_observations": len(validation),
        "max_label_end_time_train_utc": (
            datetime.fromtimestamp(max_end / MICROSECONDS, UTC) if max_end is not None else None
        ),
        "invariant_passed": invariant,
    }
    return train, validation, audit


def nested_folds(
    timestamps: np.ndarray,
    start: datetime,
    outer_validation_start: datetime,
    eligible_outer_train: np.ndarray,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    span = outer_validation_start - start
    points = [start + span * value for value in (0.40, 0.60, 0.80, 1.00)]
    folds = []
    audits = []
    for number, (validation_start, validation_end) in enumerate(
        zip(points[:-1], points[1:], strict=True), 1
    ):
        train, validation, audit = _purged_indices(
            timestamps, validation_start, validation_end, eligible_outer_train
        )
        if not len(train) or not len(validation):
            raise RuntimeError("D.13 nested fold is empty")
        audit["nested_fold"] = number
        folds.append((train, validation))
        audits.append(audit)
    return folds, audits


def _target_arrays(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    base_cost = frame["base_cost_points"].to_numpy().astype(float)
    spread = frame["spread_points"].to_numpy().astype(float)
    commission = np.maximum(base_cost - spread - 2.0, 0.0)
    long_exec = frame["h30_long_executable_return_points"].to_numpy().astype(float)
    short_exec = frame["h30_short_executable_return_points"].to_numpy().astype(float)
    signed = frame["h30_signed_executable_return_points"].to_numpy().astype(float)
    return {
        "regime": (
            frame["h60_future_tradable_movement_ratio"].to_numpy().astype(float) > 2.0
        ).astype(int),
        "movement": frame["h60_future_tradable_movement_ratio"].to_numpy().astype(float),
        "direction": signed,
        "spread": spread,
        "base_cost": base_cost,
        "commission_base": commission,
        "long_exec": long_exec,
        "short_exec": short_exec,
        "long_base": long_exec - 2.0 - commission,
        "short_base": short_exec - 2.0 - commission,
        "long_stress": long_exec - 4.0 - 2.0 * commission,
        "short_stress": short_exec - 4.0 - 2.0 * commission,
    }


def _fit_predictions(
    frame: pl.DataFrame,
    targets: dict[str, np.ndarray],
    train: np.ndarray,
    validation: np.ndarray,
    audit_sink: list[dict[str, Any]],
    scope: str,
) -> dict[str, Any]:
    regime_reduction = reduce_redundant_features(frame, train, REGIME_FEATURE_PRIORITY)
    direction_reduction = reduce_redundant_features(frame, train, DIRECTION_FEATURE_PRIORITY)
    regime_features = cast(list[str], regime_reduction["retained"])
    direction_features = cast(list[str], direction_reduction["retained"])

    regime_train_matrix = frame.select(regime_features).to_numpy().astype(float)[train]
    regime_validation_matrix = frame.select(regime_features).to_numpy().astype(float)[validation]
    direction_train_matrix = frame.select(direction_features).to_numpy().astype(float)[train]
    direction_validation_matrix = (
        frame.select(direction_features).to_numpy().astype(float)[validation]
    )
    regime_pre = TrainPreprocessor.fit(regime_train_matrix)
    direction_pre = TrainPreprocessor.fit(direction_train_matrix)
    regime_train_raw, regime_train_scaled = regime_pre.transform(regime_train_matrix)
    regime_validation_raw, regime_validation_scaled = regime_pre.transform(regime_validation_matrix)
    direction_train_raw, direction_train_scaled = direction_pre.transform(direction_train_matrix)
    direction_validation_raw, direction_validation_scaled = direction_pre.transform(
        direction_validation_matrix
    )
    y_regime = targets["regime"][train]
    y_direction = targets["direction"][train]
    if len(np.unique(y_regime)) < 2:
        raise RuntimeError("D.13 regime TRAIN contains only one class")

    logistic = LogisticRegression(C=0.1, max_iter=500, random_state=RANDOM_SEED_D13).fit(
        regime_train_scaled, y_regime
    )
    hgb_classifier = HistGradientBoostingClassifier(
        max_iter=75,
        max_depth=3,
        learning_rate=0.05,
        random_state=RANDOM_SEED_D13,
        early_stopping=False,
    ).fit(regime_train_raw, y_regime)
    ridge = Ridge(alpha=10.0).fit(direction_train_scaled, y_direction)
    hgb_regressor = HistGradientBoostingRegressor(
        max_iter=75,
        max_depth=3,
        learning_rate=0.05,
        random_state=RANDOM_SEED_D13,
        early_stopping=False,
    ).fit(direction_train_raw, y_direction)

    logistic_probability = np.clip(
        logistic.predict_proba(regime_validation_scaled)[:, 1], 1e-6, 0.999999
    )
    hgb_probability = np.clip(
        hgb_classifier.predict_proba(regime_validation_raw)[:, 1], 1e-6, 0.999999
    )
    ridge_prediction = ridge.predict(direction_validation_scaled)
    hgb_prediction = hgb_regressor.predict(direction_validation_raw)
    ridge_mae = float(mean_absolute_error(targets["direction"][validation], ridge_prediction))
    ridge_coefficients_finite = bool(np.all(np.isfinite(ridge.coef_)))
    ridge_predictions_finite = bool(np.all(np.isfinite(ridge_prediction)))
    ridge_valid = ridge_coefficients_finite and ridge_predictions_finite and ridge_mae <= 1000.0
    if not np.all(np.isfinite(hgb_prediction)):
        raise RuntimeError("D.13 HGB direction prediction is non-finite")

    audit_sink.append(
        {
            "scope": scope,
            "train_observations": len(train),
            "validation_observations": len(validation),
            "regime_retained": regime_features,
            "direction_retained": direction_features,
            "regime_quasi_singular_dropped": [
                feature
                for feature, active in zip(regime_features, regime_pre.active, strict=True)
                if not active
            ],
            "direction_quasi_singular_dropped": [
                feature
                for feature, active in zip(direction_features, direction_pre.active, strict=True)
                if not active
            ],
            "ridge_coefficients_finite": ridge_coefficients_finite,
            "ridge_predictions_finite": ridge_predictions_finite,
            "ridge_validation_mae_points": ridge_mae,
            "ridge_valid": ridge_valid,
            "scaled_train_max_abs": float(
                max(np.max(np.abs(regime_train_scaled)), np.max(np.abs(direction_train_scaled)))
            ),
            "scaled_validation_max_abs": float(
                max(
                    np.max(np.abs(regime_validation_scaled)),
                    np.max(np.abs(direction_validation_scaled)),
                )
            ),
        }
    )
    return {
        "indices": validation,
        "regime": {"LOGISTIC_L2": logistic_probability, "HGB_CLASSIFIER": hgb_probability},
        "direction": {"RIDGE": ridge_prediction, "HGB_REGRESSOR": hgb_prediction},
        "ridge_valid": ridge_valid,
        "regime_reduction": regime_reduction,
        "direction_reduction": direction_reduction,
    }


def economic_arrays(
    predictions: dict[str, Any],
    targets: dict[str, np.ndarray],
    configuration: dict[str, Any],
) -> dict[str, np.ndarray]:
    indices = cast(np.ndarray, predictions["indices"])
    probability = cast(np.ndarray, predictions["regime"][configuration["regime_model"]])
    signed_prediction = cast(np.ndarray, predictions["direction"][configuration["direction_model"]])
    base_cost = targets["base_cost"][indices]
    predicted_long = signed_prediction - base_cost
    predicted_short = -signed_prediction - base_cost
    long_side = predicted_long >= predicted_short
    predicted_edge = np.where(long_side, predicted_long, predicted_short)
    regime_pass = probability >= float(configuration["regime_threshold"])
    candidate = regime_pass & (predicted_edge > float(configuration["uncertainty_buffer_points"]))
    actual_base = np.where(long_side, targets["long_base"][indices], targets["short_base"][indices])
    actual_stress = np.where(
        long_side, targets["long_stress"][indices], targets["short_stress"][indices]
    )
    executable = np.where(long_side, targets["long_exec"][indices], targets["short_exec"][indices])
    market = np.where(long_side, targets["direction"][indices], -targets["direction"][indices])
    spread_realized = market - executable
    reconstructed_base = market - spread_realized - 2.0 - targets["commission_base"][indices]
    return {
        "indices": indices,
        "probability": probability,
        "signed_prediction": signed_prediction,
        "predicted_edge": predicted_edge,
        "regime_pass": regime_pass,
        "candidate": candidate,
        "long_side": long_side,
        "actual_base": actual_base,
        "actual_stress": actual_stress,
        "market": market,
        "executable": executable,
        "spread_realized": spread_realized,
        "slippage_base": np.full(len(indices), 2.0),
        "commission_base": targets["commission_base"][indices],
        "reconstructed_base": reconstructed_base,
    }


def _configuration_metrics(
    arrays: dict[str, np.ndarray], targets: dict[str, np.ndarray]
) -> dict[str, Any]:
    indices = arrays["indices"]
    regime_pass = arrays["regime_pass"]
    candidate = arrays["candidate"]
    y = targets["regime"][indices]
    true_positive = int(np.sum(regime_pass & (y == 1)))
    precision = true_positive / int(np.sum(regime_pass)) if np.any(regime_pass) else 0.0
    recall = true_positive / int(np.sum(y == 1)) if np.any(y == 1) else 0.0
    count = int(np.sum(candidate))
    base = arrays["actual_base"][candidate]
    movement = targets["movement"][indices][candidate]
    base_rate = float(np.mean(y))
    expectancy = float(np.mean(base)) if count else None
    precision_lift = np.clip(precision / base_rate - 1.0, -1.0, 1.0) if base_rate else 0.0
    candidate_coverage = min(count / 150.0, 1.0)
    movement_mean = float(np.mean(movement)) if count else 0.0
    movement_quality = float(np.clip(movement_mean / 2.0 - 1.0, -1.0, 1.0))
    economic_quality = math.tanh((expectancy or 0.0) / 10.0)
    score = (
        0.30 * precision_lift
        + 0.20 * recall
        + 0.15 * candidate_coverage
        + 0.20 * movement_quality
        + 0.15 * economic_quality
    )
    return {
        "candidate_count": count,
        "regime_base_rate": base_rate,
        "regime_precision": precision,
        "regime_recall": recall,
        "mean_future_movement_to_base_cost": movement_mean if count else None,
        "base_net_expectancy_points": expectancy,
        "composite_score": score if count >= 75 else None,
        "eligible_minimum_candidates": count >= 75,
    }


def _all_configurations() -> list[dict[str, Any]]:
    return [
        {
            "regime_model": regime,
            "direction_model": direction,
            "regime_threshold": threshold,
            "uncertainty_buffer_points": buffer,
        }
        for regime in REGIME_MODELS
        for direction in DIRECTION_MODELS
        for threshold in REGIME_THRESHOLDS
        for buffer in UNCERTAINTY_BUFFERS
    ]


def _select_configuration(
    predictions: list[dict[str, Any]], targets: dict[str, np.ndarray]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluations = []
    for config in _all_configurations():
        arrays_by_fold = [economic_arrays(item, targets, config) for item in predictions]
        combined = {
            key: np.concatenate([item[key] for item in arrays_by_fold]) for key in arrays_by_fold[0]
        }
        metrics = _configuration_metrics(combined, targets)
        ridge_valid = config["direction_model"] != "RIDGE" or all(
            item["ridge_valid"] for item in predictions
        )
        metrics["ridge_valid"] = ridge_valid
        if not ridge_valid:
            metrics["composite_score"] = None
        evaluations.append({**config, **metrics})
    eligible = [item for item in evaluations if item["composite_score"] is not None]
    if eligible:
        selected = max(
            eligible,
            key=lambda item: (
                item["composite_score"],
                item["candidate_count"],
                item["regime_model"] == "LOGISTIC_L2",
                item["direction_model"] == "RIDGE",
                -item["regime_threshold"],
                -item["uncertainty_buffer_points"],
            ),
        )
        selected = {**selected, "selection_eligible": True}
    else:
        selected = next(
            item
            for item in evaluations
            if item["regime_model"] == "LOGISTIC_L2"
            and item["direction_model"] == "RIDGE"
            and item["regime_threshold"] == 0.50
            and item["uncertainty_buffer_points"] == 0.0
        )
        selected = {**selected, "selection_eligible": False, "fallback_used": True}
    return selected, evaluations


def _records_from_arrays(
    frame: pl.DataFrame,
    arrays: dict[str, np.ndarray],
    fold: int,
    role: str,
    configuration: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations = []
    candidates = []
    timestamps = frame["timestamp_utc"].to_list()
    days = frame["day"].to_list()
    weeks = frame["week"].to_list()
    sessions = frame["session"].to_list()
    for local, source_index in enumerate(arrays["indices"]):
        source = int(source_index)
        candidate = bool(arrays["candidate"][local])
        record = {
            "timestamp_utc": timestamps[source],
            "day": days[source],
            "week": weeks[source],
            "session": sessions[source],
            "fold": fold,
            "role": role,
            "regime_probability": float(arrays["probability"][local]),
            "predicted_signed_return_points": float(arrays["signed_prediction"][local]),
            "regime_pass": bool(arrays["regime_pass"][local]),
            "research_candidate": candidate,
            "research_side": ("RESEARCH_LONG" if arrays["long_side"][local] else "RESEARCH_SHORT")
            if candidate
            else "RESEARCH_SKIP",
            "predicted_base_net_edge_points": float(arrays["predicted_edge"][local]),
            "actual_base_pnl_points": float(arrays["actual_base"][local]),
            "actual_stress_pnl_points": float(arrays["actual_stress"][local]),
            "market_pnl_points": float(arrays["market"][local]),
            "executable_bid_ask_pnl_points": float(arrays["executable"][local]),
            "realized_spread_cost_points": float(arrays["spread_realized"][local]),
            "base_slippage_cost_points": float(arrays["slippage_base"][local]),
            "base_commission_cost_points": float(arrays["commission_base"][local]),
            "base_reconstructed_pnl_points": float(arrays["reconstructed_base"][local]),
            "regime_model": configuration["regime_model"],
            "direction_model": configuration["direction_model"],
            "regime_threshold": configuration["regime_threshold"],
            "uncertainty_buffer_points": configuration["uncertainty_buffer_points"],
        }
        observations.append(record)
        if candidate:
            candidates.append(record)
    return observations, candidates


def _bootstrap_ci(records: list[dict[str, Any]], field: str, seed: int) -> dict[str, Any]:
    if not records:
        return {"lower": None, "upper": None, "replications": 2000, "blocks": 0}
    groups: dict[str, list[float]] = {}
    for row in records:
        groups.setdefault(str(row["day"]), []).append(float(row[field]))
    days = sorted(groups)
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(2000):
        sampled = rng.choice(days, size=len(days), replace=True)
        values = [value for day in sampled for value in groups[str(day)]]
        results.append(float(np.mean(values)))
    lower, upper = np.quantile(results, (0.025, 0.975))
    return {
        "lower": float(lower),
        "upper": float(upper),
        "replications": 2000,
        "blocks": len(days),
        "seed": seed,
    }


def _economic_metrics(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    if not records:
        return {
            "candidate_count": 0,
            "active_days": 0,
            "active_weeks": 0,
            "base_net_expectancy_points": None,
            "stress_net_expectancy_points": None,
            "profit_factor_base": None,
            "positive_week_fraction": None,
            "largest_session_share": None,
            "base_expectancy_block_bootstrap_ci95": _bootstrap_ci([], "", seed),
        }
    base = np.asarray([row["actual_base_pnl_points"] for row in records], dtype=float)
    stress = np.asarray([row["actual_stress_pnl_points"] for row in records], dtype=float)
    positive = float(np.sum(base[base > 0]))
    negative = float(-np.sum(base[base < 0]))
    weekly: dict[str, float] = {}
    sessions: dict[str, int] = {}
    for row in records:
        weekly[str(row["week"])] = weekly.get(str(row["week"]), 0.0) + float(
            row["actual_base_pnl_points"]
        )
        sessions[str(row["session"])] = sessions.get(str(row["session"]), 0) + 1
    return {
        "candidate_count": len(records),
        "long_count": sum(row["research_side"] == "RESEARCH_LONG" for row in records),
        "short_count": sum(row["research_side"] == "RESEARCH_SHORT" for row in records),
        "active_days": len({row["day"] for row in records}),
        "active_weeks": len(weekly),
        "base_net_expectancy_points": float(np.mean(base)),
        "stress_net_expectancy_points": float(np.mean(stress)),
        "profit_factor_base": positive / negative if negative > 0 else None,
        "win_rate": float(np.mean(base > 0)),
        "positive_week_fraction": sum(value > 0 for value in weekly.values()) / len(weekly),
        "largest_session_share": max(sessions.values()) / len(records),
        "weekly": [
            {
                "week": week,
                "candidate_count": sum(row["week"] == week for row in records),
                "base_total_points": total,
            }
            for week, total in sorted(weekly.items())
        ],
        "sessions": [
            {
                "session": session,
                "candidate_count": count,
                "share": count / len(records),
                "base_expectancy_points": float(
                    np.mean(
                        [
                            row["actual_base_pnl_points"]
                            for row in records
                            if row["session"] == session
                        ]
                    )
                ),
            }
            for session, count in sorted(sessions.items())
        ],
        "base_expectancy_block_bootstrap_ci95": _bootstrap_ci(
            records, "actual_base_pnl_points", seed
        ),
        "stress_expectancy_block_bootstrap_ci95": _bootstrap_ci(
            records, "actual_stress_pnl_points", seed + 1
        ),
    }


def _baseline_records(
    frame: pl.DataFrame,
    targets: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    train: np.ndarray,
    fold: int,
    configuration: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    indices = arrays["indices"]
    timestamps = frame["timestamp_utc"].to_list()
    days = frame["day"].to_list()
    weeks = frame["week"].to_list()
    sessions = frame["session"].to_list()
    result: dict[str, list[dict[str, Any]]] = {
        "RANDOM_SIDE": [],
        "UNCONDITIONAL_SIDE": [],
        "CONTRARIAN_RETURN_5S": [],
        "CONTRARIAN_RUN_LENGTH": [],
    }

    def append(name: str, local: int, long_side: bool) -> None:
        source = int(indices[local])
        result[name].append(
            {
                "timestamp_utc": timestamps[source],
                "day": days[source],
                "week": weeks[source],
                "session": sessions[source],
                "fold": fold,
                "research_side": "RESEARCH_LONG" if long_side else "RESEARCH_SHORT",
                "actual_base_pnl_points": float(
                    targets["long_base"][source] if long_side else targets["short_base"][source]
                ),
                "actual_stress_pnl_points": float(
                    targets["long_stress"][source] if long_side else targets["short_stress"][source]
                ),
            }
        )

    rng = np.random.default_rng(RANDOM_SEED_D13 + fold)
    candidate_positions = np.flatnonzero(arrays["candidate"])
    for local, long_side in zip(
        candidate_positions, rng.random(len(candidate_positions)) < 0.5, strict=True
    ):
        append("RANDOM_SIDE", int(local), bool(long_side))

    unconditional_long = float(np.mean(targets["direction"][train])) >= 0
    for local in np.flatnonzero(arrays["regime_pass"]):
        append("UNCONDITIONAL_SIDE", int(local), unconditional_long)

    magnitude = float(np.median(np.abs(targets["direction"][train])))
    buffer = float(configuration["uncertainty_buffer_points"])
    for name, feature in (
        ("CONTRARIAN_RETURN_5S", "return_5s_points"),
        ("CONTRARIAN_RUN_LENGTH", "current_signed_run_length"),
    ):
        values = frame[feature].to_numpy().astype(float)[indices]
        eligible = arrays["regime_pass"] & (magnitude - targets["base_cost"][indices] > buffer)
        for local in np.flatnonzero(eligible & np.isfinite(values) & (values != 0)):
            append(name, int(local), bool(values[local] < 0))
    return result


def _calibration_bins(probability: np.ndarray, target: np.ndarray) -> list[dict[str, Any]]:
    order = np.argsort(probability, kind="stable")
    rows = []
    for number, positions in enumerate(np.array_split(order, 10), 1):
        if len(positions):
            rows.append(
                {
                    "bin": number,
                    "N": len(positions),
                    "mean_probability": float(np.mean(probability[positions])),
                    "observed_rate": float(np.mean(target[positions])),
                }
            )
    return rows


def _model_diagnostics(
    observation_records: list[dict[str, Any]], targets: dict[str, np.ndarray]
) -> dict[str, Any]:
    indices = np.asarray([row["source_index"] for row in observation_records], dtype=int)
    probability = np.asarray([row["regime_probability"] for row in observation_records])
    prediction = np.asarray([row["predicted_signed_return_points"] for row in observation_records])
    regime = targets["regime"][indices]
    direction = targets["direction"][indices]
    return {
        "regime": {
            "N": len(indices),
            "unconditional_base_rate": float(np.mean(regime)),
            "pr_auc": float(average_precision_score(regime, probability)),
            "brier": float(brier_score_loss(regime, probability)),
            "calibration_bins": _calibration_bins(probability, regime),
        },
        "direction": {
            "N": len(indices),
            "spearman": _safe_spearman(prediction, direction),
            "mae_points": float(mean_absolute_error(direction, prediction)),
            "predictions_finite": bool(np.all(np.isfinite(prediction))),
        },
    }


class ResearchD13Engine:
    """Develop and freeze a research-only hierarchical V4 specification."""

    def run(
        self,
        data_root: Path,
        start_utc: datetime,
        end_utc: datetime,
        protocol_path: Path,
    ) -> ResearchD13Report:
        started = perf_counter()
        tracemalloc.start()
        dataset = _dataset_path(data_root, protocol_path)
        expected_start = datetime(2026, 6, 10, tzinfo=UTC)
        expected_end = datetime(2026, 9, 8, tzinfo=UTC)
        if start_utc != expected_start or end_utc != expected_end:
            raise ValueError("D.13 accepts only the locked EURUSD RESEARCH interval")
        frame_all = pl.read_parquet(dataset).sort("timestamp_utc")
        primary = frame_all.filter(
            (pl.col("event_minute_baseline") == 1.0) | (pl.col("event_acceleration_event") == 1.0)
        ).unique(subset=["timestamp_utc"], keep="first", maintain_order=True)
        if primary["timestamp_utc"].n_unique() != primary.height:
            raise RuntimeError("D.13 primary event universe was not deduplicated")
        frame = primary.with_row_index("source_index")
        targets = _target_arrays(frame)
        timestamps = frame["timestamp_utc"].cast(pl.Int64).to_numpy().astype(np.int64)
        design_end = start_utc + (end_utc - start_utc) * 0.60
        design_end_us = int(design_end.timestamp() * MICROSECONDS)
        design_indices = np.flatnonzero(timestamps + PURGE_SECONDS * MICROSECONDS <= design_end_us)
        final_regime_reduction = reduce_redundant_features(
            frame, design_indices, REGIME_FEATURE_PRIORITY
        )
        final_direction_reduction = reduce_redundant_features(
            frame, design_indices, DIRECTION_FEATURE_PRIORITY
        )

        preprocessing_audits: list[dict[str, Any]] = []
        outer_audits = []
        nested_audits = []
        outer_results = []
        observation_records: list[dict[str, Any]] = []
        candidate_records: list[dict[str, Any]] = []
        baseline_records: dict[str, list[dict[str, Any]]] = {
            "RANDOM_SIDE": [],
            "UNCONDITIONAL_SIDE": [],
            "CONTRARIAN_RETURN_5S": [],
            "CONTRARIAN_RUN_LENGTH": [],
        }

        for fold, (validation_start, validation_end) in enumerate(
            _fold_boundaries(start_utc, end_utc), 1
        ):
            train, validation, outer_audit = _purged_indices(
                timestamps, validation_start, validation_end
            )
            if not len(train) or not len(validation):
                raise RuntimeError("D.13 outer fold is empty")
            role = "MODEL_DEVELOPMENT" if fold < 4 else "INTERNAL_RESEARCH_CHECK"
            outer_audit.update({"fold": fold, "role": role})
            outer_audits.append(outer_audit)
            nested, audits = nested_folds(timestamps, start_utc, validation_start, train)
            nested_predictions = []
            for nested_number, ((nested_train, nested_validation), audit) in enumerate(
                zip(nested, audits, strict=True), 1
            ):
                audit.update({"outer_fold": fold, "role": role})
                nested_audits.append(audit)
                nested_predictions.append(
                    _fit_predictions(
                        frame,
                        targets,
                        nested_train,
                        nested_validation,
                        preprocessing_audits,
                        f"OUTER_{fold}_NESTED_{nested_number}",
                    )
                )
            selected, grid = _select_configuration(nested_predictions, targets)
            outer_predictions = _fit_predictions(
                frame,
                targets,
                train,
                validation,
                preprocessing_audits,
                f"OUTER_{fold}",
            )
            arrays = economic_arrays(outer_predictions, targets, selected)
            observations, candidates = _records_from_arrays(frame, arrays, fold, role, selected)
            for record, source_index in zip(observations, arrays["indices"], strict=True):
                record["source_index"] = int(source_index)
            observation_records.extend(observations)
            candidate_records.extend(candidates)
            fold_baselines = _baseline_records(frame, targets, arrays, train, fold, selected)
            for name, records in fold_baselines.items():
                baseline_records[name].extend(records)
            outer_results.append(
                {
                    "fold": fold,
                    "role": role,
                    "selected_configuration": selected,
                    "nested_grid": grid,
                    "validation_observations": len(validation),
                    "candidate_metrics": _economic_metrics(candidates, RANDOM_SEED_D13 + fold),
                }
            )

        # Select the single manifest configuration on DESIGN TRAIN only, before reading its check.
        final_nested, final_audits = nested_folds(timestamps, start_utc, design_end, design_indices)
        final_nested_predictions = []
        for nested_number, ((train, validation), audit) in enumerate(
            zip(final_nested, final_audits, strict=True), 1
        ):
            audit.update({"outer_fold": "FINAL_DESIGN", "role": "FREEZE_SELECTION_TRAIN_ONLY"})
            nested_audits.append(audit)
            final_nested_predictions.append(
                _fit_predictions(
                    frame,
                    targets,
                    train,
                    validation,
                    preprocessing_audits,
                    f"FINAL_DESIGN_NESTED_{nested_number}",
                )
            )
        final_selection, final_grid = _select_configuration(final_nested_predictions, targets)
        internal_validation = np.flatnonzero(timestamps >= design_end_us)
        final_predictions = _fit_predictions(
            frame,
            targets,
            design_indices,
            internal_validation,
            preprocessing_audits,
            "FINAL_DESIGN_TO_INTERNAL_RESEARCH_CHECK",
        )
        internal_arrays = economic_arrays(final_predictions, targets, final_selection)
        _, internal_candidates = _records_from_arrays(
            frame,
            internal_arrays,
            4,
            "INTERNAL_RESEARCH_CHECK",
            final_selection,
        )

        research_economics = _economic_metrics(candidate_records, RANDOM_SEED_D13 + 100)
        baseline_metrics = {
            "ALWAYS_SKIP": {
                "candidate_count": 0,
                "base_net_expectancy_points": 0.0,
                "note": "No exposure and no directional prediction.",
            },
            **{
                name: _economic_metrics(records, RANDOM_SEED_D13 + 200 + number)
                for number, (name, records) in enumerate(sorted(baseline_records.items()))
            },
        }
        d10_path = data_root / "reports" / "research-d10-1781049600000-1788825600000.json"
        if d10_path.exists() and "holdout" not in str(d10_path.resolve()).lower():
            d10 = json.loads(d10_path.read_text(encoding="utf-8"))
            baseline_metrics["D10_FROZEN_SYSTEM"] = {
                "original_universe": "FIRST_TICK_PER_ACTIVE_UTC_MINUTE",
                "original_horizon_seconds": 300,
                "directly_comparable": False,
                "decision": d10.get("conclusion"),
                "internal_freeze_check": d10.get("internal_freeze_check", {}).get(
                    "primary_candidates"
                ),
            }
        diagnostics = _model_diagnostics(observation_records, targets)
        return_baseline = baseline_metrics["CONTRARIAN_RETURN_5S"]["base_net_expectancy_points"]
        run_baseline = baseline_metrics["CONTRARIAN_RUN_LENGTH"]["base_net_expectancy_points"]
        v4_expectancy = research_economics["base_net_expectancy_points"]
        lower_ci = research_economics["base_expectancy_block_bootstrap_ci95"]["lower"]
        final_ridge_valid = (
            final_selection["direction_model"] != "RIDGE" or final_predictions["ridge_valid"]
        )
        checks = {
            "minimum_candidates": research_economics["candidate_count"] >= 150,
            "minimum_active_weeks": research_economics["active_weeks"] >= 8,
            "positive_base_expectancy": v4_expectancy is not None and v4_expectancy > 0,
            "positive_lower_95_ci_base": lower_ci is not None and lower_ci > 0,
            "nonnegative_stress_expectancy": research_economics["stress_net_expectancy_points"]
            is not None
            and research_economics["stress_net_expectancy_points"] >= 0,
            "profit_factor_base_gte_1_15": research_economics["profit_factor_base"] is not None
            and research_economics["profit_factor_base"] >= 1.15,
            "positive_week_fraction_gte_0_60": research_economics["positive_week_fraction"]
            is not None
            and research_economics["positive_week_fraction"] >= 0.60,
            "largest_session_share_lte_0_60": research_economics["largest_session_share"]
            is not None
            and research_economics["largest_session_share"] <= 0.60,
            "direction_better_than_return5_contrarian": v4_expectancy is not None
            and return_baseline is not None
            and v4_expectancy > return_baseline,
            "direction_better_than_run_length_contrarian": v4_expectancy is not None
            and run_baseline is not None
            and v4_expectancy > run_baseline,
            "regime_pr_auc_better_than_base_rate": diagnostics["regime"]["pr_auc"]
            > diagnostics["regime"]["unconditional_base_rate"],
            "ridge_numeric_checks": final_ridge_valid,
            "all_purge_invariants": all(item["invariant_passed"] for item in outer_audits)
            and all(item["invariant_passed"] for item in nested_audits),
            "final_nested_selection_eligible": bool(final_selection["selection_eligible"]),
        }
        conclusion: Literal["V4_NOT_READY_TO_FREEZE", "V4_FREEZE_READY"] = (
            "V4_FREEZE_READY" if all(checks.values()) else "V4_NOT_READY_TO_FREEZE"
        )

        output = data_root / "evaluations" / "research-d13-20260610T000000Z-20260908T000000Z"
        output.mkdir(parents=True, exist_ok=True)
        predictions_path = output / "outer-oof-predictions.parquet"
        pl.DataFrame(observation_records, infer_schema_length=None).write_parquet(
            predictions_path, compression="zstd"
        )
        features_source = Path(__file__).resolve().parent / "research_d12.py"
        manifest = {
            "name": "V4_FREEZE_MANIFEST",
            "status": (
                "READY_FOR_EXPLICIT_OOS_AUTHORIZATION"
                if conclusion == "V4_FREEZE_READY"
                else "LOCKED_RESEARCH_SNAPSHOT_NOT_AUTHORIZED_FOR_HOLDOUT"
            ),
            "research_verdict": conclusion,
            "event_universe": {
                "primary": ["MINUTE_BASELINE", "ACCELERATION_EVENT"],
                "deduplication": "TIMESTAMP",
                "diagnostic_only": [
                    "VOLATILITY_EVENT",
                    "BREAKOUT_EVENT",
                    "COMPRESSION_RELEASE_EVENT",
                ],
            },
            "feature_list": {
                "regime": final_regime_reduction["retained"],
                "direction": final_direction_reduction["retained"],
            },
            "feature_formulas_source_hash": _sha256(features_source),
            "redundancy_removals": {
                "regime": final_regime_reduction["removed"],
                "direction": final_direction_reduction["removed"],
            },
            "horizons": {"regime_seconds": 60, "direction_seconds": 30},
            "targets": {
                "regime": "FUTURE_TRADABLE_MOVEMENT_GT_2_TIMES_BASE_COST",
                "direction": "EXPECTED_SIGNED_EXECUTABLE_RETURN_POINTS",
            },
            "model_families": {
                "regime": final_selection["regime_model"],
                "direction": final_selection["direction_model"],
            },
            "hyperparameters": {
                "LOGISTIC_L2": {"C": 0.1, "max_iter": 500},
                "HGB_CLASSIFIER": {"max_iter": 75, "max_depth": 3, "learning_rate": 0.05},
                "RIDGE": {"alpha": 10.0},
                "HGB_REGRESSOR": {"max_iter": 75, "max_depth": 3, "learning_rate": 0.05},
                "random_seed": RANDOM_SEED_D13,
            },
            "preprocessing": {
                "imputation": "TRAIN_MEDIAN",
                "clipping": "TRAIN_QUANTILES_0_005_0_995",
                "scaling": "TRAIN_STANDARD_SCALER",
                "standardized_clip": [-10.0, 10.0],
                "quasi_singular_scale_floor": 1e-8,
            },
            "thresholds": {"regime_probability": final_selection["regime_threshold"]},
            "probability_calibration": {
                "method": "NATIVE_NO_POSTHOC",
                "bounds": [1e-6, 0.999999],
            },
            "costs": {
                "profile": "SIMULATED_RESEARCH_PROFILE_NOT_METAQUOTES_DEMO",
                "BASE": {"slippage_points_per_side": 1, "commission_eur_per_lot_side": 2},
                "STRESS": {"slippage_points_per_side": 2, "commission_eur_per_lot_side": 4},
                "spread": "EXECUTABLE_BID_ASK_AND_OBSERVED_AT_T_FOR_PREDICTED_EDGE",
            },
            "uncertainty_buffer_points": final_selection["uncertainty_buffer_points"],
            "candidate_rule": "REGIME_PASS_AND_MAX_SIDE_PREDICTED_BASE_EDGE_GT_BUFFER",
            "acceptance_criteria": {
                "minimum_candidates": 150,
                "minimum_active_weeks": 8,
                "base_expectancy_gt": 0,
                "lower_ci_gt": 0,
                "stress_expectancy_gte": 0,
                "profit_factor_gte": 1.15,
                "positive_week_fraction_gte": 0.60,
                "largest_session_share_lte": 0.60,
                "beats_simple_contrarian_baselines": True,
                "regime_beats_base_rate": True,
            },
            "holdout_state": "SEALED_NOT_OPENED_NOT_EVALUATED",
            "research_performance_only": True,
            "phase_e_started": False,
            "live_trading_enabled": False,
        }
        manifest["freeze_manifest_sha256"] = _freeze_hash(manifest)
        manifest_path = output / "V4_FREEZE_MANIFEST.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
        )
        peak_memory = tracemalloc.get_traced_memory()[1] / (1024**2)
        tracemalloc.stop()
        engine_hash = _sha256(Path(__file__))
        protocol_hash = _sha256(protocol_path)
        return ResearchD13Report(
            requested_start_utc=start_utc,
            requested_end_utc=end_utc,
            protocol_path=str(protocol_path.resolve()),
            protocol_sha256=protocol_hash,
            source_hashes={
                "research_d13.py": engine_hash,
                "research_protocol_d13.yaml": protocol_hash,
                "d12_events.parquet": _sha256(dataset),
                "d12_feature_formulas": _sha256(features_source),
                "outer_oof_predictions.parquet": _sha256(predictions_path),
            },
            architecture={
                "hierarchy": [
                    "RegimeOpportunityModel",
                    "ContrarianDirectionModel",
                    "EconomicExecutionGate",
                ],
                "regime_primary_target": "MOVE_GT_2_BASE_COST_AT_60S",
                "direction_primary_target": "SIGNED_EXECUTABLE_RETURN_AT_30S",
                "direction_model_is_naturally_signed_not_target_inverted": True,
                "direction_used_economically_only_after_regime_pass": True,
                "research_sides_only": [
                    "RESEARCH_LONG",
                    "RESEARCH_SHORT",
                    "RESEARCH_SKIP",
                ],
            },
            event_universe={
                "source_observations": frame_all.height,
                "primary_union_deduplicated_observations": frame.height,
                "primary": ["MINUTE_BASELINE", "ACCELERATION_EVENT"],
                "diagnostic_only": [
                    "VOLATILITY_EVENT",
                    "BREAKOUT_EVENT",
                    "COMPRESSION_RELEASE_EVENT",
                ],
                "timestamp_duplicates": 0,
            },
            feature_reduction={
                "fit_scope": "FIRST_60_PERCENT_DESIGN_TRAIN_ONLY",
                "regime": final_regime_reduction,
                "direction": final_direction_reduction,
            },
            model_policy={
                "regime_candidates": list(REGIME_MODELS),
                "direction_candidates": list(DIRECTION_MODELS),
                "model_combinations": 4,
                "thresholds": list(REGIME_THRESHOLDS),
                "buffers_points": list(UNCERTAINTY_BUFFERS),
                "configurations": 36,
                "selection": "NESTED_PURGED_COMPOSITE_PRE_REGISTERED",
            },
            preprocessing_audit={
                "fits": preprocessing_audits,
                "all_scaled_values_bounded_by_10": all(
                    item["scaled_train_max_abs"] <= 10.0
                    and item["scaled_validation_max_abs"] <= 10.0
                    for item in preprocessing_audits
                ),
                "all_ridge_coefficients_finite": all(
                    item["ridge_coefficients_finite"] for item in preprocessing_audits
                ),
                "all_ridge_predictions_finite": all(
                    item["ridge_predictions_finite"] for item in preprocessing_audits
                ),
                "all_ridge_mae_sane": all(item["ridge_valid"] for item in preprocessing_audits),
            },
            probability_calibration={
                "method": "NATIVE_NO_POSTHOC",
                "bounds": [1e-6, 0.999999],
                "diagnostics": diagnostics["regime"],
            },
            outer_purge_audit=tuple(outer_audits),
            nested_purge_audit=tuple(nested_audits),
            outer_fold_results=tuple(outer_results),
            nested_research_metrics={
                "final_design_grid": final_grid,
                "selected_on_first_60_percent_only": final_selection,
            },
            selected_configuration=final_selection,
            model_diagnostics=diagnostics,
            baselines=baseline_metrics,
            research_economics={
                **research_economics,
                "scope": "POOLED_OUTER_OOF_RESEARCH_PERFORMANCE_ONLY",
                "cost_decomposition": {
                    "formula": "market_pnl-realized_BidAsk_spread-slippage-commission=BASE_net",
                    "maximum_absolute_crosscheck_error_points": max(
                        abs(row["actual_base_pnl_points"] - row["base_reconstructed_pnl_points"])
                        for row in observation_records
                    )
                    if observation_records
                    else None,
                    "BASE_slippage_round_trip_points": 2.0,
                    "STRESS_slippage_round_trip_points": 4.0,
                    "commission": "DERIVED_FROM_D12_BASE_COST_AND_DOUBLED_FOR_STRESS",
                },
            },
            internal_research_check={
                "configuration_selected_without_internal_check": final_selection,
                "metrics": _economic_metrics(internal_candidates, RANDOM_SEED_D13 + 300),
                "authority": "RESEARCH_ONLY_NOT_FINAL_VALIDATION",
            },
            acceptance={
                "scope": "POOLED_OUTER_OOF_RESEARCH_PERFORMANCE_ONLY",
                "checks": checks,
                "all_passed": all(checks.values()),
                "criteria_changed_after_results": False,
            },
            freeze_manifest_path=str(manifest_path.resolve()),
            freeze_manifest_sha256=cast(str, manifest["freeze_manifest_sha256"]),
            freeze_manifest=manifest,
            future_forward_dataset={
                "id": "FORWARD_V4",
                "start_must_be_strictly_after": expected_end,
                "results_produced_during_d13": False,
                "note": (
                    "Post-development forward validation is temporally stronger than the "
                    "pre-TRAIN historical HOLDOUT."
                ),
            },
            artifacts={
                "source_events": str(dataset),
                "outer_oof_predictions": str(predictions_path.resolve()),
                "freeze_manifest": str(manifest_path.resolve()),
            },
            performance=D13Performance(
                elapsed_seconds=perf_counter() - started,
                source_observations=frame_all.height,
                primary_universe_observations=frame.height,
                outer_oof_observations=len(observation_records),
                peak_python_memory_mb=peak_memory,
            ),
            conclusion=conclusion,
            limitations=(
                "All reported performance is RESEARCH PERFORMANCE ONLY.",
                "The historical HOLDOUT remained sealed and was neither opened nor evaluated.",
                "D.12 already influenced feature and horizon choices, so RESEARCH is not "
                "independent evidence.",
                "No actionable BUY/SELL output, sizing, Phase E or live trading path was created.",
                "V4_FREEZE_READY, if reached, means specification completeness rather than "
                "profitability.",
            ),
        )


def render_research_d13_report(report: ResearchD13Report) -> str:
    economics = report.research_economics
    internal = report.internal_research_check["metrics"]
    return "\n".join(
        [
            "# SNIPER Phase D.13 — Hierarchical V4 Research",
            "",
            f"Verdict : **{report.conclusion}**",
            "",
            "Toutes les performances ci-dessous sont RESEARCH PERFORMANCE ONLY.",
            "",
            "## Architecture gelée ou snapshot verrouillé",
            "",
            "`RegimeOpportunityModel → ContrarianDirectionModel → EconomicExecutionGate`",
            "",
            f"- Features régime : {', '.join(report.feature_reduction['regime']['retained'])}",
            "- Features direction : "
            f"{', '.join(report.feature_reduction['direction']['retained'])}",
            f"- Modèle régime : {report.selected_configuration['regime_model']}",
            f"- Modèle direction : {report.selected_configuration['direction_model']}",
            f"- Seuil régime : {report.selected_configuration['regime_threshold']}",
            f"- Buffer : {report.selected_configuration['uncertainty_buffer_points']} points",
            "",
            "## Économie OOF RESEARCH",
            "",
            f"- Candidats : {economics['candidate_count']}",
            f"- Expectancy BASE : {economics['base_net_expectancy_points']}",
            f"- IC 95 % BASE : {economics['base_expectancy_block_bootstrap_ci95']}",
            f"- Expectancy STRESS : {economics['stress_net_expectancy_points']}",
            f"- Profit factor BASE : {economics['profit_factor_base']}",
            f"- Fraction semaines positives : {economics['positive_week_fraction']}",
            f"- Part plus grande session : {economics['largest_session_share']}",
            "",
            "## Contrôle interne RESEARCH",
            "",
            f"- Candidats : {internal['candidate_count']}",
            f"- Expectancy BASE : {internal['base_net_expectancy_points']}",
            f"- Expectancy STRESS : {internal['stress_net_expectancy_points']}",
            "",
            "## Garde-fous",
            "",
            "- HOLDOUT SEALED / NOT OPENED / NOT EVALUATED",
            "- aucun BUY/SELL actionnable, aucun sizing",
            "- Phase E NOT STARTED ; LIVE DISABLED",
            f"- Freeze manifest SHA-256 : `{report.freeze_manifest_sha256}`",
            "",
        ]
    )
