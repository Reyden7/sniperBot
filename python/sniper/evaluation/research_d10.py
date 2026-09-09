"""Phase D.10 three-outcome expected-value research on the RESEARCH dataset only."""

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
from sklearn.ensemble import (  # type: ignore[import-untyped]
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import (  # type: ignore[import-untyped]
    ElasticNet,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

from sniper.backtest.execution_model import BrokerSimulationConfig
from sniper.data.time import as_utc
from sniper.domain.edge_validation import EdgeSimulationAssumptions
from sniper.domain.research_d10 import D10Conclusion, D10Performance, ResearchD10Report
from sniper.domain.research_v3 import BarrierConfiguration
from sniper.evaluation.edge_validation import MICROSECONDS
from sniper.evaluation.research_v3 import (
    BARRIER_CONFIGURATIONS,
    FEATURE_DEFINITIONS,
    V3_FEATURES,
    _pipeline,
    _prefix,
)

RANDOM_SEED_D10 = 20260910
OUTCOMES = ("TARGET_FIRST", "STOP_FIRST", "NEITHER")
OUTCOME_TO_INT = {name: index for index, name in enumerate(OUTCOMES)}
BOOTSTRAP_REPLICATIONS = 2000
PRIMARY_CONFIGURATION_ID = "B02_PRIMARY"
SOURCE_TICKS = 17_941_326


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def build_d10_freeze_manifest(protocol_path: Path) -> dict[str, Any]:
    """Return the complete decision manifest to persist before any D.10 fit."""
    source_path = Path(__file__)
    v3_source_path = source_path.with_name("research_v3.py")
    manifest: dict[str, Any] = {
        "status": "FROZEN_BEFORE_FIRST_D10_TRAINING",
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256_file(protocol_path),
        "features": list(V3_FEATURES),
        "feature_definitions": FEATURE_DEFINITIONS,
        "features_sha256": _sha256_bytes(
            json.dumps(list(V3_FEATURES), separators=(",", ":")).encode()
        ),
        "primary_configuration": {
            "id": "B02_PRIMARY",
            "stop_atr": 1.0,
            "target_atr": 2.0,
            "minimum_target_to_base_cost": 2.0,
            "timeout_seconds": 300,
        },
        "primary_model": {
            "family": "MULTINOMIAL_LOGISTIC_REGRESSION",
            "C": 1.0,
            "penalty": "L2",
            "max_iter": 500,
            "random_state": RANDOM_SEED_D10,
        },
        "calibration": "NATIVE_SOFTMAX_NO_POSTHOC",
        "horizon_seconds": 300,
        "cost_scenario": "BASE",
        "ev_formula": "SUM(P_OUTCOME * TRAIN_MEAN_BASE_PNL_GIVEN_OUTCOME)",
        "safety_buffer": ("max(0,-mean_daily_residual+1.645*sd_daily_residual/sqrt(D))"),
        "candidate_rule": "predicted_EV - safety_buffer > 0; max side else SKIP",
        "acceptance_criteria": {
            "minimum_candidates": 100,
            "minimum_active_weeks": 6,
            "base_net_expectancy_gt": 0,
            "lower_95_ci_base_expectancy_gt": 0,
            "stress_net_expectancy_gte": 0,
            "profit_factor_base_gte": 1.10,
            "positive_week_fraction_gte": 0.60,
            "largest_session_share_lte": 0.60,
        },
        "source_hashes": {
            "evaluation/research_d10.py": _sha256_file(source_path),
            "evaluation/research_v3.py": _sha256_file(v3_source_path),
        },
        "holdout_access_permitted": False,
        "phase_e_started": False,
        "live_trading_enabled": False,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["freeze_manifest_sha256"] = _sha256_bytes(canonical)
    return manifest


def _validate_research_scope(data_root: Path, start: datetime, end: datetime) -> Path:
    resolved_root = data_root.resolve()
    if any("holdout" in part.lower() for part in resolved_root.parts):
        raise PermissionError("D.10 refuses any HOLDOUT-rooted data path")
    if start != datetime(2026, 6, 10, tzinfo=UTC) or end != datetime(2026, 9, 8, tzinfo=UTC):
        raise PermissionError("D.10 may only open the declared RESEARCH interval")
    dataset = (
        resolved_root
        / "evaluations"
        / "research-v3-20260610T000000Z-20260908T000000Z"
        / "observations.parquet"
    )
    expected_parent = (resolved_root / "evaluations").resolve()
    if dataset.parent.parent.resolve() != expected_parent:
        raise PermissionError("D.10 dataset escaped the RESEARCH evaluations root")
    if not dataset.exists():
        raise ValueError("Purged V3 RESEARCH observations are required before D.10")
    return dataset


def _primary_configuration() -> BarrierConfiguration:
    return next(item for item in BARRIER_CONFIGURATIONS if item.id == PRIMARY_CONFIGURATION_ID)


def d10_temporal_folds(
    frame: pl.DataFrame, start: datetime, end: datetime
) -> tuple[list[dict[str, Any]], tuple[dict[str, Any], ...]]:
    """Create three development folds and the untouched 40% freeze-check window."""
    duration = end - start
    boundaries = [start + duration * fraction for fraction in (0.3, 0.4, 0.5, 0.6, 1.0)]
    timestamps_us = frame["timestamp_utc"].cast(pl.Int64).to_numpy().astype(np.int64)
    configuration = _primary_configuration()
    valid = (
        frame[f"{_prefix(configuration, 'LONG')}_complete"].fill_null(False).to_numpy()
        & frame[f"{_prefix(configuration, 'SHORT')}_complete"].fill_null(False).to_numpy()
    ).astype(bool)
    folds: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for index in range(4):
        validation_start = boundaries[index]
        validation_end = boundaries[index + 1]
        validation_start_us = int(validation_start.timestamp() * MICROSECONDS)
        validation_end_us = int(validation_end.timestamp() * MICROSECONDS)
        raw_train = np.flatnonzero((timestamps_us < validation_start_us) & valid)
        train = raw_train[
            timestamps_us[raw_train] + configuration.timeout_seconds * MICROSECONDS
            <= validation_start_us
        ]
        validation = np.flatnonzero(
            (timestamps_us >= validation_start_us) & (timestamps_us < validation_end_us) & valid
        )
        max_label_end = (
            int(np.max(timestamps_us[train])) + configuration.timeout_seconds * MICROSECONDS
            if len(train)
            else None
        )
        invariant = max_label_end is None or max_label_end <= validation_start_us
        if not invariant:
            raise RuntimeError("D.10 outer purge invariant failed")
        role = "MODEL_DEVELOPMENT" if index < 3 else "INTERNAL_FREEZE_CHECK"
        folds.append(
            {
                "fold": index + 1,
                "role": role,
                "train_indices": train,
                "validation_indices": validation,
                "validation_start_utc": validation_start,
                "validation_end_utc": validation_end,
            }
        )
        audit.append(
            {
                "fold": index + 1,
                "role": role,
                "timeout_seconds": configuration.timeout_seconds,
                "validation_start_utc": validation_start,
                "original_train_observations": len(raw_train),
                "purged_observations": len(raw_train) - len(train),
                "retained_train_observations": len(train),
                "validation_observations": len(validation),
                "max_label_end_time_train_utc": (
                    datetime.fromtimestamp(max_label_end / MICROSECONDS, UTC)
                    if max_label_end is not None
                    else None
                ),
                "invariant_passed": invariant,
            }
        )
    return folds, tuple(audit)


def _nested_folds(
    frame: pl.DataFrame,
    outer_train: np.ndarray,
    start: datetime,
    outer_validation_start: datetime,
    outer_fold: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], tuple[dict[str, Any], ...]]:
    timestamps_us = frame["timestamp_utc"].cast(pl.Int64).to_numpy().astype(np.int64)
    span = outer_validation_start - start
    boundaries = [start + span * fraction for fraction in (0.4, 0.6, 0.8, 1.0)]
    universe = np.zeros(frame.height, dtype=bool)
    universe[outer_train] = True
    nested: list[tuple[np.ndarray, np.ndarray]] = []
    audits: list[dict[str, Any]] = []
    for nested_index in range(3):
        validation_start = boundaries[nested_index]
        validation_end = boundaries[nested_index + 1]
        validation_start_us = int(validation_start.timestamp() * MICROSECONDS)
        validation_end_us = int(validation_end.timestamp() * MICROSECONDS)
        raw_train = np.flatnonzero((timestamps_us < validation_start_us) & universe)
        train = raw_train[timestamps_us[raw_train] + 300 * MICROSECONDS <= validation_start_us]
        validation = np.flatnonzero(
            (timestamps_us >= validation_start_us) & (timestamps_us < validation_end_us) & universe
        )
        max_label_end = int(np.max(timestamps_us[train])) + 300 * MICROSECONDS
        invariant = max_label_end <= validation_start_us
        if not invariant:
            raise RuntimeError("D.10 nested purge invariant failed")
        nested.append((train, validation))
        audits.append(
            {
                "outer_fold": outer_fold,
                "nested_fold": nested_index + 1,
                "validation_start_utc": validation_start,
                "original_train_observations": len(raw_train),
                "purged_observations": len(raw_train) - len(train),
                "retained_train_observations": len(train),
                "validation_observations": len(validation),
                "max_label_end_time_train_utc": datetime.fromtimestamp(
                    max_label_end / MICROSECONDS, UTC
                ),
                "invariant_passed": invariant,
            }
        )
    return nested, tuple(audits)


def _encoded_outcomes(frame: pl.DataFrame, prefix: str) -> np.ndarray:
    return np.array(
        [OUTCOME_TO_INT.get(str(value), -1) for value in frame[f"{prefix}_outcome"].to_numpy()],
        dtype=int,
    )


def _probabilities(model: Pipeline, x: Any) -> np.ndarray:
    raw = model.predict_proba(x)
    classes = cast(np.ndarray, model.named_steps["model"].classes_).astype(int)
    ordered = np.zeros((len(x), len(OUTCOMES)), dtype=float)
    ordered[:, classes] = raw
    if not np.all(np.isfinite(ordered)) or not np.allclose(ordered.sum(axis=1), 1.0, atol=1e-10):
        raise RuntimeError("D.10 multiclass probabilities do not sum to one")
    return ordered


def _calibration(
    y: np.ndarray, probabilities: np.ndarray, class_index: int
) -> list[dict[str, Any]]:
    actual = (y == class_index).astype(float)
    probability = probabilities[:, class_index]
    rows = []
    for bin_index in range(10):
        lower = bin_index / 10
        upper = (bin_index + 1) / 10
        mask = (probability >= lower) & (
            probability <= upper if bin_index == 9 else probability < upper
        )
        if np.any(mask):
            rows.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "observations": int(np.sum(mask)),
                    "mean_probability": float(np.mean(probability[mask])),
                    "observed_rate": float(np.mean(actual[mask])),
                }
            )
    return rows


def _multiclass_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    one_hot = np.eye(len(OUTCOMES))[y]
    prediction = np.argmax(probabilities, axis=1)
    counts = {name: int(np.sum(y == index)) for index, name in enumerate(OUTCOMES)}
    return {
        "observations": len(y),
        "class_distribution": {
            name: {"count": counts[name], "proportion": counts[name] / len(y)} for name in OUTCOMES
        },
        "accuracy": float(accuracy_score(y, prediction)),
        "multiclass_log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2])),
        "multiclass_brier": float(np.mean(np.sum((one_hot - probabilities) ** 2, axis=1))),
        "probability_sum_max_abs_error": float(np.max(np.abs(probabilities.sum(axis=1) - 1.0))),
        "calibration": {
            name: _calibration(y, probabilities, index) for index, name in enumerate(OUTCOMES)
        },
    }


def _payoff_means(
    frame: pl.DataFrame, train: np.ndarray, prefix: str
) -> tuple[np.ndarray, dict[str, Any]]:
    outcomes = frame[f"{prefix}_outcome"].to_numpy()[train]
    pnl = frame[f"{prefix}_base_net_points"].to_numpy().astype(float)[train]
    fallback = float(np.mean(pnl))
    values = []
    report: dict[str, Any] = {}
    for outcome in OUTCOMES:
        selected = pnl[outcomes == outcome]
        missing = len(selected) == 0
        mean = fallback if missing else float(np.mean(selected))
        values.append(mean)
        report[outcome] = {
            "observations": len(selected),
            "mean_BASE_net_pnl_points": mean,
            "used_missing_class_fallback": missing,
        }
    return np.array(values, dtype=float), report


def _primary_classifier() -> Pipeline:
    return _pipeline(
        LogisticRegression(C=1.0, penalty="l2", max_iter=500, random_state=RANDOM_SEED_D10)
    )


def _hgb_classifier() -> Pipeline:
    return _pipeline(
        HistGradientBoostingClassifier(
            max_iter=50,
            max_depth=3,
            learning_rate=0.08,
            random_state=RANDOM_SEED_D10,
        )
    )


def _daily_safety_buffer(
    frame: pl.DataFrame,
    x_all: Any,
    y_all: np.ndarray,
    prefix: str,
    nested_folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, dict[str, Any]]:
    residuals: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    actual_all = frame[f"{prefix}_base_net_points"].to_numpy().astype(float)
    for train, validation in nested_folds:
        model = _primary_classifier()
        model.fit(x_all.iloc[train], y_all[train])
        probabilities = _probabilities(model, x_all.iloc[validation])
        payoffs, _ = _payoff_means(frame, train, prefix)
        predicted_ev = probabilities @ payoffs
        residuals.append(actual_all[validation] - predicted_ev)
        indices.append(validation)
    combined_residual = np.concatenate(residuals)
    combined_indices = np.concatenate(indices)
    days = frame[combined_indices]["timestamp_utc"].dt.strftime("%Y-%m-%d").to_numpy()
    daily = np.array(
        [np.mean(combined_residual[days == day]) for day in np.unique(days)], dtype=float
    )
    standard_error = float(np.std(daily, ddof=1) / math.sqrt(len(daily))) if len(daily) > 1 else 0.0
    mean_residual = float(np.mean(daily))
    buffer = max(0.0, -mean_residual + 1.645 * standard_error)
    return buffer, {
        "nested_oof_observations": len(combined_residual),
        "utc_days": len(daily),
        "mean_daily_residual_points": mean_residual,
        "daily_residual_standard_error_points": standard_error,
        "one_sided_z": 1.645,
        "safety_buffer_points": buffer,
    }


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    return {
        "observations": len(actual),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "mean_error_actual_minus_predicted": float(np.mean(actual - predicted)),
        "predicted_mean": float(np.mean(predicted)),
        "actual_mean": float(np.mean(actual)),
    }


def _fit_side(
    frame: pl.DataFrame,
    x_all: Any,
    train: np.ndarray,
    validation: np.ndarray,
    nested: list[tuple[np.ndarray, np.ndarray]],
    side: Literal["LONG", "SHORT"],
) -> dict[str, Any]:
    configuration = _primary_configuration()
    prefix = _prefix(configuration, side)
    y_all = _encoded_outcomes(frame, prefix)
    primary = _primary_classifier()
    challenger = _hgb_classifier()
    primary.fit(x_all.iloc[train], y_all[train])
    challenger.fit(x_all.iloc[train], y_all[train])
    primary_probability = _probabilities(primary, x_all.iloc[validation])
    challenger_probability = _probabilities(challenger, x_all.iloc[validation])
    payoffs, payoff_report = _payoff_means(frame, train, prefix)
    predicted_ev = primary_probability @ payoffs
    buffer, buffer_report = _daily_safety_buffer(frame, x_all, y_all, prefix, nested)
    actual_base = frame[f"{prefix}_base_net_points"].to_numpy().astype(float)
    direct_predictions: dict[str, np.ndarray] = {
        "train_mean": np.full(len(validation), float(np.mean(actual_base[train])))
    }
    direct_models: dict[str, Any] = {
        "ridge": Ridge(alpha=1.0),
        "elastic_net": ElasticNet(
            alpha=0.001, l1_ratio=0.5, max_iter=2000, random_state=RANDOM_SEED_D10
        ),
        "hist_gradient_boosting_regressor": HistGradientBoostingRegressor(
            max_iter=50,
            max_depth=3,
            learning_rate=0.08,
            random_state=RANDOM_SEED_D10,
        ),
    }
    for name, estimator in direct_models.items():
        model = _pipeline(estimator)
        model.fit(x_all.iloc[train], actual_base[train])
        direct_predictions[name] = model.predict(x_all.iloc[validation])
    binary = _pipeline(LogisticRegression(C=1.0, max_iter=500, random_state=20260909))
    binary.fit(x_all.iloc[train], (y_all[train] == 0).astype(int))
    v3_probability = binary.predict_proba(x_all.iloc[validation])[:, 1]
    return {
        "side": side,
        "primary_probability": primary_probability,
        "challenger_probability": challenger_probability,
        "predicted_ev": predicted_ev,
        "safety_buffer": buffer,
        "conservative_ev": predicted_ev - buffer,
        "v3_probability": v3_probability,
        "y": y_all[validation],
        "payoffs": payoff_report,
        "buffer_report": buffer_report,
        "primary_metrics": _multiclass_metrics(y_all[validation], primary_probability),
        "challenger_metrics": _multiclass_metrics(y_all[validation], challenger_probability),
        "direct_predictions": direct_predictions,
        "direct_metrics": {
            name: _regression_metrics(actual_base[validation], prediction)
            for name, prediction in direct_predictions.items()
        },
    }


def _candidate_frame(
    frame: pl.DataFrame,
    indices: np.ndarray,
    candidate_mask: np.ndarray,
    choose_long: np.ndarray,
    *,
    fold: int,
    role: str,
    predicted_ev: np.ndarray | None = None,
    conservative_ev: np.ndarray | None = None,
) -> pl.DataFrame:
    positions = np.flatnonzero(candidate_mask)
    if not len(positions):
        return pl.DataFrame()
    scoped = frame[indices]
    selected = scoped[positions]
    selected_long = choose_long[positions]
    configuration = _primary_configuration()
    long_prefix = _prefix(configuration, "LONG")
    short_prefix = _prefix(configuration, "SHORT")

    def values(suffix: str) -> np.ndarray:
        return np.where(
            selected_long,
            selected[f"{long_prefix}_{suffix}"].to_numpy(),
            selected[f"{short_prefix}_{suffix}"].to_numpy(),
        )

    timestamps = selected["timestamp_utc"]
    data: dict[str, Any] = {
        "timestamp_utc": timestamps,
        "day": timestamps.dt.strftime("%Y-%m-%d"),
        "week": selected["week"],
        "session": selected["session"],
        "fold": np.full(len(positions), fold),
        "role": np.full(len(positions), role),
        "side": np.where(selected_long, "LONG", "SHORT"),
        "gross": values("gross_return_points").astype(float),
        "executable": values("executable_return_points").astype(float),
        "base": values("base_net_points").astype(float),
        "stress": values("stress_net_points").astype(float),
        "outcome": values("outcome").astype(str),
        "mfe": values("mfe_points").astype(float),
        "mae": values("mae_points").astype(float),
    }
    if predicted_ev is not None:
        data["predicted_ev"] = predicted_ev[positions]
    if conservative_ev is not None:
        data["conservative_ev"] = conservative_ev[positions]
    return pl.DataFrame(data)


def _distribution(frame: pl.DataFrame, column: str) -> tuple[dict[str, Any], ...]:
    if frame.is_empty():
        return ()
    rows = []
    for value in sorted(frame[column].unique().to_list()):
        group = frame.filter(pl.col(column) == value)
        rows.append(
            {
                column: str(value),
                "candidates": group.height,
                "BASE_expectancy_points": float(str(group["base"].mean())),
                "BASE_total_points": float(str(group["base"].sum())),
                "STRESS_expectancy_points": float(str(group["stress"].mean())),
            }
        )
    return tuple(rows)


def _block_bootstrap(frame: pl.DataFrame, seed_offset: int) -> dict[str, Any]:
    if frame.is_empty():
        return {
            name: {"lower": None, "upper": None}
            for name in ("BASE_expectancy", "STRESS_expectancy", "profit_factor_BASE", "win_rate")
        }
    days = frame["day"].unique(maintain_order=True).to_list()
    groups = [frame.filter(pl.col("day") == day) for day in days]
    rng = np.random.default_rng(RANDOM_SEED_D10 + seed_offset)
    samples: dict[str, list[float]] = {
        "BASE_expectancy": [],
        "STRESS_expectancy": [],
        "profit_factor_BASE": [],
        "win_rate": [],
    }
    for _ in range(BOOTSTRAP_REPLICATIONS):
        picks = rng.integers(0, len(groups), size=len(groups))
        base = np.concatenate([groups[index]["base"].to_numpy() for index in picks])
        stress = np.concatenate([groups[index]["stress"].to_numpy() for index in picks])
        positive = float(np.sum(base[base > 0]))
        negative = float(-np.sum(base[base < 0]))
        samples["BASE_expectancy"].append(float(np.mean(base)))
        samples["STRESS_expectancy"].append(float(np.mean(stress)))
        samples["win_rate"].append(float(np.mean(base > 0)))
        if negative > 0:
            samples["profit_factor_BASE"].append(positive / negative)
    return {
        name: {
            "lower": float(np.quantile(values, 0.025)) if values else None,
            "upper": float(np.quantile(values, 0.975)) if values else None,
            "replications": len(values),
            "block": "UTC_DAY",
            "seed": RANDOM_SEED_D10 + seed_offset,
        }
        for name, values in samples.items()
    }


def _economic_metrics(frame: pl.DataFrame, seed_offset: int) -> dict[str, Any]:
    if frame.is_empty():
        return {
            "candidate_count": 0,
            "LONG_count": 0,
            "SHORT_count": 0,
            "gross_expectancy_points": None,
            "executable_expectancy_points": None,
            "BASE_net_expectancy_points": None,
            "STRESS_net_expectancy_points": None,
            "median_BASE_pnl_points": None,
            "profit_factor_BASE": None,
            "win_rate": None,
            "outcome_proportions": {name: None for name in OUTCOMES},
            "average_MFE_points": None,
            "average_MAE_points": None,
            "maximum_drawdown_BASE_points": None,
            "active_weeks": 0,
            "positive_week_fraction": None,
            "largest_session_share": None,
            "weekly_distribution": (),
            "session_distribution": (),
            "confidence_intervals_95": _block_bootstrap(frame, seed_offset),
        }
    ordered = frame.sort("timestamp_utc")
    base = ordered["base"].to_numpy().astype(float)
    cumulative = np.cumsum(base)
    peaks = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))
    drawdowns = peaks[1:] - cumulative
    positive = float(np.sum(base[base > 0]))
    negative = float(-np.sum(base[base < 0]))
    weekly = _distribution(ordered, "week")
    sessions = _distribution(ordered, "session")
    positive_weeks = sum(cast(float, item["BASE_total_points"]) > 0 for item in weekly)
    return {
        "candidate_count": ordered.height,
        "LONG_count": int((ordered["side"] == "LONG").sum()),
        "SHORT_count": int((ordered["side"] == "SHORT").sum()),
        "gross_expectancy_points": float(np.mean(ordered["gross"].to_numpy())),
        "executable_expectancy_points": float(np.mean(ordered["executable"].to_numpy())),
        "BASE_net_expectancy_points": float(np.mean(base)),
        "STRESS_net_expectancy_points": float(np.mean(ordered["stress"].to_numpy())),
        "median_BASE_pnl_points": float(np.median(base)),
        "profit_factor_BASE": positive / negative if negative > 0 else None,
        "win_rate": float(np.mean(base > 0)),
        "outcome_proportions": {
            name: float(np.mean(ordered["outcome"].to_numpy() == name)) for name in OUTCOMES
        },
        "average_MFE_points": float(np.mean(ordered["mfe"].to_numpy())),
        "average_MAE_points": float(np.mean(ordered["mae"].to_numpy())),
        "maximum_drawdown_BASE_points": float(np.max(drawdowns)),
        "active_weeks": len(weekly),
        "positive_week_fraction": positive_weeks / len(weekly),
        "largest_session_share": max(cast(int, item["candidates"]) for item in sessions)
        / ordered.height,
        "weekly_distribution": weekly,
        "session_distribution": sessions,
        "confidence_intervals_95": _block_bootstrap(ordered, seed_offset),
    }


def _aggregate_multiclass(parts: list[dict[str, Any]], model_key: str) -> dict[str, Any]:
    y = np.concatenate([part["y"] for part in parts])
    probability = np.concatenate([part[model_key] for part in parts])
    return _multiclass_metrics(y, probability)


def _aggregate_direct(
    parts: list[dict[str, Any]], frame: pl.DataFrame, side: str
) -> dict[str, Any]:
    configuration = _primary_configuration()
    prefix = _prefix(configuration, side)
    actual = np.concatenate(
        [
            frame[part["validation_indices"]][f"{prefix}_base_net_points"].to_numpy()
            for part in parts
        ]
    ).astype(float)
    names = tuple(parts[0][side]["direct_predictions"])
    return {
        name: _regression_metrics(
            actual,
            np.concatenate([part[side]["direct_predictions"][name] for part in parts]),
        )
        for name in names
    }


def _scope_report(
    parts: list[dict[str, Any]],
    candidate_frames: dict[str, list[pl.DataFrame]],
    frame: pl.DataFrame,
    seed_offset: int,
) -> dict[str, Any]:
    def concat(name: str) -> pl.DataFrame:
        values = [value for value in candidate_frames[name] if not value.is_empty()]
        return pl.concat(values, how="vertical") if values else pl.DataFrame()

    return {
        "folds": [part["fold"] for part in parts],
        "probabilistic_models": {
            side: {
                "primary_multinomial_logistic": _aggregate_multiclass(
                    [part[side] for part in parts], "primary_probability"
                ),
                "challenger_hist_gradient_boosting": _aggregate_multiclass(
                    [part[side] for part in parts], "challenger_probability"
                ),
            }
            for side in ("LONG", "SHORT")
        },
        "direct_pnl_challengers": {
            side: _aggregate_direct(parts, frame, side) for side in ("LONG", "SHORT")
        },
        "primary_candidates": _economic_metrics(concat("D10_PRIMARY"), seed_offset),
        "baselines": {
            name: _economic_metrics(concat(name), seed_offset + index + 1)
            for index, name in enumerate(
                ("ALWAYS_SKIP", "RANDOM_SIDE", "UNCONDITIONAL_TRAIN_EV", "V3_METAGATE")
            )
        },
    }


class ResearchD10Engine:
    def run(
        self,
        *,
        data_root: Path,
        start_utc: datetime,
        end_utc: datetime,
        protocol_path: Path,
        freeze_manifest_path: Path,
    ) -> ResearchD10Report:
        start, end = as_utc(start_utc), as_utc(end_utc)
        dataset_path = _validate_research_scope(data_root, start, end)
        manifest = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
        expected_manifest = build_d10_freeze_manifest(protocol_path)
        if manifest != expected_manifest:
            raise RuntimeError("D.10 freeze manifest differs from the pre-training specification")
        tracemalloc.start()
        started = perf_counter()
        frame = pl.read_parquet(dataset_path).sort("timestamp_utc")
        x_all = frame.select(V3_FEATURES).to_pandas()
        folds, outer_audit = d10_temporal_folds(frame, start, end)
        fold_parts: list[dict[str, Any]] = []
        nested_audit: list[dict[str, Any]] = []
        frames_by_role: dict[str, dict[str, list[pl.DataFrame]]] = {
            role: {
                name: []
                for name in (
                    "D10_PRIMARY",
                    "ALWAYS_SKIP",
                    "RANDOM_SIDE",
                    "UNCONDITIONAL_TRAIN_EV",
                    "V3_METAGATE",
                )
            }
            for role in ("MODEL_DEVELOPMENT", "INTERNAL_FREEZE_CHECK")
        }
        configuration = _primary_configuration()
        for fold in folds:
            fold_number = cast(int, fold["fold"])
            role = cast(str, fold["role"])
            train = cast(np.ndarray, fold["train_indices"])
            validation = cast(np.ndarray, fold["validation_indices"])
            nested, audits = _nested_folds(
                frame,
                train,
                start,
                cast(datetime, fold["validation_start_utc"]),
                fold_number,
            )
            nested_audit.extend(audits)
            long = _fit_side(frame, x_all, train, validation, nested, "LONG")
            short = _fit_side(frame, x_all, train, validation, nested, "SHORT")
            long_conservative = cast(np.ndarray, long["conservative_ev"])
            short_conservative = cast(np.ndarray, short["conservative_ev"])
            long_eligible = long_conservative > 0
            short_eligible = short_conservative > 0
            choose_long = long_eligible & (
                ~short_eligible | (long_conservative >= short_conservative)
            )
            choose_short = short_eligible & ~choose_long
            candidate = choose_long | choose_short
            predicted = np.where(choose_long, long["predicted_ev"], short["predicted_ev"])
            conservative = np.where(choose_long, long_conservative, short_conservative)
            frames_by_role[role]["D10_PRIMARY"].append(
                _candidate_frame(
                    frame,
                    validation,
                    candidate,
                    choose_long,
                    fold=fold_number,
                    role=role,
                    predicted_ev=predicted,
                    conservative_ev=conservative,
                )
            )
            frames_by_role[role]["ALWAYS_SKIP"].append(pl.DataFrame())
            random_choose_long = (
                np.random.default_rng(RANDOM_SEED_D10 + fold_number).random(len(validation)) < 0.5
            )
            frames_by_role[role]["RANDOM_SIDE"].append(
                _candidate_frame(
                    frame,
                    validation,
                    np.ones(len(validation), dtype=bool),
                    random_choose_long,
                    fold=fold_number,
                    role=role,
                )
            )
            long_prefix = _prefix(configuration, "LONG")
            short_prefix = _prefix(configuration, "SHORT")
            long_mean = float(np.mean(frame[train][f"{long_prefix}_base_net_points"].to_numpy()))
            short_mean = float(np.mean(frame[train][f"{short_prefix}_base_net_points"].to_numpy()))
            unconditional_positive = max(long_mean, short_mean) > 0
            unconditional_long = np.full(len(validation), long_mean >= short_mean)
            frames_by_role[role]["UNCONDITIONAL_TRAIN_EV"].append(
                _candidate_frame(
                    frame,
                    validation,
                    np.full(len(validation), unconditional_positive),
                    unconditional_long,
                    fold=fold_number,
                    role=role,
                )
            )
            scoped = frame[validation]
            base_extra = scoped["cost_base_points"].to_numpy().astype(float) - scoped[
                "cost_optimistic_points"
            ].to_numpy().astype(float)
            long_v3_probability = cast(np.ndarray, long["v3_probability"])
            short_v3_probability = cast(np.ndarray, short["v3_probability"])
            long_v3_ev = (
                long_v3_probability
                * scoped[f"{long_prefix}_target_points"].to_numpy().astype(float)
                - (1 - long_v3_probability)
                * scoped[f"{long_prefix}_stop_points"].to_numpy().astype(float)
                - base_extra
            )
            short_v3_ev = (
                short_v3_probability
                * scoped[f"{short_prefix}_target_points"].to_numpy().astype(float)
                - (1 - short_v3_probability)
                * scoped[f"{short_prefix}_stop_points"].to_numpy().astype(float)
                - base_extra
            )
            long_v3_eligible = (long_v3_probability >= 0.60) & (long_v3_ev > 0)
            short_v3_eligible = (short_v3_probability >= 0.60) & (short_v3_ev > 0)
            choose_long_v3 = long_v3_eligible & (~short_v3_eligible | (long_v3_ev >= short_v3_ev))
            choose_short_v3 = short_v3_eligible & ~choose_long_v3
            frames_by_role[role]["V3_METAGATE"].append(
                _candidate_frame(
                    frame,
                    validation,
                    choose_long_v3 | choose_short_v3,
                    choose_long_v3,
                    fold=fold_number,
                    role=role,
                )
            )
            fold_parts.append(
                {
                    "fold": fold_number,
                    "role": role,
                    "validation_start_utc": fold["validation_start_utc"],
                    "validation_end_utc": fold["validation_end_utc"],
                    "train_observations": len(train),
                    "validation_observations": len(validation),
                    "validation_indices": validation,
                    "LONG": long,
                    "SHORT": short,
                    "primary_candidate_count": int(np.sum(candidate)),
                    "primary_LONG_count": int(np.sum(choose_long)),
                    "primary_SHORT_count": int(np.sum(choose_short)),
                }
            )
        development_parts = [part for part in fold_parts if part["role"] == "MODEL_DEVELOPMENT"]
        freeze_parts = [part for part in fold_parts if part["role"] == "INTERNAL_FREEZE_CHECK"]
        development = _scope_report(
            development_parts, frames_by_role["MODEL_DEVELOPMENT"], frame, 100
        )
        freeze = _scope_report(freeze_parts, frames_by_role["INTERNAL_FREEZE_CHECK"], frame, 200)
        criteria = cast(dict[str, Any], manifest["acceptance_criteria"])
        primary = freeze["primary_candidates"]
        ci_lower = primary["confidence_intervals_95"]["BASE_expectancy"]["lower"]
        checks = {
            "minimum_candidates": primary["candidate_count"] >= criteria["minimum_candidates"],
            "minimum_active_weeks": primary["active_weeks"] >= criteria["minimum_active_weeks"],
            "positive_BASE_expectancy": (
                primary["BASE_net_expectancy_points"] is not None
                and primary["BASE_net_expectancy_points"] > 0
            ),
            "positive_lower_95_ci_BASE": ci_lower is not None and ci_lower > 0,
            "nonnegative_STRESS_expectancy": (
                primary["STRESS_net_expectancy_points"] is not None
                and primary["STRESS_net_expectancy_points"] >= 0
            ),
            "minimum_profit_factor_BASE": (
                primary["profit_factor_BASE"] is not None
                and primary["profit_factor_BASE"] >= criteria["profit_factor_base_gte"]
            ),
            "minimum_positive_week_fraction": (
                primary["positive_week_fraction"] is not None
                and primary["positive_week_fraction"] >= criteria["positive_week_fraction_gte"]
            ),
            "maximum_largest_session_share": (
                primary["largest_session_share"] is not None
                and primary["largest_session_share"] <= criteria["largest_session_share_lte"]
            ),
        }
        all_passed = all(checks.values())
        if all_passed:
            conclusion: D10Conclusion = "D10_FREEZE_CANDIDATE"
        elif not checks["minimum_candidates"] or not checks["minimum_active_weeks"]:
            conclusion = "D10_INSUFFICIENT_EVIDENCE"
        elif checks["positive_BASE_expectancy"] and checks["nonnegative_STRESS_expectancy"]:
            conclusion = "D10_NEEDS_MORE_RESEARCH"
        else:
            conclusion = "D10_RESEARCH_REJECTED"
        elapsed = perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        broker = BrokerSimulationConfig()

        def public_fold(part: dict[str, Any]) -> dict[str, Any]:
            return {
                key: value
                for key, value in part.items()
                if key not in {"validation_indices"} and key not in {"LONG", "SHORT"}
            } | {
                "conditional_TRAIN_payoffs": {
                    side: part[side]["payoffs"] for side in ("LONG", "SHORT")
                },
                "safety_buffers": {side: part[side]["buffer_report"] for side in ("LONG", "SHORT")},
                "probabilistic_metrics": {
                    side: {
                        "primary": part[side]["primary_metrics"],
                        "challenger": part[side]["challenger_metrics"],
                    }
                    for side in ("LONG", "SHORT")
                },
                "direct_pnl_metrics": {
                    side: part[side]["direct_metrics"] for side in ("LONG", "SHORT")
                },
            }

        return ResearchD10Report(
            requested_start_utc=start,
            requested_end_utc=end,
            protocol_path=str(protocol_path.resolve()),
            protocol_sha256=cast(str, manifest["protocol_sha256"]),
            freeze_manifest_path=str(freeze_manifest_path.resolve()),
            freeze_manifest_sha256=cast(str, manifest["freeze_manifest_sha256"]),
            source_hashes=cast(dict[str, str], manifest["source_hashes"]),
            primary_configuration=cast(dict[str, Any], manifest["primary_configuration"]),
            model_policy={
                "primary": manifest["primary_model"],
                "challenger": "FIXED_HIST_GRADIENT_BOOSTING_MULTICLASS",
                "calibration": manifest["calibration"],
                "automatic_challenger_promotion": False,
            },
            ev_policy={
                "formula": manifest["ev_formula"],
                "payoffs": "TRAIN_MEAN_BASE_PNL_BY_SIDE_AND_OUTCOME",
                "NEITHER_uses_real_timeout_pnl": True,
                "nominal_barrier_substitution": False,
            },
            safety_buffer_policy={
                "formula": manifest["safety_buffer"],
                "source": "NESTED_PURGED_TRAIN_OOF_RESIDUALS",
                "block": "UTC_DAY",
            },
            split_policy={
                "MODEL_DEVELOPMENT": "three expanding folds, validation 30-40/40-50/50-60%",
                "INTERNAL_FREEZE_CHECK": "train 0-60%, validation 60-100%",
                "purge_seconds": 300,
                "nested_folds_per_outer_fold": 3,
            },
            simulation_profile=EdgeSimulationAssumptions(
                broker_profile=broker.profile(),
                evaluation_volume_lots=float(broker.volumes.minimum),
                slippage_points_per_side=1,
                commission_eur_per_lot_per_side=2,
                commission_minimum_eur_per_side=0,
                disclaimer="Simulated research profile only; not MetaQuotes-Demo capability.",
            ),
            folds=tuple(
                {
                    "fold": fold["fold"],
                    "role": fold["role"],
                    "validation_start_utc": fold["validation_start_utc"],
                    "validation_end_utc": fold["validation_end_utc"],
                    "train_observations": len(cast(np.ndarray, fold["train_indices"])),
                    "validation_observations": len(cast(np.ndarray, fold["validation_indices"])),
                }
                for fold in folds
            ),
            outer_purge_audit=outer_audit,
            nested_purge_audit=tuple(nested_audit),
            fold_results=tuple(public_fold(part) for part in fold_parts),
            model_development=development,
            internal_freeze_check=freeze,
            freeze_acceptance={
                "criteria": criteria,
                "observed_checks": checks,
                "all_passed": all_passed,
            },
            performance=D10Performance(
                elapsed_seconds=elapsed,
                source_ticks=SOURCE_TICKS,
                observations=frame.height,
                eligible_labeled_observations=sum(
                    cast(int, fold["validation_observations"]) for fold in outer_audit
                ),
                peak_python_memory_mb=peak / 1024 / 1024,
            ),
            dataset_path=str(dataset_path),
            conclusion=conclusion,
            limitations=(
                "All D.10 results remain RESEARCH and are not external holdout evidence.",
                "The HOLDOUT remained sealed and was neither opened nor evaluated.",
                "Native softmax probabilities were evaluated without post-hoc calibration.",
                "Conditional payoff regressors were not promoted; TRAIN means remain primary.",
                "No sizing, Phase E, order execution or live trading was implemented.",
            ),
        )


def render_research_d10_report(report: ResearchD10Report) -> str:
    development = report.model_development["primary_candidates"]
    freeze = report.internal_freeze_check["primary_candidates"]

    def metric(value: object) -> str:
        return "N/A" if value is None else f"{float(cast(float, value)):.6f}"

    lines = [
        "# SNIPER Phase D.10 — Three-Outcome Expected Value Engine",
        "",
        f"Verdict : **{report.conclusion}**",
        "",
        "## Résultats économiques primaires",
        "",
        "| Scope | Candidates L/S | Exécutable | BASE | STRESS | PF BASE | Win rate | Drawdown |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| MODEL_DEVELOPMENT | {development['candidate_count']} "
        f"({development['LONG_count']}/{development['SHORT_count']}) | "
        f"{metric(development['executable_expectancy_points'])} | "
        f"{metric(development['BASE_net_expectancy_points'])} | "
        f"{metric(development['STRESS_net_expectancy_points'])} | "
        f"{metric(development['profit_factor_BASE'])} | {metric(development['win_rate'])} | "
        f"{metric(development['maximum_drawdown_BASE_points'])} |",
        f"| INTERNAL_FREEZE_CHECK | {freeze['candidate_count']} "
        f"({freeze['LONG_count']}/{freeze['SHORT_count']}) | "
        f"{metric(freeze['executable_expectancy_points'])} | "
        f"{metric(freeze['BASE_net_expectancy_points'])} | "
        f"{metric(freeze['STRESS_net_expectancy_points'])} | "
        f"{metric(freeze['profit_factor_BASE'])} | {metric(freeze['win_rate'])} | "
        f"{metric(freeze['maximum_drawdown_BASE_points'])} |",
        "",
        "## Métriques multiclasses",
        "",
        "| Scope | Side | Modèle | Log-loss | Brier multiclass | Accuracy |",
        "|---|---|---|---:|---:|---:|",
    ]
    for scope_name, scope in (
        ("MODEL_DEVELOPMENT", report.model_development),
        ("INTERNAL_FREEZE_CHECK", report.internal_freeze_check),
    ):
        for side in ("LONG", "SHORT"):
            models = scope["probabilistic_models"][side]
            for model_name, values in models.items():
                lines.append(
                    f"| {scope_name} | {side} | {model_name} | "
                    f"{metric(values['multiclass_log_loss'])} | "
                    f"{metric(values['multiclass_brier'])} | {metric(values['accuracy'])} |"
                )
    lines.extend(
        [
            "",
            "## Critères INTERNAL_FREEZE_CHECK",
            "",
            *[
                f"- {name}: {'PASS' if passed else 'FAIL'}"
                for name, passed in report.freeze_acceptance["observed_checks"].items()
            ],
            "",
            "## Garde-fous",
            "",
            f"- Freeze manifest : `{report.freeze_manifest_sha256}`",
            "- B02_PRIMARY uniquement pour la décision",
            "- HOLDOUT : SEALED, non ouvert, non évalué",
            "- Phase E : NON COMMENCÉE",
            "- Trading live : DÉSACTIVÉ",
            "- Sizing : RISK_ENGINE_ONLY",
            "",
        ]
    )
    return "\n".join(lines)
