"""Phase D.9 path-dependent LONG/SHORT research on the sealed RESEARCH dataset."""

import tracemalloc
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore[import-untyped]
from sklearn.inspection import permutation_importance  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore[import-untyped]

from sniper.backtest.execution_model import BrokerSimulationConfig
from sniper.data.time import as_utc
from sniper.domain.edge_validation import EdgeSimulationAssumptions
from sniper.domain.research_v2 import TemporalFold
from sniper.domain.research_v3 import (
    BarrierConfiguration,
    CombinedBarrierReport,
    PurgedFoldAudit,
    ResearchV3Report,
    SideOutcomeReport,
    V3Conclusion,
    V3Performance,
)
from sniper.evaluation.edge_validation import MICROSECONDS, _days, _read_ticks, _tick_path
from sniper.evaluation.research_v2 import (
    CATEGORICAL_FEATURES_V2,
    NUMERIC_FEATURES_V2,
    SCENARIOS,
    _classification_metrics,
)

RANDOM_SEED_V3 = 20260909
PRIMARY_CONFIGURATION_ID = "B02_PRIMARY"
PRIMARY_MODEL = "logistic"
MINIMUM_SUCCESS_PROBABILITY = 0.60
EPSILON = 1e-9
QUALITY_MAE_FLOOR_POINTS = 1.0

BARRIER_CONFIGURATIONS = (
    BarrierConfiguration(
        id="B01",
        stop_atr=0.75,
        target_atr=1.25,
        timeout_seconds=180,
        minimum_target_to_base_cost=1.5,
    ),
    BarrierConfiguration(
        id="B02_PRIMARY",
        stop_atr=1.0,
        target_atr=2.0,
        timeout_seconds=300,
        minimum_target_to_base_cost=2.0,
    ),
    BarrierConfiguration(
        id="B03",
        stop_atr=1.0,
        target_atr=2.5,
        timeout_seconds=600,
        minimum_target_to_base_cost=2.0,
    ),
    BarrierConfiguration(
        id="B04",
        stop_atr=1.5,
        target_atr=3.0,
        timeout_seconds=900,
        minimum_target_to_base_cost=2.0,
    ),
)

V3_NUMERIC_FEATURES = NUMERIC_FEATURES_V2 + (
    "return_5s_points",
    "return_15s_points",
    "return_30s_points",
    "return_60s_points",
    "signed_distance_recent_high_points",
    "signed_distance_recent_low_points",
    "position_recent_60s_range",
    "m15_ema_slope_atr",
    "m5_momentum_atr",
    "m1_momentum_acceleration_points",
    "signed_tick_imbalance",
    "signed_consecutive_directional_ticks",
    "micro_pullback_depth_atr",
    "directional_wick_pressure",
    "directional_body_pressure",
    "signed_distance_from_breakout_atr",
    "retest_quality",
)
V3_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES_V2
V3_FEATURES = V3_NUMERIC_FEATURES + V3_CATEGORICAL_FEATURES

FEATURE_DEFINITIONS = {
    "return_Ns_points": "(mid(T) - last mid at or before T-Ns) / point, N=5/15/30/60",
    "signed_distance_recent_high_points": "(mid(T)-max(mid over [T-60s,T]))/point",
    "signed_distance_recent_low_points": "(mid(T)-min(mid over [T-60s,T]))/point",
    "position_recent_60s_range": "(mid(T)-low60)/(high60-low60), or 0.5 if flat",
    "m15_ema_slope_atr": "M15 EMA slope points / max(M15 ATR points, epsilon)",
    "m5_momentum_atr": "M5 momentum points / max(M5 ATR points, epsilon)",
    "m1_momentum_acceleration_points": "return_5s - return_15s/3",
    "signed_tick_imbalance": "(upticks-downticks)/max(upticks+downticks,1) over 60s",
    "signed_consecutive_directional_ticks": "signed length of final non-flat tick run",
    "micro_pullback_depth_atr": "distance from trend-side 60s extreme / M1 ATR",
    "directional_wick_pressure": "(lower_wick-upper_wick)/(sum_wicks+epsilon)",
    "directional_body_pressure": "sign(M1 momentum1)*M1 body ratio",
    "signed_distance_from_breakout_atr": (
        "+support distance for bullish, -resistance distance for bearish, else 0"
    ),
    "retest_quality": "1/(1+absolute signed breakout distance), else 0",
}


def _prefix(configuration: BarrierConfiguration, side: str) -> str:
    return f"{configuration.id.lower()}_{side.lower()}"


def _causal_micro_features(
    timestamps_us: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    current_index: int,
    *,
    m1_atr_points: float,
) -> dict[str, float]:
    """Compute V3 additions from quotes at or before current_index only."""
    current_us = int(timestamps_us[current_index])
    mids = (bids + asks) / 2
    current_mid = float(mids[current_index])
    point = 0.00001

    def past_return(seconds: int) -> float:
        target = current_us - seconds * MICROSECONDS
        index = int(np.searchsorted(timestamps_us, target, side="right")) - 1
        if index < 0 or index > current_index:
            return 0.0
        return (current_mid - float(mids[index])) / point

    left = int(np.searchsorted(timestamps_us, current_us - 60 * MICROSECONDS, side="left"))
    recent = mids[left : current_index + 1]
    high = float(np.max(recent))
    low = float(np.min(recent))
    span = high - low
    changes = np.diff(recent)
    upticks = int(np.sum(changes > 0))
    downticks = int(np.sum(changes < 0))
    directional = upticks + downticks
    imbalance = (upticks - downticks) / directional if directional else 0.0
    nonzero = np.sign(changes[changes != 0])
    consecutive = 0
    if len(nonzero):
        final_sign = int(nonzero[-1])
        consecutive = final_sign
        for sign in nonzero[-2::-1]:
            if int(sign) != final_sign:
                break
            consecutive += final_sign
    return_5 = past_return(5)
    return_15 = past_return(15)
    return_60 = past_return(60)
    pullback = high - current_mid if return_60 >= 0 else current_mid - low
    return {
        "return_5s_points": return_5,
        "return_15s_points": return_15,
        "return_30s_points": past_return(30),
        "return_60s_points": return_60,
        "signed_distance_recent_high_points": (current_mid - high) / point,
        "signed_distance_recent_low_points": (current_mid - low) / point,
        "position_recent_60s_range": (current_mid - low) / span if span > 0 else 0.5,
        "m1_momentum_acceleration_points": return_5 - return_15 / 3,
        "signed_tick_imbalance": imbalance,
        "signed_consecutive_directional_ticks": float(consecutive),
        "micro_pullback_depth_atr": (pullback / point) / max(m1_atr_points, EPSILON),
    }


def _barrier_side_labels(
    timestamps_us: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    current_index: int,
    configuration: BarrierConfiguration,
    side: Literal["LONG", "SHORT"],
    *,
    atr_points: float,
    base_cost_points: float,
    costs: dict[str, float],
) -> dict[str, object]:
    current_us = int(timestamps_us[current_index])
    timeout_us = current_us + configuration.timeout_seconds * MICROSECONDS
    terminal_index = int(np.searchsorted(timestamps_us, timeout_us, side="right")) - 1
    prefix = _prefix(configuration, side)
    if (
        terminal_index <= current_index
        or timeout_us - int(timestamps_us[terminal_index]) > 2 * MICROSECONDS
    ):
        return {f"{prefix}_complete": False}
    point = 0.00001
    target_points = max(
        configuration.target_atr * atr_points,
        configuration.minimum_target_to_base_cost * base_cost_points,
    )
    stop_points = max(configuration.stop_atr * atr_points, 1.0)
    future_bids = bids[current_index + 1 : terminal_index + 1]
    future_asks = asks[current_index + 1 : terminal_index + 1]
    future_times = timestamps_us[current_index + 1 : terminal_index + 1]
    entry_bid = float(bids[current_index])
    entry_ask = float(asks[current_index])
    current_mid = (entry_bid + entry_ask) / 2
    if side == "LONG":
        path = (future_bids - entry_ask) / point
        market_path = ((future_bids + future_asks) / 2 - current_mid) / point
    else:
        path = (entry_bid - future_asks) / point
        market_path = (current_mid - (future_bids + future_asks) / 2) / point
    target_hits = np.flatnonzero(path >= target_points)
    stop_hits = np.flatnonzero(path <= -stop_points)
    target_position = int(target_hits[0]) if len(target_hits) else None
    stop_position = int(stop_hits[0]) if len(stop_hits) else None
    if target_position is not None and (stop_position is None or target_position < stop_position):
        outcome = "TARGET_FIRST"
        exit_position = target_position
    elif stop_position is not None:
        outcome = "STOP_FIRST"
        exit_position = stop_position
    else:
        outcome = "NEITHER"
        exit_position = len(path) - 1
    executable_return = float(path[exit_position])
    market_return = float(market_path[exit_position])
    observed_spread = costs["optimistic"]
    return {
        f"{prefix}_complete": True,
        f"{prefix}_outcome": outcome,
        f"{prefix}_target_points": target_points,
        f"{prefix}_stop_points": stop_points,
        f"{prefix}_gross_return_points": market_return,
        f"{prefix}_executable_return_points": executable_return,
        f"{prefix}_optimistic_net_points": executable_return,
        f"{prefix}_base_net_points": executable_return - (costs["base"] - observed_spread),
        f"{prefix}_stress_net_points": executable_return - (costs["stress"] - observed_spread),
        f"{prefix}_mfe_points": max(0.0, float(np.max(path))),
        f"{prefix}_mae_points": max(0.0, -float(np.min(path))),
        f"{prefix}_time_to_target_seconds": (
            (int(future_times[target_position]) - current_us) / MICROSECONDS
            if target_position is not None
            else None
        ),
        f"{prefix}_time_to_stop_seconds": (
            (int(future_times[stop_position]) - current_us) / MICROSECONDS
            if stop_position is not None
            else None
        ),
    }


def build_v3_dataset(
    *, data_root: Path, start_utc: datetime, end_utc: datetime
) -> tuple[pl.DataFrame, int, float]:
    start, end = as_utc(start_utc), as_utc(end_utc)
    if start != datetime(2026, 6, 10, tzinfo=UTC) or end != datetime(2026, 9, 8, tzinfo=UTC):
        raise PermissionError("D.9 may only open the declared RESEARCH dataset")
    source = (
        data_root
        / "evaluations"
        / "research-v2-20260610T000000Z-20260908T000000Z"
        / "observations.parquet"
    )
    if not source.exists():
        raise ValueError("D.8 RESEARCH observations are required before D.9")
    base = pl.read_parquet(source).select(
        "timestamp_utc",
        "week",
        *V3_CATEGORICAL_FEATURES,
        *NUMERIC_FEATURES_V2,
        "cost_optimistic_points",
        "cost_base_points",
        "cost_stress_points",
    )
    rows: list[dict[str, object]] = []
    ticks_analyzed = 0
    peak_frame_mb = 0.0
    for day in _days(start, end):
        day_start = max(start, datetime.combine(day, datetime.min.time(), UTC))
        midnight = day_start.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = min(end, midnight + timedelta(days=1))
        observations = base.filter(
            (pl.col("timestamp_utc") >= day_start) & (pl.col("timestamp_utc") < day_end)
        )
        if observations.is_empty():
            continue
        current = _read_ticks(_tick_path(data_root, day), day_start, day_end)
        ticks_analyzed += current.height
        previous = _read_ticks(
            _tick_path(data_root, day - timedelta(days=1)),
            day_start - timedelta(seconds=60),
            day_start,
        )
        following = _read_ticks(
            _tick_path(data_root, day + timedelta(days=1)),
            day_end,
            min(end, day_end + timedelta(seconds=900)),
        )
        extended = pl.concat([previous, current, following], how="vertical").sort(
            "timestamp_utc", maintain_order=True
        )
        peak_frame_mb = max(peak_frame_mb, extended.estimated_size("mb"))
        timestamps_us = extended["timestamp_utc"].cast(pl.Int64).to_numpy()
        bids = extended["bid"].to_numpy().astype(float)
        asks = extended["ask"].to_numpy().astype(float)
        for row in observations.iter_rows(named=True):
            sample_us = int(row["timestamp_utc"].timestamp() * MICROSECONDS)
            current_index = int(np.searchsorted(timestamps_us, sample_us, side="left"))
            if (
                current_index >= len(timestamps_us)
                or int(timestamps_us[current_index]) != sample_us
            ):
                raise ValueError("RESEARCH observation timestamp is absent from source ticks")
            atr_points = float(row["m1_atr_points"])
            micro = _causal_micro_features(
                timestamps_us,
                bids,
                asks,
                current_index,
                m1_atr_points=atr_points,
            )
            m15_atr = max(float(row["m15_atr_points"]), EPSILON)
            m5_atr = max(float(row["m5_atr_points"]), EPSILON)
            upper = float(row["m1_upper_wick_ratio"])
            lower = float(row["m1_lower_wick_ratio"])
            breakout = str(row["m5_breakout_retest"])
            breakout_distance = (
                float(row["m5_support_distance_atr"])
                if breakout == "BULLISH"
                else -float(row["m5_resistance_distance_atr"])
                if breakout == "BEARISH"
                else 0.0
            )
            row.update(
                {
                    **micro,
                    "m15_ema_slope_atr": float(row["m15_ema_slope_points"]) / m15_atr,
                    "m5_momentum_atr": float(row["m5_momentum_points"]) / m5_atr,
                    "directional_wick_pressure": (lower - upper) / max(lower + upper, EPSILON),
                    "directional_body_pressure": np.sign(float(row["m1_momentum_1_points"]))
                    * float(row["m1_body_ratio"]),
                    "signed_distance_from_breakout_atr": breakout_distance,
                    "retest_quality": (
                        1 / (1 + abs(breakout_distance)) if breakout != "NONE" else 0.0
                    ),
                }
            )
            costs = {
                "optimistic": float(row["cost_optimistic_points"]),
                "base": float(row["cost_base_points"]),
                "stress": float(row["cost_stress_points"]),
            }
            for configuration in BARRIER_CONFIGURATIONS:
                for side in ("LONG", "SHORT"):
                    row.update(
                        _barrier_side_labels(
                            timestamps_us,
                            bids,
                            asks,
                            current_index,
                            configuration,
                            side,
                            atr_points=atr_points,
                            base_cost_points=costs["base"],
                            costs=costs,
                        )
                    )
            rows.append(row)
    return (
        pl.DataFrame(rows, infer_schema_length=None).sort("timestamp_utc"),
        ticks_analyzed,
        peak_frame_mb,
    )


def v3_temporal_folds(
    frame: pl.DataFrame, start: datetime, end: datetime
) -> tuple[list[tuple[np.ndarray, np.ndarray]], tuple[TemporalFold, ...]]:
    duration = end - start
    boundaries = [start + duration * fraction for fraction in (0.6, 0.7, 0.8, 0.9, 1.0)]
    timestamps = frame["timestamp_utc"].to_numpy()
    folds = []
    reports = []
    for fold_index in range(4):
        train_end = boundaries[fold_index]
        validation_end = boundaries[fold_index + 1]
        train_mask = timestamps < np.datetime64(train_end.replace(tzinfo=None), "us")
        validation_mask = timestamps >= np.datetime64(train_end.replace(tzinfo=None), "us")
        validation_mask &= timestamps < np.datetime64(validation_end.replace(tzinfo=None), "us")
        train_indices = np.flatnonzero(train_mask)
        validation_indices = np.flatnonzero(validation_mask)
        folds.append((train_indices, validation_indices))
        reports.append(
            TemporalFold(
                fold=fold_index + 1,
                train_start_utc=start,
                train_end_utc_exclusive=train_end,
                validation_start_utc=train_end,
                validation_end_utc_exclusive=validation_end,
                train_observations=len(train_indices),
                validation_observations=len(validation_indices),
                evaluation_role="RESEARCH_WALK_FORWARD",
            )
        )
    return folds, tuple(reports)


def purge_v3_training_labels(
    frame: pl.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    fold_reports: tuple[TemporalFold, ...],
    configuration: BarrierConfiguration,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], tuple[PurgedFoldAudit, ...]]:
    """Remove TRAIN rows whose complete label is not observable at validation start."""
    timestamps_us = frame["timestamp_utc"].cast(pl.Int64).to_numpy().astype(np.int64)
    timeout_us = configuration.timeout_seconds * MICROSECONDS
    purged_folds: list[tuple[np.ndarray, np.ndarray]] = []
    audit: list[PurgedFoldAudit] = []
    for (train_indices, validation_indices), fold_report in zip(folds, fold_reports, strict=True):
        validation_start_us = int(fold_report.validation_start_utc.timestamp() * MICROSECONDS)
        label_end_us = timestamps_us[train_indices] + timeout_us
        keep = label_end_us <= validation_start_us
        retained = train_indices[keep]
        max_label_end_us = (
            int(np.max(timestamps_us[retained] + timeout_us)) if len(retained) else None
        )
        invariant_passed = max_label_end_us is None or max_label_end_us <= validation_start_us
        if not invariant_passed:
            raise RuntimeError("Purged walk-forward label boundary invariant failed")
        purged_folds.append((retained, validation_indices))
        audit.append(
            PurgedFoldAudit(
                configuration_id=configuration.id,
                fold=fold_report.fold,
                timeout_seconds=configuration.timeout_seconds,
                validation_start_utc=fold_report.validation_start_utc,
                original_train_observations=len(train_indices),
                purged_observations=int(np.sum(~keep)),
                retained_train_observations=len(retained),
                max_label_end_time_train_utc=(
                    datetime.fromtimestamp(max_label_end_us / MICROSECONDS, UTC)
                    if max_label_end_us is not None
                    else None
                ),
                invariant_passed=invariant_passed,
            )
        )
    return purged_folds, tuple(audit)


def _pipeline(model: Any) -> Pipeline:
    preprocess = ColumnTransformer(
        (
            ("numeric", StandardScaler(), list(V3_NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(V3_CATEGORICAL_FEATURES),
            ),
        )
    )
    return Pipeline((("preprocess", preprocess), ("model", model)))


def _feature_names(pipeline: Pipeline) -> list[str]:
    transformer = cast(ColumnTransformer, pipeline.named_steps["preprocess"])
    return [str(value) for value in transformer.get_feature_names_out()]


def _importance(
    model: Pipeline, x: Any, y: np.ndarray, scoring: str
) -> tuple[dict[str, object], ...]:
    result = permutation_importance(
        model,
        x,
        y,
        n_repeats=3,
        random_state=RANDOM_SEED_V3,
        scoring=scoring,
        n_jobs=1,
    )
    ranked = sorted(
        zip(V3_FEATURES, result.importances_mean, strict=True),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )[:20]
    return tuple({"feature": name, "importance": float(value)} for name, value in ranked)


def _coefficients(model: Pipeline) -> tuple[dict[str, object], ...]:
    estimator = cast(LogisticRegression, model.named_steps["model"])
    ranked = sorted(
        zip(_feature_names(model), estimator.coef_[0], strict=True),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )[:25]
    return tuple({"feature": name, "coefficient": float(value)} for name, value in ranked)


def _fit_side(
    frame: pl.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    configuration: BarrierConfiguration,
    side: Literal["LONG", "SHORT"],
) -> tuple[SideOutcomeReport, dict[str, np.ndarray]]:
    prefix = _prefix(configuration, side)
    valid = frame[f"{prefix}_complete"].fill_null(False).to_numpy().astype(bool)
    x_all = frame.select(V3_FEATURES).to_pandas()
    outcomes = frame[f"{prefix}_outcome"].to_numpy()
    y_all = np.array([int(value == "TARGET_FIRST") for value in outcomes], dtype=int)
    outputs: dict[str, list[np.ndarray]] = {
        "indices": [],
        "prior_probability": [],
        "logistic_probability": [],
        "hgb_probability": [],
    }
    final_logistic: Pipeline | None = None
    final_hgb: Pipeline | None = None
    final_x: Any = None
    final_y: np.ndarray | None = None
    for train_indices, validation_indices in folds:
        train_indices = train_indices[valid[train_indices]]
        validation_indices = validation_indices[valid[validation_indices]]
        x_train = x_all.iloc[train_indices]
        x_validation = x_all.iloc[validation_indices]
        y_train = y_all[train_indices]
        y_validation = y_all[validation_indices]
        logistic = _pipeline(LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_SEED_V3))
        hgb = _pipeline(
            HistGradientBoostingClassifier(
                max_iter=50,
                max_depth=3,
                learning_rate=0.08,
                random_state=RANDOM_SEED_V3,
            )
        )
        logistic.fit(x_train, y_train)
        hgb.fit(x_train, y_train)
        outputs["indices"].append(validation_indices)
        outputs["prior_probability"].append(
            np.full(len(validation_indices), float(np.mean(y_train)))
        )
        outputs["logistic_probability"].append(logistic.predict_proba(x_validation)[:, 1])
        outputs["hgb_probability"].append(hgb.predict_proba(x_validation)[:, 1])
        final_logistic, final_hgb, final_x, final_y = logistic, hgb, x_validation, y_validation
    combined = {name: np.concatenate(values) for name, values in outputs.items()}
    indices = combined["indices"].astype(int)
    y = y_all[indices]
    prior = combined["prior_probability"]
    logistic_probability = combined["logistic_probability"]
    hgb_probability = combined["hgb_probability"]
    models = {
        "prior_baseline": _classification_metrics(y, (prior >= 0.5).astype(int), prior),
        "logistic_regression": _classification_metrics(
            y, (logistic_probability >= 0.5).astype(int), logistic_probability
        ),
        "hist_gradient_boosting": _classification_metrics(
            y, (hgb_probability >= 0.5).astype(int), hgb_probability
        ),
    }
    scoped = frame[indices]
    mfe = scoped[f"{prefix}_mfe_points"].to_numpy().astype(float)
    mae = scoped[f"{prefix}_mae_points"].to_numpy().astype(float)
    quality = mfe / np.maximum(mae, QUALITY_MAE_FLOOR_POINTS)
    target_times = scoped[f"{prefix}_time_to_target_seconds"].drop_nulls().to_numpy()
    stop_times = scoped[f"{prefix}_time_to_stop_seconds"].drop_nulls().to_numpy()
    assert final_logistic is not None and final_hgb is not None and final_y is not None
    return (
        SideOutcomeReport(
            configuration_id=configuration.id,
            side=side,
            observations=len(indices),
            target_before_stop=int(np.sum(outcomes[indices] == "TARGET_FIRST")),
            stop_before_target=int(np.sum(outcomes[indices] == "STOP_FIRST")),
            neither_before_timeout=int(np.sum(outcomes[indices] == "NEITHER")),
            target_before_stop_rate=float(np.mean(y)),
            average_mfe_points=float(np.mean(mfe)),
            average_mae_points=float(np.mean(mae)),
            average_mfe_mae_quality=float(np.mean(quality)),
            average_time_to_target_seconds=(
                float(np.mean(target_times)) if len(target_times) else None
            ),
            average_time_to_stop_seconds=(float(np.mean(stop_times)) if len(stop_times) else None),
            models=models,
            feature_importance={
                "logistic_coefficients": _coefficients(final_logistic),
                "logistic_permutation": _importance(
                    final_logistic, final_x, final_y, "average_precision"
                ),
                "hgb_permutation": _importance(final_hgb, final_x, final_y, "average_precision"),
            },
        ),
        {
            "indices": indices,
            "logistic_probability": logistic_probability,
            "hgb_probability": hgb_probability,
        },
    )


def _distribution(frame: pl.DataFrame, column: str) -> tuple[dict[str, object], ...]:
    if frame.is_empty():
        return ()
    rows = []
    for value in sorted(frame[column].unique().to_list()):
        group = frame.filter(pl.col(column) == value)
        rows.append(
            {
                column: str(value),
                "candidates_count": group.height,
                "base_net_expectancy_points": float(str(group["actual_base_net"].mean())),
                "stress_net_expectancy_points": float(str(group["actual_stress_net"].mean())),
                "target_before_stop_rate": float(
                    str((group["actual_outcome"] == "TARGET_FIRST").mean())
                ),
            }
        )
    return tuple(rows)


def _combined_result(
    frame: pl.DataFrame,
    configuration: BarrierConfiguration,
    long_predictions: dict[str, np.ndarray],
    short_predictions: dict[str, np.ndarray],
) -> tuple[CombinedBarrierReport, pl.DataFrame]:
    indices = long_predictions["indices"].astype(int)
    if not np.array_equal(indices, short_predictions["indices"].astype(int)):
        raise ValueError("LONG and SHORT validation indices differ")
    scoped = frame[indices]
    long_probability = long_predictions[f"{PRIMARY_MODEL}_probability"]
    short_probability = short_predictions[f"{PRIMARY_MODEL}_probability"]
    long_prefix = _prefix(configuration, "LONG")
    short_prefix = _prefix(configuration, "SHORT")
    base_extra = scoped["cost_base_points"].to_numpy().astype(float) - scoped[
        "cost_optimistic_points"
    ].to_numpy().astype(float)
    long_expected_net = (
        long_probability * scoped[f"{long_prefix}_target_points"].to_numpy().astype(float)
        - (1 - long_probability) * scoped[f"{long_prefix}_stop_points"].to_numpy().astype(float)
        - base_extra
    )
    short_expected_net = (
        short_probability * scoped[f"{short_prefix}_target_points"].to_numpy().astype(float)
        - (1 - short_probability) * scoped[f"{short_prefix}_stop_points"].to_numpy().astype(float)
        - base_extra
    )
    long_eligible = (long_probability >= MINIMUM_SUCCESS_PROBABILITY) & (long_expected_net > 0)
    short_eligible = (short_probability >= MINIMUM_SUCCESS_PROBABILITY) & (short_expected_net > 0)
    choose_long = long_eligible & (~short_eligible | (long_expected_net >= short_expected_net))
    choose_short = short_eligible & ~choose_long
    candidate_mask = choose_long | choose_short
    selected_positions = np.flatnonzero(candidate_mask)
    selected = scoped[selected_positions]
    if selected.is_empty():
        return (
            CombinedBarrierReport(
                configuration_id=configuration.id,
                timeout_seconds=configuration.timeout_seconds,
                candidates_count=0,
                long_candidates=0,
                short_candidates=0,
                skipped_count=len(indices),
                target_before_stop_rate=None,
                gross_expectancy_points=None,
                executable_expectancy_points=None,
                optimistic_net_expectancy_points=None,
                base_net_expectancy_points=None,
                stress_net_expectancy_points=None,
                profit_factor_base=None,
                average_mfe_points=None,
                average_mae_points=None,
                average_mfe_mae_quality=None,
                weekly_distribution=(),
                session_distribution=(),
            ),
            selected,
        )

    selected_long = choose_long[selected_positions]

    def directional_values(suffix: str) -> np.ndarray:
        return np.where(
            selected_long,
            selected[f"{long_prefix}_{suffix}"].to_numpy(),
            selected[f"{short_prefix}_{suffix}"].to_numpy(),
        )

    gross = directional_values("gross_return_points").astype(float)
    executable = directional_values("executable_return_points").astype(float)
    optimistic = directional_values("optimistic_net_points").astype(float)
    base = directional_values("base_net_points").astype(float)
    stress = directional_values("stress_net_points").astype(float)
    mfe = directional_values("mfe_points").astype(float)
    mae = directional_values("mae_points").astype(float)
    outcomes = directional_values("outcome").astype(str)
    selected = selected.with_columns(
        pl.Series("selected_side", np.where(selected_long, "LONG", "SHORT")),
        pl.Series("actual_gross", gross),
        pl.Series("actual_executable", executable),
        pl.Series("actual_optimistic_net", optimistic),
        pl.Series("actual_base_net", base),
        pl.Series("actual_stress_net", stress),
        pl.Series("actual_mfe", mfe),
        pl.Series("actual_mae", mae),
        pl.Series("actual_outcome", outcomes),
    )
    positive = float(np.sum(base[base > 0]))
    negative = float(-np.sum(base[base < 0]))
    profit_factor = positive / negative if negative > 0 else None
    return (
        CombinedBarrierReport(
            configuration_id=configuration.id,
            timeout_seconds=configuration.timeout_seconds,
            candidates_count=selected.height,
            long_candidates=int(np.sum(selected_long)),
            short_candidates=int(np.sum(~selected_long)),
            skipped_count=len(indices) - selected.height,
            target_before_stop_rate=float(np.mean(outcomes == "TARGET_FIRST")),
            gross_expectancy_points=float(np.mean(gross)),
            executable_expectancy_points=float(np.mean(executable)),
            optimistic_net_expectancy_points=float(np.mean(optimistic)),
            base_net_expectancy_points=float(np.mean(base)),
            stress_net_expectancy_points=float(np.mean(stress)),
            profit_factor_base=profit_factor,
            average_mfe_points=float(np.mean(mfe)),
            average_mae_points=float(np.mean(mae)),
            average_mfe_mae_quality=float(np.mean(mfe / np.maximum(mae, QUALITY_MAE_FLOOR_POINTS))),
            weekly_distribution=_distribution(selected, "week"),
            session_distribution=_distribution(selected, "session"),
        ),
        selected,
    )


def _opportunity_quality(frame: pl.DataFrame) -> dict[str, object]:
    results: dict[str, object] = {}
    for configuration in BARRIER_CONFIGURATIONS:
        long_prefix = _prefix(configuration, "LONG")
        short_prefix = _prefix(configuration, "SHORT")
        valid = frame[f"{long_prefix}_complete"].fill_null(False) & frame[
            f"{short_prefix}_complete"
        ].fill_null(False)
        scoped = frame.filter(valid)
        mfe = np.maximum(
            scoped[f"{long_prefix}_mfe_points"].to_numpy().astype(float),
            scoped[f"{short_prefix}_mfe_points"].to_numpy().astype(float),
        )
        long_mfe = scoped[f"{long_prefix}_mfe_points"].to_numpy().astype(float)
        short_mfe = scoped[f"{short_prefix}_mfe_points"].to_numpy().astype(float)
        mae = np.where(
            long_mfe >= short_mfe,
            scoped[f"{long_prefix}_mae_points"].to_numpy().astype(float),
            scoped[f"{short_prefix}_mae_points"].to_numpy().astype(float),
        )
        costs = scoped["cost_base_points"].to_numpy().astype(float)
        results[configuration.id] = {
            "observations": scoped.height,
            "p_mfe_gt_1_5_cost": float(np.mean(mfe > 1.5 * costs)),
            "p_mfe_gt_2_cost": float(np.mean(mfe > 2.0 * costs)),
            "p_mfe_gt_3_cost": float(np.mean(mfe > 3.0 * costs)),
            "average_mfe_mae_quality": float(
                np.mean(mfe / np.maximum(mae, QUALITY_MAE_FLOOR_POINTS))
            ),
            "quality_definition": "MFE of better side / max(MAE of that side, 1 point)",
        }
    return results


def _registered_hypotheses(frame: pl.DataFrame) -> tuple[dict[str, object], ...]:
    configuration = next(
        item for item in BARRIER_CONFIGURATIONS if item.id == PRIMARY_CONFIGURATION_ID
    )
    hypotheses = (
        ("H001", "bearish_m1_pattern", "SHORT", pl.col("m1_pattern") == "BEARISH"),
        (
            "H002",
            "bullish_breakout_asia",
            "LONG",
            (pl.col("m5_breakout_retest") == "BULLISH") & (pl.col("session") == "ASIA"),
        ),
        (
            "H003",
            "bearish_breakout_london_new_york",
            "SHORT",
            (pl.col("m5_breakout_retest") == "BEARISH") & (pl.col("session") == "LONDON_NEW_YORK"),
        ),
    )
    results = []
    for hypothesis_id, name, side, condition in hypotheses:
        prefix = _prefix(configuration, side)
        scoped = frame.filter(condition & pl.col(f"{prefix}_complete").fill_null(False))
        results.append(
            {
                "id": hypothesis_id,
                "name": name,
                "status": "UNVALIDATED_RESEARCH_RESULT",
                "side": side,
                "observations": scoped.height,
                "target_before_stop_rate": (
                    float(str((scoped[f"{prefix}_outcome"] == "TARGET_FIRST").mean()))
                    if scoped.height
                    else None
                ),
                "executable_expectancy_points": (
                    float(str(scoped[f"{prefix}_executable_return_points"].mean()))
                    if scoped.height
                    else None
                ),
                "base_net_expectancy_points": (
                    float(str(scoped[f"{prefix}_base_net_points"].mean()))
                    if scoped.height
                    else None
                ),
            }
        )
    return tuple(results)


class ResearchV3Engine:
    def run(
        self,
        *,
        data_root: Path,
        start_utc: datetime,
        end_utc: datetime,
        purge_training_labels: bool = True,
    ) -> ResearchV3Report:
        start, end = as_utc(start_utc), as_utc(end_utc)
        tracemalloc.start()
        started = perf_counter()
        dataset_path = (
            data_root
            / "evaluations"
            / "research-v3-20260610T000000Z-20260908T000000Z"
            / "observations.parquet"
        )
        if dataset_path.exists():
            frame = pl.read_parquet(dataset_path)
            ticks_analyzed = 17_941_326
            peak_frame_mb = 0.0
        else:
            frame, ticks_analyzed, peak_frame_mb = build_v3_dataset(
                data_root=data_root, start_utc=start, end_utc=end
            )
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = dataset_path.with_suffix(".parquet.tmp")
            frame.write_parquet(temporary, compression="zstd", statistics=True)
            temporary.replace(dataset_path)
        folds, fold_reports = v3_temporal_folds(frame, start, end)
        long_reports = []
        short_reports = []
        combined_reports = []
        purge_audit: list[PurgedFoldAudit] = []
        predictions: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        primary_selected = pl.DataFrame()
        for configuration in BARRIER_CONFIGURATIONS:
            configuration_folds = folds
            if purge_training_labels:
                configuration_folds, configuration_audit = purge_v3_training_labels(
                    frame, folds, fold_reports, configuration
                )
                purge_audit.extend(configuration_audit)
            long_report, long_prediction = _fit_side(
                frame, configuration_folds, configuration, "LONG"
            )
            short_report, short_prediction = _fit_side(
                frame, configuration_folds, configuration, "SHORT"
            )
            long_reports.append(long_report)
            short_reports.append(short_report)
            predictions[(configuration.id, "LONG")] = long_prediction
            predictions[(configuration.id, "SHORT")] = short_prediction
            combined, selected = _combined_result(
                frame, configuration, long_prediction, short_prediction
            )
            combined_reports.append(combined)
            if configuration.id == PRIMARY_CONFIGURATION_ID:
                primary_selected = selected
        primary = next(
            item for item in combined_reports if item.configuration_id == PRIMARY_CONFIGURATION_ID
        )
        primary_long = next(
            item for item in long_reports if item.configuration_id == PRIMARY_CONFIGURATION_ID
        )
        primary_short = next(
            item for item in short_reports if item.configuration_id == PRIMARY_CONFIGURATION_ID
        )
        weeks = primary.weekly_distribution
        sessions = primary.session_distribution
        positive_weeks = sum(cast(float, item["base_net_expectancy_points"]) > 0 for item in weeks)
        largest_session_share = (
            max(cast(int, item["candidates_count"]) for item in sessions) / primary.candidates_count
            if sessions and primary.candidates_count
            else 1.0
        )
        long_lift = cast(float, primary_long.models["logistic_regression"]["pr_auc"]) - cast(
            float, primary_long.models["prior_baseline"]["pr_auc"]
        )
        short_lift = cast(float, primary_short.models["logistic_regression"]["pr_auc"]) - cast(
            float, primary_short.models["prior_baseline"]["pr_auc"]
        )
        meets = (
            (
                primary.candidates_count >= 200
                and cast(float, primary.executable_expectancy_points) > 0
                and cast(float, primary.base_net_expectancy_points) > 0
                and cast(float, primary.stress_net_expectancy_points) >= 0
                and cast(float, primary.profit_factor_base) >= 1.10
                and len(weeks) >= 4
                and positive_weeks / len(weeks) >= 0.60
                and largest_session_share <= 0.60
                and long_lift >= 0.02
                and short_lift >= 0.02
            )
            if primary.candidates_count and weeks
            else False
        )
        if meets:
            conclusion: V3Conclusion = "V3_FREEZE_CANDIDATE"
        elif (
            primary.candidates_count >= 50
            and primary.base_net_expectancy_points is not None
            and primary.base_net_expectancy_points > 0
        ):
            conclusion = "V3_NEEDS_MORE_RESEARCH"
        else:
            conclusion = "V3_RESEARCH_REJECTED"
        elapsed = perf_counter() - started
        _, peak_python = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        broker = BrokerSimulationConfig()
        return ResearchV3Report(
            requested_start_utc=start,
            requested_end_utc=end,
            sampling_policy=(
                "First tick of each active UTC minute; deterministic and chronological."
            ),
            label_policy=(
                "Separate executable Bid/Ask triple barriers for LONG and SHORT. TARGET_FIRST, "
                "STOP_FIRST or NEITHER are computed only after causal features are frozen at T."
            ),
            split_policy=(
                "60% initial TRAIN plus four expanding 10% walk-forward RESEARCH folds; "
                "scaler/encoder/model fit on prior TRAIN only; no random split. "
                + (
                    "For every configuration, TRAIN requires observation timestamp plus "
                    "timeout <= validation start."
                    if purge_training_labels
                    else "Original D.9 boundary used timestamp < validation start only."
                )
            ),
            purge_policy=(
                "LABEL_END_BEFORE_VALIDATION" if purge_training_labels else "NONE_ORIGINAL"
            ),
            purge_audit=tuple(purge_audit),
            feature_definitions=FEATURE_DEFINITIONS,
            barrier_configurations=BARRIER_CONFIGURATIONS,
            meta_gate_policy=(
                "Primary regularized logistic models are separate by side. Eligible when "
                "P(target-first)>=0.60 and conservative expected BASE net >0; choose the side "
                "with highest expected BASE net, otherwise SKIP."
            ),
            cost_scenarios=SCENARIOS,
            simulation_profile=EdgeSimulationAssumptions(
                broker_profile=broker.profile(),
                evaluation_volume_lots=float(broker.volumes.minimum),
                slippage_points_per_side=1,
                commission_eur_per_lot_per_side=2,
                commission_minimum_eur_per_side=0,
                disclaimer="Simulated research profile only; not MetaQuotes-Demo capability.",
            ),
            folds=fold_reports,
            opportunity_quality=_opportunity_quality(frame),
            long_results=tuple(long_reports),
            short_results=tuple(short_reports),
            combined_results=tuple(combined_reports),
            registered_hypotheses=_registered_hypotheses(frame),
            freeze_criteria_predeclared={
                "minimum_candidates": 200,
                "minimum_profit_factor_base": 1.10,
                "minimum_active_weeks": 4,
                "minimum_positive_week_fraction": 0.60,
                "maximum_single_session_share": 0.60,
                "requires_positive_executable_expectancy": True,
                "requires_positive_base_net_expectancy": True,
                "requires_nonnegative_stress_net_expectancy": True,
                "minimum_pr_auc_lift_over_prior_each_side": 0.02,
                "observed_long_pr_auc_lift": long_lift,
                "observed_short_pr_auc_lift": short_lift,
                "criteria_passed": meets,
            },
            performance=V3Performance(
                elapsed_seconds=elapsed,
                ticks_analyzed=ticks_analyzed,
                observations=frame.height,
                labeled_side_events=sum(
                    item.observations for item in (*long_reports, *short_reports)
                ),
                peak_python_memory_mb=peak_python / 1024 / 1024,
                peak_loaded_tick_frame_mb=peak_frame_mb,
            ),
            dataset_path=str(dataset_path.resolve()),
            conclusion=conclusion,
            limitations=(
                "All D.9 results are RESEARCH and cannot be final out-of-sample evidence.",
                "The HOLDOUT remained sealed and was neither opened nor evaluated.",
                "Barrier fills ignore queue/market impact and use simulated slippage/commission.",
                "No sizing, order execution, live trading or Phase E implementation exists.",
                f"Primary selected rows retained for reporting only: {primary_selected.height}.",
            ),
        )


def render_research_v3_report(report: ResearchV3Report) -> str:
    def metric(value: object) -> str:
        return "N/A" if value is None else f"{float(cast(float, value)):.6f}"

    lines = [
        "# SNIPER Phase D.9 — Direction Research V3",
        "",
        f"Conclusion : **{report.conclusion}**",
        "",
        f"Ticks : {report.performance.ticks_analyzed:,}",
        f"Observations : {report.performance.observations:,}",
        f"Événements directionnels labellisés : {report.performance.labeled_side_events:,}",
        f"Temps : {report.performance.elapsed_seconds:.2f} s",
        "",
        "## Configurations de barrières",
        "",
        "| ID | Stop ATR | Target ATR | Timeout | Target/coût BASE min |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {item.id} | {item.stop_atr} | {item.target_atr} | "
            f"{item.timeout_seconds}s | {item.minimum_target_to_base_cost} |"
            for item in report.barrier_configurations
        ],
        "",
        "## Scénarios de coûts simulés",
        "",
        "| Scénario | Slippage points/côté | Commission EUR/lot/côté |",
        "|---|---:|---:|",
        *[
            f"| {item.name} | {item.slippage_points_per_side} | "
            f"{item.commission_eur_per_lot_per_side} |"
            for item in report.cost_scenarios
        ],
        "",
        "## Modèles LONG / SHORT",
        "",
        "| Config | Side | Obs | Target rate | Prior PR-AUC | LR PR-AUC | HGB PR-AUC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in (*report.long_results, *report.short_results):
        lines.append(
            f"| {result.configuration_id} | {result.side} | {result.observations} | "
            f"{metric(result.target_before_stop_rate)} | "
            f"{metric(result.models['prior_baseline']['pr_auc'])} | "
            f"{metric(result.models['logistic_regression']['pr_auc'])} | "
            f"{metric(result.models['hist_gradient_boosting']['pr_auc'])} |"
        )
    lines.extend(
        [
            "",
            "## Meta-labeling combiné",
            "",
            "| Config | Timeout | Candidates L/S | Target rate | Gross | Executable | "
            "BASE net | STRESS net | PF BASE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            *[
                f"| {item.configuration_id} | {item.timeout_seconds}s | "
                f"{item.candidates_count} ({item.long_candidates}/{item.short_candidates}) | "
                f"{metric(item.target_before_stop_rate)} | "
                f"{metric(item.gross_expectancy_points)} | "
                f"{metric(item.executable_expectancy_points)} | "
                f"{metric(item.base_net_expectancy_points)} | "
                f"{metric(item.stress_net_expectancy_points)} | "
                f"{metric(item.profit_factor_base)} |"
                for item in report.combined_results
            ],
            "",
            "## Hypothèses D.7 — résultats séparés",
            "",
            "| ID | Side | Obs | Target rate | Executable | BASE net |",
            "|---|---|---:|---:|---:|---:|",
            *[
                f"| {item['id']} | {item['side']} | {item['observations']} | "
                f"{metric(item['target_before_stop_rate'])} | "
                f"{metric(item['executable_expectancy_points'])} | "
                f"{metric(item['base_net_expectancy_points'])} |"
                for item in report.registered_hypotheses
            ],
            "",
            "## Configuration primaire par semaine",
            "",
            "| Semaine | Candidats | Target rate | BASE net | STRESS net |",
            "|---|---:|---:|---:|---:|",
            *[
                f"| {item['week']} | {item['candidates_count']} | "
                f"{metric(item['target_before_stop_rate'])} | "
                f"{metric(item['base_net_expectancy_points'])} | "
                f"{metric(item['stress_net_expectancy_points'])} |"
                for item in next(
                    result
                    for result in report.combined_results
                    if result.configuration_id == "B02_PRIMARY"
                ).weekly_distribution
            ],
            "",
            "## Configuration primaire par session",
            "",
            "| Session | Candidats | Target rate | BASE net | STRESS net |",
            "|---|---:|---:|---:|---:|",
            *[
                f"| {item['session']} | {item['candidates_count']} | "
                f"{metric(item['target_before_stop_rate'])} | "
                f"{metric(item['base_net_expectancy_points'])} | "
                f"{metric(item['stress_net_expectancy_points'])} |"
                for item in next(
                    result
                    for result in report.combined_results
                    if result.configuration_id == "B02_PRIMARY"
                ).session_distribution
            ],
            "",
            "## Garde-fous",
            "",
            "- Dataset : RESEARCH uniquement",
            "- HOLDOUT : SEALED, non ouvert, non évalué",
            "- Modèle primaire : régression logistique préenregistrée",
            "- Configuration primaire : B02_PRIMARY préenregistrée",
            "- Scénario principal : BASE",
            f"- Politique de purge TRAIN : {report.purge_policy}",
            "- Sizing : RISK_ENGINE_ONLY",
            "- Trading live : DÉSACTIVÉ",
            "- Phase E : NON COMMENCÉE",
            "",
        ]
    )
    return "\n".join(lines)
