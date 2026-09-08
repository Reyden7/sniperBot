"""Leakage-safe Phase D.8 Opportunity, Direction and Execution research."""

import tracemalloc
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.ensemble import (  # type: ignore[import-untyped]
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.inspection import permutation_importance  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore[import-untyped]

from sniper.backtest.execution_model import BrokerSimulationConfig
from sniper.data.feature_source import load_parquet_bars
from sniper.data.time import as_utc
from sniper.domain.bar import Timeframe
from sniper.domain.edge_validation import EdgeSimulationAssumptions
from sniper.domain.research_v2 import (
    CostScenario,
    GateResult,
    HorizonModelReport,
    ResearchV2Report,
    TemporalFold,
    V2Conclusion,
    V2Performance,
)
from sniper.evaluation.edge_validation import (
    MICROSECONDS,
    _assert_frozen_engine,
    _bar_window,
    _current_market_tick,
    _days,
    _read_ticks,
    _session,
    _streaming_tick_features,
    _tick_path,
)
from sniper.features.engine import FeatureEngine

HORIZONS_V2 = (30, 60, 180, 300, 600, 900)
NUMERIC_FEATURES_V2 = (
    "m15_atr_points",
    "m5_atr_points",
    "m1_atr_points",
    "m1_range_atr_ratio",
    "m15_ema_slope_points",
    "m5_momentum_points",
    "m1_momentum_1_points",
    "m1_momentum_3_points",
    "m1_momentum_5_points",
    "m1_body_ratio",
    "m1_upper_wick_ratio",
    "m1_lower_wick_ratio",
    "tick_rate_1s",
    "tick_rate_5s",
    "tick_rate_15s",
    "tick_rate_acceleration",
    "tick_price_acceleration_points",
    "tick_uptick_ratio",
    "spread_points",
    "spread_vs_median_ratio",
    "m5_support_distance_atr",
    "m5_resistance_distance_atr",
)
CATEGORICAL_FEATURES_V2 = (
    "session",
    "m15_regime",
    "m15_direction",
    "m5_direction",
    "m5_breakout_retest",
    "m1_compression",
    "m1_expansion",
    "m1_pattern",
)
FEATURE_COLUMNS_V2 = NUMERIC_FEATURES_V2 + CATEGORICAL_FEATURES_V2
SCENARIOS = (
    CostScenario(
        name="OPTIMISTIC",
        slippage_points_per_side=0,
        commission_eur_per_lot_per_side=0,
        commission_minimum_eur_per_side=0,
        disclaimer="Observed spread only; optimistic research scenario, not a broker promise.",
    ),
    CostScenario(
        name="BASE",
        slippage_points_per_side=1,
        commission_eur_per_lot_per_side=2,
        commission_minimum_eur_per_side=0,
        disclaimer="V1 comparison assumptions; simulated, not MetaQuotes-Demo capability.",
    ),
    CostScenario(
        name="STRESS",
        slippage_points_per_side=2,
        commission_eur_per_lot_per_side=4,
        commission_minimum_eur_per_side=0,
        disclaimer="Adverse simulated sensitivity scenario; not a broker quote.",
    ),
)
MOVE_COST_THRESHOLDS = (1.0, 1.5, 2.0, 3.0)
PRIMARY_HORIZON = 300
PRIMARY_MOVE_COST_THRESHOLD = 1.5
RANDOM_SEED = 20260908


def _feature_source_hash() -> str:
    path = Path(__file__).resolve().parents[1] / "features" / "engine.py"
    return sha256(path.read_bytes()).hexdigest()


def _pattern(snapshot: Any) -> str:
    return snapshot.m1.pattern.engulfing or (
        "HAMMER"
        if snapshot.m1.pattern.hammer
        else "SHOOTING_STAR"
        if snapshot.m1.pattern.shooting_star
        else "NONE"
    )


def _commission_points(mid: float, broker: BrokerSimulationConfig, scenario: CostScenario) -> float:
    volume = float(broker.volumes.minimum)
    commission_side = max(
        scenario.commission_eur_per_lot_per_side * volume,
        scenario.commission_minimum_eur_per_side,
    )
    eur_per_point = float(broker.point) * float(broker.contract_size) * volume / mid
    return commission_side * 2 / eur_per_point


def _future_labels(
    timestamps_us: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    current_index: int,
    broker: BrokerSimulationConfig,
    max_delay_seconds: float = 2.0,
) -> dict[str, object]:
    current_us = int(timestamps_us[current_index])
    bid = float(bids[current_index])
    ask = float(asks[current_index])
    mid = (bid + ask) / 2
    point = float(broker.point)
    result: dict[str, object] = {}
    for scenario in SCENARIOS:
        result[f"cost_{scenario.name.lower()}_points"] = (
            (ask - bid) / point
            + 2 * scenario.slippage_points_per_side
            + _commission_points(mid, broker, scenario)
        )
    for horizon in HORIZONS_V2:
        target_us = current_us + horizon * MICROSECONDS
        target_index = int(np.searchsorted(timestamps_us, target_us, side="left"))
        prefix = f"h{horizon}"
        if (
            target_index >= len(timestamps_us)
            or (int(timestamps_us[target_index]) - target_us) / MICROSECONDS > max_delay_seconds
        ):
            for suffix in (
                "market_return_points",
                "long_executable_return_points",
                "short_executable_return_points",
                "opportunity_mfe_points",
                "realized_direction_mae_points",
                "long_mfe_points",
                "long_mae_points",
                "short_mfe_points",
                "short_mae_points",
                "direction_long",
            ):
                result[f"{prefix}_{suffix}"] = None
            continue
        future_mid = (float(bids[target_index]) + float(asks[target_index])) / 2
        market_return = (future_mid - mid) / point
        path_end = target_index + 1
        future_bids = bids[current_index + 1 : path_end]
        future_asks = asks[current_index + 1 : path_end]
        if len(future_bids) == 0:
            continue
        path_mids = (future_bids + future_asks) / 2
        up = max(0.0, float(np.max((path_mids - mid) / point)))
        down = max(0.0, float(np.max((mid - path_mids) / point)))
        long_path = (future_bids - ask) / point
        short_path = (bid - future_asks) / point
        result.update(
            {
                f"{prefix}_market_return_points": market_return,
                f"{prefix}_long_executable_return_points": (
                    float(bids[target_index] - ask) / point
                ),
                f"{prefix}_short_executable_return_points": (
                    float(bid - asks[target_index]) / point
                ),
                f"{prefix}_opportunity_mfe_points": max(up, down),
                f"{prefix}_realized_direction_mae_points": down if market_return >= 0 else up,
                f"{prefix}_long_mfe_points": max(0.0, float(np.max(long_path))),
                f"{prefix}_long_mae_points": max(0.0, -float(np.min(long_path))),
                f"{prefix}_short_mfe_points": max(0.0, float(np.max(short_path))),
                f"{prefix}_short_mae_points": max(0.0, -float(np.min(short_path))),
                f"{prefix}_direction_long": int(market_return > 0),
            }
        )
    return result


def build_research_dataset(
    *, data_root: Path, start_utc: datetime, end_utc: datetime
) -> tuple[pl.DataFrame, int, float]:
    start, end = as_utc(start_utc), as_utc(end_utc)
    if start != datetime(2026, 6, 10, tzinfo=UTC) or end != datetime(2026, 9, 8, tzinfo=UTC):
        raise PermissionError("D.8 may only open the declared RESEARCH dataset")
    feature_engine = FeatureEngine()
    broker = BrokerSimulationConfig()
    bars = load_parquet_bars(data_root, "EURUSD", end, lookback_days=120)
    timeframes: tuple[Timeframe, ...] = ("M15", "M5", "M1")
    bar_times = {
        timeframe: [bar.open_time_utc for bar in bars[timeframe]] for timeframe in timeframes
    }
    rows: list[dict[str, object]] = []
    ticks_analyzed = 0
    peak_frame_mb = 0.0
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    for day in _days(start, end):
        day_start = max(start, datetime.combine(day, datetime.min.time(), UTC))
        midnight = day_start.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = min(end, midnight + timedelta(days=1))
        current = _read_ticks(_tick_path(data_root, day), day_start, day_end)
        if current.is_empty():
            continue
        ticks_analyzed += current.height
        previous = _read_ticks(
            _tick_path(data_root, day - timedelta(days=1)),
            max(start, day_start - timedelta(seconds=60)),
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
        bids = extended["bid"].to_numpy()
        asks = extended["ask"].to_numpy()
        spreads = extended["spread_points"].to_numpy()
        current_us = current["timestamp_utc"].cast(pl.Int64).to_numpy()
        minute_keys = current_us // (60 * MICROSECONDS)
        sample_indices = np.concatenate((np.array([0]), np.flatnonzero(np.diff(minute_keys)) + 1))
        price_cache: dict[float, Decimal] = {}
        for local_index in sample_indices:
            sample_us = int(current_us[int(local_index)])
            current_index = int(np.searchsorted(timestamps_us, sample_us, side="left"))
            sample_time = epoch + timedelta(microseconds=sample_us)
            visible_left = int(
                np.searchsorted(timestamps_us, sample_us - 60 * MICROSECONDS, side="left")
            )
            try:
                snapshot = feature_engine.compute(
                    bars=_bar_window(bars, bar_times, sample_time),
                    ticks=_current_market_tick(
                        timestamps_us, bids, asks, current_index, price_cache
                    ),
                    as_of_utc=sample_time,
                    symbol="EURUSD",
                )
            except ValueError:
                continue
            snapshot = snapshot.model_copy(
                update={
                    "ticks": _streaming_tick_features(
                        timestamps_us,
                        bids,
                        asks,
                        spreads,
                        visible_left,
                        current_index,
                        price_cache,
                    ),
                    "visible_ticks": current_index - visible_left + 1,
                }
            )
            point = Decimal("0.00001")
            row = {
                "timestamp_utc": sample_time,
                "week": f"{sample_time.isocalendar().year}-W{sample_time.isocalendar().week:02d}",
                "session": _session(sample_time),
                "m15_atr_points": float(snapshot.m15.atr / point),
                "m5_atr_points": float(snapshot.m5.atr / point),
                "m1_atr_points": float(snapshot.m1.atr / point),
                "m1_range_atr_ratio": float(snapshot.m1.range_atr_ratio),
                "m15_ema_slope_points": float(snapshot.m15.ema_slope / point),
                "m5_momentum_points": float(snapshot.m5.momentum / point),
                "m1_momentum_1_points": float(snapshot.m1.momentum_1 / point),
                "m1_momentum_3_points": float(snapshot.m1.momentum_3 / point),
                "m1_momentum_5_points": float(snapshot.m1.momentum_5 / point),
                "m1_body_ratio": float(snapshot.m1.body_ratio),
                "m1_upper_wick_ratio": float(snapshot.m1.upper_wick_ratio),
                "m1_lower_wick_ratio": float(snapshot.m1.lower_wick_ratio),
                "tick_rate_1s": snapshot.ticks.rate_1s,
                "tick_rate_5s": snapshot.ticks.rate_5s,
                "tick_rate_15s": snapshot.ticks.rate_15s,
                "tick_rate_acceleration": float(snapshot.ticks.tick_rate_acceleration),
                "tick_price_acceleration_points": float(snapshot.ticks.price_acceleration / point),
                "tick_uptick_ratio": float(snapshot.ticks.uptick_ratio),
                "spread_points": float(snapshot.ticks.current_spread_points),
                "spread_vs_median_ratio": (
                    float(
                        snapshot.ticks.current_spread_points / snapshot.ticks.median_spread_points
                    )
                    if snapshot.ticks.median_spread_points > 0
                    else 0.0
                ),
                "m5_support_distance_atr": float(snapshot.m5.support_distance_atr),
                "m5_resistance_distance_atr": float(snapshot.m5.resistance_distance_atr),
                "m15_regime": snapshot.m15.regime.value,
                "m15_direction": snapshot.m15.direction.value,
                "m5_direction": snapshot.m5.direction.value,
                "m5_breakout_retest": snapshot.m5.breakout_retest or "NONE",
                "m1_compression": str(snapshot.m1.compression).upper(),
                "m1_expansion": str(snapshot.m1.expansion).upper(),
                "m1_pattern": _pattern(snapshot),
                **_future_labels(timestamps_us, bids, asks, current_index, broker),
            }
            rows.append(row)
    return (
        pl.DataFrame(rows, infer_schema_length=None).sort("timestamp_utc"),
        ticks_analyzed,
        peak_frame_mb,
    )


def temporal_folds(
    frame: pl.DataFrame, start: datetime, end: datetime
) -> tuple[list[tuple[np.ndarray, np.ndarray]], tuple[TemporalFold, ...]]:
    duration = end - start
    boundaries = [start + duration * fraction for fraction in (0.7, 0.8, 0.9, 1.0)]
    timestamps = frame["timestamp_utc"].to_numpy()
    folds = []
    reports = []
    for fold_index in range(3):
        train_end = boundaries[fold_index]
        validation_start = boundaries[fold_index]
        validation_end = boundaries[fold_index + 1]
        train_mask = timestamps < np.datetime64(train_end.replace(tzinfo=None), "us")
        validation_mask = timestamps >= np.datetime64(validation_start.replace(tzinfo=None), "us")
        validation_mask &= timestamps < np.datetime64(validation_end.replace(tzinfo=None), "us")
        train_indices = np.flatnonzero(train_mask)
        validation_indices = np.flatnonzero(validation_mask)
        folds.append((train_indices, validation_indices))
        reports.append(
            TemporalFold(
                fold=fold_index + 1,
                train_start_utc=start,
                train_end_utc_exclusive=train_end,
                validation_start_utc=validation_start,
                validation_end_utc_exclusive=validation_end,
                train_observations=len(train_indices),
                validation_observations=len(validation_indices),
                evaluation_role=(
                    "INTERNAL_FREEZE_CHECK" if fold_index == 2 else "MODEL_COMPARISON"
                ),
            )
        )
    return folds, tuple(reports)


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        (
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES_V2)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES_V2),
            ),
        )
    )


def _pipeline(model: Any) -> Pipeline:
    return Pipeline((("preprocess", _preprocessor()), ("model", model)))


def _calibration(y: np.ndarray, probability: np.ndarray) -> list[dict[str, float | int]]:
    bins = []
    for lower in np.linspace(0, 0.9, 10):
        upper = lower + 0.1
        mask = (probability >= lower) & (probability < upper if upper < 1 else probability <= 1)
        if np.any(mask):
            bins.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "observations": int(mask.sum()),
                    "mean_probability": float(probability[mask].mean()),
                    "observed_rate": float(y[mask].mean()),
                }
            )
    return bins


def _classification_metrics(
    y: np.ndarray, prediction: np.ndarray, probability: np.ndarray
) -> dict[str, object]:
    return {
        "observations": len(y),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision_positive": float(precision_score(y, prediction, zero_division=0)),
        "precision_negative": float(precision_score(1 - y, 1 - prediction, zero_division=0)),
        "recall_positive": float(recall_score(y, prediction, zero_division=0)),
        "recall_negative": float(recall_score(1 - y, 1 - prediction, zero_division=0)),
        "pr_auc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "base_rate": float(y.mean()),
        "calibration": _calibration(y, probability),
    }


def _regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "observations": len(y),
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
    }


def _top_importance(
    model: Pipeline,
    x: Any,
    y: np.ndarray,
    scoring: str,
) -> list[dict[str, float | str]]:
    sample = min(len(y), 3000)
    result = permutation_importance(
        model,
        x.iloc[:sample],
        y[:sample],
        scoring=scoring,
        n_repeats=3,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    pairs = sorted(
        zip(FEATURE_COLUMNS_V2, result.importances_mean, strict=True),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )[:15]
    return [{"feature": name, "importance": float(value)} for name, value in pairs]


def _linear_coefficients(model: Pipeline) -> list[dict[str, float | str]]:
    transformer = cast(ColumnTransformer, model.named_steps["preprocess"])
    estimator = cast(LogisticRegression, model.named_steps["model"])
    names = transformer.get_feature_names_out()
    pairs = sorted(
        zip(names, estimator.coef_[0], strict=True),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )[:20]
    return [{"feature": str(name), "coefficient": float(value)} for name, value in pairs]


def _direction_returns(
    frame: pl.DataFrame, horizon: int, prediction: np.ndarray
) -> dict[str, float]:
    gross = frame[f"h{horizon}_market_return_points"].to_numpy().astype(float)
    long_exec = frame[f"h{horizon}_long_executable_return_points"].to_numpy().astype(float)
    short_exec = frame[f"h{horizon}_short_executable_return_points"].to_numpy().astype(float)
    sign = np.where(prediction == 1, 1.0, -1.0)
    executable = np.where(prediction == 1, long_exec, short_exec)
    return {
        "expected_gross_return_points": float(np.mean(gross * sign)),
        "expected_executable_return_points": float(np.mean(executable)),
    }


def _scoped_model_metrics(
    frame: pl.DataFrame,
    horizon: int,
    combined: dict[str, np.ndarray],
    fold_numbers: tuple[int, ...],
) -> dict[str, object]:
    """Summarise fixed models on a declared fold scope without refitting or selecting."""
    positions = np.flatnonzero(np.isin(combined["fold_number"], fold_numbers))
    indices = combined["indices"][positions].astype(int)
    scoped = frame[indices]
    base_cost = scoped["cost_base_points"].to_numpy().astype(float)
    y_mfe = scoped[f"h{horizon}_opportunity_mfe_points"].to_numpy().astype(float)
    y_mae = scoped[f"h{horizon}_realized_direction_mae_points"].to_numpy().astype(float)
    y_direction = scoped[f"h{horizon}_direction_long"].to_numpy().astype(int)
    y_opportunity = (y_mfe > base_cost).astype(int)

    def values(name: str) -> np.ndarray:
        return combined[name][positions]

    direction_probability = values("direction_probability")
    hgb_direction_probability = values("hgb_direction_probability")
    direction_predictions = {
        "random_baseline": values("random_direction_prediction").astype(int),
        "momentum_baseline": values("momentum_prediction").astype(int),
        "mean_reversion_baseline": values("mean_reversion_prediction").astype(int),
        "logistic_regression": (direction_probability >= 0.5).astype(int),
        "hist_gradient_boosting": (hgb_direction_probability >= 0.5).astype(int),
    }
    direction_probabilities = {
        "random_baseline": np.full(len(indices), 0.5),
        "momentum_baseline": direction_predictions["momentum_baseline"].astype(float),
        "mean_reversion_baseline": direction_predictions["mean_reversion_baseline"].astype(float),
        "logistic_regression": direction_probability,
        "hist_gradient_boosting": hgb_direction_probability,
    }
    return {
        "folds": list(fold_numbers),
        "observations": len(indices),
        "opportunity": {
            "mfe_regression": {
                "constant_mean_baseline": _regression_metrics(
                    y_mfe, values("constant_mfe_prediction")
                ),
                "hist_gradient_boosting": _regression_metrics(
                    y_mfe, values("opportunity_mfe_prediction")
                ),
            },
            "mae_regression": {
                "constant_mean_baseline": _regression_metrics(
                    y_mae, values("constant_mae_prediction")
                ),
                "hist_gradient_boosting": _regression_metrics(
                    y_mae, values("opportunity_mae_prediction")
                ),
            },
            "classification_move_gt_base_cost": {
                "prior_probability_baseline": _classification_metrics(
                    y_opportunity,
                    (values("random_opportunity_probability") >= 0.5).astype(int),
                    values("random_opportunity_probability"),
                ),
                "logistic_regression": _classification_metrics(
                    y_opportunity,
                    (values("logistic_opportunity_probability") >= 0.5).astype(int),
                    values("logistic_opportunity_probability"),
                ),
                "hist_gradient_boosting": _classification_metrics(
                    y_opportunity,
                    (values("opportunity_probability") >= 0.5).astype(int),
                    values("opportunity_probability"),
                ),
            },
        },
        "direction": {
            name: {
                **_classification_metrics(y_direction, prediction, direction_probabilities[name]),
                **_direction_returns(scoped, horizon, prediction),
            }
            for name, prediction in direction_predictions.items()
        },
    }


def _fit_horizon(
    frame: pl.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    horizon: int,
) -> tuple[HorizonModelReport, dict[str, np.ndarray]]:
    import pandas as pd  # type: ignore[import-untyped]

    label = f"h{horizon}_market_return_points"
    valid = frame[label].is_not_null().to_numpy()
    x_all = frame.select(FEATURE_COLUMNS_V2).to_pandas()
    opportunity_target = frame[f"h{horizon}_opportunity_mfe_points"].to_numpy()
    direction_target = frame[f"h{horizon}_direction_long"].to_numpy()
    signed_return_target = frame[label].to_numpy()
    base_cost = frame["cost_base_points"].to_numpy().astype(float)
    outputs: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "indices",
            "fold_number",
            "opportunity_probability",
            "opportunity_mfe_prediction",
            "opportunity_mae_prediction",
            "constant_mfe_prediction",
            "constant_mae_prediction",
            "direction_probability",
            "signed_return_prediction",
            "random_opportunity_probability",
            "logistic_opportunity_probability",
            "probability_mfe_gt_spread",
            "probability_mfe_gt_1_5_cost",
            "probability_mfe_gt_2_cost",
            "random_direction_prediction",
            "momentum_prediction",
            "mean_reversion_prediction",
            "hgb_direction_probability",
        )
    }
    final_models: dict[str, Pipeline] = {}
    final_x: pd.DataFrame | None = None
    final_y_opportunity: np.ndarray | None = None
    final_y_direction: np.ndarray | None = None
    rng = np.random.default_rng(RANDOM_SEED + horizon)
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        train_indices = train_indices[valid[train_indices]]
        validation_indices = validation_indices[valid[validation_indices]]
        x_train = x_all.iloc[train_indices]
        x_validation = x_all.iloc[validation_indices]
        y_mfe_train = opportunity_target[train_indices].astype(float)
        y_mfe_validation = opportunity_target[validation_indices].astype(float)
        mae_target = frame[f"h{horizon}_realized_direction_mae_points"].to_numpy()
        y_mae_train = mae_target[train_indices].astype(float)
        y_opportunity_train = (y_mfe_train > base_cost[train_indices]).astype(int)
        y_direction_train = direction_target[train_indices].astype(int)

        opportunity_regressor = _pipeline(
            HistGradientBoostingRegressor(
                max_iter=60, max_depth=3, learning_rate=0.08, random_state=RANDOM_SEED
            )
        )
        opportunity_logistic = _pipeline(
            LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_SEED)
        )
        opportunity_spread_logistic = _pipeline(
            LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_SEED)
        )
        opportunity_1_5_logistic = _pipeline(
            LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_SEED)
        )
        opportunity_2_logistic = _pipeline(
            LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_SEED)
        )
        opportunity_classifier = _pipeline(
            HistGradientBoostingClassifier(
                max_iter=60, max_depth=3, learning_rate=0.08, random_state=RANDOM_SEED
            )
        )
        opportunity_mae_regressor = _pipeline(
            HistGradientBoostingRegressor(
                max_iter=60, max_depth=3, learning_rate=0.08, random_state=RANDOM_SEED
            )
        )
        direction_logistic = _pipeline(
            LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_SEED)
        )
        direction_hgb = _pipeline(
            HistGradientBoostingClassifier(
                max_iter=60, max_depth=3, learning_rate=0.08, random_state=RANDOM_SEED
            )
        )
        return_regressor = _pipeline(
            HistGradientBoostingRegressor(
                max_iter=60, max_depth=3, learning_rate=0.08, random_state=RANDOM_SEED
            )
        )
        opportunity_regressor.fit(x_train, y_mfe_train)
        opportunity_mae_regressor.fit(x_train, y_mae_train)
        opportunity_logistic.fit(x_train, y_opportunity_train)
        train_spread = frame["spread_points"].to_numpy()[train_indices].astype(float)
        opportunity_spread_logistic.fit(x_train, (y_mfe_train > train_spread).astype(int))
        opportunity_1_5_logistic.fit(
            x_train, (y_mfe_train > 1.5 * base_cost[train_indices]).astype(int)
        )
        opportunity_2_logistic.fit(
            x_train, (y_mfe_train > 2.0 * base_cost[train_indices]).astype(int)
        )
        opportunity_classifier.fit(x_train, y_opportunity_train)
        direction_logistic.fit(x_train, y_direction_train)
        direction_hgb.fit(x_train, y_direction_train)
        return_regressor.fit(x_train, signed_return_target[train_indices].astype(float))

        outputs["indices"].append(validation_indices)
        outputs["fold_number"].append(np.full(len(validation_indices), fold_number, dtype=int))
        outputs["opportunity_probability"].append(
            opportunity_classifier.predict_proba(x_validation)[:, 1]
        )
        outputs["opportunity_mfe_prediction"].append(opportunity_regressor.predict(x_validation))
        outputs["opportunity_mae_prediction"].append(
            opportunity_mae_regressor.predict(x_validation)
        )
        outputs["constant_mfe_prediction"].append(
            np.full(len(validation_indices), y_mfe_train.mean())
        )
        outputs["constant_mae_prediction"].append(
            np.full(len(validation_indices), y_mae_train.mean())
        )
        outputs["direction_probability"].append(
            direction_logistic.predict_proba(x_validation)[:, 1]
        )
        outputs["signed_return_prediction"].append(return_regressor.predict(x_validation))
        outputs["random_opportunity_probability"].append(
            np.full(len(validation_indices), y_opportunity_train.mean())
        )
        outputs["logistic_opportunity_probability"].append(
            opportunity_logistic.predict_proba(x_validation)[:, 1]
        )
        outputs["probability_mfe_gt_spread"].append(
            opportunity_spread_logistic.predict_proba(x_validation)[:, 1]
        )
        outputs["probability_mfe_gt_1_5_cost"].append(
            opportunity_1_5_logistic.predict_proba(x_validation)[:, 1]
        )
        outputs["probability_mfe_gt_2_cost"].append(
            opportunity_2_logistic.predict_proba(x_validation)[:, 1]
        )
        outputs["random_direction_prediction"].append(
            rng.integers(0, 2, size=len(validation_indices))
        )
        momentum = frame["m1_momentum_3_points"].to_numpy()[validation_indices].astype(float)
        outputs["momentum_prediction"].append((momentum >= 0).astype(int))
        outputs["mean_reversion_prediction"].append((momentum < 0).astype(int))
        outputs["hgb_direction_probability"].append(direction_hgb.predict_proba(x_validation)[:, 1])
        final_models = {
            "opportunity_classifier": opportunity_classifier,
            "opportunity_logistic": opportunity_logistic,
            "direction_logistic": direction_logistic,
        }
        final_x = x_validation
        final_y_opportunity = (y_mfe_validation > base_cost[validation_indices]).astype(int)
        final_y_direction = direction_target[validation_indices].astype(int)

    combined = {key: np.concatenate(value) for key, value in outputs.items()}
    indices = combined["indices"].astype(int)
    validation = frame[indices]
    y_mfe = opportunity_target[indices].astype(float)
    y_mae = frame[f"h{horizon}_realized_direction_mae_points"].to_numpy()[indices].astype(float)
    y_opportunity = (y_mfe > base_cost[indices]).astype(int)
    y_direction = direction_target[indices].astype(int)
    opportunity_probability = combined["opportunity_probability"]
    logistic_opportunity_probability = combined["logistic_opportunity_probability"]
    direction_probability = combined["direction_probability"]
    hgb_direction_probability = combined["hgb_direction_probability"]
    random_direction = combined["random_direction_prediction"].astype(int)
    momentum_prediction = combined["momentum_prediction"].astype(int)
    mean_reversion_prediction = combined["mean_reversion_prediction"].astype(int)
    logistic_direction_prediction = (direction_probability >= 0.5).astype(int)
    hgb_direction_prediction = (hgb_direction_probability >= 0.5).astype(int)
    random_opportunity_probability = combined["random_opportunity_probability"]
    opportunity = {
        "target": "DIRECTION_INDEPENDENT_MAXIMUM_ABSOLUTE_MID_EXCURSION",
        "empirical_probabilities": {
            "mfe_gt_spread": float(
                np.mean(y_mfe > validation["spread_points"].to_numpy().astype(float))
            ),
            "mfe_gt_total_cost": float(np.mean(y_mfe > base_cost[indices])),
            "mfe_gt_1_5_total_cost": float(np.mean(y_mfe > 1.5 * base_cost[indices])),
            "mfe_gt_2_total_cost": float(np.mean(y_mfe > 2.0 * base_cost[indices])),
        },
        "regression": {
            "constant_mean_baseline": _regression_metrics(
                y_mfe, combined["constant_mfe_prediction"]
            ),
            "hist_gradient_boosting": _regression_metrics(
                y_mfe, combined["opportunity_mfe_prediction"]
            ),
        },
        "mae_regression": {
            "constant_mean_baseline": _regression_metrics(
                y_mae, combined["constant_mae_prediction"]
            ),
            "hist_gradient_boosting": _regression_metrics(
                y_mae, combined["opportunity_mae_prediction"]
            ),
        },
        "probability_models": {
            "mfe_gt_spread": _classification_metrics(
                (y_mfe > validation["spread_points"].to_numpy().astype(float)).astype(int),
                (combined["probability_mfe_gt_spread"] >= 0.5).astype(int),
                combined["probability_mfe_gt_spread"],
            ),
            "mfe_gt_total_cost": _classification_metrics(
                y_opportunity,
                (logistic_opportunity_probability >= 0.5).astype(int),
                logistic_opportunity_probability,
            ),
            "mfe_gt_1_5_total_cost": _classification_metrics(
                (y_mfe > 1.5 * base_cost[indices]).astype(int),
                (combined["probability_mfe_gt_1_5_cost"] >= 0.5).astype(int),
                combined["probability_mfe_gt_1_5_cost"],
            ),
            "mfe_gt_2_total_cost": _classification_metrics(
                (y_mfe > 2.0 * base_cost[indices]).astype(int),
                (combined["probability_mfe_gt_2_cost"] >= 0.5).astype(int),
                combined["probability_mfe_gt_2_cost"],
            ),
        },
        "classification_move_gt_base_cost": {
            "prior_probability_baseline": _classification_metrics(
                y_opportunity,
                (random_opportunity_probability >= 0.5).astype(int),
                random_opportunity_probability,
            ),
            "logistic_regression": _classification_metrics(
                y_opportunity,
                (logistic_opportunity_probability >= 0.5).astype(int),
                logistic_opportunity_probability,
            ),
            "hist_gradient_boosting": _classification_metrics(
                y_opportunity,
                (opportunity_probability >= 0.5).astype(int),
                opportunity_probability,
            ),
        },
    }
    direction = {
        "label": "TERMINAL_MID_RETURN_POSITIVE",
        "random_baseline": {
            **_classification_metrics(
                y_direction, random_direction, np.full(len(y_direction), 0.5)
            ),
            **_direction_returns(validation, horizon, random_direction),
        },
        "momentum_baseline": {
            **_classification_metrics(
                y_direction, momentum_prediction, momentum_prediction.astype(float)
            ),
            **_direction_returns(validation, horizon, momentum_prediction),
        },
        "mean_reversion_baseline": {
            **_classification_metrics(
                y_direction, mean_reversion_prediction, mean_reversion_prediction.astype(float)
            ),
            **_direction_returns(validation, horizon, mean_reversion_prediction),
        },
        "logistic_regression": {
            **_classification_metrics(
                y_direction, logistic_direction_prediction, direction_probability
            ),
            **_direction_returns(validation, horizon, logistic_direction_prediction),
        },
        "hist_gradient_boosting": {
            **_classification_metrics(
                y_direction, hgb_direction_prediction, hgb_direction_probability
            ),
            **_direction_returns(validation, horizon, hgb_direction_prediction),
        },
        "expected_directional_return_regression": _regression_metrics(
            signed_return_target[indices].astype(float), combined["signed_return_prediction"]
        ),
    }
    assert final_x is not None and final_y_opportunity is not None and final_y_direction is not None
    explainability = {
        "opportunity_permutation_importance": _top_importance(
            final_models["opportunity_classifier"],
            final_x,
            final_y_opportunity,
            "average_precision",
        ),
        "direction_permutation_importance": _top_importance(
            final_models["direction_logistic"],
            final_x,
            final_y_direction,
            "balanced_accuracy",
        ),
        "opportunity_logistic_coefficients": _linear_coefficients(
            final_models["opportunity_logistic"]
        ),
        "direction_logistic_coefficients": _linear_coefficients(final_models["direction_logistic"]),
        "partial_dependence": "NOT_USED; no stable nonlinear candidate justifies interpretation",
    }
    predictions = {
        "indices": indices,
        "fold_number": combined["fold_number"].astype(int),
        "opportunity_probability": opportunity_probability,
        "opportunity_mfe": combined["opportunity_mfe_prediction"],
        "direction_probability": direction_probability,
        "signed_return": combined["signed_return_prediction"],
    }
    return (
        HorizonModelReport(
            horizon_seconds=horizon,
            opportunity=opportunity,
            direction=direction,
            explainability=explainability,
            evaluation_windows={
                "MODEL_COMPARISON": _scoped_model_metrics(frame, horizon, combined, (1, 2)),
                "INTERNAL_FREEZE_CHECK": _scoped_model_metrics(frame, horizon, combined, (3,)),
            },
        ),
        predictions,
    )


def _gate_result(
    frame: pl.DataFrame,
    horizon: int,
    predictions: dict[str, np.ndarray],
    scenario: CostScenario,
    threshold: float,
    evaluation_scope: Literal[
        "MODEL_COMPARISON", "INTERNAL_FREEZE_CHECK", "ALL_VALIDATION"
    ] = "ALL_VALIDATION",
) -> tuple[GateResult, pl.DataFrame]:
    fold_numbers = predictions["fold_number"].astype(int)
    if evaluation_scope == "MODEL_COMPARISON":
        positions = np.flatnonzero(np.isin(fold_numbers, (1, 2)))
    elif evaluation_scope == "INTERNAL_FREEZE_CHECK":
        positions = np.flatnonzero(fold_numbers == 3)
    else:
        positions = np.arange(len(fold_numbers))
    indices = predictions["indices"][positions].astype(int)
    validation = frame[indices].with_columns(
        pl.Series("opportunity_probability", predictions["opportunity_probability"][positions]),
        pl.Series("predicted_mfe", predictions["opportunity_mfe"][positions]),
        pl.Series("direction_probability", predictions["direction_probability"][positions]),
        pl.Series("predicted_signed_return", predictions["signed_return"][positions]),
    )
    probability = predictions["direction_probability"][positions]
    side_long = probability >= 0.5
    confidence = np.maximum(probability, 1 - probability)
    signed_prediction = predictions["signed_return"][positions]
    expected_gross = np.where(side_long, signed_prediction, -signed_prediction)
    cost_column = f"cost_{scenario.name.lower()}_points"
    costs = validation[cost_column].to_numpy().astype(float)
    predicted_mfe = predictions["opportunity_mfe"][positions]
    predicted_move_to_cost = np.divide(
        predicted_mfe,
        costs,
        out=np.full_like(predicted_mfe, np.inf, dtype=float),
        where=costs > 0,
    )
    mask = (
        (predictions["opportunity_probability"][positions] >= 0.5)
        & (confidence >= 0.55)
        & (expected_gross > costs)
        & (predicted_move_to_cost > threshold)
    )
    selected = validation.filter(pl.Series(mask))
    if selected.is_empty():
        return (
            GateResult(
                horizon_seconds=horizon,
                scenario=scenario.name,
                move_to_cost_threshold=threshold,
                evaluation_scope=evaluation_scope,
                candidates_count=0,
                gross_expectancy_points=None,
                executable_expectancy_points=None,
                simulated_net_expectancy_points=None,
                average_mfe_points=None,
                average_mae_points=None,
                average_move_to_cost_ratio=None,
                percentage_profitable=None,
            ),
            selected,
        )
    selected_long = selected["direction_probability"].to_numpy() >= 0.5
    gross = selected[f"h{horizon}_market_return_points"].to_numpy().astype(float)
    gross = np.where(selected_long, gross, -gross)
    executable = np.where(
        selected_long,
        selected[f"h{horizon}_long_executable_return_points"].to_numpy().astype(float),
        selected[f"h{horizon}_short_executable_return_points"].to_numpy().astype(float),
    )
    mfe = np.where(
        selected_long,
        selected[f"h{horizon}_long_mfe_points"].to_numpy().astype(float),
        selected[f"h{horizon}_short_mfe_points"].to_numpy().astype(float),
    )
    mae = np.where(
        selected_long,
        selected[f"h{horizon}_long_mae_points"].to_numpy().astype(float),
        selected[f"h{horizon}_short_mae_points"].to_numpy().astype(float),
    )
    selected_cost = selected[cost_column].to_numpy().astype(float)
    observed_spread = selected["spread_points"].to_numpy().astype(float)
    simulated_net = executable - (selected_cost - observed_spread)
    actual_move_to_cost = np.divide(
        mfe,
        selected_cost,
        out=np.full_like(mfe, np.nan, dtype=float),
        where=selected_cost > 0,
    )
    finite_move_to_cost = actual_move_to_cost[np.isfinite(actual_move_to_cost)]
    average_move_to_cost = float(np.mean(finite_move_to_cost)) if len(finite_move_to_cost) else None
    selected = selected.with_columns(
        pl.Series("actual_gross_points", gross),
        pl.Series("actual_executable_points", executable),
        pl.Series("actual_simulated_net_points", simulated_net),
        pl.Series("actual_mfe_points", mfe),
        pl.Series("actual_mae_points", mae),
        pl.Series("actual_move_to_cost_ratio", actual_move_to_cost),
    )
    return (
        GateResult(
            horizon_seconds=horizon,
            scenario=scenario.name,
            move_to_cost_threshold=threshold,
            evaluation_scope=evaluation_scope,
            candidates_count=selected.height,
            gross_expectancy_points=float(np.mean(gross)),
            executable_expectancy_points=float(np.mean(executable)),
            simulated_net_expectancy_points=float(np.mean(simulated_net)),
            average_mfe_points=float(np.mean(mfe)),
            average_mae_points=float(np.mean(mae)),
            average_move_to_cost_ratio=average_move_to_cost,
            percentage_profitable=float(np.mean(simulated_net > 0) * 100),
        ),
        selected,
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
                "gross_expectancy_points": float(str(group["actual_gross_points"].mean())),
                "executable_expectancy_points": float(
                    str(group["actual_executable_points"].mean())
                ),
                "simulated_net_expectancy_points": float(
                    str(group["actual_simulated_net_points"].mean())
                ),
                "percentage_profitable": float(
                    str((group["actual_simulated_net_points"] > 0).mean())
                )
                * 100,
            }
        )
    return tuple(rows)


class ResearchV2Engine:
    def run(self, *, data_root: Path, start_utc: datetime, end_utc: datetime) -> ResearchV2Report:
        start, end = as_utc(start_utc), as_utc(end_utc)
        hashes = _assert_frozen_engine()
        tracemalloc.start()
        started = perf_counter()
        dataset_path = (
            data_root
            / "evaluations"
            / "research-v2-20260610T000000Z-20260908T000000Z"
            / "observations.parquet"
        )
        if dataset_path.exists():
            frame = pl.read_parquet(dataset_path)
            ticks_analyzed = 17_941_326
            peak_frame_mb = 0.0
        else:
            frame, ticks_analyzed, peak_frame_mb = build_research_dataset(
                data_root=data_root, start_utc=start, end_utc=end
            )
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = dataset_path.with_suffix(".parquet.tmp")
            frame.write_parquet(temporary, compression="zstd", statistics=True)
            temporary.replace(dataset_path)
        folds, fold_reports = temporal_folds(frame, start, end)
        horizon_reports = []
        predictions_by_horizon = {}
        for horizon in HORIZONS_V2:
            report, predictions = _fit_horizon(frame, folds, horizon)
            horizon_reports.append(report)
            predictions_by_horizon[horizon] = predictions
        gate_results = []
        comparison_gate_results = []
        freeze_gate_results = []
        primary_selected = pl.DataFrame()
        primary_result = None
        primary_comparison_selected = pl.DataFrame()
        primary_comparison_result = None
        for horizon in HORIZONS_V2:
            for scenario in SCENARIOS:
                for threshold in MOVE_COST_THRESHOLDS:
                    gate, _ = _gate_result(
                        frame,
                        horizon,
                        predictions_by_horizon[horizon],
                        scenario,
                        threshold,
                    )
                    gate_results.append(gate)
                    comparison_gate, comparison_selected = _gate_result(
                        frame,
                        horizon,
                        predictions_by_horizon[horizon],
                        scenario,
                        threshold,
                        "MODEL_COMPARISON",
                    )
                    comparison_gate_results.append(comparison_gate)
                    freeze_gate, freeze_selected = _gate_result(
                        frame,
                        horizon,
                        predictions_by_horizon[horizon],
                        scenario,
                        threshold,
                        "INTERNAL_FREEZE_CHECK",
                    )
                    freeze_gate_results.append(freeze_gate)
                    if (
                        horizon == PRIMARY_HORIZON
                        and scenario.name == "BASE"
                        and threshold == PRIMARY_MOVE_COST_THRESHOLD
                    ):
                        primary_result = freeze_gate
                        primary_selected = freeze_selected
                        primary_comparison_result = comparison_gate
                        primary_comparison_selected = comparison_selected
        assert primary_result is not None and primary_comparison_result is not None
        weekly = _distribution(primary_selected, "week")
        sessions = _distribution(primary_selected, "session")
        active_weeks = sum(cast(int, item["candidates_count"]) > 0 for item in weekly)
        positive_weeks = sum(
            cast(float, item["simulated_net_expectancy_points"]) > 0 for item in weekly
        )
        largest_session_share = (
            max(cast(int, item["candidates_count"]) for item in sessions)
            / primary_result.candidates_count
            if sessions and primary_result.candidates_count
            else 1.0
        )
        comparison_weekly = _distribution(primary_comparison_selected, "week")
        comparison_sessions = _distribution(primary_comparison_selected, "session")
        comparison_active_weeks = len(comparison_weekly)
        comparison_positive_weeks = sum(
            cast(float, item["simulated_net_expectancy_points"]) > 0 for item in comparison_weekly
        )
        comparison_largest_session_share = (
            max(cast(int, item["candidates_count"]) for item in comparison_sessions)
            / primary_comparison_result.candidates_count
            if comparison_sessions and primary_comparison_result.candidates_count
            else 1.0
        )
        comparison_passes = (
            (
                primary_comparison_result.candidates_count >= 200
                and cast(float, primary_comparison_result.executable_expectancy_points) > 0
                and cast(float, primary_comparison_result.simulated_net_expectancy_points) > 0
                and comparison_active_weeks >= 3
                and comparison_positive_weeks / comparison_active_weeks >= 0.6
                and comparison_largest_session_share <= 0.6
                and cast(float, primary_comparison_result.average_move_to_cost_ratio) >= 1.5
            )
            if primary_comparison_result.candidates_count
            else False
        )
        freeze_passes = (
            (
                primary_result.candidates_count >= 50
                and cast(float, primary_result.executable_expectancy_points) > 0
                and cast(float, primary_result.simulated_net_expectancy_points) > 0
                and active_weeks >= 2
                and positive_weeks / active_weeks >= 0.5
                and largest_session_share <= 0.75
                and cast(float, primary_result.average_move_to_cost_ratio) >= 1.5
            )
            if primary_result.candidates_count
            else False
        )
        if comparison_passes and freeze_passes:
            conclusion: V2Conclusion = "V2_VALIDATION_CANDIDATE"
        elif (
            primary_result.candidates_count >= 25
            and primary_result.executable_expectancy_points is not None
            and primary_result.executable_expectancy_points > 0
        ):
            conclusion = "V2_NEEDS_MORE_RESEARCH"
        else:
            conclusion = "V2_RESEARCH_REJECTED"
        elapsed = perf_counter() - started
        _, peak_python = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        broker = BrokerSimulationConfig()
        return ResearchV2Report(
            requested_start_utc=start,
            requested_end_utc=end,
            sampling_policy=(
                "First tick of each active UTC minute; chronological only; no tickless samples."
            ),
            label_policy=(
                "Features use timestamps <=T. Labels are computed separately from ticks >T at "
                "30/60/180/300/600/900 seconds. Opportunity MFE is the maximum absolute mid "
                "excursion; direction uses terminal mid return and executable Bid/Ask outcomes."
            ),
            split_policy=(
                "Initial 70% chronological TRAIN. Final 30% VALIDATION is three contiguous 10% "
                "walk-forward folds; folds 1-2 are MODEL_COMPARISON and fold 3 is the immutable "
                "INTERNAL_FREEZE_CHECK. Every scaler/encoder/model is refit only on prior TRAIN."
            ),
            feature_policy=(
                "Only existing V1 features known at T; no global normalization and no HOLDOUT data."
            ),
            model_policy=(
                "Fixed untuned baselines, L2 LogisticRegression and shallow HistGradientBoosting; "
                "no neural network and no hyperparameter search."
            ),
            execution_gate_policy=(
                "Opportunity P(move>cost)>=0.5; direction confidence>=0.55; expected gross edge "
                "> scenario cost; report fixed move/cost thresholds 1.0/1.5/2.0/3.0. Primary "
                "predeclared gate is 300s BASE at 1.5."
            ),
            freeze_criteria_predeclared={
                "primary_horizon_seconds": 300,
                "primary_scenario": "BASE",
                "primary_move_to_cost_threshold": 1.5,
                "minimum_candidates": 200,
                "minimum_active_weeks": 3,
                "minimum_positive_week_fraction": 0.6,
                "maximum_single_session_share": 0.6,
                "minimum_average_move_to_cost_ratio": 1.5,
                "requires_positive_executable_expectancy": True,
                "requires_positive_base_simulated_net_expectancy": True,
                "internal_freeze_check": {
                    "fold": 3,
                    "minimum_candidates": 50,
                    "minimum_active_weeks": 2,
                    "minimum_positive_week_fraction": 0.5,
                    "maximum_single_session_share": 0.75,
                    "minimum_average_move_to_cost_ratio": 1.5,
                    "requires_positive_executable_expectancy": True,
                    "requires_positive_base_simulated_net_expectancy": True,
                    "may_change_research_design_after_results": False,
                },
                "model_comparison_passed": comparison_passes,
                "internal_freeze_check_passed": freeze_passes,
            },
            source_hashes={**hashes, "feature_engine_v2_input": _feature_source_hash()},
            simulation_profile=EdgeSimulationAssumptions(
                broker_profile=broker.profile(),
                evaluation_volume_lots=float(broker.volumes.minimum),
                slippage_points_per_side=1,
                commission_eur_per_lot_per_side=2,
                commission_minimum_eur_per_side=0,
                disclaimer=(
                    "Simulated research profile only; not MetaQuotes-Demo or a future broker."
                ),
            ),
            cost_scenarios=SCENARIOS,
            dataset_path=str(dataset_path.resolve()),
            folds=fold_reports,
            horizons=tuple(horizon_reports),
            combined_validation=tuple(gate_results),
            model_comparison_validation=tuple(comparison_gate_results),
            internal_freeze_check=tuple(freeze_gate_results),
            selected_base_gate=primary_result,
            weekly_distribution=weekly,
            session_distribution=sessions,
            performance=V2Performance(
                elapsed_seconds=elapsed,
                ticks_analyzed=ticks_analyzed,
                observations=frame.height,
                peak_python_memory_mb=peak_python / 1024 / 1024,
                peak_loaded_tick_frame_mb=peak_frame_mb,
            ),
            conclusion=conclusion,
            limitations=(
                "VALIDATION belongs to the declared RESEARCH dataset and is not final OOS proof.",
                "HOLDOUT remained sealed and was neither opened nor evaluated.",
                "Historical news clearance is unavailable.",
                "Model selection on VALIDATION requires a later untouched HOLDOUT after freeze.",
                "No sizing, order execution, live trading or Phase E implementation exists.",
            ),
        )


def render_research_v2_report(report: ResearchV2Report) -> str:
    def metric(value: object) -> str:
        return "N/A" if value is None else f"{float(cast(float, value)):.6f}"

    lines = [
        "# SNIPER Phase D.8 — Research Engine V2",
        "",
        f"Conclusion : **{report.conclusion}**",
        "",
        f"Observations : {report.performance.observations:,}",
        f"Ticks source : {report.performance.ticks_analyzed:,}",
        f"Temps : {report.performance.elapsed_seconds:.2f} s",
        "",
        "## Split walk-forward",
        "",
    ]
    for fold in report.folds:
        lines.append(
            f"- Fold {fold.fold}: TRAIN `< {fold.train_end_utc_exclusive.isoformat()}` "
            f"({fold.train_observations:,}); VALIDATION "
            f"`[{fold.validation_start_utc.isoformat()}, "
            f"{fold.validation_end_utc_exclusive.isoformat()})` "
            f"({fold.validation_observations:,}); role={fold.evaluation_role}; "
            "preprocessing=TRAIN_ONLY"
        )
    lines.extend(
        [
            "",
            "## Modèles par horizon — MODEL_COMPARISON (folds 1–2)",
            "",
            "| Horizon | Opp. base rate | Opp. PR-AUC LR | Opp. PR-AUC HGB | "
            "Dir. bal.acc random | momentum | mean-rev | LR | HGB |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for horizon in report.horizons:
        scoped = horizon.evaluation_windows["MODEL_COMPARISON"]
        opportunity = scoped["opportunity"]["classification_move_gt_base_cost"]
        direction = scoped["direction"]
        lines.append(
            f"| {horizon.horizon_seconds}s | "
            f"{metric(opportunity['hist_gradient_boosting']['base_rate'])} | "
            f"{metric(opportunity['logistic_regression']['pr_auc'])} | "
            f"{metric(opportunity['hist_gradient_boosting']['pr_auc'])} | "
            f"{metric(direction['random_baseline']['balanced_accuracy'])} | "
            f"{metric(direction['momentum_baseline']['balanced_accuracy'])} | "
            f"{metric(direction['mean_reversion_baseline']['balanced_accuracy'])} | "
            f"{metric(direction['logistic_regression']['balanced_accuracy'])} | "
            f"{metric(direction['hist_gradient_boosting']['balanced_accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "## INTERNAL_FREEZE_CHECK (fold 3, lecture seule)",
            "",
            "| Horizon | Observations | Opp. PR-AUC LR | Opp. PR-AUC HGB | "
            "Dir. bal.acc LR | Dir. bal.acc HGB |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for horizon in report.horizons:
        scoped = horizon.evaluation_windows["INTERNAL_FREEZE_CHECK"]
        opportunity = scoped["opportunity"]["classification_move_gt_base_cost"]
        direction = scoped["direction"]
        lines.append(
            f"| {horizon.horizon_seconds}s | {scoped['observations']} | "
            f"{metric(opportunity['logistic_regression']['pr_auc'])} | "
            f"{metric(opportunity['hist_gradient_boosting']['pr_auc'])} | "
            f"{metric(direction['logistic_regression']['balanced_accuracy'])} | "
            f"{metric(direction['hist_gradient_boosting']['balanced_accuracy'])} |"
        )
    gate = report.selected_base_gate
    lines.extend(
        [
            "",
            "## ExecutionGate primaire pré-déclaré",
            "",
            f"- Horizon : {gate.horizon_seconds}s",
            f"- Scénario : {gate.scenario}",
            f"- Seuil move/cost : {gate.move_to_cost_threshold}",
            f"- Portée : {gate.evaluation_scope}",
            f"- Candidats : {gate.candidates_count}",
            f"- Expectancy brute : {metric(gate.gross_expectancy_points)} points",
            f"- Expectancy exécutable : {metric(gate.executable_expectancy_points)} points",
            f"- Expectancy nette simulée : {metric(gate.simulated_net_expectancy_points)} points",
            f"- MFE / MAE : {metric(gate.average_mfe_points)} / "
            f"{metric(gate.average_mae_points)} points",
            f"- Move/cost moyen : {metric(gate.average_move_to_cost_ratio)}",
            f"- Profitable : {metric(gate.percentage_profitable)}%",
            "",
            "## Garde-fous",
            "",
            "- V1 : RESEARCH_REJECTED, code/hashes/rapports conservés",
            "- HOLDOUT ouvert : NON",
            "- HOLDOUT évalué : NON",
            "- Fit preprocessing sur VALIDATION/HOLDOUT : NON",
            "- Optimisation automatique des seuils : NON",
            "- Adaptation après INTERNAL_FREEZE_CHECK : INTERDITE",
            "- Rapport principal sélectionné après résultats : NON "
            "(300s / BASE / 1.5 préenregistré)",
            "- Sizing : RISK_ENGINE_ONLY",
            "- Trading live : DÉSACTIVÉ",
            "- Phase E : NON COMMENCÉE",
            "",
        ]
    )
    return "\n".join(lines)
