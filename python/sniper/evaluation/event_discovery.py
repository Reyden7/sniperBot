"""Phase D.7 event-based discovery for the frozen V1 Signal Engine."""

import tracemalloc
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Literal, cast

import numpy as np
import polars as pl
from pydantic import Field

from sniper.backtest.execution_model import BrokerSimulationConfig
from sniper.config import Model, NonNegative, Positive
from sniper.data.feature_source import load_parquet_bars
from sniper.data.time import as_utc
from sniper.domain.bar import Timeframe
from sniper.domain.edge_validation import EdgeSimulationAssumptions
from sniper.domain.event_discovery import (
    CostFeasibility,
    DirectionDiagnostic,
    DirectionMode,
    EpisodeMoment,
    EpisodeSummary,
    EventDiscoveryPerformance,
    EventDiscoveryReport,
    GroupDiagnostic,
    HorizonMeans,
    NumericFeatureDiagnostic,
)
from sniper.domain.signal import Bias, FeatureSnapshot, SignalDecision, TickFeatures
from sniper.evaluation.edge_validation import (
    HORIZONS,
    MICROSECONDS,
    _assert_frozen_engine,
    _bar_window,
    _current_market_tick,
    _days,
    _future_metrics,
    _read_ticks,
    _session,
    _spearman,
    _streaming_tick_features,
    _tick_path,
)
from sniper.features.engine import FeatureEngine
from sniper.strategy.filters import EvaluationContext
from sniper.strategy.scorer import SignalEngine


class EventDiscoveryConfig(Model):
    sampling_interval_seconds: int = Field(default=1, ge=1)
    activation_score: int = Field(default=90, ge=0, le=100)
    rearm_score: int = Field(default=85, ge=0, le=100)
    max_horizon_tick_delay_seconds: Positive = Decimal(2)
    point: Positive = Decimal("0.00001")
    max_spread_points: Positive = Decimal(2)
    max_tick_age_seconds: Positive = Decimal(2)
    slippage_points_per_side: NonNegative = Decimal(1)
    commission_eur_per_lot_per_side: NonNegative = Decimal(2)
    commission_minimum_eur_per_side: NonNegative = Decimal(0)
    minimum_group_sample: int = Field(default=30, ge=10)

    def validate_methodology(self) -> None:
        if self.sampling_interval_seconds != 1:
            raise ValueError("Phase D.7 methodology is fixed to first tick per active UTC second")
        if self.activation_score != 90 or self.rearm_score != 85:
            raise ValueError("Phase D.7 activation/rearm methodology is fixed at 90/85")


@dataclass(slots=True)
class _Episode:
    episode_id: int
    direction: Bias
    first: dict[str, object]
    peak: dict[str, object]
    evaluated_states: int = 1


def _minute_score_upper_bounds(
    snapshot: FeatureSnapshot,
    context: EvaluationContext,
    signal_engine: SignalEngine,
    point: Decimal,
) -> dict[Bias, int]:
    """Exact reachable score bounds using unchanged scoring and maximally favorable ticks."""
    result: dict[Bias, int] = {}
    for direction in (Bias.BUY, Bias.SELL):
        buy = direction == Bias.BUY
        ticks = TickFeatures(
            rate_1s=1,
            rate_5s=1,
            rate_15s=1,
            uptick_ratio=Decimal("0.60") if buy else Decimal("0.40"),
            downtick_ratio=Decimal("0.40") if buy else Decimal("0.60"),
            price_acceleration=point if buy else -point,
            tick_rate_acceleration=Decimal("0.8"),
            last_tick_timestamp_utc=snapshot.as_of_utc,
            last_tick_age_seconds=Decimal(0),
            current_spread_points=Decimal(0),
            median_spread_points=Decimal(1),
        )
        candidate = signal_engine.evaluate(
            snapshot.model_copy(
                update={"ticks": ticks, "visible_ticks": max(3, snapshot.visible_ticks)}
            ),
            context,
            point=point,
        )
        if candidate.trigger_direction == direction:
            result[direction] = candidate.score
    return result


def _opposite(direction: Bias) -> Bias:
    return Bias.SELL if direction == Bias.BUY else Bias.BUY


def _sign(value: Decimal) -> str:
    return "UP" if value > 0 else "DOWN" if value < 0 else "FLAT"


def _feature_values(snapshot: FeatureSnapshot, direction: Bias, session: str) -> dict[str, object]:
    direction_sign = Decimal(1) if direction == Bias.BUY else Decimal(-1)
    point = Decimal("0.00001")
    breakout = snapshot.m5.breakout_retest or "NONE"
    pattern = snapshot.m1.pattern.engulfing or (
        "HAMMER"
        if snapshot.m1.pattern.hammer
        else "SHOOTING_STAR"
        if snapshot.m1.pattern.shooting_star
        else "NONE"
    )
    volatility_state = (
        "COMPRESSION"
        if snapshot.m1.compression
        else "EXPANSION"
        if snapshot.m1.expansion
        else "NORMAL"
    )
    spread_state = (
        "ACCEPTABLE_RELATIVE"
        if snapshot.ticks.current_spread_points
        <= snapshot.ticks.median_spread_points * Decimal("1.25")
        else "ELEVATED_RELATIVE"
    )
    m1_momentum_sign = _sign(snapshot.m1.momentum_3)
    tick_acceleration_sign = _sign(snapshot.ticks.price_acceleration)
    return {
        "m15_regime": snapshot.m15.regime.value,
        "m15_direction": snapshot.m15.direction.value,
        "m15_ema_slope_aligned_points": float(snapshot.m15.ema_slope / point * direction_sign),
        "m15_atr_points": float(snapshot.m15.atr / point),
        "m5_direction": snapshot.m5.direction.value,
        "m5_momentum_aligned_points": float(snapshot.m5.momentum / point * direction_sign),
        "m5_atr_points": float(snapshot.m5.atr / point),
        "m5_support_distance_atr": float(snapshot.m5.support_distance_atr),
        "m5_resistance_distance_atr": float(snapshot.m5.resistance_distance_atr),
        "m5_breakout_retest": breakout,
        "m1_momentum_1_aligned_points": float(snapshot.m1.momentum_1 / point * direction_sign),
        "m1_momentum_3_aligned_points": float(snapshot.m1.momentum_3 / point * direction_sign),
        "m1_momentum_5_aligned_points": float(snapshot.m1.momentum_5 / point * direction_sign),
        "m1_momentum_sign": m1_momentum_sign,
        "m1_range_atr_ratio": float(snapshot.m1.range_atr_ratio),
        "m1_atr_points": float(snapshot.m1.atr / point),
        "m1_body_ratio": float(snapshot.m1.body_ratio),
        "m1_upper_wick_ratio": float(snapshot.m1.upper_wick_ratio),
        "m1_lower_wick_ratio": float(snapshot.m1.lower_wick_ratio),
        "m1_pattern": pattern,
        "m1_compression": str(snapshot.m1.compression).upper(),
        "m1_expansion": str(snapshot.m1.expansion).upper(),
        "volatility_state": volatility_state,
        "tick_price_acceleration_aligned_points": float(
            snapshot.ticks.price_acceleration / point * direction_sign
        ),
        "tick_acceleration_sign": tick_acceleration_sign,
        "tick_rate_1s": snapshot.ticks.rate_1s,
        "tick_rate_5s": snapshot.ticks.rate_5s,
        "tick_rate_15s": snapshot.ticks.rate_15s,
        "tick_rate_acceleration": float(snapshot.ticks.tick_rate_acceleration),
        "tick_directional_ratio": float(
            snapshot.ticks.uptick_ratio if direction == Bias.BUY else snapshot.ticks.downtick_ratio
        ),
        "spread_points": float(snapshot.ticks.current_spread_points),
        "spread_vs_median_ratio": (
            float(snapshot.ticks.current_spread_points / snapshot.ticks.median_spread_points)
            if snapshot.ticks.median_spread_points > 0
            else 0.0
        ),
        "spread_state": spread_state,
        "session": session,
        "interaction_m15_regime_x_m5_direction": (
            f"{snapshot.m15.regime.value}|{snapshot.m5.direction.value}"
        ),
        "interaction_m5_direction_x_m1_momentum": (
            f"{snapshot.m5.direction.value}|{m1_momentum_sign}"
        ),
        "interaction_m1_momentum_x_tick_acceleration": (
            f"{m1_momentum_sign}|{tick_acceleration_sign}"
        ),
        "interaction_breakout_retest_x_session": f"{breakout}|{session}",
        "interaction_volatility_x_spread": f"{volatility_state}|{spread_state}",
    }


def _prefixed_metrics(prefix: str, metrics: dict[str, object] | None) -> dict[str, object]:
    keys = (
        *(f"return_after_{horizon}s_points" for horizon in HORIZONS),
        *(f"market_return_after_{horizon}s_points" for horizon in HORIZONS),
        "directional_accuracy",
        "expectancy_before_costs_points",
        "expectancy_after_spread_points",
        "expectancy_after_slippage_points",
        "expectancy_after_commission_points",
        "expectancy_after_commission_eur",
        "mfe_30s_points",
        "mae_30s_points",
        "mfe_30s_eur",
        "mae_30s_eur",
        "commission_round_trip_eur",
        "commission_round_trip_points",
        "stop_target_outcome",
    )
    return {f"{prefix}_{key}": None if metrics is None else metrics.get(key) for key in keys}


def _point(
    *,
    snapshot: FeatureSnapshot,
    decision: SignalDecision,
    timestamps_us: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    current_index: int,
    config: EventDiscoveryConfig,
    broker: BrokerSimulationConfig,
) -> dict[str, object]:
    direction = decision.trigger_direction
    original = None
    inverse = None
    if (
        direction != Bias.NEUTRAL
        and decision.proposed_stop_distance_points is not None
        and decision.proposed_target_distance_points is not None
    ):
        original = _future_metrics(
            timestamps_us,
            bids,
            asks,
            current_index,
            direction,
            decision.proposed_stop_distance_points,
            decision.proposed_target_distance_points,
            config,
            broker,
        )
        inverse = _future_metrics(
            timestamps_us,
            bids,
            asks,
            current_index,
            _opposite(direction),
            decision.proposed_stop_distance_points,
            decision.proposed_target_distance_points,
            config,
            broker,
        )
    instant = snapshot.as_of_utc
    return {
        "timestamp_utc": instant,
        "direction": direction.value,
        "score": decision.score,
        "score_tier": decision.tier.value,
        "current_bid": float(bids[current_index]),
        "current_ask": float(asks[current_index]),
        "proposed_stop_points": decision.proposed_stop_distance_points,
        "proposed_target_points": decision.proposed_target_distance_points,
        "reasons": ",".join(decision.reasons),
        **_feature_values(snapshot, direction, _session(instant)),
        **_prefixed_metrics("original", original),
        **_prefixed_metrics("inverse", inverse),
    }


def _complete_episode(
    episode: _Episode,
    *,
    end_time: datetime,
    reason: str,
    rows: list[dict[str, object]],
) -> None:
    first_time = episode.first["timestamp_utc"]
    assert isinstance(first_time, datetime)
    wall_duration = max(0.0, (end_time - first_time).total_seconds())
    for moment, point in (("FIRST_CROSSING", episode.first), ("PEAK_SCORE", episode.peak)):
        rows.append(
            {
                "episode_id": episode.episode_id,
                "moment": moment,
                "tradable": moment == "FIRST_CROSSING",
                "episode_end_utc": end_time,
                "episode_end_reason": reason,
                "episode_wall_duration_seconds": wall_duration,
                "episode_evaluated_states": episode.evaluated_states,
                "episode_peak_score": episode.peak["score"],
                **point,
            }
        )


def _mean(frame: pl.DataFrame, column: str) -> float | None:
    if frame.is_empty() or column not in frame.columns:
        return None
    value = frame[column].mean()
    return float(str(value)) if value is not None else None


def _horizon_means(frame: pl.DataFrame, prefix: str) -> HorizonMeans:
    return HorizonMeans(
        seconds_5=_mean(frame, f"{prefix}_5s_points"),
        seconds_10=_mean(frame, f"{prefix}_10s_points"),
        seconds_30=_mean(frame, f"{prefix}_30s_points"),
        seconds_60=_mean(frame, f"{prefix}_60s_points"),
        seconds_180=_mean(frame, f"{prefix}_180s_points"),
    )


def _direction_diagnostic(
    frame: pl.DataFrame,
    moment: EpisodeMoment,
    mode: Literal["original", "inverse"],
) -> DirectionDiagnostic:
    selected = frame.filter(pl.col("moment") == moment)
    usable = selected.filter(pl.col(f"{mode}_return_after_30s_points").is_not_null())
    gross_accuracy = (
        float((usable[f"{mode}_market_return_after_30s_points"] > 0).sum()) / usable.height * 100
        if not usable.is_empty()
        else None
    )
    executable_accuracy = _mean(usable, f"{mode}_directional_accuracy")
    direction_mode: DirectionMode = "ORIGINAL" if mode == "original" else "INVERSE_DIAGNOSTIC"
    return DirectionDiagnostic(
        moment=moment,
        direction_mode=direction_mode,
        tradable=moment == "FIRST_CROSSING" and mode == "original",
        observations=selected.height,
        usable=usable.height,
        executable_accuracy_30s_pct=(
            executable_accuracy * 100 if executable_accuracy is not None else None
        ),
        gross_accuracy_30s_pct=gross_accuracy,
        gross_expectancy_points=_horizon_means(usable, f"{mode}_market_return_after"),
        executable_expectancy_points=_horizon_means(usable, f"{mode}_return_after"),
        simulated_net_expectancy_30s_points=_mean(
            usable, f"{mode}_expectancy_after_commission_points"
        ),
        average_mfe_30s_points=_mean(usable, f"{mode}_mfe_30s_points"),
        average_mae_30s_points=_mean(usable, f"{mode}_mae_30s_points"),
    )


def _correlation(frame: pl.DataFrame, left: str, right: str) -> float | None:
    selected = frame.select(left, right).drop_nulls()
    if selected.is_empty():
        return None
    return _spearman(
        selected[left].to_numpy().astype(float), selected[right].to_numpy().astype(float)
    )


NUMERIC_FEATURES = (
    ("m15_ema_slope_aligned_points", "ALIGNED_TO_ORIGINAL_DIRECTION"),
    ("m15_atr_points", "RAW_MAGNITUDE"),
    ("m5_momentum_aligned_points", "ALIGNED_TO_ORIGINAL_DIRECTION"),
    ("m5_atr_points", "RAW_MAGNITUDE"),
    ("m5_support_distance_atr", "RAW_MAGNITUDE"),
    ("m5_resistance_distance_atr", "RAW_MAGNITUDE"),
    ("m1_momentum_1_aligned_points", "ALIGNED_TO_ORIGINAL_DIRECTION"),
    ("m1_momentum_3_aligned_points", "ALIGNED_TO_ORIGINAL_DIRECTION"),
    ("m1_momentum_5_aligned_points", "ALIGNED_TO_ORIGINAL_DIRECTION"),
    ("m1_range_atr_ratio", "RAW_MAGNITUDE"),
    ("m1_atr_points", "RAW_MAGNITUDE"),
    ("m1_body_ratio", "RAW_MAGNITUDE"),
    ("m1_upper_wick_ratio", "RAW_MAGNITUDE"),
    ("m1_lower_wick_ratio", "RAW_MAGNITUDE"),
    ("tick_price_acceleration_aligned_points", "ALIGNED_TO_ORIGINAL_DIRECTION"),
    ("tick_rate_1s", "RAW_MAGNITUDE"),
    ("tick_rate_5s", "RAW_MAGNITUDE"),
    ("tick_rate_15s", "RAW_MAGNITUDE"),
    ("tick_rate_acceleration", "RAW_MAGNITUDE"),
    ("tick_directional_ratio", "ALIGNED_TO_ORIGINAL_DIRECTION"),
    ("spread_points", "RAW_MAGNITUDE"),
    ("spread_vs_median_ratio", "RAW_MAGNITUDE"),
)
CATEGORICAL_FEATURES = (
    "m15_regime",
    "m15_direction",
    "m5_direction",
    "m5_breakout_retest",
    "m1_momentum_sign",
    "m1_pattern",
    "m1_compression",
    "m1_expansion",
    "volatility_state",
    "tick_acceleration_sign",
    "spread_state",
    "session",
)
INTERACTIONS = (
    "interaction_m15_regime_x_m5_direction",
    "interaction_m5_direction_x_m1_momentum",
    "interaction_m1_momentum_x_tick_acceleration",
    "interaction_breakout_retest_x_session",
    "interaction_volatility_x_spread",
)


def _numeric_diagnostics(frame: pl.DataFrame) -> tuple[NumericFeatureDiagnostic, ...]:
    usable = frame.filter(
        (pl.col("moment") == "FIRST_CROSSING")
        & pl.col("original_return_after_30s_points").is_not_null()
    )
    result = []
    for feature, transformation in NUMERIC_FEATURES:
        result.append(
            NumericFeatureDiagnostic(
                feature=feature,
                transformation=transformation,
                observations=usable.height,
                spearman_vs_return=HorizonMeans(
                    **{
                        f"seconds_{horizon}": _correlation(
                            usable, feature, f"original_return_after_{horizon}s_points"
                        )
                        for horizon in HORIZONS
                    }
                ),
                spearman_vs_mfe_30s=_correlation(usable, feature, "original_mfe_30s_points"),
                spearman_vs_mae_30s=_correlation(usable, feature, "original_mae_30s_points"),
            )
        )
    return tuple(result)


def _group_diagnostics(
    frame: pl.DataFrame,
    features: tuple[str, ...],
    minimum_sample: int,
) -> tuple[GroupDiagnostic, ...]:
    usable = frame.filter(
        (pl.col("moment") == "FIRST_CROSSING")
        & pl.col("original_return_after_30s_points").is_not_null()
    )
    result = []
    for feature in features:
        for value in sorted(str(item) for item in usable[feature].unique().to_list()):
            group = usable.filter(pl.col(feature).cast(pl.String) == value)
            group_accuracy = _mean(group, "original_directional_accuracy")
            result.append(
                GroupDiagnostic(
                    analysis=feature,
                    value=value,
                    observations=group.height,
                    sample_sufficient=group.height >= minimum_sample,
                    executable_accuracy_30s_pct=(
                        group_accuracy * 100 if group_accuracy is not None else None
                    ),
                    gross_expectancy_points=_horizon_means(group, "original_market_return_after"),
                    average_mfe_30s_points=_mean(group, "original_mfe_30s_points"),
                    average_mae_30s_points=_mean(group, "original_mae_30s_points"),
                )
            )
    return tuple(result)


def _episode_summary(frame: pl.DataFrame) -> EpisodeSummary:
    first = frame.filter(pl.col("moment") == "FIRST_CROSSING")
    durations = first["episode_wall_duration_seconds"].to_list() if not first.is_empty() else []
    return EpisodeSummary(
        episodes=first.height,
        completed_episodes=int((first["episode_end_reason"] != "RANGE_END").sum()),
        open_at_range_end=int((first["episode_end_reason"] == "RANGE_END").sum()),
        average_duration_seconds=float(np.mean(durations)) if durations else None,
        median_duration_seconds=float(median(durations)) if durations else None,
        maximum_duration_seconds=float(max(durations)) if durations else None,
        average_first_score=_mean(first, "score"),
        average_peak_score=_mean(first, "episode_peak_score"),
    )


def _cost_feasibility(frame: pl.DataFrame) -> CostFeasibility:
    usable = frame.filter(
        (pl.col("moment") == "FIRST_CROSSING")
        & pl.col("original_return_after_30s_points").is_not_null()
    )
    cost_frame = usable.with_columns(
        (
            pl.col("original_expectancy_before_costs_points")
            - pl.col("original_expectancy_after_commission_points")
        ).alias("total_round_trip_cost_points")
    )
    cost = _mean(cost_frame, "total_round_trip_cost_points")
    mfe = _mean(usable, "original_mfe_30s_points")
    exceeds = (
        float((usable["original_mfe_30s_points"] > cost).sum()) / usable.height * 100
        if not usable.is_empty() and cost is not None
        else None
    )
    return CostFeasibility(
        observations=usable.height,
        average_total_round_trip_cost_points=cost,
        average_mfe_30s_points=mfe,
        mfe_to_cost_ratio=(mfe / cost if mfe is not None and cost else None),
        fraction_mfe_exceeding_cost_pct=exceeds,
        note=(
            "Total cost is market return before costs minus simulated net return, so it includes "
            "the observed executable spread, two-sided simulated slippage and commission."
        ),
    )


class EventCandidateDiscoverer:
    def __init__(self, config: EventDiscoveryConfig | None = None) -> None:
        self.config = config or EventDiscoveryConfig()
        self.config.validate_methodology()
        self.feature_engine = FeatureEngine(point=self.config.point)
        self.signal_engine = SignalEngine()
        self.broker = BrokerSimulationConfig(point=self.config.point)

    def run(
        self, *, data_root: Path, start_utc: datetime, end_utc: datetime
    ) -> EventDiscoveryReport:
        start, end = as_utc(start_utc), as_utc(end_utc)
        if start >= end:
            raise ValueError("event discovery requires a nonempty UTC range")
        hashes = _assert_frozen_engine()
        tracemalloc.start()
        started = perf_counter()
        bars = load_parquet_bars(
            data_root, "EURUSD", end, lookback_days=max(1, (end - start).days + 30)
        )
        timeframes: tuple[Timeframe, ...] = ("M15", "M5", "M1")
        bar_times = {
            timeframe: [bar.open_time_utc for bar in bars[timeframe]] for timeframe in timeframes
        }
        context = EvaluationContext(
            max_spread_points=self.config.max_spread_points,
            max_tick_age_seconds=self.config.max_tick_age_seconds,
            session_allowed=True,
            news_clear=True,
        )
        ticks_analyzed = 0
        active_seconds_scanned = 0
        signal_engine_evaluations = 0
        bar_feature_snapshots = 0
        peak_tick_frame_mb = 0.0
        rows: list[dict[str, object]] = []
        active: _Episode | None = None
        next_episode_id = 1
        armed = False
        previous_below_activation = False
        initialized = False
        last_time = start

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
                min(end, day_end + timedelta(seconds=180)),
            )
            extended = pl.concat([previous, current, following], how="vertical").sort(
                "timestamp_utc", maintain_order=True
            )
            peak_tick_frame_mb = max(peak_tick_frame_mb, extended.estimated_size("mb"))
            timestamps_us = extended["timestamp_utc"].cast(pl.Int64).to_numpy()
            bids = extended["bid"].to_numpy()
            asks = extended["ask"].to_numpy()
            spreads = extended["spread_points"].to_numpy()
            current_us = current["timestamp_utc"].cast(pl.Int64).to_numpy()
            second_keys = current_us // MICROSECONDS
            sample_local_indices = np.concatenate(
                (np.array([0]), np.flatnonzero(np.diff(second_keys)) + 1)
            )
            price_cache: dict[float, Decimal] = {}
            base_snapshot: FeatureSnapshot | None = None
            base_minute: int | None = None
            score_upper_bounds: dict[Bias, int] = {}
            epoch = datetime(1970, 1, 1, tzinfo=UTC)
            for local_index in sample_local_indices:
                sample_us = int(current_us[int(local_index)])
                current_index = int(np.searchsorted(timestamps_us, sample_us, side="left"))
                sample_time = epoch + timedelta(microseconds=sample_us)
                active_seconds_scanned += 1
                minute = sample_us // (60 * MICROSECONDS)
                if minute != base_minute:
                    try:
                        base_snapshot = self.feature_engine.compute(
                            bars=_bar_window(bars, bar_times, sample_time),
                            ticks=_current_market_tick(
                                timestamps_us, bids, asks, current_index, price_cache
                            ),
                            as_of_utc=sample_time,
                            symbol="EURUSD",
                        )
                    except ValueError:
                        base_snapshot = None
                        score_upper_bounds = {}
                    else:
                        score_upper_bounds = _minute_score_upper_bounds(
                            base_snapshot, context, self.signal_engine, self.config.point
                        )
                        signal_engine_evaluations += 2
                    base_minute = minute
                    bar_feature_snapshots += 1
                if base_snapshot is None:
                    continue
                maximum_score = max(score_upper_bounds.values(), default=0)
                requires_exact_evaluation = maximum_score >= self.config.activation_score or (
                    not armed and maximum_score >= self.config.rearm_score
                )
                if not requires_exact_evaluation:
                    if active is not None:
                        _complete_episode(
                            active,
                            end_time=sample_time,
                            reason="SCORE_BELOW_90_PROVEN_BY_UPPER_BOUND",
                            rows=rows,
                        )
                        active = None
                    if maximum_score < self.config.rearm_score:
                        armed = True
                    initialized = True
                    previous_below_activation = True
                    last_time = sample_time
                    continue
                visible_left = int(
                    np.searchsorted(timestamps_us, sample_us - 60 * MICROSECONDS, side="left")
                )
                snapshot = base_snapshot.model_copy(
                    update={
                        "as_of_utc": sample_time,
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
                decision = self.signal_engine.evaluate(snapshot, context, point=self.config.point)
                signal_engine_evaluations += 1
                last_time = sample_time
                below_activation = decision.score < self.config.activation_score
                below_rearm = decision.score < self.config.rearm_score
                compatible = (
                    not decision.blockers
                    and decision.trigger_direction in (Bias.BUY, Bias.SELL)
                    and decision.score >= self.config.activation_score
                )

                if not initialized:
                    initialized = True
                    armed = below_rearm
                    previous_below_activation = below_activation
                    continue

                if active is not None:
                    same_direction = decision.trigger_direction == active.direction
                    if compatible and same_direction:
                        active.evaluated_states += 1
                        if decision.score > cast(int, active.peak["score"]):
                            active.peak = _point(
                                snapshot=snapshot,
                                decision=decision,
                                timestamps_us=timestamps_us,
                                bids=bids,
                                asks=asks,
                                current_index=current_index,
                                config=self.config,
                                broker=self.broker,
                            )
                    else:
                        reason = (
                            "BLOCKER"
                            if decision.blockers
                            else "DIRECTION_CHANGE"
                            if not same_direction
                            else "SCORE_BELOW_90"
                        )
                        _complete_episode(active, end_time=sample_time, reason=reason, rows=rows)
                        active = None
                        armed = below_rearm
                elif not armed:
                    if below_rearm:
                        armed = True
                elif previous_below_activation and compatible:
                    point = _point(
                        snapshot=snapshot,
                        decision=decision,
                        timestamps_us=timestamps_us,
                        bids=bids,
                        asks=asks,
                        current_index=current_index,
                        config=self.config,
                        broker=self.broker,
                    )
                    active = _Episode(
                        episode_id=next_episode_id,
                        direction=decision.trigger_direction,
                        first=point,
                        peak=point,
                    )
                    next_episode_id += 1
                    armed = False
                previous_below_activation = below_activation

        if not initialized:
            raise ValueError("no active tick partitions were found in the requested range")
        if active is not None:
            _complete_episode(active, end_time=last_time, reason="RANGE_END", rows=rows)
        if rows:
            events = pl.DataFrame(rows, infer_schema_length=None).sort("timestamp_utc")
        else:
            raise ValueError("no candidate episode was discovered in the requested range")
        run_name = f"{start:%Y%m%dT%H%M%SZ}-{end:%Y%m%dT%H%M%SZ}-1s"
        event_path = data_root / "evaluations" / f"event-discovery-{run_name}" / "episodes.parquet"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = event_path.with_suffix(".parquet.tmp")
        events.write_parquet(temporary, compression="zstd", statistics=True)
        temporary.replace(event_path)

        moments: tuple[EpisodeMoment, ...] = ("FIRST_CROSSING", "PEAK_SCORE")
        modes: tuple[Literal["original", "inverse"], ...] = ("original", "inverse")
        direction_diagnostics = tuple(
            _direction_diagnostic(events, moment, mode) for moment in moments for mode in modes
        )
        numeric = _numeric_diagnostics(events)
        categorical = _group_diagnostics(
            events, CATEGORICAL_FEATURES, self.config.minimum_group_sample
        )
        interactions = _group_diagnostics(events, INTERACTIONS, self.config.minimum_group_sample)
        summary = _episode_summary(events)
        feasibility = _cost_feasibility(events)
        first_original = direction_diagnostics[0]
        first_inverse = direction_diagnostics[1]
        original_gross = first_original.gross_expectancy_points.seconds_30
        inverse_gross = first_inverse.gross_expectancy_points.seconds_30
        original_executable = first_original.executable_expectancy_points.seconds_30
        if first_original.usable < 200:
            conclusion = "INSUFFICIENT_FIRST_CROSSING_SAMPLE_FOR_EDGE_CONCLUSION"
        elif (
            original_gross is not None
            and original_gross > 0
            and original_executable is not None
            and original_executable <= 0
        ):
            conclusion = "GROSS_EFFECT_TOO_SMALL_TO_SURVIVE_OBSERVED_SPREAD"
        elif original_gross is not None and original_gross > 0:
            conclusion = "ORIGINAL_DIRECTION_GROSS_EFFECT_REQUIRES_FULL_COST_VALIDATION"
        elif inverse_gross is not None and inverse_gross > 0:
            conclusion = "INVERSE_DIRECTION_DIAGNOSTIC_ONLY; STRATEGY_NOT_REVERSED"
        else:
            conclusion = "NO_POSITIVE_GROSS_DIRECTIONAL_EFFECT_DETECTED"
        elapsed = perf_counter() - started
        _, peak_python = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        profile = self.broker.profile()
        return EventDiscoveryReport(
            requested_start_utc=start,
            requested_end_utc=end,
            sampling_policy=(
                "FIRST_TICK_OF_EACH_ACTIVE_UTC_SECOND; at most one evaluation per UTC second; "
                "all ticks remain available to the unchanged 60-second TickFeatures window; "
                "no evaluation is emitted for tickless seconds"
            ),
            episode_policy=(
                "An armed episode starts only on an observed score crossing from <90 to >=90 "
                "with no blocker. It remains active while score>=90, no blocker and the same "
                "direction persist. It ends on score<90, blocker or direction change."
            ),
            first_crossing_policy=(
                "FIRST_CROSSING is the only tradable observation. The range-start state is "
                "left-censored and cannot create an episode without an in-range crossing."
            ),
            peak_score_policy=(
                "PEAK_SCORE is the earliest maximum score observed inside the completed episode. "
                "It uses future episode knowledge and is diagnostic, never tradable."
            ),
            analysis_policy=(
                "After any counted episode, discovery remains disarmed until score<85. The 85 "
                "rearm threshold is fixed methodology, not optimized on outcomes. Feature tests "
                "use FIRST_CROSSING only; interaction cells require N>=30 to be sufficient."
            ),
            session_policy=(
                "UTC is canonical; IANA DST conversion defines Europe/London, America/New_York "
                "and Asia/Tokyo 08:00-17:00 local sessions; server time is never used."
            ),
            signal_engine_source_sha256=hashes,
            evaluation_context_assumptions={
                "max_spread_points": str(self.config.max_spread_points),
                "max_tick_age_seconds": str(self.config.max_tick_age_seconds),
                "session_allowed": "true for descriptive cross-session research",
                "news_clear": (
                    "true as an explicit evaluation assumption; historical news clearance is "
                    "unavailable and therefore not verified"
                ),
            },
            simulation=EdgeSimulationAssumptions(
                broker_profile=profile,
                evaluation_volume_lots=float(self.broker.volumes.minimum),
                slippage_points_per_side=float(self.config.slippage_points_per_side),
                commission_eur_per_lot_per_side=float(self.config.commission_eur_per_lot_per_side),
                commission_minimum_eur_per_side=float(self.config.commission_minimum_eur_per_side),
                disclaimer=(
                    "All volume, leverage, slippage and commission values are simulated research "
                    "assumptions. They are not MetaQuotes-Demo capabilities and do not describe "
                    "any future real broker."
                ),
            ),
            event_parquet_path=str(event_path.resolve()),
            performance=EventDiscoveryPerformance(
                elapsed_seconds=elapsed,
                ticks_analyzed=ticks_analyzed,
                active_seconds_scanned=active_seconds_scanned,
                signal_engine_evaluations=signal_engine_evaluations,
                bar_feature_snapshots=bar_feature_snapshots,
                ticks_per_second=ticks_analyzed / elapsed if elapsed else 0,
                scanned_seconds_per_second=active_seconds_scanned / elapsed if elapsed else 0,
                peak_python_memory_mb=peak_python / 1024 / 1024,
                peak_loaded_tick_frame_mb=peak_tick_frame_mb,
                memory_measurement_note=(
                    "Python peak is tracemalloc-only and excludes native Polars buffers; the "
                    "largest concurrently loaded four-column tick frame is reported separately."
                ),
            ),
            episode_summary=summary,
            direction_diagnostics=direction_diagnostics,
            numeric_feature_diagnostics=numeric,
            categorical_feature_diagnostics=categorical,
            interaction_diagnostics=interactions,
            cost_feasibility=feasibility,
            conclusion=conclusion,
            limitations=(
                "PEAK_SCORE is retrospective and cannot be interpreted as a live entry.",
                "Inverse direction is a diagnostic requested after D.6 and is not a strategy.",
                "Historical news clearance is unavailable and explicitly assumed.",
                "No weights, thresholds, features or Signal Engine rules were changed or tuned.",
                "One-second sampling can miss sub-second score crossings by explicit design.",
            ),
        )


def render_event_discovery_report(report: EventDiscoveryReport) -> str:
    def value(number: float | None, suffix: str = "") -> str:
        return "N/A" if number is None else f"{number:.6f}{suffix}"

    lines = [
        "# SNIPER Phase D.7 — Event-Based Candidate Discovery",
        "",
        f"Conclusion : **{report.conclusion}**",
        "",
        f"Période UTC : `{report.requested_start_utc.isoformat()}` → "
        f"`{report.requested_end_utc.isoformat()}`",
        f"Ticks analysés : {report.performance.ticks_analyzed:,}",
        f"Secondes actives scannées : {report.performance.active_seconds_scanned:,}",
        f"Évaluations Signal Engine : {report.performance.signal_engine_evaluations:,}",
        f"Épisodes : {report.episode_summary.episodes:,}",
        f"Temps : {report.performance.elapsed_seconds:.2f} s "
        f"({report.performance.ticks_per_second:,.0f} ticks/s)",
        "",
        "## Original vs inverse",
        "",
        "| Moment | Direction | Tradable | N | Accuracy | Gross 30 s | Net 30 s | MFE | MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for direction_item in report.direction_diagnostics:
        lines.append(
            f"| {direction_item.moment} | {direction_item.direction_mode} | "
            f"{'OUI' if direction_item.tradable else 'NON'} | {direction_item.usable} | "
            f"{value(direction_item.executable_accuracy_30s_pct, '%')} | "
            f"{value(direction_item.gross_expectancy_points.seconds_30)} | "
            f"{value(direction_item.simulated_net_expectancy_30s_points)} | "
            f"{value(direction_item.average_mfe_30s_points)} | "
            f"{value(direction_item.average_mae_30s_points)} |"
        )
    lines.extend(
        [
            "",
            "### Expectancy par horizon",
            "",
            "| Moment | Direction | Type | 5 s | 10 s | 30 s | 60 s | 180 s |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for direction_item in report.direction_diagnostics:
        for metric_type, horizons in (
            ("GROSS_MID", direction_item.gross_expectancy_points),
            ("EXECUTABLE", direction_item.executable_expectancy_points),
        ):
            lines.append(
                f"| {direction_item.moment} | {direction_item.direction_mode} | "
                f"{metric_type} | {value(horizons.seconds_5)} | "
                f"{value(horizons.seconds_10)} | {value(horizons.seconds_30)} | "
                f"{value(horizons.seconds_60)} | {value(horizons.seconds_180)} |"
            )
    lines.extend(
        [
            "",
            "## Coûts vs mouvement",
            "",
            "- Coût total simulé : "
            f"{value(report.cost_feasibility.average_total_round_trip_cost_points)} points",
            f"- MFE moyen 30 s : {value(report.cost_feasibility.average_mfe_30s_points)} points",
            f"- Ratio MFE/coût : {value(report.cost_feasibility.mfe_to_cost_ratio)}",
            "- MFE supérieur au coût : "
            f"{value(report.cost_feasibility.fraction_mfe_exceeding_cost_pct, '%')}",
            "",
            "## Features numériques — Spearman, FIRST_CROSSING original",
            "",
            "| Feature | N | 5 s | 10 s | 30 s | 60 s | 180 s | MFE | MAE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for numeric_item in report.numeric_feature_diagnostics:
        correlation = numeric_item.spearman_vs_return
        lines.append(
            f"| {numeric_item.feature} | {numeric_item.observations} | "
            f"{value(correlation.seconds_5)} | "
            f"{value(correlation.seconds_10)} | {value(correlation.seconds_30)} | "
            f"{value(correlation.seconds_60)} | {value(correlation.seconds_180)} | "
            f"{value(numeric_item.spearman_vs_mfe_30s)} | "
            f"{value(numeric_item.spearman_vs_mae_30s)} |"
        )
    lines.extend(["", "## Features catégorielles", ""])
    for category_item in report.categorical_feature_diagnostics:
        lines.append(
            f"- `{category_item.analysis}={category_item.value}` — "
            f"N={category_item.observations}, "
            f"suffisant={'OUI' if category_item.sample_sufficient else 'NON'}, "
            f"gross 30 s={value(category_item.gross_expectancy_points.seconds_30)}, "
            f"accuracy={value(category_item.executable_accuracy_30s_pct, '%')}"
        )
    lines.extend(["", "## Interactions descriptives", ""])
    for interaction_item in report.interaction_diagnostics:
        lines.append(
            f"- `{interaction_item.analysis}={interaction_item.value}` — "
            f"N={interaction_item.observations}, "
            f"suffisant={'OUI' if interaction_item.sample_sufficient else 'NON'}, "
            f"gross 30 s={value(interaction_item.gross_expectancy_points.seconds_30)}, "
            f"accuracy={value(interaction_item.executable_accuracy_30s_pct, '%')}"
        )
    lines.extend(
        [
            "",
            "## Méthode et garde-fous",
            "",
            f"- Échantillonnage : {report.sampling_policy}",
            f"- Épisodes : {report.episode_policy}",
            f"- Réarmement/analyse : {report.analysis_policy}",
            f"- PEAK_SCORE : {report.peak_score_policy}",
            "- Paramètres Signal Engine modifiés : NON",
            "- Inversion de stratégie : NON",
            "- Optimisation : NON",
            "- Trading live : DÉSACTIVÉ",
            f"- Hypothèses simulées : {report.simulation.disclaimer}",
            "",
        ]
    )
    return "\n".join(lines)
