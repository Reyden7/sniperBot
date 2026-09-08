"""Streaming Phase D.6 validation of the frozen V1 Signal Engine."""

import tracemalloc
from bisect import bisect_right
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from pydantic import Field

from sniper.backtest.execution_model import BrokerSimulationConfig
from sniper.config import Model, NonNegative, Positive
from sniper.data.feature_source import load_parquet_bars
from sniper.data.time import as_utc
from sniper.domain.bar import Bar, Timeframe
from sniper.domain.edge_validation import (
    AggregateSlice,
    BootstrapConfidenceIntervals,
    ConfidenceInterval,
    DiagnosticEvidence,
    EdgeDiagnosis,
    EdgePerformance,
    EdgeSimulationAssumptions,
    EdgeValidationReport,
    SpearmanDiagnostics,
)
from sniper.domain.signal import Bias, TickFeatures
from sniper.domain.trade import MarketTick
from sniper.features.engine import FeatureEngine
from sniper.strategy.filters import EvaluationContext
from sniper.strategy.scorer import SignalEngine

FROZEN_SIGNAL_ENGINE_HASHES = {
    "strategy/scorer.py": "fb0019dbdaa652ba324774113b0a661514982b65802cf6022af3551ed3bf894a",
    "features/engine.py": "5f32abef8fba5a929ce710b3b2d6af0b342348b7d45256616fd3f2c62037736c",
    "strategy/filters.py": "388681851ba059d57997d1ed2a0a16d1c325d119b438ecbf991faa07eef36919",
}
HORIZONS = (5, 10, 30, 60, 180)
BUCKETS = ((0, 49), (50, 59), (60, 69), (70, 79), (80, 89), (90, 94), (95, 100))
SESSION_ORDER = ("ASIA", "LONDON", "LONDON_NEW_YORK", "NEW_YORK", "OTHER")
WEEKDAY_ORDER = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MICROSECONDS = 1_000_000


class EdgeValidationConfig(Model):
    sampling_interval_seconds: int = Field(default=60, ge=60)
    max_horizon_tick_delay_seconds: Positive = Decimal(2)
    point: Positive = Decimal("0.00001")
    max_spread_points: Positive = Decimal(2)
    max_tick_age_seconds: Positive = Decimal(2)
    slippage_points_per_side: NonNegative = Decimal(1)
    commission_eur_per_lot_per_side: NonNegative = Decimal(2)
    commission_minimum_eur_per_side: NonNegative = Decimal(0)
    bootstrap_seed: int = 20260908
    bootstrap_resamples: int = Field(default=5000, ge=1000)
    minimum_bucket_sample: int = Field(default=100, ge=30)
    minimum_usable_observations: int = Field(default=5000, ge=1000)
    minimum_candidate_observations: int = Field(default=200, ge=30)
    minimum_active_weeks: int = Field(default=8, ge=4)


class FutureMetricsConfig(Protocol):
    @property
    def max_horizon_tick_delay_seconds(self) -> Decimal: ...

    @property
    def point(self) -> Decimal: ...

    @property
    def slippage_points_per_side(self) -> Decimal: ...

    @property
    def commission_eur_per_lot_per_side(self) -> Decimal: ...

    @property
    def commission_minimum_eur_per_side(self) -> Decimal: ...


def _source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    return {
        relative: sha256((package / relative).read_bytes()).hexdigest()
        for relative in FROZEN_SIGNAL_ENGINE_HASHES
    }


def _assert_frozen_engine() -> dict[str, str]:
    observed = _source_hashes()
    if observed != FROZEN_SIGNAL_ENGINE_HASHES:
        raise RuntimeError("Signal Engine source differs from the frozen Phase D.6 baseline")
    return observed


def _days(start: datetime, end: datetime) -> list[date]:
    result = []
    cursor = start.date()
    while cursor < end.date() or (cursor == end.date() and end.time() != datetime.min.time()):
        result.append(cursor)
        cursor += timedelta(days=1)
    return result


def _tick_path(root: Path, day: date) -> Path:
    return root / "ticks" / "symbol=EURUSD" / f"date={day.isoformat()}" / "part-000.parquet"


def _read_ticks(path: Path, start: datetime, end: datetime) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(
            schema={
                "timestamp_utc": pl.Datetime("us", "UTC"),
                "bid": pl.Float64,
                "ask": pl.Float64,
                "spread_points": pl.Float64,
            }
        )
    columns = ["timestamp_utc", "bid", "ask"]
    has_spread_points = "spread_points" in pl.read_parquet_schema(path)
    if has_spread_points:
        columns.append("spread_points")
    frame = pl.read_parquet(path, columns=columns)
    if not has_spread_points:
        frame = frame.with_columns(
            ((pl.col("ask") - pl.col("bid")) / 0.00001).alias("spread_points")
        )
    return frame.filter((pl.col("timestamp_utc") >= start) & (pl.col("timestamp_utc") < end)).sort(
        "timestamp_utc", maintain_order=True
    )


def _session(timestamp: datetime) -> str:
    london = timestamp.astimezone(ZoneInfo("Europe/London"))
    new_york = timestamp.astimezone(ZoneInfo("America/New_York"))
    tokyo = timestamp.astimezone(ZoneInfo("Asia/Tokyo"))
    london_open = 8 <= london.hour < 17
    new_york_open = 8 <= new_york.hour < 17
    tokyo_open = 8 <= tokyo.hour < 17
    if london_open and new_york_open:
        return "LONDON_NEW_YORK"
    if london_open:
        return "LONDON"
    if new_york_open:
        return "NEW_YORK"
    if tokyo_open:
        return "ASIA"
    return "OTHER"


def _bucket(score: int) -> str:
    for minimum, maximum in BUCKETS:
        if minimum <= score <= maximum:
            return f"{minimum}-{maximum}"
    raise ValueError("score outside 0..100")


@dataclass(frozen=True, slots=True)
class _FeatureTick:
    timestamp_utc: datetime
    bid: Decimal
    ask: Decimal

    @property
    def spread(self) -> Decimal:
        return self.ask - self.bid

    @property
    def mid(self) -> Decimal:
        return (self.ask + self.bid) / 2


def _decimal_price(value: float, cache: dict[float, Decimal]) -> Decimal:
    result = cache.get(value)
    if result is None:
        result = Decimal(str(value))
        cache[value] = result
    return result


def _current_market_tick(
    timestamps_us: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    index: int,
    price_cache: dict[float, Decimal],
) -> list[MarketTick]:
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    tick = _FeatureTick(
        timestamp_utc=epoch + timedelta(microseconds=int(timestamps_us[index])),
        bid=_decimal_price(float(bids[index]), price_cache),
        ask=_decimal_price(float(asks[index]), price_cache),
    )
    return cast(list[MarketTick], [tick])


def _streaming_tick_features(
    timestamps_us: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    spread_points: np.ndarray,
    left: int,
    right: int,
    price_cache: dict[float, Decimal],
) -> TickFeatures:
    current_us = int(timestamps_us[right])
    window_times = timestamps_us[left : right + 1]
    rate_1 = int(np.count_nonzero(window_times >= current_us - MICROSECONDS))
    rate_5 = int(np.count_nonzero(window_times >= current_us - 5 * MICROSECONDS))
    rate_15 = int(np.count_nonzero(window_times >= current_us - 15 * MICROSECONDS))
    mids = (bids[left : right + 1] + asks[left : right + 1]) / 2
    changes = np.diff(mids)
    upticks = int(np.count_nonzero(changes > 0))
    downticks = int(np.count_nonzero(changes < 0))
    directional = upticks + downticks
    uptick_ratio = Decimal(upticks) / directional if directional else Decimal("0.5")
    downtick_ratio = Decimal(downticks) / directional if directional else Decimal("0.5")
    price_acceleration = Decimal(0)
    if right - left >= 2:
        last_mids = []
        for index in range(right - 2, right + 1):
            bid = _decimal_price(float(bids[index]), price_cache)
            ask = _decimal_price(float(asks[index]), price_cache)
            last_mids.append((bid + ask) / 2)
        price_acceleration = (last_mids[2] - last_mids[1]) - (last_mids[1] - last_mids[0])
    spreads = spread_points[left : right + 1]
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    return TickFeatures(
        rate_1s=rate_1,
        rate_5s=rate_5,
        rate_15s=rate_15,
        uptick_ratio=uptick_ratio,
        downtick_ratio=downtick_ratio,
        price_acceleration=price_acceleration,
        tick_rate_acceleration=Decimal(rate_1) - Decimal(rate_5) / 5,
        last_tick_timestamp_utc=epoch + timedelta(microseconds=current_us),
        last_tick_age_seconds=Decimal(0),
        current_spread_points=Decimal(str(float(spreads[-1]))),
        median_spread_points=Decimal(str(float(np.median(spreads)))),
    )


def _bar_window(
    bars: dict[Timeframe, list[Bar]],
    bar_times: dict[Timeframe, list[datetime]],
    timestamp: datetime,
) -> dict[Timeframe, list[Bar]]:
    result: dict[Timeframe, list[Bar]] = {}
    for timeframe in ("M15", "M5", "M1"):
        cutoff = bisect_right(bar_times[timeframe], timestamp)
        result[timeframe] = bars[timeframe][max(0, cutoff - 64) : cutoff]
    return result


def _future_metrics(
    timestamps_us: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    current_index: int,
    direction: Bias,
    stop_points: int,
    target_points: int,
    config: FutureMetricsConfig,
    broker: BrokerSimulationConfig,
) -> dict[str, object] | None:
    current_us = int(timestamps_us[current_index])
    current_bid = float(bids[current_index])
    current_ask = float(asks[current_index])
    current_mid = (current_bid + current_ask) / 2
    sign = 1.0 if direction == Bias.BUY else -1.0
    point = float(config.point)
    future_indices: dict[int, int] = {}
    for horizon in HORIZONS:
        target_us = current_us + horizon * MICROSECONDS
        index = int(np.searchsorted(timestamps_us, target_us, side="left"))
        if index >= len(timestamps_us):
            return None
        delay_seconds = (int(timestamps_us[index]) - target_us) / MICROSECONDS
        if delay_seconds > float(config.max_horizon_tick_delay_seconds):
            return None
        future_indices[horizon] = index

    executable_returns: dict[int, float] = {}
    market_returns: dict[int, float] = {}
    for horizon, index in future_indices.items():
        future_bid = float(bids[index])
        future_ask = float(asks[index])
        future_mid = (future_bid + future_ask) / 2
        market_returns[horizon] = sign * (future_mid - current_mid) / point
        executable_returns[horizon] = (
            (future_bid - current_ask) / point
            if direction == Bias.BUY
            else (current_bid - future_ask) / point
        )

    end_30 = int(np.searchsorted(timestamps_us, current_us + 30 * MICROSECONDS, side="right"))
    start_window = current_index + 1
    if end_30 <= start_window:
        return None
    if direction == Bias.BUY:
        pnl_30 = (bids[start_window:end_30] - current_ask) / point
    else:
        pnl_30 = (current_bid - asks[start_window:end_30]) / point
    mfe = max(0.0, float(np.max(pnl_30)))
    mae = max(0.0, -float(np.min(pnl_30)))

    end_180 = int(np.searchsorted(timestamps_us, current_us + 180 * MICROSECONDS, side="right"))
    if direction == Bias.BUY:
        path_pnl = (bids[start_window:end_180] - current_ask) / point
    else:
        path_pnl = (current_bid - asks[start_window:end_180]) / point
    target_hits = np.flatnonzero(path_pnl >= target_points)
    stop_hits = np.flatnonzero(path_pnl <= -stop_points)
    first_target = int(target_hits[0]) if len(target_hits) else None
    first_stop = int(stop_hits[0]) if len(stop_hits) else None
    if first_target is not None and (first_stop is None or first_target < first_stop):
        stop_target_outcome = "TARGET_HIT_FIRST"
    elif first_stop is not None and (first_target is None or first_stop < first_target):
        stop_target_outcome = "STOP_HIT_FIRST"
    else:
        stop_target_outcome = "NEITHER_HIT"

    volume = broker.volumes.minimum
    commission_side = max(
        config.commission_eur_per_lot_per_side * volume,
        config.commission_minimum_eur_per_side,
    )
    commission_round_trip_eur = float(commission_side * 2)
    usd_per_point = point * float(broker.contract_size) * float(volume)
    eur_per_point = usd_per_point / current_mid
    commission_points = commission_round_trip_eur / eur_per_point
    before = market_returns[30]
    after_spread = executable_returns[30]
    after_slippage = after_spread - 2 * float(config.slippage_points_per_side)
    after_commission = after_slippage - commission_points
    return {
        **{f"return_after_{horizon}s_points": executable_returns[horizon] for horizon in HORIZONS},
        **{
            f"market_return_after_{horizon}s_points": market_returns[horizon]
            for horizon in HORIZONS
        },
        "directional_accuracy": float(after_spread > 0),
        "expectancy_before_costs_points": before,
        "expectancy_after_spread_points": after_spread,
        "expectancy_after_slippage_points": after_slippage,
        "expectancy_after_commission_points": after_commission,
        "expectancy_after_commission_eur": after_commission * eur_per_point,
        "mfe_30s_points": mfe,
        "mae_30s_points": mae,
        "mfe_30s_eur": mfe * eur_per_point,
        "mae_30s_eur": mae * eur_per_point,
        "commission_round_trip_eur": commission_round_trip_eur,
        "commission_round_trip_points": commission_points,
        "stop_target_outcome": stop_target_outcome,
    }


def _none_metrics() -> dict[str, object]:
    return {
        **{f"return_after_{horizon}s_points": None for horizon in HORIZONS},
        **{f"market_return_after_{horizon}s_points": None for horizon in HORIZONS},
        "directional_accuracy": None,
        "expectancy_before_costs_points": None,
        "expectancy_after_spread_points": None,
        "expectancy_after_slippage_points": None,
        "expectancy_after_commission_points": None,
        "expectancy_after_commission_eur": None,
        "mfe_30s_points": None,
        "mae_30s_points": None,
        "mfe_30s_eur": None,
        "mae_30s_eur": None,
        "commission_round_trip_eur": None,
        "commission_round_trip_points": None,
        "stop_target_outcome": None,
    }


def _write_observations(rows: list[dict[str, object]], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".parquet.tmp")
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(
        temporary, compression="zstd", statistics=True
    )
    temporary.replace(target)


def _mean(frame: pl.DataFrame, column: str) -> float | None:
    if frame.is_empty():
        return None
    value = frame[column].mean()
    return float(str(value)) if value is not None else None


def _aggregate(frame: pl.DataFrame, key: str, config: EdgeValidationConfig) -> AggregateSlice:
    usable = frame.filter(pl.col("usable"))
    side = frame["side"] if not frame.is_empty() else pl.Series([], dtype=pl.String)
    outcomes = (
        usable["stop_target_outcome"] if not usable.is_empty() else pl.Series([], dtype=pl.String)
    )
    accuracy = _mean(usable, "directional_accuracy")
    return AggregateSlice(
        key=key,
        observations=frame.height,
        blocked=int(frame["blocked"].sum()) if not frame.is_empty() else 0,
        usable=usable.height,
        buy=int((side == "BUY").sum()),
        sell=int((side == "SELL").sum()),
        wait=int((side == "WAIT").sum()),
        sample_sufficient=usable.height >= config.minimum_bucket_sample,
        directional_accuracy_pct=accuracy * 100 if accuracy is not None else None,
        average_return_after_5s_points=_mean(usable, "return_after_5s_points"),
        average_return_after_10s_points=_mean(usable, "return_after_10s_points"),
        average_return_after_30s_points=_mean(usable, "return_after_30s_points"),
        average_return_after_60s_points=_mean(usable, "return_after_60s_points"),
        average_return_after_180s_points=_mean(usable, "return_after_180s_points"),
        expectancy_before_costs_points=_mean(usable, "expectancy_before_costs_points"),
        expectancy_after_spread_points=_mean(usable, "expectancy_after_spread_points"),
        expectancy_after_slippage_points=_mean(usable, "expectancy_after_slippage_points"),
        expectancy_after_commission_points=_mean(usable, "expectancy_after_commission_points"),
        expectancy_after_commission_eur=_mean(usable, "expectancy_after_commission_eur"),
        average_mfe_30s_points=_mean(usable, "mfe_30s_points"),
        average_mae_30s_points=_mean(usable, "mae_30s_points"),
        average_mfe_30s_eur=_mean(usable, "mfe_30s_eur"),
        average_mae_30s_eur=_mean(usable, "mae_30s_eur"),
        target_hit_first=int((outcomes == "TARGET_HIT_FIRST").sum()),
        stop_hit_first=int((outcomes == "STOP_HIT_FIRST").sum()),
        neither_hit=int((outcomes == "NEITHER_HIT").sum()),
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and sorted_values[end] == sorted_values[cursor]:
            end += 1
        ranks[order[cursor:end]] = (cursor + end - 1) / 2 + 1
        cursor = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    return float(np.corrcoef(_average_ranks(left), _average_ranks(right))[0, 1])


def _spearman_diagnostics(frame: pl.DataFrame) -> SpearmanDiagnostics:
    usable = frame.filter(pl.col("usable"))
    if usable.is_empty():
        return SpearmanDiagnostics(
            observations=0,
            score_vs_directional_accuracy=None,
            score_vs_net_expectancy=None,
            score_vs_mfe=None,
            score_vs_mae=None,
            desired_relationships_met=0,
        )
    score = usable["score"].to_numpy().astype(float)
    accuracy = usable["directional_accuracy"].to_numpy().astype(float)
    expectancy = usable["expectancy_after_commission_points"].to_numpy().astype(float)
    mfe = usable["mfe_30s_points"].to_numpy().astype(float)
    mae = usable["mae_30s_points"].to_numpy().astype(float)
    correlations = (
        _spearman(score, accuracy),
        _spearman(score, expectancy),
        _spearman(score, mfe),
        _spearman(score, mae),
    )
    met = sum(
        value is not None and (value >= 0.05 if index < 3 else value <= -0.05)
        for index, value in enumerate(correlations)
    )
    return SpearmanDiagnostics(
        observations=usable.height,
        score_vs_directional_accuracy=correlations[0],
        score_vs_net_expectancy=correlations[1],
        score_vs_mfe=correlations[2],
        score_vs_mae=correlations[3],
        desired_relationships_met=met,
    )


def _ci(values: np.ndarray, estimate: float) -> ConfidenceInterval:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return ConfidenceInterval(estimate=estimate, lower_95=float(lower), upper_95=float(upper))


def _bootstrap(
    frame: pl.DataFrame, config: EdgeValidationConfig
) -> BootstrapConfidenceIntervals | None:
    usable = frame.filter(pl.col("usable"))
    if usable.is_empty():
        return None
    daily = (
        usable.group_by("date", maintain_order=True)
        .agg(
            pl.len().alias("count"),
            pl.col("directional_accuracy").sum().alias("accuracy"),
            pl.col("expectancy_after_commission_points").sum().alias("net"),
            pl.col("mfe_30s_points").sum().alias("mfe"),
            pl.col("mae_30s_points").sum().alias("mae"),
        )
        .sort("date")
    )
    counts = daily["count"].to_numpy().astype(float)
    sums = np.vstack(
        [daily[column].to_numpy().astype(float) for column in ("accuracy", "net", "mfe", "mae")]
    )
    generator = np.random.default_rng(config.bootstrap_seed)
    samples = generator.integers(0, len(counts), size=(config.bootstrap_resamples, len(counts)))
    sampled_counts = counts[samples].sum(axis=1)
    estimates = np.vstack([values[samples].sum(axis=1) / sampled_counts for values in sums])
    observed = sums.sum(axis=1) / counts.sum()
    return BootstrapConfidenceIntervals(
        seed=config.bootstrap_seed,
        resamples=config.bootstrap_resamples,
        active_days=len(counts),
        directional_accuracy_pct=_ci(estimates[0] * 100, float(observed[0] * 100)),
        net_expectancy_points=_ci(estimates[1], float(observed[1])),
        mfe_30s_points=_ci(estimates[2], float(observed[2])),
        mae_30s_points=_ci(estimates[3], float(observed[3])),
    )


def _group_slices(
    frame: pl.DataFrame, column: str, keys: list[str], config: EdgeValidationConfig
) -> tuple[AggregateSlice, ...]:
    return tuple(_aggregate(frame.filter(pl.col(column) == key), key, config) for key in keys)


def _diagnosis(
    overall: AggregateSlice,
    candidate_overall: AggregateSlice,
    buckets: tuple[AggregateSlice, ...],
    candidate_weeks: tuple[AggregateSlice, ...],
    ci: BootstrapConfidenceIntervals | None,
    spearman: SpearmanDiagnostics,
    config: EdgeValidationConfig,
) -> tuple[EdgeDiagnosis, DiagnosticEvidence]:
    candidate_observations = sum(
        bucket.usable for bucket in buckets if bucket.key in ("90-94", "95-100")
    )
    active_weeks = sum(week.usable > 0 for week in candidate_weeks)
    positive_weeks = sum(
        week.expectancy_after_commission_points is not None
        and week.expectancy_after_commission_points > 0
        for week in candidate_weeks
    )
    positive_fraction = positive_weeks / active_weeks if active_weeks else None
    net_positive = (
        candidate_overall.expectancy_after_commission_points is not None
        and candidate_overall.expectancy_after_commission_points > 0
    )
    ci_lower_positive = ci is not None and ci.net_expectancy_points.lower_95 > 0
    stable = (
        active_weeks >= config.minimum_active_weeks
        and positive_fraction is not None
        and positive_fraction >= 0.60
    )
    score_coherent = spearman.desired_relationships_met >= 3 and (
        spearman.score_vs_net_expectancy is not None and spearman.score_vs_net_expectancy > 0
    )
    sufficient = (
        overall.usable >= config.minimum_usable_observations
        and candidate_observations >= config.minimum_candidate_observations
        and active_weeks >= config.minimum_active_weeks
    )
    evidence = DiagnosticEvidence(
        minimum_usable_observations=config.minimum_usable_observations,
        minimum_candidate_observations=config.minimum_candidate_observations,
        minimum_active_weeks=config.minimum_active_weeks,
        usable_observations=overall.usable,
        candidate_observations=candidate_observations,
        active_weeks=active_weeks,
        positive_net_weeks=positive_weeks,
        positive_net_week_fraction=positive_fraction,
        net_expectancy_positive=net_positive,
        net_expectancy_ci_lower_positive=ci_lower_positive,
        stable_across_weeks=stable,
        resists_simulated_costs=net_positive,
        score_quality_coherent=score_coherent,
    )
    if not sufficient:
        diagnosis: EdgeDiagnosis = "INSUFFICIENT_DATA"
    elif net_positive and ci_lower_positive and stable and score_coherent:
        diagnosis = "EDGE_CANDIDATE"
    elif not net_positive:
        diagnosis = "EDGE_NOT_DETECTED"
    else:
        diagnosis = "EDGE_WEAK"
    return diagnosis, evidence


class EdgeValidator:
    def __init__(self, config: EdgeValidationConfig | None = None) -> None:
        self.config = config or EdgeValidationConfig()
        self.feature_engine = FeatureEngine(point=self.config.point)
        self.signal_engine = SignalEngine()
        self.broker = BrokerSimulationConfig(point=self.config.point)

    def run(
        self,
        *,
        data_root: Path,
        start_utc: datetime,
        end_utc: datetime,
    ) -> EdgeValidationReport:
        start, end = as_utc(start_utc), as_utc(end_utc)
        if start >= end:
            raise ValueError("edge validation requires a nonempty UTC range")
        hashes = _assert_frozen_engine()
        tracemalloc.start()
        started = perf_counter()
        bars = load_parquet_bars(
            data_root,
            "EURUSD",
            end,
            lookback_days=max(1, (end - start).days + 30),
        )
        timeframes: tuple[Timeframe, ...] = ("M15", "M5", "M1")
        bar_times: dict[Timeframe, list[datetime]] = {
            timeframe: [bar.open_time_utc for bar in bars[timeframe]] for timeframe in timeframes
        }
        context = EvaluationContext(
            max_spread_points=self.config.max_spread_points,
            max_tick_age_seconds=self.config.max_tick_age_seconds,
            session_allowed=True,
            news_clear=True,
        )
        run_name = f"{start:%Y%m%dT%H%M%SZ}-{end:%Y%m%dT%H%M%SZ}-60s"
        observation_root = data_root / "evaluations" / f"edge-validation-{run_name}"
        observation_paths: list[Path] = []
        ticks_analyzed = 0
        observations_produced = 0
        samples_skipped_no_feature_base = 0
        peak_tick_frame_mb = 0.0

        for day in _days(start, end):
            day_start = max(start, datetime.combine(day, datetime.min.time(), UTC))
            next_midnight = day_start.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = min(end, next_midnight + timedelta(days=1))
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
            spread_points = extended["spread_points"].to_numpy()
            current_us = current["timestamp_utc"].cast(pl.Int64).to_numpy()
            minute_keys = current_us // (self.config.sampling_interval_seconds * MICROSECONDS)
            sample_local_indices = np.concatenate(
                (np.array([0]), np.flatnonzero(np.diff(minute_keys)) + 1)
            )
            rows: list[dict[str, object]] = []
            price_cache: dict[float, Decimal] = {}
            epoch = datetime(1970, 1, 1, tzinfo=UTC)
            for local_index in sample_local_indices:
                sample_us = int(current_us[int(local_index)])
                current_index = int(np.searchsorted(timestamps_us, sample_us, side="left"))
                sample_time = epoch + timedelta(microseconds=sample_us)
                visible_left = int(
                    np.searchsorted(timestamps_us, sample_us - 60 * MICROSECONDS, side="left")
                )
                current_tick = _current_market_tick(
                    timestamps_us,
                    bids,
                    asks,
                    current_index,
                    price_cache,
                )
                try:
                    snapshot = self.feature_engine.compute(
                        bars=_bar_window(bars, bar_times, sample_time),
                        ticks=current_tick,
                        as_of_utc=sample_time,
                        symbol="EURUSD",
                    )
                except ValueError:
                    samples_skipped_no_feature_base += 1
                    continue
                snapshot = snapshot.model_copy(
                    update={
                        "ticks": _streaming_tick_features(
                            timestamps_us,
                            bids,
                            asks,
                            spread_points,
                            visible_left,
                            current_index,
                            price_cache,
                        ),
                        "visible_ticks": current_index - visible_left + 1,
                    }
                )
                decision = self.signal_engine.evaluate(snapshot, context, point=self.config.point)
                blocked = bool(decision.blockers)
                metrics: dict[str, object] | None = None
                if (
                    not blocked
                    and decision.trigger_direction != Bias.NEUTRAL
                    and decision.proposed_stop_distance_points is not None
                    and decision.proposed_target_distance_points is not None
                ):
                    metrics = _future_metrics(
                        timestamps_us,
                        bids,
                        asks,
                        current_index,
                        decision.trigger_direction,
                        decision.proposed_stop_distance_points,
                        decision.proposed_target_distance_points,
                        self.config,
                        self.broker,
                    )
                usable = metrics is not None
                iso = sample_time.isocalendar()
                row: dict[str, object] = {
                    "timestamp_utc": sample_time,
                    "date": sample_time.date().isoformat(),
                    "week": f"{iso.year}-W{iso.week:02d}",
                    "weekday": sample_time.strftime("%A"),
                    "session": _session(sample_time),
                    "score": decision.score,
                    "score_bucket": _bucket(decision.score),
                    "side": decision.side.value,
                    "market_bias": decision.market_bias.value,
                    "trigger_direction": decision.trigger_direction.value,
                    "blocked": blocked,
                    "usable": usable,
                    "outcome_status": (
                        "SIGNAL_BLOCKED"
                        if blocked
                        else "COMPLETE"
                        if usable
                        else "FUTURE_DATA_INCOMPLETE"
                    ),
                    "blockers": ",".join(decision.blockers),
                    "reasons": ",".join(decision.reasons),
                    "current_bid": float(bids[current_index]),
                    "current_ask": float(asks[current_index]),
                    "spread_points": float(snapshot.ticks.current_spread_points),
                    "m1_atr_points": float(snapshot.m1.atr / self.config.point),
                    "proposed_stop_points": decision.proposed_stop_distance_points,
                    "proposed_target_points": decision.proposed_target_distance_points,
                    **{
                        f"component_{name}": value
                        for name, value in decision.components.model_dump().items()
                    },
                    **(metrics if metrics is not None else _none_metrics()),
                }
                rows.append(row)
            target = observation_root / f"date={day.isoformat()}" / "part-000.parquet"
            _write_observations(rows, target)
            observation_paths.append(target)
            observations_produced += len(rows)

        if not observation_paths:
            raise ValueError("no active tick partitions were found in the requested range")
        observations = pl.concat(
            [pl.read_parquet(path) for path in observation_paths], how="vertical_relaxed"
        ).sort("timestamp_utc")
        score_distribution = {
            str(row["score"]): int(row["len"])
            for row in observations.group_by("score").len().sort("score").iter_rows(named=True)
        }
        overall = _aggregate(observations, "GLOBAL", self.config)
        candidates = observations.filter(pl.col("side") != "WAIT")
        candidate_overall = _aggregate(candidates, "BUY_SELL_CANDIDATES", self.config)
        bucket_slices = _group_slices(
            observations,
            "score_bucket",
            [f"{minimum}-{maximum}" for minimum, maximum in BUCKETS],
            self.config,
        )
        session_slices = _group_slices(observations, "session", list(SESSION_ORDER), self.config)
        candidate_session_slices = _group_slices(
            candidates, "session", list(SESSION_ORDER), self.config
        )
        week_keys = sorted(observations["week"].unique().to_list())
        week_slices = _group_slices(observations, "week", week_keys, self.config)
        candidate_week_slices = _group_slices(candidates, "week", week_keys, self.config)
        weekday_slices = _group_slices(observations, "weekday", list(WEEKDAY_ORDER), self.config)
        confidence = _bootstrap(candidates, self.config)
        spearman = _spearman_diagnostics(observations)
        diagnosis, evidence = _diagnosis(
            overall,
            candidate_overall,
            bucket_slices,
            candidate_week_slices,
            confidence,
            spearman,
            self.config,
        )
        elapsed = perf_counter() - started
        _, peak_python = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        profile = self.broker.profile()
        return EdgeValidationReport(
            requested_start_utc=start,
            requested_end_utc=end,
            sampling_policy=(
                "FIRST_TICK_OF_EACH_ACTIVE_UTC_MINUTE; at most one observation per 60-second "
                "UTC bucket; no observations during tickless minutes; each signal sees only "
                "ticks and completed bars timestamped at or before the sampled tick; samples "
                "before the first computable M15/M5/M1 feature base are counted and skipped"
            ),
            session_policy=(
                "UTC is canonical. Each UTC instant is converted with IANA DST rules to "
                "Europe/London, America/New_York and Asia/Tokyo; local session windows are "
                "08:00-17:00. LONDON_NEW_YORK has priority when both are open, then LONDON, "
                "NEW_YORK, ASIA, and OTHER. Server time is never used."
            ),
            max_horizon_tick_delay_seconds=float(self.config.max_horizon_tick_delay_seconds),
            signal_engine_source_sha256=hashes,
            evaluation_context_assumptions={
                "max_spread_points": str(self.config.max_spread_points),
                "max_tick_age_seconds": str(self.config.max_tick_age_seconds),
                "session_allowed": "true for research comparison across all session buckets",
                "news_clear": (
                    "true as an explicit evaluation assumption; no historical news calendar "
                    "was collected, so this is not verified news clearance"
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
            observation_parquet_root=str(observation_root.resolve()),
            performance=EdgePerformance(
                elapsed_seconds=elapsed,
                ticks_analyzed=ticks_analyzed,
                observations_produced=observations_produced,
                samples_skipped_no_feature_base=samples_skipped_no_feature_base,
                ticks_per_second=ticks_analyzed / elapsed if elapsed else 0,
                peak_python_memory_mb=peak_python / 1024 / 1024,
                peak_loaded_tick_frame_mb=peak_tick_frame_mb,
                memory_measurement_note=(
                    "Python peak is tracemalloc-only and excludes native Polars buffers; the "
                    "largest concurrently loaded four-column tick frame is reported separately."
                ),
            ),
            score_distribution=score_distribution,
            overall=overall,
            candidate_overall=candidate_overall,
            score_buckets=bucket_slices,
            sessions=session_slices,
            candidate_sessions=candidate_session_slices,
            weeks=week_slices,
            candidate_weeks=candidate_week_slices,
            weekdays=weekday_slices,
            confidence_intervals=confidence,
            spearman=spearman,
            diagnostic_evidence=evidence,
            diagnosis=diagnosis,
            limitations=(
                "Historical news clearance is unavailable and explicitly assumed for evaluation.",
                "Execution costs and nano-lot monetary values use a simulated broker profile.",
                "Bootstrap resamples UTC days to preserve intraday dependence but does not prove "
                "future generalization.",
                "Stop/target first-hit outcomes are limited to 180 seconds.",
                "No weights, thresholds, features or Signal Engine rules were optimized.",
            ),
        )


def render_edge_report(report: EdgeValidationReport) -> str:
    def value(number: float | None, suffix: str = "") -> str:
        return "N/A" if number is None else f"{number:.6f}{suffix}"

    lines = [
        "# SNIPER Phase D.6 — Edge Validation",
        "",
        f"Diagnostic final : **{report.diagnosis}**",
        "",
        f"Periode UTC : `{report.requested_start_utc.isoformat()}` → "
        f"`{report.requested_end_utc.isoformat()}`",
        f"Ticks analyses : {report.performance.ticks_analyzed:,}",
        f"Observations : {report.performance.observations_produced:,}",
        f"Temps : {report.performance.elapsed_seconds:.2f} s "
        f"({report.performance.ticks_per_second:,.0f} ticks/s)",
        f"Memoire Python peak : {report.performance.peak_python_memory_mb:.2f} MiB; "
        f"plus grand frame ticks : {report.performance.peak_loaded_tick_frame_mb:.2f} MiB",
        "",
        "## Résultat global",
        "",
        f"Candidats BUY/SELL utilisables : {report.candidate_overall.usable:,}; "
        f"accuracy 30 s : {value(report.candidate_overall.directional_accuracy_pct, '%')}; "
        f"expectancy nette : "
        f"{value(report.candidate_overall.expectancy_after_commission_points, ' points')}",
        "",
        "| Étape de coûts (candidats) | Expectancy points |",
        "|---|---:|",
        f"| Avant coûts | {value(report.candidate_overall.expectancy_before_costs_points)} |",
        f"| Après spread | {value(report.candidate_overall.expectancy_after_spread_points)} |",
        f"| Après slippage | {value(report.candidate_overall.expectancy_after_slippage_points)} |",
        f"| Après commission | "
        f"{value(report.candidate_overall.expectancy_after_commission_points)} |",
        "",
        "| Horizon candidat | Rendement exécutable moyen (points) |",
        "|---|---:|",
        f"| 5 s | {value(report.candidate_overall.average_return_after_5s_points)} |",
        f"| 10 s | {value(report.candidate_overall.average_return_after_10s_points)} |",
        f"| 30 s | {value(report.candidate_overall.average_return_after_30s_points)} |",
        f"| 60 s | {value(report.candidate_overall.average_return_after_60s_points)} |",
        f"| 180 s | {value(report.candidate_overall.average_return_after_180s_points)} |",
        "",
        "## Buckets de score",
        "",
        "| Score | Obs. | Bloquées | Usables | BUY | SELL | WAIT | Accuracy | "
        "Expectancy nette | MFE | MAE | Target/Stop/Neither |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.score_buckets:
        lines.append(
            f"| {row.key} | {row.observations} | {row.blocked} | {row.usable} | "
            f"{row.buy} | {row.sell} | {row.wait} | "
            f"{value(row.directional_accuracy_pct, '%')} | "
            f"{value(row.expectancy_after_commission_points)} | "
            f"{value(row.average_mfe_30s_points)} | {value(row.average_mae_30s_points)} | "
            f"{row.target_hit_first}/{row.stop_hit_first}/{row.neither_hit} |"
        )
    lines.extend(
        [
            "",
            "## Sessions — candidats BUY/SELL",
            "",
            "| Session | Usables | Accuracy | Expectancy nette | MFE | MAE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report.candidate_sessions:
        lines.append(
            f"| {row.key} | {row.usable} | {value(row.directional_accuracy_pct, '%')} | "
            f"{value(row.expectancy_after_commission_points)} | "
            f"{value(row.average_mfe_30s_points)} | {value(row.average_mae_30s_points)} |"
        )
    lines.extend(
        [
            "",
            "## Stabilité hebdomadaire — candidats BUY/SELL",
            "",
            "| Semaine | Observations | Usables | Accuracy | Expectancy nette |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report.candidate_weeks:
        lines.append(
            f"| {row.key} | {row.observations} | {row.usable} | "
            f"{value(row.directional_accuracy_pct, '%')} | "
            f"{value(row.expectancy_after_commission_points)} |"
        )
    lines.extend(
        [
            "",
            "## Jour de semaine — toutes observations utilisables",
            "",
            "| Jour | Usables | Accuracy | Expectancy nette |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report.weekdays:
        lines.append(
            f"| {row.key} | {row.usable} | {value(row.directional_accuracy_pct, '%')} | "
            f"{value(row.expectancy_after_commission_points)} |"
        )
    lines.extend(
        [
            "",
            "## Intervalles de confiance 95 %",
            "",
        ]
    )
    if report.confidence_intervals is not None:
        ci = report.confidence_intervals
        for name, interval in (
            ("Directional accuracy (%)", ci.directional_accuracy_pct),
            ("Net expectancy (points)", ci.net_expectancy_points),
            ("MFE 30 s (points)", ci.mfe_30s_points),
            ("MAE 30 s (points)", ci.mae_30s_points),
        ):
            lines.append(
                f"- {name}: {interval.estimate:.6f} "
                f"[{interval.lower_95:.6f}, {interval.upper_95:.6f}]"
            )
    lines.extend(
        [
            "",
            "## Monotonie — Spearman",
            "",
            f"- Score / accuracy : {value(report.spearman.score_vs_directional_accuracy)}",
            f"- Score / expectancy nette : {value(report.spearman.score_vs_net_expectancy)}",
            f"- Score / MFE : {value(report.spearman.score_vs_mfe)}",
            f"- Score / MAE : {value(report.spearman.score_vs_mae)}",
            f"- Relations attendues satisfaites : {report.spearman.desired_relationships_met}/4",
            "",
            "## Distribution exacte des scores",
            "",
            ", ".join(f"{score}: {count}" for score, count in report.score_distribution.items()),
        ]
    )
    lines.extend(
        [
            "",
            "## Garde-fous",
            "",
            "- Paramètres du Signal Engine modifiés : NON",
            "- Optimisation effectuée : NON",
            "- Trading live : DÉSACTIVÉ",
            f"- Échantillonnage : {report.sampling_policy}",
            f"- Sessions : {report.session_policy}",
            f"- Hypothèses simulées : {report.simulation.disclaimer}",
        ]
    )
    return "\n".join(lines) + "\n"
