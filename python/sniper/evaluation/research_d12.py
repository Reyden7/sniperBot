"""Phase D.12 causal feature and regime discovery on EURUSD RESEARCH only."""

import hashlib
import json
import math
import tracemalloc
import warnings
from collections import deque
from datetime import UTC, datetime, time, timedelta
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
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from sniper.backtest.execution_model import BrokerSimulationConfig
from sniper.data.time import as_utc
from sniper.domain.research_d12 import D12Conclusion, D12Performance, ResearchD12Report
from sniper.evaluation.edge_validation import MICROSECONDS, _days, _read_ticks, _tick_path
from sniper.evaluation.research_v2 import SCENARIOS, _commission_points

POINT = 0.00001
HORIZONS_D12 = (30, 60, 180, 300, 600, 900)
MOVE_THRESHOLDS_D12 = (1.5, 2.0, 3.0, 4.0)
BOOTSTRAP_REPLICATIONS_D12 = 500
RANDOM_SEED_D12 = 20260912
EXPECTED_RESEARCH_TICKS = 17_941_326
WINDOWS = (300, 900, 1800, 3600, 14400)
EVENT_COOLDOWNS = {
    "MINUTE_BASELINE": 60,
    "VOLATILITY_EVENT": 180,
    "BREAKOUT_EVENT": 300,
    "ACCELERATION_EVENT": 120,
    "COMPRESSION_RELEASE_EVENT": 600,
}
BASE_SCENARIO = next(item for item in SCENARIOS if item.name == "BASE")


def _feature_groups() -> dict[str, str]:
    groups: dict[str, str] = {}
    for window in WINDOWS:
        groups[f"realized_volatility_{window}s_points"] = "volatility"
        groups[f"range_{window}s_points"] = "volatility"
        groups[f"trend_efficiency_{window}s"] = "trend"
        groups[f"absolute_directional_persistence_{window}s"] = "trend"
    groups.update(
        {
            "atr_normalized_300s": "volatility",
            "volatility_percentile_300s": "regime",
            "range_expansion_ratio_300s": "volatility",
            "range_compression_ratio_300s": "volatility",
            "return_autocorrelation_lag1_900s": "trend",
            "variance_ratio_5_900s": "trend",
            "tick_activity_percentile_15s": "regime",
            "spread_percentile_4h": "spread_cost",
            "spread_to_realized_volatility_300s": "spread_cost",
            "realized_move_to_base_cost_300s": "spread_cost",
            "distance_to_daily_high_points": "trend",
            "distance_to_daily_low_points": "trend",
            "position_in_daily_range": "trend",
            "distance_to_intraday_twap_points": "trend",
            "intraday_twap_slope_900s_points": "trend",
            "session_age_minutes": "session_time",
            "minutes_since_london_open": "session_time",
            "minutes_since_new_york_open": "session_time",
            "london_new_york_overlap": "session_time",
            "asian_range_points": "session_time",
            "london_first_30m_realized_volatility": "session_time",
            "london_first_60m_range": "session_time",
            "london_open_displacement_points": "session_time",
        }
    )
    for seconds in (1, 5, 15, 30, 60):
        groups[f"signed_tick_imbalance_{seconds}s"] = "microstructure"
    groups.update(
        {
            "imbalance_acceleration_5s_vs_30s": "microstructure",
            "run_length_mean_60s": "microstructure",
            "run_length_max_60s": "microstructure",
            "current_signed_run_length": "microstructure",
            "transition_probability_up_up_60s": "microstructure",
            "transition_probability_down_down_60s": "microstructure",
            "interarrival_mean_5s": "microstructure",
            "interarrival_mean_60s": "microstructure",
            "interarrival_cv_60s": "microstructure",
            "interarrival_acceleration_5s_vs_60s": "microstructure",
            "quote_change_rate_5s": "microstructure",
            "quote_change_rate_60s": "microstructure",
            "spread_change_rate_60s": "microstructure",
            "spread_delta_vs_median_60s": "spread_cost",
            "displacement_per_tick_15s_points": "momentum",
            "micro_pullback_amplitude_60s_points": "breakout_retest",
            "time_since_local_high_60s": "breakout_retest",
            "time_since_local_low_60s": "breakout_retest",
            "failed_breakout_count_60s": "breakout_retest",
            "breakout_velocity_15s_points_per_second": "breakout_retest",
            "retest_latency_seconds": "breakout_retest",
            "return_5s_points": "momentum",
            "return_15s_points": "momentum",
            "return_30s_points": "momentum",
            "return_60s_points": "momentum",
            "tick_rate_1s": "microstructure",
            "tick_rate_5s": "microstructure",
            "tick_rate_15s": "microstructure",
            "tick_rate_acceleration": "microstructure",
            "tick_uptick_ratio_60s": "microstructure",
            "spread_points": "spread_cost",
            "spread_vs_median_60s": "spread_cost",
        }
    )
    groups.update(
        {
            "return_60s_x_high_volatility": "regime",
            "breakout_velocity_x_low_spread_vol": "regime",
            "micro_pullback_x_trend_efficiency_900s": "regime",
            "imbalance_15s_x_compression_release": "regime",
            "imbalance_acceleration_x_range_expansion": "regime",
            "signed_persistence_900s_x_volatility_percentile": "regime",
            "london_displacement_x_overlap": "regime",
            "distance_twap_x_trend_efficiency_3600s": "regime",
        }
    )
    return groups


FEATURE_GROUPS = _feature_groups()
FEATURE_NAMES = tuple(FEATURE_GROUPS)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_research_scope(data_root: Path, start: datetime, end: datetime) -> Path:
    resolved = data_root.resolve()
    if "holdout" in str(resolved).lower():
        raise PermissionError("D.12 refuses any HOLDOUT path before reading files or hashes")
    expected_start = datetime(2026, 6, 10, tzinfo=UTC)
    expected_end = datetime(2026, 9, 8, tzinfo=UTC)
    if start != expected_start or end != expected_end:
        raise PermissionError("D.12 may only use EURUSD_RESEARCH_20260610_20260908")
    tick_root = (resolved / "ticks" / "symbol=EURUSD").resolve()
    if tick_root.parent.parent != resolved or tick_root.name != "symbol=EURUSD":
        raise PermissionError("D.12 RESEARCH tick root escaped the declared data root")
    if not tick_root.exists():
        raise ValueError("D.12 RESEARCH EURUSD ticks are missing")
    return tick_root


def _rolling_sum(times: np.ndarray, values: np.ndarray, seconds: int) -> np.ndarray:
    starts = np.searchsorted(times, times - seconds * MICROSECONDS, side="left")
    prefix = np.concatenate(([0.0], np.cumsum(values.astype(float))))
    return prefix[np.arange(1, len(times) + 1)] - prefix[starts]


def _rolling_extreme(
    times: np.ndarray,
    values: np.ndarray,
    seconds: int,
    *,
    maximum: bool,
    exclude_current: bool = False,
) -> np.ndarray:
    output = np.full(len(values), np.nan)
    candidates: deque[int] = deque()
    for index, value in enumerate(values):
        cutoff = times[index] - seconds * MICROSECONDS
        while candidates and times[candidates[0]] < cutoff:
            candidates.popleft()
        if exclude_current:
            if candidates:
                output[index] = values[candidates[0]]
        while candidates and (
            values[candidates[-1]] <= value if maximum else values[candidates[-1]] >= value
        ):
            candidates.pop()
        candidates.append(index)
        if not exclude_current:
            output[index] = values[candidates[0]]
    return output


def _past_value(times: np.ndarray, values: np.ndarray, seconds: int) -> np.ndarray:
    indices = np.searchsorted(times, times - seconds * MICROSECONDS, side="right") - 1
    output = np.full(len(values), np.nan)
    valid = indices >= 0
    output[valid] = values[indices[valid]]
    return output


def _first_indices(keys: np.ndarray) -> np.ndarray:
    if not len(keys):
        return np.array([], dtype=int)
    return np.concatenate((np.array([0]), np.flatnonzero(np.diff(keys)) + 1)).astype(int)


def _causal_minute_stat(
    times: np.ndarray,
    minute_positions: np.ndarray,
    values: np.ndarray,
    statistic: Literal["median", "p20"],
) -> np.ndarray:
    result = np.full(len(times), np.nan)
    for number, position in enumerate(minute_positions):
        left_time = times[position] - 14400 * MICROSECONDS
        left = int(np.searchsorted(times[minute_positions], left_time, side="left"))
        reference_positions = minute_positions[left:number]
        reference = values[reference_positions]
        reference = reference[np.isfinite(reference)]
        value = np.nan
        if len(reference) >= 20:
            value = float(
                np.median(reference) if statistic == "median" else np.quantile(reference, 0.2)
            )
        stop = (
            int(minute_positions[number + 1]) if number + 1 < len(minute_positions) else len(times)
        )
        result[int(position) : stop] = value
    return result


def _causal_percentile(current: float, reference: np.ndarray) -> float | None:
    finite = reference[np.isfinite(reference)]
    if len(finite) < 20 or not math.isfinite(current):
        return None
    return float((np.sum(finite < current) + 0.5 * np.sum(finite == current)) / len(finite))


def _apply_cooldown(
    positions: np.ndarray, mask: np.ndarray, times: np.ndarray, cooldown_seconds: int
) -> np.ndarray:
    selected = []
    last = -(10**30)
    for position in positions[mask]:
        timestamp = int(times[position])
        if timestamp >= last + cooldown_seconds * MICROSECONDS:
            selected.append(int(position))
            last = timestamp
    return np.asarray(selected, dtype=int)


def _session(instant: datetime) -> tuple[str, float]:
    minute = instant.hour * 60 + instant.minute + instant.second / 60
    if minute < 7 * 60:
        return "ASIA", minute
    if minute < 12 * 60:
        return "LONDON", minute - 7 * 60
    if minute < 16 * 60:
        return "LONDON_NEW_YORK", minute - 12 * 60
    if minute < 21 * 60:
        return "NEW_YORK", minute - 16 * 60
    return "OTHER", minute - 21 * 60


def _run_lengths(signs: np.ndarray) -> np.ndarray:
    directional = signs[signs != 0]
    if not len(directional):
        return np.array([], dtype=float)
    changes = np.concatenate((np.array([True]), directional[1:] != directional[:-1]))
    starts = np.flatnonzero(changes)
    return np.diff(np.concatenate((starts, np.array([len(directional)])))).astype(float)


def _micro_features(
    times: np.ndarray,
    mids: np.ndarray,
    spreads: np.ndarray,
    current_index: int,
) -> dict[str, float]:
    current_time = int(times[current_index])
    result: dict[str, float] = {}
    for seconds in (1, 5, 15, 30, 60):
        left = int(np.searchsorted(times, current_time - seconds * MICROSECONDS, side="left"))
        changes = np.diff(mids[left : current_index + 1])
        up, down = int(np.sum(changes > 0)), int(np.sum(changes < 0))
        result[f"signed_tick_imbalance_{seconds}s"] = (up - down) / max(up + down, 1)
        result[f"tick_rate_{seconds}s"] = float(current_index - left + 1)
    left_60 = int(np.searchsorted(times, current_time - 60 * MICROSECONDS, side="left"))
    recent_mid = mids[left_60 : current_index + 1]
    recent_times = times[left_60 : current_index + 1]
    recent_spreads = spreads[left_60 : current_index + 1]
    signs = np.sign(np.diff(recent_mid))
    lengths = _run_lengths(signs)
    directional = signs[signs != 0]
    current_run = 0.0
    if len(directional):
        current_sign = directional[-1]
        current_run = float(current_sign)
        for value in directional[-2::-1]:
            if value != current_sign:
                break
            current_run += float(current_sign)
    transitions = list(zip(directional[:-1], directional[1:], strict=True))
    up_origins = sum(left == 1 for left, _ in transitions)
    down_origins = sum(left == -1 for left, _ in transitions)
    arrivals = np.diff(recent_times).astype(float) / MICROSECONDS
    left_5 = int(np.searchsorted(times, current_time - 5 * MICROSECONDS, side="left"))
    arrivals_5 = np.diff(times[left_5 : current_index + 1]).astype(float) / MICROSECONDS
    changes_5 = np.diff(mids[left_5 : current_index + 1])
    spread_changes = np.diff(recent_spreads)
    high_index, low_index = int(np.argmax(recent_mid)), int(np.argmin(recent_mid))
    high, low, current = (
        float(np.max(recent_mid)),
        float(np.min(recent_mid)),
        float(mids[current_index]),
    )
    result.update(
        {
            "imbalance_acceleration_5s_vs_30s": result["signed_tick_imbalance_5s"]
            - result["signed_tick_imbalance_30s"],
            "run_length_mean_60s": float(np.mean(lengths)) if len(lengths) else 0.0,
            "run_length_max_60s": float(np.max(lengths)) if len(lengths) else 0.0,
            "current_signed_run_length": current_run,
            "transition_probability_up_up_60s": (
                sum(left == 1 and right == 1 for left, right in transitions) / max(up_origins, 1)
            ),
            "transition_probability_down_down_60s": (
                sum(left == -1 and right == -1 for left, right in transitions)
                / max(down_origins, 1)
            ),
            "interarrival_mean_5s": float(np.mean(arrivals_5)) if len(arrivals_5) else 5.0,
            "interarrival_mean_60s": float(np.mean(arrivals)) if len(arrivals) else 60.0,
            "interarrival_cv_60s": (
                float(np.std(arrivals) / np.mean(arrivals))
                if len(arrivals) and np.mean(arrivals) > 0
                else 0.0
            ),
            "quote_change_rate_5s": float(np.mean(changes_5 != 0)) if len(changes_5) else 0.0,
            "quote_change_rate_60s": float(np.mean(np.diff(recent_mid) != 0))
            if len(recent_mid) > 1
            else 0.0,
            "spread_change_rate_60s": float(np.mean(spread_changes != 0))
            if len(spread_changes)
            else 0.0,
            "spread_delta_vs_median_60s": float(spreads[current_index] - np.median(recent_spreads)),
            "micro_pullback_amplitude_60s_points": max(high - current, current - low) / POINT,
            "time_since_local_high_60s": (current_time - int(recent_times[high_index]))
            / MICROSECONDS,
            "time_since_local_low_60s": (current_time - int(recent_times[low_index]))
            / MICROSECONDS,
        }
    )
    result["interarrival_acceleration_5s_vs_60s"] = (
        result["interarrival_mean_60s"] - result["interarrival_mean_5s"]
    )
    return result


def _failed_breakouts_and_retest(
    times: np.ndarray, mids: np.ndarray, current: int
) -> tuple[float, float]:
    current_time = int(times[current])
    left = int(np.searchsorted(times, current_time - 60 * MICROSECONDS, side="left"))
    window_times = times[left : current + 1]
    window_mid = mids[left : current + 1]
    failures = 0
    latest_breakout: int | None = None
    for index in range(1, len(window_mid)):
        prior_left = int(
            np.searchsorted(window_times, window_times[index] - 15 * MICROSECONDS, side="left")
        )
        prior = window_mid[prior_left:index]
        if not len(prior):
            continue
        high, low = float(np.max(prior)), float(np.min(prior))
        value = float(window_mid[index])
        if value > high:
            latest_breakout = index
            if float(window_mid[-1]) <= high:
                failures += 1
        elif value < low:
            latest_breakout = index
            if float(window_mid[-1]) >= low:
                failures += 1
    latency = (
        (current_time - int(window_times[latest_breakout])) / MICROSECONDS
        if latest_breakout is not None
        else 60.0
    )
    return float(failures), float(latency)


def _second_context(raw: pl.DataFrame) -> dict[str, Any]:
    times = raw["timestamp_utc"].cast(pl.Int64).to_numpy().astype(np.int64)
    bids = raw["bid"].to_numpy().astype(float)
    asks = raw["ask"].to_numpy().astype(float)
    mids = (bids + asks) / 2
    spreads = raw["spread_points"].to_numpy().astype(float)
    first = _first_indices(times // MICROSECONDS)
    second_times = times[first]
    second_mids = mids[first]
    second_spreads = spreads[first]
    counts = np.diff(np.concatenate((first, np.array([len(times)])))).astype(float)
    delta = np.zeros(len(first), dtype=float)
    if len(first) > 1:
        delta[1:] = np.diff(second_mids) / POINT
    squared = delta**2
    absolute = np.abs(delta)
    up = (delta > 0).astype(float)
    down = (delta < 0).astype(float)
    context: dict[str, Any] = {
        "raw_times": times,
        "raw_bids": bids,
        "raw_asks": asks,
        "raw_mids": mids,
        "raw_spreads": spreads,
        "first": first,
        "times": second_times,
        "mids": second_mids,
        "spreads": second_spreads,
        "counts": counts,
        "delta": delta,
    }
    for window in WINDOWS:
        rv = np.sqrt(_rolling_sum(second_times, squared, window))
        high = _rolling_extreme(second_times, second_mids, window, maximum=True)
        low = _rolling_extreme(second_times, second_mids, window, maximum=False)
        path = _rolling_sum(second_times, absolute, window)
        past = _past_value(second_times, second_mids, window)
        displacement = np.abs(second_mids - past) / POINT
        directional_up = _rolling_sum(second_times, up, window)
        directional_down = _rolling_sum(second_times, down, window)
        context[f"rv_{window}"] = rv
        context[f"range_{window}"] = (high - low) / POINT
        context[f"efficiency_{window}"] = np.divide(
            displacement,
            path,
            out=np.zeros_like(displacement),
            where=np.isfinite(displacement) & (path > 0),
        )
        context[f"persistence_{window}"] = np.divide(
            np.abs(directional_up - directional_down),
            directional_up + directional_down,
            out=np.zeros_like(directional_up),
            where=(directional_up + directional_down) > 0,
        )
    context["rv_60"] = np.sqrt(_rolling_sum(second_times, squared, 60))
    high60 = _rolling_extreme(second_times, second_mids, 60, maximum=True)
    low60 = _rolling_extreme(second_times, second_mids, 60, maximum=False)
    context["range_60"] = (high60 - low60) / POINT
    for seconds in (1, 5, 15):
        rate = _rolling_sum(second_times, counts, seconds) - counts + 1
        context[f"rate_{seconds}"] = rate
    for seconds in (5, 15, 30, 60):
        context[f"return_{seconds}"] = (
            second_mids - _past_value(second_times, second_mids, seconds)
        ) / POINT
    minute_positions = _first_indices(second_times // (60 * MICROSECONDS))
    context["minute_positions"] = minute_positions
    context["range_median_4h"] = _causal_minute_stat(
        second_times, minute_positions, cast(np.ndarray, context["range_300"]), "median"
    )
    context["range_p20_4h"] = _causal_minute_stat(
        second_times, minute_positions, cast(np.ndarray, context["range_300"]), "p20"
    )
    context["rate5_median_4h"] = _causal_minute_stat(
        second_times, minute_positions, cast(np.ndarray, context["rate_5"]), "median"
    )
    return context


def _event_positions(
    context: dict[str, Any], day_start_us: int, day_end_us: int
) -> tuple[dict[int, tuple[str, ...]], dict[str, int], int]:
    times = cast(np.ndarray, context["times"])
    mids = cast(np.ndarray, context["mids"])
    current = np.flatnonzero((times >= day_start_us) & (times < day_end_us))
    baseline_local = _first_indices(times[current] // (60 * MICROSECONDS))
    selected: dict[str, np.ndarray] = {"MINUTE_BASELINE": current[baseline_local]}
    rv300 = cast(np.ndarray, context["rv_300"])
    rv900 = cast(np.ndarray, context["rv_900"])
    return60 = cast(np.ndarray, context["return_60"])
    return5 = cast(np.ndarray, context["return_5"])
    history900 = _past_value(times, mids, 900)
    history1800 = _past_value(times, mids, 1800)
    history14400 = _past_value(times, mids, 14400)
    volatility_mask = (
        np.isfinite(history900[current])
        & np.isfinite(return60[current])
        & (np.abs(return60[current]) >= 1.5 * rv300[current])
    )
    selected["VOLATILITY_EVENT"] = _apply_cooldown(
        current, volatility_mask, times, EVENT_COOLDOWNS["VOLATILITY_EVENT"]
    )
    prior_high = _rolling_extreme(times, mids, 900, maximum=True, exclude_current=True)
    prior_low = _rolling_extreme(times, mids, 900, maximum=False, exclude_current=True)
    breakout_mask = np.isfinite(history900[current]) & (
        (mids[current] > prior_high[current] + 0.25 * rv900[current] * POINT)
        | (mids[current] < prior_low[current] - 0.25 * rv900[current] * POINT)
    )
    selected["BREAKOUT_EVENT"] = _apply_cooldown(
        current, breakout_mask, times, EVENT_COOLDOWNS["BREAKOUT_EVENT"]
    )
    rate5 = cast(np.ndarray, context["rate_5"])
    rate5_median = cast(np.ndarray, context["rate5_median_4h"])
    acceleration_mask = (
        np.isfinite(history1800[current])
        & np.isfinite(rate5_median[current])
        & (rate5[current] >= 2 * rate5_median[current])
        & (np.abs(return5[current]) >= 0.5 * cast(np.ndarray, context["rv_60"])[current])
    )
    selected["ACCELERATION_EVENT"] = _apply_cooldown(
        current, acceleration_mask, times, EVENT_COOLDOWNS["ACCELERATION_EVENT"]
    )
    previous_range = _past_value(times, cast(np.ndarray, context["range_300"]), 60)
    compression_limit = cast(np.ndarray, context["range_p20_4h"])
    release_mask = (
        np.isfinite(history14400[current])
        & np.isfinite(previous_range[current])
        & np.isfinite(compression_limit[current])
        & (previous_range[current] <= compression_limit[current])
        & (cast(np.ndarray, context["range_60"])[current] >= 1.5 * previous_range[current])
    )
    selected["COMPRESSION_RELEASE_EVENT"] = _apply_cooldown(
        current, release_mask, times, EVENT_COOLDOWNS["COMPRESSION_RELEASE_EVENT"]
    )
    merged: dict[int, list[str]] = {}
    counts = {}
    for name, positions in selected.items():
        counts[name] = len(positions)
        for position in positions:
            merged.setdefault(int(position), []).append(name)
    return (
        {position: tuple(sorted(names)) for position, names in merged.items()},
        counts,
        len(current),
    )


def _event_labels(
    times: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    timestamp_us: int,
    broker: BrokerSimulationConfig,
) -> dict[str, Any] | None:
    mids = (bids + asks) / 2
    current = int(np.searchsorted(times, timestamp_us, side="left"))
    if current >= len(times) or int(times[current]) != timestamp_us:
        return None
    entry_bid, entry_ask = float(bids[current]), float(asks[current])
    entry_mid = (entry_bid + entry_ask) / 2
    base_cost = (
        (entry_ask - entry_bid) / POINT
        + 2 * BASE_SCENARIO.slippage_points_per_side
        + _commission_points(entry_mid, broker, BASE_SCENARIO)
    )
    result: dict[str, Any] = {"base_cost_points": base_cost}
    for horizon in HORIZONS_D12:
        target_us = timestamp_us + horizon * MICROSECONDS
        target = int(np.searchsorted(times, target_us, side="left"))
        if (
            target >= len(times)
            or (int(times[target]) - target_us) / MICROSECONDS > 2
            or target <= current
        ):
            return None
        future_mid = mids[current + 1 : target + 1]
        gross_long = max(0.0, float(np.max(future_mid) - entry_mid) / POINT)
        gross_short = max(0.0, float(entry_mid - np.min(future_mid)) / POINT)
        long_mfe = max(0.0, float(np.max(bids[current + 1 : target + 1]) - entry_ask) / POINT)
        long_mae = max(0.0, float(entry_ask - np.min(bids[current + 1 : target + 1])) / POINT)
        short_mfe = max(0.0, float(entry_bid - np.min(asks[current + 1 : target + 1])) / POINT)
        short_mae = max(0.0, float(np.max(asks[current + 1 : target + 1]) - entry_bid) / POINT)
        long_return = float(bids[target] - entry_ask) / POINT
        short_return = float(entry_bid - asks[target]) / POINT
        favorable = max(gross_long, gross_short)
        adverse = gross_short if gross_long >= gross_short else gross_long
        prefix = f"h{horizon}"
        result.update(
            {
                f"{prefix}_long_executable_return_points": long_return,
                f"{prefix}_short_executable_return_points": short_return,
                f"{prefix}_signed_executable_return_points": (long_return - short_return) / 2,
                f"{prefix}_mfe_long_points": long_mfe,
                f"{prefix}_mae_long_points": long_mae,
                f"{prefix}_mfe_short_points": short_mfe,
                f"{prefix}_mae_short_points": short_mae,
                f"{prefix}_gross_mfe_long_points": gross_long,
                f"{prefix}_gross_mfe_short_points": gross_short,
                f"{prefix}_future_tradable_movement_ratio": favorable / max(base_cost, 1e-9),
                f"{prefix}_path_quality_ratio": favorable / max(adverse, 1.0),
            }
        )
    return result


def _day_context(context: dict[str, Any], day_start_us: int, day_end_us: int) -> dict[str, Any]:
    times = cast(np.ndarray, context["times"])
    mids = cast(np.ndarray, context["mids"])
    left = int(np.searchsorted(times, day_start_us, side="left"))
    right = int(np.searchsorted(times, day_end_us, side="left"))
    values = mids[left:right]
    cumulative_sum = np.cumsum(values)
    result: dict[str, Any] = {
        "left": left,
        "right": right,
        "high": np.maximum.accumulate(values),
        "low": np.minimum.accumulate(values),
        "twap": cumulative_sum / np.arange(1, len(values) + 1),
    }
    london_open = day_start_us + 7 * 3600 * MICROSECONDS
    london_30 = london_open + 1800 * MICROSECONDS
    london_60 = london_open + 3600 * MICROSECONDS
    asia_stop = int(np.searchsorted(times, london_open, side="left"))
    open_index = int(np.searchsorted(times, london_open, side="left"))
    stop_30 = int(np.searchsorted(times, london_30, side="left"))
    stop_60 = int(np.searchsorted(times, london_60, side="left"))
    asian = mids[left : min(asia_stop, right)]
    first_30_delta = cast(np.ndarray, context["delta"])[open_index : min(stop_30, right)]
    first_60 = mids[open_index : min(stop_60, right)]
    result.update(
        {
            "london_open_us": london_open,
            "london_30_us": london_30,
            "london_60_us": london_60,
            "london_open_mid": float(mids[open_index]) if open_index < right else None,
            "asian_range": (float(np.max(asian) - np.min(asian)) / POINT) if len(asian) else None,
            "london_30_rv": float(np.sqrt(np.sum(first_30_delta**2)))
            if len(first_30_delta)
            else None,
            "london_60_range": (float(np.max(first_60) - np.min(first_60)) / POINT)
            if len(first_60)
            else None,
        }
    )
    return result


def _event_feature_row(
    context: dict[str, Any],
    day: dict[str, Any],
    position: int,
    event_types: tuple[str, ...],
) -> dict[str, Any]:
    times = cast(np.ndarray, context["times"])
    mids = cast(np.ndarray, context["mids"])
    spreads = cast(np.ndarray, context["spreads"])
    current_time = int(times[position])
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    instant = epoch + timedelta(microseconds=current_time)
    session, session_age = _session(instant)
    row: dict[str, Any] = {
        "timestamp_utc": instant,
        "day": instant.strftime("%Y-%m-%d"),
        "week": f"{instant.isocalendar().year}-W{instant.isocalendar().week:02d}",
        "session": session,
        "session_age_minutes": session_age,
        "event_types": "|".join(event_types),
    }
    for name in EVENT_COOLDOWNS:
        row[f"event_{name.lower()}"] = float(name in event_types)
    for window in WINDOWS:
        row[f"realized_volatility_{window}s_points"] = float(
            cast(np.ndarray, context[f"rv_{window}"])[position]
        )
        row[f"range_{window}s_points"] = float(
            cast(np.ndarray, context[f"range_{window}"])[position]
        )
        row[f"trend_efficiency_{window}s"] = float(
            cast(np.ndarray, context[f"efficiency_{window}"])[position]
        )
        row[f"absolute_directional_persistence_{window}s"] = float(
            cast(np.ndarray, context[f"persistence_{window}"])[position]
        )
    minute_positions = cast(np.ndarray, context["minute_positions"])
    minute_times = times[minute_positions]
    minute_right = int(np.searchsorted(minute_times, current_time, side="left"))
    minute_left = int(
        np.searchsorted(minute_times, current_time - 14400 * MICROSECONDS, side="left")
    )
    reference_positions = minute_positions[minute_left:minute_right]
    range_reference = cast(np.ndarray, context["range_300"])[reference_positions]
    rv_reference = cast(np.ndarray, context["rv_300"])[reference_positions]
    rate_reference = cast(np.ndarray, context["rate_15"])[reference_positions]
    spread_reference = spreads[reference_positions]
    range_current = row["range_300s_points"]
    rv_current = row["realized_volatility_300s_points"]
    range_median = float(np.median(range_reference)) if len(range_reference) >= 20 else math.nan
    row.update(
        {
            "atr_normalized_300s": range_current / max(range_median, 1e-9),
            "volatility_percentile_300s": _causal_percentile(rv_current, rv_reference),
            "range_expansion_ratio_300s": range_current / max(range_median, 1e-9),
            "range_compression_ratio_300s": range_median / max(range_current, 1e-9),
            "tick_activity_percentile_15s": _causal_percentile(
                float(cast(np.ndarray, context["rate_15"])[position]), rate_reference
            ),
            "spread_percentile_4h": _causal_percentile(float(spreads[position]), spread_reference),
            "spread_to_realized_volatility_300s": float(spreads[position]) / max(rv_current, 1e-9),
            "spread_points": float(spreads[position]),
        }
    )
    raw_index = int(cast(np.ndarray, context["first"])[position])
    micro = _micro_features(
        cast(np.ndarray, context["raw_times"]),
        cast(np.ndarray, context["raw_mids"]),
        cast(np.ndarray, context["raw_spreads"]),
        raw_index,
    )
    row.update(micro)
    row["tick_rate_acceleration"] = row["tick_rate_1s"] - row["tick_rate_5s"] / 5
    row["tick_uptick_ratio_60s"] = (row["signed_tick_imbalance_60s"] + 1) / 2
    row["spread_vs_median_60s"] = float(spreads[position]) / max(
        float(spreads[position] - row["spread_delta_vs_median_60s"]), 1e-9
    )
    for seconds in (5, 15, 30, 60):
        row[f"return_{seconds}s_points"] = float(
            cast(np.ndarray, context[f"return_{seconds}"])[position]
        )
    raw_times = cast(np.ndarray, context["raw_times"])
    raw_left_15 = int(np.searchsorted(raw_times, current_time - 15 * MICROSECONDS, side="left"))
    raw_count_15 = max(raw_index - raw_left_15 + 1, 1)
    row["displacement_per_tick_15s_points"] = row["return_15s_points"] / raw_count_15
    failed, latency = _failed_breakouts_and_retest(times, mids, position)
    row["failed_breakout_count_60s"] = failed
    row["retest_latency_seconds"] = latency
    row["breakout_velocity_15s_points_per_second"] = row["return_15s_points"] / 15
    slice_900_left = int(np.searchsorted(times, current_time - 900 * MICROSECONDS, side="left"))
    returns = cast(np.ndarray, context["delta"])[slice_900_left : position + 1]
    if len(returns) >= 10 and np.std(returns[:-1]) > 0 and np.std(returns[1:]) > 0:
        row["return_autocorrelation_lag1_900s"] = float(
            np.corrcoef(returns[:-1], returns[1:])[0, 1]
        )
    else:
        row["return_autocorrelation_lag1_900s"] = 0.0
    five_second = (
        np.add.reduceat(returns, np.arange(0, len(returns), 5)) if len(returns) else np.array([])
    )
    row["variance_ratio_5_900s"] = (
        float(np.var(five_second) / (5 * np.var(returns)))
        if len(five_second) >= 3 and np.var(returns) > 0
        else 0.0
    )
    local = position - cast(int, day["left"])
    daily_high = float(cast(np.ndarray, day["high"])[local])
    daily_low = float(cast(np.ndarray, day["low"])[local])
    current_mid = float(mids[position])
    daily_span = daily_high - daily_low
    twap = float(cast(np.ndarray, day["twap"])[local])
    prior_twap_index = (
        int(np.searchsorted(times, current_time - 900 * MICROSECONDS, side="right")) - 1
    )
    prior_local = max(0, prior_twap_index - cast(int, day["left"]))
    prior_twap = float(cast(np.ndarray, day["twap"])[min(prior_local, local)])
    row.update(
        {
            "distance_to_daily_high_points": (current_mid - daily_high) / POINT,
            "distance_to_daily_low_points": (current_mid - daily_low) / POINT,
            "position_in_daily_range": (current_mid - daily_low) / daily_span
            if daily_span > 0
            else 0.5,
            "distance_to_intraday_twap_points": (current_mid - twap) / POINT,
            "intraday_twap_slope_900s_points": (twap - prior_twap) / POINT,
            "minutes_since_london_open": (instant.hour * 60 + instant.minute - 7 * 60),
            "minutes_since_new_york_open": (instant.hour * 60 + instant.minute - 12 * 60),
            "london_new_york_overlap": float(12 <= instant.hour < 16),
            "asian_range_points": day["asian_range"]
            if current_time >= day["london_open_us"]
            else None,
            "london_first_30m_realized_volatility": day["london_30_rv"]
            if current_time >= day["london_30_us"]
            else None,
            "london_first_60m_range": day["london_60_range"]
            if current_time >= day["london_60_us"]
            else None,
            "london_open_displacement_points": (
                (current_mid - cast(float, day["london_open_mid"])) / POINT
                if day["london_open_mid"] is not None and current_time >= day["london_open_us"]
                else None
            ),
        }
    )
    row["realized_move_to_base_cost_300s"] = row["range_300s_points"] / max(
        float(spreads[position]) + 2.0, 1e-9
    )
    volatility_percentile = row["volatility_percentile_300s"]
    row["volatility_regime"] = (
        "UNKNOWN"
        if volatility_percentile is None
        else "LOW"
        if volatility_percentile < 1 / 3
        else "MEDIUM"
        if volatility_percentile < 2 / 3
        else "HIGH"
    )
    row["trend_regime"] = "TREND" if row["trend_efficiency_900s"] >= 0.30 else "RANGE"
    signed_persistence = math.copysign(
        row["absolute_directional_persistence_900s"], row["return_60s_points"]
    )
    row.update(
        {
            "return_60s_x_high_volatility": row["return_60s_points"]
            * float(volatility_percentile is not None and volatility_percentile >= 2 / 3),
            "breakout_velocity_x_low_spread_vol": row["breakout_velocity_15s_points_per_second"]
            * float(row["spread_to_realized_volatility_300s"] <= 0.5),
            "micro_pullback_x_trend_efficiency_900s": row["micro_pullback_amplitude_60s_points"]
            * row["trend_efficiency_900s"],
            "imbalance_15s_x_compression_release": row["signed_tick_imbalance_15s"]
            * float("COMPRESSION_RELEASE_EVENT" in event_types),
            "imbalance_acceleration_x_range_expansion": row["imbalance_acceleration_5s_vs_30s"]
            * row["range_expansion_ratio_300s"],
            "signed_persistence_900s_x_volatility_percentile": signed_persistence
            * (volatility_percentile if volatility_percentile is not None else 0.0),
            "london_displacement_x_overlap": (row["london_open_displacement_points"] or 0.0)
            * row["london_new_york_overlap"],
            "distance_twap_x_trend_efficiency_3600s": row["distance_to_intraday_twap_points"]
            * row["trend_efficiency_3600s"],
        }
    )
    return row


def build_d12_event_dataset(
    *, data_root: Path, start: datetime, end: datetime, output_dir: Path
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Stream RESEARCH by UTC day and materialize only causally sampled events."""
    broker = BrokerSimulationConfig()
    history: pl.DataFrame | None = None
    part_paths: list[Path] = []
    ticks_analyzed = 0
    detector_seconds = 0
    triggered_events = 0
    complete_events = 0
    peak_frame_mb = 0.0
    triggered_by_scheme = {name: 0 for name in EVENT_COOLDOWNS}
    complete_by_scheme = {name: 0 for name in EVENT_COOLDOWNS}
    parts_dir = output_dir / "event-parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for current_day in _days(start, end):
        day_start = max(start, datetime.combine(current_day, time.min, UTC))
        day_end = min(
            end, day_start.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        )
        current = _read_ticks(_tick_path(data_root, current_day), day_start, day_end)
        if current.is_empty():
            if history is not None:
                history = history.filter(
                    pl.col("timestamp_utc") >= day_end - timedelta(seconds=14400)
                )
            continue
        ticks_analyzed += current.height
        following_day = current_day + timedelta(days=1)
        following = _read_ticks(
            _tick_path(data_root, following_day),
            day_end,
            min(end, day_end + timedelta(seconds=900)),
        )
        past = (
            current
            if history is None or history.is_empty()
            else pl.concat([history, current], how="vertical").sort("timestamp_utc")
        )
        future = (
            current
            if following.is_empty()
            else pl.concat([current, following], how="vertical").sort("timestamp_utc")
        )
        peak_frame_mb = max(peak_frame_mb, past.estimated_size("mb") + future.estimated_size("mb"))
        context = _second_context(past)
        day_start_us = int(day_start.timestamp() * MICROSECONDS)
        day_end_us = int(day_end.timestamp() * MICROSECONDS)
        event_positions, scheme_counts, day_detector_seconds = _event_positions(
            context, day_start_us, day_end_us
        )
        detector_seconds += day_detector_seconds
        triggered_events += len(event_positions)
        for name, count in scheme_counts.items():
            triggered_by_scheme[name] += count
        day_features = _day_context(context, day_start_us, day_end_us)
        future_times = future["timestamp_utc"].cast(pl.Int64).to_numpy().astype(np.int64)
        future_bids = future["bid"].to_numpy().astype(float)
        future_asks = future["ask"].to_numpy().astype(float)
        rows = []
        second_times = cast(np.ndarray, context["times"])
        for position, event_types in sorted(event_positions.items()):
            labels = _event_labels(
                future_times, future_bids, future_asks, int(second_times[position]), broker
            )
            if labels is None:
                continue
            row = _event_feature_row(context, day_features, position, event_types)
            row.update(labels)
            row["realized_move_to_base_cost_300s"] = row["range_300s_points"] / max(
                row["base_cost_points"], 1e-9
            )
            rows.append(row)
            complete_events += 1
            for name in event_types:
                complete_by_scheme[name] += 1
        if rows:
            part = parts_dir / f"date={current_day.isoformat()}.parquet"
            pl.DataFrame(rows, infer_schema_length=None).write_parquet(
                part, compression="zstd", statistics=True
            )
            part_paths.append(part)
        history = past.filter(pl.col("timestamp_utc") >= day_end - timedelta(seconds=14400))
    if not part_paths:
        raise RuntimeError("D.12 sampling produced no complete RESEARCH events")
    frame = pl.concat([pl.read_parquet(path) for path in part_paths], how="diagonal_relaxed").sort(
        "timestamp_utc"
    )
    if ticks_analyzed != EXPECTED_RESEARCH_TICKS:
        raise RuntimeError(
            f"D.12 expected {EXPECTED_RESEARCH_TICKS} RESEARCH ticks, got {ticks_analyzed}"
        )
    missing = sorted(set(FEATURE_NAMES) - set(frame.columns))
    if missing:
        raise RuntimeError(f"D.12 feature materialization incomplete: {missing}")
    return frame, {
        "ticks_analyzed": ticks_analyzed,
        "detector_seconds": detector_seconds,
        "triggered_events": triggered_events,
        "complete_events": complete_events,
        "triggered_by_scheme": triggered_by_scheme,
        "complete_by_scheme": complete_by_scheme,
        "peak_loaded_tick_frame_mb": peak_frame_mb,
    }


def _fold_boundaries(start: datetime, end: datetime) -> tuple[tuple[datetime, datetime], ...]:
    duration = end - start
    points = [start + duration * fraction for fraction in (0.3, 0.4, 0.5, 0.6, 1.0)]
    return tuple((points[index], points[index + 1]) for index in range(4))


def _with_folds(frame: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
    expression = pl.lit(0)
    for fold, (fold_start, fold_end) in enumerate(_fold_boundaries(start, end), 1):
        expression = (
            pl.when((pl.col("timestamp_utc") >= fold_start) & (pl.col("timestamp_utc") < fold_end))
            .then(fold)
            .otherwise(expression)
        )
    return frame.with_columns(expression.alias("fold"))


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = spearmanr(x, y)
    effect = float(result.statistic)
    pvalue = float(result.pvalue)
    return (effect if np.isfinite(effect) else None, pvalue if np.isfinite(pvalue) else None)


def _rank_bins(values: np.ndarray, bins: int = 10) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    output = np.empty(len(values), dtype=int)
    output[order] = np.minimum(np.arange(len(values)) * bins // max(len(values), 1), bins - 1)
    return output


def _discrete_mutual_information(x: np.ndarray, y: np.ndarray) -> float | None:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 50 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(mutual_info_score(_rank_bins(x), _rank_bins(y)))


def _block_correlation_ci(
    x: np.ndarray,
    y: np.ndarray,
    blocks: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    values = []
    for block in sorted(set(map(str, blocks))):
        mask = blocks.astype(str) == block
        effect, _ = _safe_spearman(x[mask], y[mask])
        if effect is not None:
            values.append(effect)
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"estimate": None, "lower": None, "upper": None, "blocks": 0, "replications": 0}
    rng = np.random.default_rng(seed)
    bootstrap = np.array(
        [
            float(np.mean(array[rng.integers(0, len(array), len(array))]))
            for _ in range(BOOTSTRAP_REPLICATIONS_D12)
        ]
    )
    return {
        "estimand": "MEAN_WITHIN_BLOCK_SPEARMAN",
        "estimate": float(np.mean(array)),
        "lower": float(np.quantile(bootstrap, 0.025)),
        "upper": float(np.quantile(bootstrap, 0.975)),
        "blocks": len(array),
        "replications": BOOTSTRAP_REPLICATIONS_D12,
    }


def _monotonic_bins(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 10 or np.std(x) == 0:
        return {"spearman": None, "groups": []}
    order = np.argsort(x, kind="stable")
    groups = []
    means = []
    for number, indices in enumerate(np.array_split(order, 5), 1):
        mean = float(np.mean(y[indices]))
        means.append(mean)
        groups.append(
            {
                "group": number,
                "N": len(indices),
                "mean_feature": float(np.mean(x[indices])),
                "mean_target": mean,
            }
        )
    effect, _ = _safe_spearman(np.arange(1, 6, dtype=float), np.asarray(means))
    return {"spearman": effect, "groups": groups}


def _benjamini_hochberg(pvalues: list[float | None]) -> list[float | None]:
    valid = [(index, value) for index, value in enumerate(pvalues) if value is not None]
    output: list[float | None] = [None] * len(pvalues)
    if not valid:
        return output
    ordered = sorted(valid, key=lambda item: item[1])
    count = len(ordered)
    running = 1.0
    for rank in range(count, 0, -1):
        index, value = ordered[rank - 1]
        running = min(running, value * count / rank)
        output[index] = running
    return output


def _univariate_matrix(frame: pl.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scoped = frame.filter(pl.col("fold") > 0)
    days = scoped["day"].to_numpy().astype(str)
    weeks = scoped["week"].to_numpy().astype(str)
    sessions = scoped["session"].to_numpy().astype(str)
    volatility_regimes = scoped["volatility_regime"].to_numpy().astype(str)
    trend_regimes = scoped["trend_regime"].to_numpy().astype(str)
    rows: list[dict[str, Any]] = []
    row_number = 0
    for horizon in HORIZONS_D12:
        targets = {
            "WITHIN_REGIME_DIRECTION": scoped[f"h{horizon}_signed_executable_return_points"]
            .to_numpy()
            .astype(float),
            "REGIME_QUALITY": scoped[f"h{horizon}_future_tradable_movement_ratio"]
            .to_numpy()
            .astype(float),
        }
        for target_name, target in targets.items():
            for feature in FEATURE_NAMES:
                row_number += 1
                values = scoped[feature].to_numpy().astype(float)
                valid = np.isfinite(values) & np.isfinite(target)
                x, y = values[valid], target[valid]
                effect, pvalue = _safe_spearman(x, y)
                fold_effects: dict[str, float | None] = {}
                for fold in range(1, 5):
                    fold_mask = valid & (scoped["fold"].to_numpy() == fold)
                    fold_effects[f"fold_{fold}"], _ = _safe_spearman(
                        values[fold_mask], target[fold_mask]
                    )
                monotonic = _monotonic_bins(x, y)
                if len(x):
                    lower, upper = np.quantile(x, (0.10, 0.90))
                    extreme = x >= upper if (effect or 0) >= 0 else x <= lower
                    extreme_sessions = sessions[valid][extreme]
                    largest_session_share = (
                        max(np.sum(extreme_sessions == value) for value in set(extreme_sessions))
                        / len(extreme_sessions)
                        if len(extreme_sessions)
                        else 1.0
                    )
                    clipped_x = np.clip(x, *np.quantile(x, (0.01, 0.99)))
                    clipped_y = np.clip(y, *np.quantile(y, (0.01, 0.99)))
                    winsorized, _ = _safe_spearman(clipped_x, clipped_y)
                else:
                    largest_session_share = 1.0
                    winsorized = None
                regime_effects = {}
                for regime in ("LOW", "MEDIUM", "HIGH"):
                    mask = valid & (volatility_regimes == regime)
                    regime_effects[f"volatility_{regime}"] = {
                        "N": int(np.sum(mask)),
                        "effect": _safe_spearman(values[mask], target[mask])[0],
                    }
                for regime in ("RANGE", "TREND"):
                    mask = valid & (trend_regimes == regime)
                    regime_effects[f"trend_{regime}"] = {
                        "N": int(np.sum(mask)),
                        "effect": _safe_spearman(values[mask], target[mask])[0],
                    }
                rows.append(
                    {
                        "feature": feature,
                        "family": FEATURE_GROUPS[feature],
                        "horizon_seconds": horizon,
                        "target": target_name,
                        "N": len(x),
                        "effect_spearman": effect,
                        "p_value": pvalue,
                        "mutual_information_10x10": _discrete_mutual_information(x, y),
                        "monotonic_bin_spearman": monotonic["spearman"],
                        "monotonic_bins": monotonic["groups"],
                        "fold_effects": fold_effects,
                        "regime_effects": regime_effects,
                        "day_block_ci95": _block_correlation_ci(
                            values[valid],
                            target[valid],
                            days[valid],
                            seed=RANDOM_SEED_D12 + row_number,
                        ),
                        "week_block_ci95": _block_correlation_ci(
                            values[valid],
                            target[valid],
                            weeks[valid],
                            seed=RANDOM_SEED_D12 + 10000 + row_number,
                        ),
                        "largest_session_share_extreme_decile": largest_session_share,
                        "winsorized_effect_spearman": winsorized,
                    }
                )
    adjusted = _benjamini_hochberg([cast(float | None, row["p_value"]) for row in rows])
    status_counts = {
        "FEATURE_NO_SIGNAL": 0,
        "FEATURE_UNSTABLE": 0,
        "FEATURE_RESEARCH_CANDIDATE": 0,
    }
    for row, qvalue in zip(rows, adjusted, strict=True):
        row["fdr_q_value"] = qvalue
        effect = cast(float | None, row["effect_spearman"])
        fold_values = [value for value in row["fold_effects"].values() if value is not None]
        sign = np.sign(effect) if effect is not None else 0
        same_sign = sum(abs(value) >= 0.005 and np.sign(value) == sign for value in fold_values)
        freeze = row["fold_effects"]["fold_4"]
        freeze_not_opposite = freeze is None or abs(freeze) < 0.005 or np.sign(freeze) == sign
        monotonic_effect = cast(float | None, row["monotonic_bin_spearman"])
        winsorized = cast(float | None, row["winsorized_effect_spearman"])
        day_ci = row["day_block_ci95"]
        day_excludes_zero = (
            day_ci["lower"] is not None
            and day_ci["upper"] is not None
            and (day_ci["lower"] > 0 or day_ci["upper"] < 0)
            and np.sign(day_ci["estimate"]) == sign
        )
        candidate_checks = {
            "minimum_N": row["N"] >= 5000,
            "non_negligible_effect": effect is not None and abs(effect) >= 0.03,
            "same_sign_in_at_least_3_folds": same_sign >= 3,
            "freeze_not_opposite": freeze_not_opposite,
            "monotonic_bins": monotonic_effect is not None
            and abs(monotonic_effect) >= 0.70
            and np.sign(monotonic_effect) == sign,
            "session_not_dominant": row["largest_session_share_extreme_decile"] <= 0.60,
            "outlier_robust": winsorized is not None
            and effect is not None
            and np.sign(winsorized) == sign
            and abs(winsorized) >= 0.5 * abs(effect),
            "day_block_ci_excludes_zero": day_excludes_zero,
            "fdr_q_lte_0_10": qvalue is not None and qvalue <= 0.10,
        }
        if all(candidate_checks.values()):
            status = "FEATURE_RESEARCH_CANDIDATE"
        elif (
            effect is not None
            and abs(effect) >= 0.02
            and (same_sign >= 3 or (qvalue is not None and qvalue <= 0.10) or day_excludes_zero)
        ):
            status = "FEATURE_UNSTABLE"
        else:
            status = "FEATURE_NO_SIGNAL"
        row["same_sign_folds"] = same_sign
        row["candidate_checks"] = candidate_checks
        row["status"] = status
        status_counts[status] += 1
    return rows, {
        "rows": len(rows),
        "features": len(FEATURE_NAMES),
        "horizons": len(HORIZONS_D12),
        "targets": 2,
        "interactions": 8,
        "statistical_tests": len(rows),
        "status_counts": status_counts,
    }


def _sampling_and_label_summary(
    frame: pl.DataFrame,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    validation = frame.filter(pl.col("fold") > 0)
    sampling = []
    for event in EVENT_COOLDOWNS:
        scoped = validation.filter(pl.col("event_types").str.contains(event, literal=True))
        sampling.append(
            {
                "scheme": event,
                "observations": scoped.height,
                "active_days": scoped["day"].n_unique(),
                "active_weeks": scoped["week"].n_unique(),
                "session_counts": {
                    str(value): scoped.filter(pl.col("session") == value).height
                    for value in sorted(scoped["session"].unique().to_list())
                },
                "horizons": [
                    {
                        "horizon_seconds": horizon,
                        "mean_future_tradable_movement_ratio": float(
                            str(scoped[f"h{horizon}_future_tradable_movement_ratio"].mean())
                        ),
                        "movement_probabilities": {
                            f"move_gt_{threshold}_cost": float(
                                (
                                    scoped[f"h{horizon}_future_tradable_movement_ratio"]
                                    .to_numpy()
                                    .astype(float)
                                    > threshold
                                ).mean()
                            )
                            for threshold in MOVE_THRESHOLDS_D12
                        },
                        "mean_signed_executable_return_points": float(
                            str(scoped[f"h{horizon}_signed_executable_return_points"].mean())
                        ),
                    }
                    for horizon in HORIZONS_D12
                ],
            }
        )
    labels = []
    for horizon in HORIZONS_D12:
        ratio = validation[f"h{horizon}_future_tradable_movement_ratio"].to_numpy().astype(float)
        labels.append(
            {
                "horizon_seconds": horizon,
                "N": len(ratio),
                "mean_future_tradable_movement_ratio": float(np.mean(ratio)),
                "median_future_tradable_movement_ratio": float(np.median(ratio)),
                "movement_probabilities": {
                    f"move_gt_{threshold}_cost": float(np.mean(ratio > threshold))
                    for threshold in MOVE_THRESHOLDS_D12
                },
                "fold_movement_probabilities": {
                    f"fold_{fold}": {
                        f"move_gt_{threshold}_cost": float(
                            np.mean(
                                validation.filter(pl.col("fold") == fold)[
                                    f"h{horizon}_future_tradable_movement_ratio"
                                ]
                                .to_numpy()
                                .astype(float)
                                > threshold
                            )
                        )
                        for threshold in MOVE_THRESHOLDS_D12
                    }
                    for fold in range(1, 5)
                },
                "mean_path_quality_ratio": float(
                    str(validation[f"h{horizon}_path_quality_ratio"].mean())
                ),
                "mean_signed_executable_return_points": float(
                    str(validation[f"h{horizon}_signed_executable_return_points"].mean())
                ),
                "mean_mfe_long_points": float(
                    str(validation[f"h{horizon}_mfe_long_points"].mean())
                ),
                "mean_mae_long_points": float(
                    str(validation[f"h{horizon}_mae_long_points"].mean())
                ),
                "mean_mfe_short_points": float(
                    str(validation[f"h{horizon}_mfe_short_points"].mean())
                ),
                "mean_mae_short_points": float(
                    str(validation[f"h{horizon}_mae_short_points"].mean())
                ),
            }
        )
    return tuple(sampling), tuple(labels)


def _impute_and_scale(
    train: np.ndarray, validation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train_imputed = np.where(np.isfinite(train), train, medians)
    validation_imputed = np.where(np.isfinite(validation), validation, medians)
    scaler = StandardScaler().fit(train_imputed)
    return (
        train_imputed,
        validation_imputed,
        scaler.transform(train_imputed),
        scaler.transform(validation_imputed),
    )


def _metric_direction(y: np.ndarray, prediction: np.ndarray) -> float:
    return _safe_spearman(prediction, y)[0] or 0.0


def _metric_regime(y: np.ndarray, probability: np.ndarray) -> float:
    return float(average_precision_score(y, probability)) if len(np.unique(y)) > 1 else 0.0


def _aggregate_importance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["group"]), []).append(row["importance"])
    return [
        {
            "model": model,
            "group": group,
            "mean_importance": float(np.mean(values)),
            "fold_repeat_measurements": len(values),
        }
        for (model, group), values in sorted(grouped.items())
    ]


def _multivariate_diagnostics(
    frame: pl.DataFrame, start: datetime, end: datetime
) -> dict[str, Any]:
    features = list(FEATURE_NAMES)
    matrix = frame.select(features).to_numpy().astype(float)
    timestamps = frame["timestamp_utc"].cast(pl.Int64).to_numpy().astype(np.int64)
    folds = frame["fold"].to_numpy().astype(int)
    groups = {
        group: np.asarray(
            [index for index, feature in enumerate(features) if FEATURE_GROUPS[feature] == group],
            dtype=int,
        )
        for group in sorted(set(FEATURE_GROUPS.values()))
    }
    reports = []
    for horizon in HORIZONS_D12:
        direction_target = (
            frame[f"h{horizon}_signed_executable_return_points"].to_numpy().astype(float)
        )
        regime_target = (
            frame[f"h{horizon}_future_tradable_movement_ratio"].to_numpy().astype(float) > 2.0
        ).astype(int)
        direction_oof: dict[str, list[np.ndarray]] = {
            "ridge": [],
            "hgb": [],
            "target": [],
            "indices": [],
        }
        regime_oof: dict[str, list[np.ndarray]] = {"logistic": [], "hgb": [], "target": []}
        fold_reports = []
        permutation_rows: list[dict[str, Any]] = []
        ablation_rows: list[dict[str, Any]] = []
        for fold, (validation_start, _) in enumerate(_fold_boundaries(start, end), 1):
            validation = np.flatnonzero(folds == fold)
            cutoff = int(validation_start.timestamp() * MICROSECONDS)
            train = np.flatnonzero(timestamps + horizon * MICROSECONDS <= cutoff)
            if not len(train) or not len(validation):
                raise RuntimeError("D.12 purged multivariate fold is empty")
            max_label_end = int(np.max(timestamps[train])) + horizon * MICROSECONDS
            if max_label_end > cutoff:
                raise RuntimeError("D.12 multivariate purge invariant failed")
            raw_train, raw_validation, scaled_train, scaled_validation = _impute_and_scale(
                matrix[train], matrix[validation]
            )
            y_direction_train = direction_target[train]
            y_direction_validation = direction_target[validation]
            y_regime_train = regime_target[train]
            y_regime_validation = regime_target[validation]
            ridge = Ridge(alpha=10.0).fit(scaled_train, y_direction_train)
            hgb_direction = HistGradientBoostingRegressor(
                max_iter=75,
                max_depth=3,
                learning_rate=0.05,
                random_state=RANDOM_SEED_D12,
                early_stopping=False,
            ).fit(raw_train, y_direction_train)
            logistic = LogisticRegression(
                C=0.1,
                max_iter=300,
                random_state=RANDOM_SEED_D12,
            ).fit(scaled_train, y_regime_train)
            hgb_regime = HistGradientBoostingClassifier(
                max_iter=75,
                max_depth=3,
                learning_rate=0.05,
                random_state=RANDOM_SEED_D12,
                early_stopping=False,
            ).fit(raw_train, y_regime_train)
            ridge_prediction = ridge.predict(scaled_validation)
            hgb_direction_prediction = hgb_direction.predict(raw_validation)
            logistic_probability = logistic.predict_proba(scaled_validation)[:, 1]
            hgb_regime_probability = hgb_regime.predict_proba(raw_validation)[:, 1]
            direction_oof["ridge"].append(ridge_prediction)
            direction_oof["hgb"].append(hgb_direction_prediction)
            direction_oof["target"].append(y_direction_validation)
            direction_oof["indices"].append(validation)
            regime_oof["logistic"].append(logistic_probability)
            regime_oof["hgb"].append(hgb_regime_probability)
            regime_oof["target"].append(y_regime_validation)
            base_metrics = {
                "ridge": _metric_direction(y_direction_validation, ridge_prediction),
                "hgb_direction": _metric_direction(
                    y_direction_validation, hgb_direction_prediction
                ),
                "logistic": _metric_regime(y_regime_validation, logistic_probability),
                "hgb_regime": _metric_regime(y_regime_validation, hgb_regime_probability),
            }
            fold_reports.append(
                {
                    "fold": fold,
                    "train_observations": len(train),
                    "validation_observations": len(validation),
                    "purge_seconds": horizon,
                    "max_label_end_time_train_utc": datetime.fromtimestamp(
                        max_label_end / MICROSECONDS, UTC
                    ),
                    "validation_start_utc": validation_start,
                    "purge_invariant_passed": True,
                    "metrics": base_metrics,
                }
            )
            for group_number, (group, columns) in enumerate(groups.items()):
                if not len(columns):
                    continue
                keep = np.asarray(
                    [index for index in range(len(features)) if index not in set(columns)],
                    dtype=int,
                )
                ablated_ridge = Ridge(alpha=10.0).fit(scaled_train[:, keep], y_direction_train)
                ablated_logistic = LogisticRegression(
                    C=0.1,
                    max_iter=300,
                    random_state=RANDOM_SEED_D12,
                ).fit(scaled_train[:, keep], y_regime_train)
                ablation_rows.extend(
                    [
                        {
                            "model": "RIDGE",
                            "group": group,
                            "importance": base_metrics["ridge"]
                            - _metric_direction(
                                y_direction_validation,
                                ablated_ridge.predict(scaled_validation[:, keep]),
                            ),
                        },
                        {
                            "model": "LOGISTIC_REGRESSION_L2",
                            "group": group,
                            "importance": base_metrics["logistic"]
                            - _metric_regime(
                                y_regime_validation,
                                ablated_logistic.predict_proba(scaled_validation[:, keep])[:, 1],
                            ),
                        },
                    ]
                )
                for repeat in range(3):
                    rng = np.random.default_rng(
                        RANDOM_SEED_D12 + horizon * 100 + fold * 10 + group_number * 3 + repeat
                    )
                    permutation = rng.permutation(len(validation))
                    permuted_raw = raw_validation.copy()
                    permuted_scaled = scaled_validation.copy()
                    permuted_raw[:, columns] = permuted_raw[permutation][:, columns]
                    permuted_scaled[:, columns] = permuted_scaled[permutation][:, columns]
                    permutation_rows.extend(
                        [
                            {
                                "model": "RIDGE",
                                "group": group,
                                "importance": base_metrics["ridge"]
                                - _metric_direction(
                                    y_direction_validation,
                                    ridge.predict(permuted_scaled),
                                ),
                            },
                            {
                                "model": "HIST_GRADIENT_BOOSTING_REGRESSOR",
                                "group": group,
                                "importance": base_metrics["hgb_direction"]
                                - _metric_direction(
                                    y_direction_validation,
                                    hgb_direction.predict(permuted_raw),
                                ),
                            },
                            {
                                "model": "LOGISTIC_REGRESSION_L2",
                                "group": group,
                                "importance": base_metrics["logistic"]
                                - _metric_regime(
                                    y_regime_validation,
                                    logistic.predict_proba(permuted_scaled)[:, 1],
                                ),
                            },
                            {
                                "model": "HIST_GRADIENT_BOOSTING_CLASSIFIER",
                                "group": group,
                                "importance": base_metrics["hgb_regime"]
                                - _metric_regime(
                                    y_regime_validation,
                                    hgb_regime.predict_proba(permuted_raw)[:, 1],
                                ),
                            },
                        ]
                    )
        direction_y = np.concatenate(direction_oof["target"])
        regime_y = np.concatenate(regime_oof["target"])
        ridge_oof = np.concatenate(direction_oof["ridge"])
        hgb_direction_oof = np.concatenate(direction_oof["hgb"])
        logistic_oof = np.concatenate(regime_oof["logistic"])
        hgb_regime_oof = np.concatenate(regime_oof["hgb"])
        validation_indices = np.concatenate(direction_oof["indices"]).astype(int)
        conditional_direction = {}
        for column, values in (
            ("volatility", frame["volatility_regime"].to_numpy().astype(str)),
            ("trend", frame["trend_regime"].to_numpy().astype(str)),
        ):
            for regime in sorted(set(values[validation_indices])):
                mask = values[validation_indices] == regime
                conditional_direction[f"{column}_{regime}"] = {
                    "N": int(np.sum(mask)),
                    "ridge_spearman": _metric_direction(direction_y[mask], ridge_oof[mask]),
                    "hgb_spearman": _metric_direction(direction_y[mask], hgb_direction_oof[mask]),
                }
        reports.append(
            {
                "horizon_seconds": horizon,
                "folds": fold_reports,
                "direction": {
                    "N": len(direction_y),
                    "ridge_spearman": _metric_direction(direction_y, ridge_oof),
                    "ridge_mae": float(mean_absolute_error(direction_y, ridge_oof)),
                    "hgb_spearman": _metric_direction(direction_y, hgb_direction_oof),
                    "hgb_mae": float(mean_absolute_error(direction_y, hgb_direction_oof)),
                    "conditional_on_causal_regime": conditional_direction,
                },
                "regime_quality_move_gt_2_cost": {
                    "N": len(regime_y),
                    "base_rate": float(np.mean(regime_y)),
                    "logistic_pr_auc": _metric_regime(regime_y, logistic_oof),
                    "logistic_brier": float(brier_score_loss(regime_y, logistic_oof)),
                    "hgb_pr_auc": _metric_regime(regime_y, hgb_regime_oof),
                    "hgb_brier": float(brier_score_loss(regime_y, hgb_regime_oof)),
                },
                "group_permutation_importance": _aggregate_importance(permutation_rows),
                "linear_group_ablation": _aggregate_importance(ablation_rows),
            }
        )
    return {
        "purpose": "DIAGNOSTIC_INTERACTIONS_ONLY_NO_TRADE_SKIP_MODEL",
        "features": len(features),
        "reports": reports,
    }


def _flatten_feature_matrix(rows: list[dict[str, Any]]) -> pl.DataFrame:
    flattened = []
    for row in rows:
        flattened.append(
            {
                "feature": row["feature"],
                "family": row["family"],
                "horizon_seconds": row["horizon_seconds"],
                "target": row["target"],
                "N": row["N"],
                "effect_spearman": row["effect_spearman"],
                "day_ci_lower": row["day_block_ci95"]["lower"],
                "day_ci_upper": row["day_block_ci95"]["upper"],
                "week_ci_lower": row["week_block_ci95"]["lower"],
                "week_ci_upper": row["week_block_ci95"]["upper"],
                "fold_1": row["fold_effects"]["fold_1"],
                "fold_2": row["fold_effects"]["fold_2"],
                "fold_3": row["fold_effects"]["fold_3"],
                "fold_4": row["fold_effects"]["fold_4"],
                "mutual_information_10x10": row["mutual_information_10x10"],
                "monotonic_bin_spearman": row["monotonic_bin_spearman"],
                "winsorized_effect_spearman": row["winsorized_effect_spearman"],
                "largest_session_share_extreme_decile": row["largest_session_share_extreme_decile"],
                "p_value": row["p_value"],
                "fdr_q_value": row["fdr_q_value"],
                "same_sign_folds": row["same_sign_folds"],
                "status": row["status"],
            }
        )
    return pl.DataFrame(flattened, infer_schema_length=None)


def _family_summary(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    summaries = []
    for target in ("REGIME_QUALITY", "WITHIN_REGIME_DIRECTION"):
        for family in sorted(set(FEATURE_GROUPS.values())):
            scoped = [row for row in rows if row["target"] == target and row["family"] == family]
            ranked = sorted(
                scoped,
                key=lambda row: abs(cast(float | None, row["effect_spearman"]) or 0.0),
                reverse=True,
            )
            summaries.append(
                {
                    "target": target,
                    "family": family,
                    "tests": len(scoped),
                    "research_candidates": sum(
                        row["status"] == "FEATURE_RESEARCH_CANDIDATE" for row in scoped
                    ),
                    "unstable": sum(row["status"] == "FEATURE_UNSTABLE" for row in scoped),
                    "top_absolute_effects": [
                        {
                            "feature": row["feature"],
                            "horizon_seconds": row["horizon_seconds"],
                            "effect": row["effect_spearman"],
                            "fdr_q_value": row["fdr_q_value"],
                            "status": row["status"],
                        }
                        for row in ranked[:5]
                    ],
                }
            )
    return tuple(summaries)


def _early_day_hypothesis(frame: pl.DataFrame) -> dict[str, Any]:
    validation = frame.filter(pl.col("fold") > 0)
    feature_names = (
        "asian_range_points",
        "london_first_30m_realized_volatility",
        "london_first_60m_range",
        "london_open_displacement_points",
        "spread_to_realized_volatility_300s",
        "tick_activity_percentile_15s",
    )
    daily_rows = []
    for day in sorted(validation["day"].unique().to_list()):
        scoped = validation.filter(pl.col("day") == day).sort("timestamp_utc")
        after_london_hour = scoped.filter(pl.col("timestamp_utc").dt.hour() >= 8)
        if after_london_hour.is_empty():
            continue
        record: dict[str, Any] = {
            "day": str(day),
            "day_mean_future_tradable_movement_ratio_h300": float(
                str(scoped["h300_future_tradable_movement_ratio"].mean())
            ),
            "day_fraction_move_gt_2_cost_h300": float(
                (
                    scoped["h300_future_tradable_movement_ratio"].to_numpy().astype(float) > 2.0
                ).mean()
            ),
        }
        for feature in feature_names:
            feature_values = after_london_hour[feature].drop_nulls()
            record[feature] = float(str(feature_values[0])) if len(feature_values) else None
        daily_rows.append(record)
    daily = pl.DataFrame(daily_rows, infer_schema_length=None)
    target = daily["day_mean_future_tradable_movement_ratio_h300"].to_numpy().astype(float)
    diagnostics = []
    for feature in feature_names:
        feature_values_array = daily[feature].to_numpy().astype(float)
        effect, pvalue = _safe_spearman(feature_values_array, target)
        diagnostics.append(
            {
                "feature": feature,
                "days": int(np.sum(np.isfinite(feature_values_array) & np.isfinite(target))),
                "spearman_vs_day_mean_tradability": effect,
                "p_value_uncorrected": pvalue,
                "status": "EXPLORATORY_DAY_LEVEL_HYPOTHESIS_NOT_VALIDATED",
            }
        )
    return {
        "purpose": "EXPLAIN_D11_DAY_MEAN_VS_OBSERVATION_LEVEL_DIFFERENCE",
        "causal_cutoff": "EARLY_FEATURES_USE_QUOTES_AVAILABLE_BY_THE_RECORDED_TIMESTAMP_ONLY",
        "days": daily.height,
        "diagnostics": diagnostics,
        "daily_rows": daily_rows,
    }


def _verdict(rows: list[dict[str, Any]]) -> tuple[D12Conclusion, dict[str, Any]]:
    evidence: dict[str, Any] = {}
    problem_passes = {}
    for target in ("REGIME_QUALITY", "WITHIN_REGIME_DIRECTION"):
        candidates = [
            row
            for row in rows
            if row["target"] == target and row["status"] == "FEATURE_RESEARCH_CANDIDATE"
        ]
        features = sorted({row["feature"] for row in candidates})
        groups = sorted({row["family"] for row in candidates})
        horizons = sorted({row["horizon_seconds"] for row in candidates})
        passes = len(features) >= 3 and len(groups) >= 2 and len(horizons) >= 2
        evidence[target] = {
            "candidate_rows": len(candidates),
            "unique_candidate_features": features,
            "candidate_feature_groups": groups,
            "candidate_horizons": horizons,
            "minimum_3_features": len(features) >= 3,
            "minimum_2_groups": len(groups) >= 2,
            "minimum_2_horizons": len(horizons) >= 2,
            "passes": passes,
        }
        problem_passes[target] = passes
    unstable_rows = sum(row["status"] == "FEATURE_UNSTABLE" for row in rows)
    candidate_rows = sum(row["status"] == "FEATURE_RESEARCH_CANDIDATE" for row in rows)
    worth = all(problem_passes.values())
    if worth:
        conclusion: D12Conclusion = "D12_FEATURES_WORTH_V4"
    elif unstable_rows or candidate_rows:
        conclusion = "D12_UNSTABLE_FEATURE_SIGNAL"
    else:
        conclusion = "D12_NO_USEFUL_FEATURE_SIGNAL"
    evidence.update(
        {
            "both_problems_pass": worth,
            "unstable_rows": unstable_rows,
            "candidate_rows": candidate_rows,
            "verdict_policy_applied_without_post_result_change": True,
        }
    )
    return conclusion, evidence


def _render_report(report: ResearchD12Report) -> str:
    direction = report.verdict_evidence["WITHIN_REGIME_DIRECTION"]
    regime = report.verdict_evidence["REGIME_QUALITY"]
    return "\n".join(
        [
            "# SNIPER Phase D.12 — V4 Feature & Regime Discovery",
            "",
            f"Verdict : **{report.conclusion}**",
            "",
            f"Ticks RESEARCH : {report.performance.ticks_analyzed:,}",
            f"Événements complets : {report.performance.complete_events:,}",
            f"Features : {report.feature_matrix_summary['features']}",
            f"Tests statistiques corrigés FDR : {report.multiple_testing['tests']}",
            f"Candidats régime : {len(regime['unique_candidate_features'])}",
            f"Candidats direction : {len(direction['unique_candidate_features'])}",
            f"Runtime : {report.performance.elapsed_seconds:.2f} s",
            f"Peak mémoire Python : {report.performance.peak_python_memory_mb:.2f} MiB",
            "",
            "## Garde-fous",
            "",
            "- Découverte diagnostique uniquement ; aucun TRADE/SKIP, BUY ou SELL",
            "- Aucun seuil, SL/TP, sizing ou moteur V4 créé",
            "- HOLDOUT SEALED, non ouvert et non évalué",
            "- Phase E non commencée ; trading live désactivé",
            "",
        ]
    )


class ResearchD12Engine:
    def run(
        self,
        *,
        data_root: Path,
        start_utc: datetime,
        end_utc: datetime,
        protocol_path: Path,
    ) -> ResearchD12Report:
        start, end = as_utc(start_utc), as_utc(end_utc)
        _validate_research_scope(data_root, start, end)
        if not protocol_path.exists():
            raise ValueError("D.12 requires its pre-registered protocol")
        protocol_hash = _sha256(protocol_path)
        tracemalloc.start()
        started = perf_counter()
        output_dir = data_root / "evaluations" / "research-d12-20260610T000000Z-20260908T000000Z"
        output_dir.mkdir(parents=True, exist_ok=True)
        events, stream = build_d12_event_dataset(
            data_root=data_root, start=start, end=end, output_dir=output_dir
        )
        events = _with_folds(events, start, end)
        event_path = output_dir / "events.parquet"
        events.write_parquet(event_path, compression="zstd", statistics=True)
        sampling, labels = _sampling_and_label_summary(events)
        sampling = tuple(
            {
                **row,
                "triggered_before_label_completeness": stream["triggered_by_scheme"][row["scheme"]],
                "complete_all_horizons": stream["complete_by_scheme"][row["scheme"]],
            }
            for row in sampling
        )
        matrix_rows, matrix_summary = _univariate_matrix(events)
        flattened = _flatten_feature_matrix(matrix_rows)
        matrix_path = output_dir / "feature-matrix.parquet"
        flattened.write_parquet(matrix_path, compression="zstd", statistics=True)
        matrix_json_path = output_dir / "feature-matrix.json"
        matrix_json_path.write_text(
            json.dumps(matrix_rows, indent=2, default=str), encoding="utf-8"
        )
        families = _family_summary(matrix_rows)
        family_path = output_dir / "family-summary.json"
        family_path.write_text(json.dumps(families, indent=2), encoding="utf-8")
        multivariate = _multivariate_diagnostics(events, start, end)
        multivariate_path = output_dir / "multivariate-diagnostics.json"
        multivariate_path.write_text(
            json.dumps(multivariate, indent=2, default=str), encoding="utf-8"
        )
        early_day = _early_day_hypothesis(events)
        early_day_path = output_dir / "early-day-hypothesis.json"
        early_day_path.write_text(json.dumps(early_day, indent=2), encoding="utf-8")
        conclusion, verdict_evidence = _verdict(matrix_rows)
        elapsed = perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        source_hashes = {
            "research_d12.py": _sha256(Path(__file__)),
            "research_protocol_d12.yaml": protocol_hash,
            "events.parquet": _sha256(event_path),
            "feature_matrix.parquet": _sha256(matrix_path),
        }
        return ResearchD12Report(
            requested_start_utc=start,
            requested_end_utc=end,
            protocol_sha256=protocol_hash,
            source_hashes=source_hashes,
            causal_feature_policy={
                "quotes_at_or_before_T_only": True,
                "daily_high_low_are_causal_to_T": True,
                "final_daily_high_low_forbidden": True,
                "volume_vwap_used": False,
                "replacement": "INTRADAY_TIME_WEIGHTED_QUOTE_MID_TWAP",
                "order_book_features_invented": False,
            },
            sampling_summary=sampling,
            label_summary=labels,
            feature_matrix_summary=matrix_summary,
            feature_status_summary=matrix_summary["status_counts"],
            family_summary=families,
            multivariate_diagnostics=multivariate,
            early_day_hypothesis=early_day,
            multiple_testing={
                "features": len(FEATURE_NAMES),
                "horizons": len(HORIZONS_D12),
                "targets": 2,
                "interactions": 8,
                "tests": len(matrix_rows),
                "correction": "BENJAMINI_HOCHBERG",
                "fdr_q": 0.10,
                "single_p_value_never_treated_as_discovery": True,
            },
            verdict_evidence=verdict_evidence,
            artifacts={
                "events_parquet": str(event_path.resolve()),
                "feature_matrix_parquet": str(matrix_path.resolve()),
                "feature_matrix_json": str(matrix_json_path.resolve()),
                "family_summary_json": str(family_path.resolve()),
                "multivariate_diagnostics_json": str(multivariate_path.resolve()),
                "early_day_hypothesis_json": str(early_day_path.resolve()),
            },
            performance=D12Performance(
                elapsed_seconds=elapsed,
                ticks_analyzed=stream["ticks_analyzed"],
                detector_seconds=stream["detector_seconds"],
                events_produced=stream["triggered_events"],
                complete_events=stream["complete_events"],
                ticks_per_second=stream["ticks_analyzed"] / max(elapsed, 1e-9),
                peak_python_memory_mb=peak / 1024 / 1024,
                peak_loaded_tick_frame_mb=stream["peak_loaded_tick_frame_mb"],
            ),
            conclusion=conclusion,
            limitations=(
                "D.12 is feature discovery, not a strategy, trading gate or profitability claim.",
                "Quote-mid TWAP replaces unavailable traded-volume VWAP; no volume was invented.",
                "Feature statuses remain RESEARCH diagnostics under multiple testing.",
                "Day-level hypothesis has few independent days and is exploratory only.",
                "The HOLDOUT remained sealed and was neither opened nor evaluated.",
                "No V4, Phase E, sizing, order execution or live trading was created.",
            ),
        )


def render_research_d12_report(report: ResearchD12Report) -> str:
    return _render_report(report)
