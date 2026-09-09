"""Deterministic causal M15/M5/M1 Binance Spot research strategy engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from math import sqrt

import numpy as np
import polars as pl


class MarketRegime(StrEnum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    UNKNOWN = "UNKNOWN"


class SetupType(StrEnum):
    MOMENTUM_PULLBACK = "MOMENTUM_PULLBACK"
    BREAKOUT_RETEST = "BREAKOUT_RETEST"
    RANGE_MEAN_REVERSION = "RANGE_MEAN_REVERSION"


@dataclass(frozen=True)
class ResearchBar:
    symbol: str
    timestamp_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades_count: int
    taker_buy_quote_volume: float


@dataclass(frozen=True)
class StrategySignal:
    symbol: str
    timestamp_utc: datetime
    side: str
    setup_type: SetupType
    regime: MarketRegime
    entry_reference: float
    invalid_level: float
    target_reference: float
    atr: float
    confidence: float
    reason: tuple[str, ...]


def bars_from_frame(frame: pl.DataFrame) -> dict[str, list[ResearchBar]]:
    """Convert canonical M1 data to sorted immutable bars."""
    result: dict[str, list[ResearchBar]] = {}
    for row in frame.iter_rows(named=True):
        timestamp = row["timestamp_utc"]
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        bar = ResearchBar(
            symbol=str(row["symbol"]),
            timestamp_utc=timestamp.astimezone(UTC),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            quote_volume=float(row["quote_volume"]),
            trades_count=int(row["trades_count"]),
            taker_buy_quote_volume=float(row["taker_buy_quote_volume"]),
        )
        result.setdefault(bar.symbol, []).append(bar)
    for bars in result.values():
        bars.sort(key=lambda item: item.timestamp_utc)
    return result


def aggregate_bars(m1: list[ResearchBar], minutes: int) -> list[ResearchBar]:
    """Build complete UTC-aligned bars; incomplete groups are discarded."""
    if minutes not in {5, 15}:
        raise ValueError("only M5 and M15 aggregation is supported")
    grouped: dict[datetime, list[ResearchBar]] = {}
    for bar in m1:
        aligned = bar.timestamp_utc.replace(
            minute=(bar.timestamp_utc.minute // minutes) * minutes,
            second=0,
            microsecond=0,
        )
        grouped.setdefault(aligned, []).append(bar)
    output = []
    for timestamp, rows in sorted(grouped.items()):
        rows.sort(key=lambda item: item.timestamp_utc)
        expected = [timestamp + timedelta(minutes=index) for index in range(minutes)]
        if [row.timestamp_utc for row in rows] != expected:
            continue
        output.append(
            ResearchBar(
                symbol=rows[0].symbol,
                timestamp_utc=timestamp,
                open=rows[0].open,
                high=max(row.high for row in rows),
                low=min(row.low for row in rows),
                close=rows[-1].close,
                volume=sum(row.volume for row in rows),
                quote_volume=sum(row.quote_volume for row in rows),
                trades_count=sum(row.trades_count for row in rows),
                taker_buy_quote_volume=sum(row.taker_buy_quote_volume for row in rows),
            )
        )
    return output


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    output = np.empty_like(values)
    output[0] = values[0]
    alpha = 2.0 / (span + 1)
    for index in range(1, len(values)):
        output[index] = alpha * values[index] + (1 - alpha) * output[index - 1]
    return output


def _atr(bars: list[ResearchBar], period: int = 14) -> np.ndarray:
    true_range = np.empty(len(bars), dtype=float)
    true_range[0] = bars[0].high - bars[0].low
    for index in range(1, len(bars)):
        true_range[index] = max(
            bars[index].high - bars[index].low,
            abs(bars[index].high - bars[index - 1].close),
            abs(bars[index].low - bars[index - 1].close),
        )
    return _ema(true_range, period * 2 - 1)


def _rolling(values: np.ndarray, window: int, operation: str) -> np.ndarray:
    result = np.full(len(values), np.nan)
    for index in range(window - 1, len(values)):
        sample = values[index - window + 1 : index + 1]
        if operation == "mean":
            result[index] = float(np.mean(sample))
        elif operation == "std":
            result[index] = float(np.std(sample, ddof=1))
        elif operation == "median":
            result[index] = float(np.median(sample))
        else:
            raise ValueError("unsupported rolling operation")
    return result


def classify_m15_regimes(m15: list[ResearchBar]) -> list[MarketRegime]:
    """Classify each closed M15 bar using only itself and earlier bars."""
    if not m15:
        return []
    closes = np.asarray([bar.close for bar in m15], dtype=float)
    fast = _ema(closes, 9)
    slow = _ema(closes, 21)
    regimes = [MarketRegime.UNKNOWN] * len(m15)
    for index in range(21, len(m15)):
        path = np.abs(np.diff(closes[index - 12 : index + 1])).sum()
        efficiency = abs(closes[index] - closes[index - 12]) / path if path > 0 else 0
        normalized_gap = abs(fast[index] - slow[index]) / closes[index]
        if efficiency >= 0.35 and normalized_gap >= 0.0005:
            if fast[index] > slow[index] and fast[index] > fast[index - 3]:
                regimes[index] = MarketRegime.TREND_UP
            elif fast[index] < slow[index] and fast[index] < fast[index - 3]:
                regimes[index] = MarketRegime.TREND_DOWN
            else:
                regimes[index] = MarketRegime.RANGE
        else:
            regimes[index] = MarketRegime.RANGE
    return regimes


class BinanceStrategyEngine:
    """Emit fixed-rule long-only Spot setups at completed M5 boundaries."""

    def generate(self, m1: list[ResearchBar]) -> tuple[list[ResearchBar], list[StrategySignal]]:
        if len(m1) < 400:
            return [], []
        m5 = aggregate_bars(m1, 5)
        m15 = aggregate_bars(m1, 15)
        regimes = classify_m15_regimes(m15)
        closes = np.asarray([bar.close for bar in m5], dtype=float)
        highs = np.asarray([bar.high for bar in m5], dtype=float)
        lows = np.asarray([bar.low for bar in m5], dtype=float)
        volumes = np.asarray([bar.quote_volume for bar in m5], dtype=float)
        fast = _ema(closes, 9)
        slow = _ema(closes, 21)
        atr = _atr(m5)
        volume_median = _rolling(volumes, 20, "median")
        mean = _rolling(closes, 20, "mean")
        std = _rolling(closes, 20, "std")
        m15_cursor = 0
        signals = []
        for index in range(25, len(m5) - 1):
            signal_time = m5[index].timestamp_utc + timedelta(minutes=5)
            while (
                m15_cursor + 1 < len(m15)
                and m15[m15_cursor + 1].timestamp_utc + timedelta(minutes=15) <= signal_time
            ):
                m15_cursor += 1
            if m15[m15_cursor].timestamp_utc + timedelta(minutes=15) > signal_time:
                continue
            regime = regimes[m15_cursor]
            m1_index = min((index + 1) * 5 - 1, len(m1) - 1)
            if m1_index < 3 or m1[m1_index].close <= m1[m1_index - 3].close:
                continue
            current = m5[index]
            previous = m5[index - 1]
            signal: StrategySignal | None = None
            breakout_level = float(np.max(highs[index - 21 : index - 1]))
            previous_broke_out = previous.close > breakout_level
            retested = current.low <= breakout_level * 1.0015 and current.close > breakout_level
            breakout_volume = previous.quote_volume >= volume_median[index - 1] * 1.20
            if previous_broke_out and retested and breakout_volume and regime != MarketRegime.TREND_DOWN:
                stop = min(current.low, breakout_level - atr[index] * 0.35)
                signal = StrategySignal(
                    current.symbol,
                    signal_time,
                    "BUY",
                    SetupType.BREAKOUT_RETEST,
                    regime,
                    current.close,
                    stop,
                    current.close + atr[index] * 2.0,
                    atr[index],
                    0.75,
                    ("M5_BREAKOUT_CONFIRMED", "M5_RETEST_HELD", "M1_MOMENTUM_POSITIVE"),
                )
            pullback = current.low <= fast[index] * 1.001 and current.close > fast[index]
            resumed = current.close > previous.close and volumes[index] >= volume_median[index] * 0.80
            if signal is None and regime == MarketRegime.TREND_UP and fast[index] > slow[index]:
                if pullback and resumed:
                    stop = min(lows[index - 2 : index + 1]) - atr[index] * 0.10
                    signal = StrategySignal(
                        current.symbol,
                        signal_time,
                        "BUY",
                        SetupType.MOMENTUM_PULLBACK,
                        regime,
                        current.close,
                        stop,
                        current.close + atr[index] * 1.5,
                        atr[index],
                        0.70,
                        ("M15_UPTREND", "M5_PULLBACK", "M5_RESUMPTION", "M1_MOMENTUM_POSITIVE"),
                    )
            lower_band = mean[index - 1] - 2.0 * std[index - 1]
            mean_reentry = previous.close < lower_band and current.close > previous.close
            if signal is None and regime == MarketRegime.RANGE and mean_reentry:
                target = mean[index]
                if target > current.close:
                    signal = StrategySignal(
                        current.symbol,
                        signal_time,
                        "BUY",
                        SetupType.RANGE_MEAN_REVERSION,
                        regime,
                        current.close,
                        min(previous.low, current.low) - atr[index] * 0.25,
                        target,
                        atr[index],
                        min(0.80, 0.55 + abs(previous.close - lower_band) / max(atr[index], 1e-12) / 10),
                        ("M15_RANGE", "M5_LOWER_EXTENSION", "M5_REENTRY", "M1_MOMENTUM_POSITIVE"),
                    )
            if signal is not None and signal.invalid_level < signal.entry_reference:
                signals.append(signal)
        return m5, signals


def annualized_sharpe(daily_returns: list[float]) -> float:
    if len(daily_returns) < 2:
        return 0.0
    deviation = float(np.std(daily_returns, ddof=1))
    return float(np.mean(daily_returns) / deviation * sqrt(365)) if deviation > 0 else 0.0
