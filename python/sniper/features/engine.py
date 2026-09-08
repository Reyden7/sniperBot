"""Point-in-time V1 feature computation for EURUSD only."""

from datetime import datetime, timedelta
from decimal import Decimal

from sniper.data.time import as_utc
from sniper.domain.bar import Bar, Timeframe
from sniper.domain.signal import (
    FeatureSnapshot,
    M1Features,
    M5Features,
    M15Features,
    MarketRegime,
    TickFeatures,
    TimeframeWarmup,
    TrendDirection,
)
from sniper.domain.trade import MarketTick
from sniper.features.momentum import ema, ema_slope, momentum
from sniper.features.patterns import candle_features
from sniper.features.structure import breakout_retest, levels, structure
from sniper.features.volatility import atr, range_regime


def _completed(bars: list[Bar], timeframe: Timeframe, as_of: datetime) -> list[Bar]:
    minutes = {"M1": 1, "M5": 5, "M15": 15}[timeframe]
    duration = timedelta(minutes=minutes)
    return sorted(
        (
            bar
            for bar in bars
            if bar.timeframe == timeframe
            and bar.is_complete
            and bar.open_time_utc + duration <= as_of
        ),
        key=lambda bar: bar.open_time_utc,
    )


def _safe_momentum(values: list[Decimal], horizon: int) -> Decimal:
    return momentum(values, horizon) if len(values) > horizon else Decimal(0)


def _direction(value: Decimal) -> TrendDirection:
    if value > 0:
        return TrendDirection.UP
    if value < 0:
        return TrendDirection.DOWN
    return TrendDirection.FLAT


WARMUP_REQUIREMENTS: dict[Timeframe, tuple[int, tuple[str, ...]]] = {
    "M15": (14, ("ema_slope", "structure", "atr", "regime")),
    "M5": (20, ("ema_short_long", "momentum", "atr", "levels", "breakout_retest")),
    "M1": (15, ("momentum_1_3_5", "atr", "candle_shape", "patterns", "range_regime")),
}
TIMEFRAMES: tuple[Timeframe, ...] = ("M15", "M5", "M1")


class FeatureEngine:
    """Pure computation: no orders, positions, volume, optimization or mutable market state."""

    def __init__(self, point: Decimal = Decimal("0.00001")) -> None:
        if not point.is_finite() or point <= 0:
            raise ValueError("point must be finite and positive")
        self.point = point

    def compute(
        self,
        *,
        bars: dict[Timeframe, list[Bar]],
        ticks: list[MarketTick],
        as_of_utc: datetime,
        symbol: str = "EURUSD",
    ) -> FeatureSnapshot:
        as_of = as_utc(as_of_utc)
        if symbol != "EURUSD":
            raise ValueError("Feature Engine V1 supports EURUSD only")
        visible_bars: dict[Timeframe, list[Bar]] = {
            "M15": _completed(bars.get("M15", []), "M15", as_of),
            "M5": _completed(bars.get("M5", []), "M5", as_of),
            "M1": _completed(bars.get("M1", []), "M1", as_of),
        }
        if not visible_bars["M15"] or not visible_bars["M5"] or not visible_bars["M1"]:
            raise ValueError("at least one completed M15, M5 and M1 bar is required")
        visible_ticks = sorted(
            (tick for tick in ticks if tick.timestamp_utc <= as_of),
            key=lambda tick: tick.timestamp_utc,
        )
        if not visible_ticks:
            raise ValueError("at least one visible tick is required")

        m15_bars = visible_bars["M15"]
        m15_closes = [bar.close for bar in m15_bars]
        if len(m15_closes) >= 2:
            m15_ema, m15_slope = ema_slope(m15_closes, 5)
        else:
            m15_ema, m15_slope = m15_closes[-1], Decimal(0)
        m15_atr = atr(m15_bars)
        m15_structure = structure(m15_bars)
        m15_direction = _direction(m15_slope)
        m15_regime = (
            MarketRegime.TREND
            if m15_structure != "MIXED"
            and m15_atr > 0
            and abs(m15_slope) >= m15_atr * Decimal("0.05")
            else MarketRegime.RANGE
        )

        m5_bars = visible_bars["M5"]
        m5_closes = [bar.close for bar in m5_bars]
        ema_short = ema(m5_closes, 3)
        ema_long = ema(m5_closes, 8)
        m5_momentum = _safe_momentum(m5_closes, 3)
        m5_atr = atr(m5_bars)
        support, resistance = levels(m5_bars)
        current_m5 = m5_closes[-1]
        support_distance = (current_m5 - support) / m5_atr if m5_atr > 0 else Decimal(0)
        resistance_distance = (resistance - current_m5) / m5_atr if m5_atr > 0 else Decimal(0)

        m1_bars = visible_bars["M1"]
        m1_closes = [bar.close for bar in m1_bars]
        m1_atr = atr(m1_bars)
        range_ratio, compression, expansion = range_regime(m1_bars, m1_atr)
        body, upper_wick, lower_wick, pattern = candle_features(m1_bars)

        recent_ticks = [
            tick for tick in visible_ticks if tick.timestamp_utc >= as_of - timedelta(seconds=60)
        ]
        current_tick = visible_ticks[-1]
        rate_1 = sum(tick.timestamp_utc >= as_of - timedelta(seconds=1) for tick in recent_ticks)
        rate_5 = sum(tick.timestamp_utc >= as_of - timedelta(seconds=5) for tick in recent_ticks)
        rate_15 = sum(tick.timestamp_utc >= as_of - timedelta(seconds=15) for tick in recent_ticks)
        changes = [
            right.mid - left.mid
            for left, right in zip(recent_ticks, recent_ticks[1:], strict=False)
        ]
        upticks = sum(change > 0 for change in changes)
        downticks = sum(change < 0 for change in changes)
        directional = upticks + downticks
        uptick_ratio = Decimal(upticks) / directional if directional else Decimal("0.5")
        downtick_ratio = Decimal(downticks) / directional if directional else Decimal("0.5")
        price_acceleration = changes[-1] - changes[-2] if len(changes) >= 2 else Decimal(0)
        tick_rate_acceleration = Decimal(rate_1) - Decimal(rate_5) / 5
        spreads = sorted(tick.spread / self.point for tick in recent_ticks)
        if not spreads:
            spreads = [current_tick.spread / self.point]
        middle = len(spreads) // 2
        median_spread = (
            spreads[middle] if len(spreads) % 2 else (spreads[middle - 1] + spreads[middle]) / 2
        )

        return FeatureSnapshot(
            as_of_utc=as_of,
            m15=M15Features(
                ema=m15_ema,
                ema_slope=m15_slope,
                atr=m15_atr,
                structure=m15_structure,
                direction=m15_direction,
                regime=m15_regime,
            ),
            m5=M5Features(
                ema_short=ema_short,
                ema_long=ema_long,
                momentum=m5_momentum,
                atr=m5_atr,
                direction=_direction(ema_short - ema_long),
                support=support,
                resistance=resistance,
                support_distance_atr=max(support_distance, Decimal(0)),
                resistance_distance_atr=max(resistance_distance, Decimal(0)),
                breakout_retest=breakout_retest(m5_bars),
            ),
            m1=M1Features(
                momentum_1=_safe_momentum(m1_closes, 1),
                momentum_3=_safe_momentum(m1_closes, 3),
                momentum_5=_safe_momentum(m1_closes, 5),
                atr=m1_atr,
                range_atr_ratio=range_ratio,
                body_ratio=body,
                upper_wick_ratio=upper_wick,
                lower_wick_ratio=lower_wick,
                pattern=pattern,
                compression=compression,
                expansion=expansion,
            ),
            ticks=TickFeatures(
                rate_1s=rate_1,
                rate_5s=rate_5,
                rate_15s=rate_15,
                uptick_ratio=uptick_ratio,
                downtick_ratio=downtick_ratio,
                price_acceleration=price_acceleration,
                tick_rate_acceleration=tick_rate_acceleration,
                last_tick_timestamp_utc=current_tick.timestamp_utc,
                last_tick_age_seconds=Decimal(
                    str((as_of - current_tick.timestamp_utc).total_seconds())
                ),
                current_spread_points=current_tick.spread / self.point,
                median_spread_points=median_spread,
            ),
            completed_bars={
                "M15": len(m15_bars),
                "M5": len(m5_bars),
                "M1": len(m1_bars),
            },
            warmup={
                timeframe: TimeframeWarmup(
                    required_bars=WARMUP_REQUIREMENTS[timeframe][0],
                    available_bars=len(visible_bars[timeframe]),
                    complete=len(visible_bars[timeframe]) >= WARMUP_REQUIREMENTS[timeframe][0],
                    feature_families=WARMUP_REQUIREMENTS[timeframe][1],
                )
                for timeframe in TIMEFRAMES
            },
            visible_ticks=len(recent_ticks),
        )
