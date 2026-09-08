"""Minimal V1 candle patterns and wick/body measurements."""

from decimal import Decimal

from sniper.domain.bar import Bar
from sniper.domain.signal import PatternFeatures


def candle_features(bars: list[Bar]) -> tuple[Decimal, Decimal, Decimal, PatternFeatures]:
    if not bars:
        raise ValueError("candle features require at least one bar")
    current = bars[-1]
    candle_range = current.high - current.low
    if candle_range == 0:
        return Decimal(0), Decimal(0), Decimal(0), PatternFeatures()
    body = abs(current.close - current.open)
    upper = current.high - max(current.open, current.close)
    lower = min(current.open, current.close) - current.low
    engulfing = None
    if len(bars) >= 2:
        previous = bars[-2]
        bullish = (
            previous.close < previous.open
            and current.close > current.open
            and current.open <= previous.close
            and current.close >= previous.open
        )
        bearish = (
            previous.close > previous.open
            and current.close < current.open
            and current.open >= previous.close
            and current.close <= previous.open
        )
        engulfing = "BULLISH" if bullish else "BEARISH" if bearish else None
    hammer = lower >= body * 2 and upper <= max(body, candle_range * Decimal("0.10"))
    shooting_star = upper >= body * 2 and lower <= max(body, candle_range * Decimal("0.10"))
    return (
        body / candle_range,
        upper / candle_range,
        lower / candle_range,
        PatternFeatures(engulfing=engulfing, hammer=hammer, shooting_star=shooting_star),
    )
