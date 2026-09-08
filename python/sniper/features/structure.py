"""Minimal market structure, levels and breakout/retest detection."""

from decimal import Decimal

from sniper.domain.bar import Bar


def structure(bars: list[Bar]) -> str:
    if len(bars) < 3:
        return "MIXED"
    recent = bars[-3:]
    if (
        recent[0].high < recent[1].high < recent[2].high
        and recent[0].low < recent[1].low < recent[2].low
    ):
        return "HH_HL"
    if (
        recent[0].high > recent[1].high > recent[2].high
        and recent[0].low > recent[1].low > recent[2].low
    ):
        return "LH_LL"
    return "MIXED"


def levels(bars: list[Bar], lookback: int = 20) -> tuple[Decimal, Decimal]:
    if not bars:
        raise ValueError("levels require bars")
    window = bars[-lookback:]
    return min(bar.low for bar in window), max(bar.high for bar in window)


def breakout_retest(bars: list[Bar], lookback: int = 5) -> str | None:
    if len(bars) < lookback + 2:
        return None
    history = bars[-lookback - 2 : -2]
    breakout = bars[-2]
    retest = bars[-1]
    resistance = max(bar.high for bar in history)
    support = min(bar.low for bar in history)
    if breakout.close > resistance and retest.low <= resistance <= retest.close:
        return "BULLISH"
    if breakout.close < support and retest.high >= support >= retest.close:
        return "BEARISH"
    return None
