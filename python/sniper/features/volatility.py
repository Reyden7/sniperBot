"""ATR and compression/expansion primitives with no future data."""

from decimal import Decimal

from sniper.domain.bar import Bar


def true_ranges(bars: list[Bar]) -> list[Decimal]:
    result: list[Decimal] = []
    previous_close = None
    for bar in bars:
        value = bar.high - bar.low
        if previous_close is not None:
            value = max(value, abs(bar.high - previous_close), abs(bar.low - previous_close))
        result.append(value)
        previous_close = bar.close
    return result


def atr(bars: list[Bar], period: int = 14) -> Decimal:
    if not bars or period < 1:
        raise ValueError("ATR requires bars and a positive period")
    values = true_ranges(bars)[-period:]
    return sum(values, Decimal(0)) / len(values)


def range_regime(bars: list[Bar], current_atr: Decimal) -> tuple[Decimal, bool, bool]:
    current_range = bars[-1].high - bars[-1].low
    ratio = current_range / current_atr if current_atr > 0 else Decimal(0)
    return ratio, ratio <= Decimal("0.60"), ratio >= Decimal("1.50")
