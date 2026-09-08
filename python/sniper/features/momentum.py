"""Small deterministic momentum primitives used by V1."""

from decimal import Decimal


def ema(values: list[Decimal], period: int) -> Decimal:
    if not values or period < 1:
        raise ValueError("EMA requires values and a positive period")
    alpha = Decimal(2) / Decimal(period + 1)
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (Decimal(1) - alpha) * result
    return result


def ema_slope(values: list[Decimal], period: int) -> tuple[Decimal, Decimal]:
    if len(values) < 2:
        raise ValueError("EMA slope requires two values")
    current = ema(values, period)
    previous = ema(values[:-1], period)
    return current, current - previous


def momentum(values: list[Decimal], bars: int) -> Decimal:
    if bars < 1 or len(values) <= bars:
        raise ValueError("insufficient values for momentum horizon")
    return values[-1] - values[-1 - bars]
