"""Seedable price-slippage models for deterministic research."""

from dataclasses import dataclass, field
from decimal import Decimal
from random import Random
from typing import Protocol

from sniper.domain.trade import Side


class SlippageModel(Protocol):
    @property
    def maximum_adverse_points(self) -> Decimal: ...

    def apply(
        self, requested_price: Decimal, point: Decimal, side: Side, *, entry: bool
    ) -> Decimal: ...


def _apply_points(
    requested_price: Decimal, point: Decimal, side: Side, entry: bool, points: Decimal
) -> Decimal:
    adverse_direction = side.sign if entry else -side.sign
    result = requested_price + adverse_direction * points * point
    if result <= 0:
        raise ValueError("slippage produced a nonpositive price")
    return result


@dataclass(frozen=True)
class NoSlippage:
    @property
    def maximum_adverse_points(self) -> Decimal:
        return Decimal(0)

    def apply(
        self, requested_price: Decimal, point: Decimal, side: Side, *, entry: bool
    ) -> Decimal:
        return requested_price


@dataclass(frozen=True)
class FixedSlippage:
    """Signed points: positive is adverse, negative is favorable."""

    points: Decimal

    def __post_init__(self) -> None:
        if not self.points.is_finite():
            raise ValueError("slippage points must be finite")

    @property
    def maximum_adverse_points(self) -> Decimal:
        return max(self.points, Decimal(0))

    def apply(
        self, requested_price: Decimal, point: Decimal, side: Side, *, entry: bool
    ) -> Decimal:
        return _apply_points(requested_price, point, side, entry, self.points)


@dataclass
class RandomSlippage:
    """Uniform signed stress slippage, reproducible for the same seed and event order."""

    max_points: int
    seed: int
    _random: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_points < 0:
            raise ValueError("max_points must be nonnegative")
        self._random = Random(self.seed)

    @property
    def maximum_adverse_points(self) -> Decimal:
        return Decimal(self.max_points)

    def apply(
        self, requested_price: Decimal, point: Decimal, side: Side, *, entry: bool
    ) -> Decimal:
        points = Decimal(self._random.randint(-self.max_points, self.max_points))
        return _apply_points(requested_price, point, side, entry, points)
