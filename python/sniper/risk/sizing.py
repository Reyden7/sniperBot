"""Conservative research sizing on the broker's volume grid."""

from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal

from sniper.domain.broker import VolumeConstraints


@dataclass(frozen=True)
class SizingResult:
    volume: Decimal
    estimated_loss: Decimal
    required_margin: Decimal
    reason: str


def floor_volume(requested: Decimal, volumes: VolumeConstraints) -> Decimal:
    if not requested.is_finite() or requested < 0:
        raise ValueError("requested volume must be finite and nonnegative")
    if requested < volumes.minimum:
        return Decimal(0)
    steps = ((min(requested, volumes.maximum) - volumes.minimum) / volumes.step).to_integral_value(
        rounding=ROUND_FLOOR
    )
    return volumes.minimum + steps * volumes.step


def size_position(
    *,
    equity: Decimal,
    target_risk_pct: Decimal,
    volumes: VolumeConstraints,
    loss_at_stop: Callable[[Decimal], Decimal],
    margin_required: Callable[[Decimal], Decimal],
    free_margin: Decimal,
    minimum_margin_buffer: Decimal = Decimal(0),
    hard_max_risk_pct: Decimal = Decimal("0.50"),
) -> SizingResult:
    """Find the largest affordable grid volume with monotone loss/margin oracles.

    loss_at_stop MUST include spread once, round-trip commission and slippage,
    evaluated at a valid stop. The oracles must be nonnegative, finite and
    nondecreasing in volume. No 1-lot pip-value approximation is made here.
    """
    values = (equity, target_risk_pct, free_margin, minimum_margin_buffer, hard_max_risk_pct)
    if not all(value.is_finite() for value in values):
        raise ValueError("sizing inputs must be finite")
    if equity <= 0 or not 0 < target_risk_pct <= hard_max_risk_pct <= Decimal("0.50"):
        raise ValueError("invalid equity or risk limits")
    if minimum_margin_buffer < 0:
        raise ValueError("negative margin buffer")
    budget = equity * target_risk_pct / 100
    available_margin = min(equity, free_margin) - minimum_margin_buffer

    def estimate(volume: Decimal) -> tuple[Decimal, Decimal]:
        loss, margin = loss_at_stop(volume), margin_required(volume)
        if any(not value.is_finite() or value < 0 for value in (loss, margin)):
            raise ValueError("invalid sizing oracle result")
        if loss == 0:
            raise ValueError("a position with a stop must have positive estimated loss")
        return loss, margin

    loss, margin = estimate(volumes.minimum)
    if loss > budget:
        return SizingResult(Decimal(0), loss, margin, "REJECT_INSUFFICIENT_GRANULARITY")
    if margin > available_margin:
        return SizingResult(Decimal(0), loss, margin, "REJECT_INSUFFICIENT_MARGIN")
    low = 0
    high = int((volumes.maximum - volumes.minimum) // volumes.step)
    while low < high:
        middle = (low + high + 1) // 2
        volume = volumes.minimum + middle * volumes.step
        loss, margin = estimate(volume)
        if loss <= budget and margin <= available_margin:
            low = middle
        else:
            high = middle - 1
    volume = volumes.minimum + low * volumes.step
    loss, margin = estimate(volume)
    # Re-evaluation prevents accepting a changed oracle result after the search.
    if loss > budget or margin > available_margin:
        return SizingResult(Decimal(0), loss, margin, "REJECT_ESTIMATE_CHANGED")
    return SizingResult(volume, loss, margin, "ACCEPT")
