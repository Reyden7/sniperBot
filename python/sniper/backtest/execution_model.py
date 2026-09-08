"""Broker constraints and explicit commission models for research execution."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Protocol

from pydantic import Field

from sniper.config import Model, Positive
from sniper.domain.broker import VolumeConstraints


class CommissionModel(Protocol):
    def per_side(self, volume: Decimal) -> Decimal: ...


@dataclass(frozen=True)
class NoCommission:
    def per_side(self, volume: Decimal) -> Decimal:
        return Decimal(0)


@dataclass(frozen=True)
class PerLotCommission:
    per_lot_per_side: Decimal

    def __post_init__(self) -> None:
        if not self.per_lot_per_side.is_finite() or self.per_lot_per_side < 0:
            raise ValueError("commission must be finite and nonnegative")

    def per_side(self, volume: Decimal) -> Decimal:
        return self.per_lot_per_side * volume


@dataclass(frozen=True)
class MinimumCommission:
    per_lot_per_side: Decimal
    minimum_per_side: Decimal

    def __post_init__(self) -> None:
        if any(
            not value.is_finite() or value < 0
            for value in (self.per_lot_per_side, self.minimum_per_side)
        ):
            raise ValueError("commission must be finite and nonnegative")

    def per_side(self, volume: Decimal) -> Decimal:
        return max(self.per_lot_per_side * volume, self.minimum_per_side)


class BrokerSimulationConfig(Model):
    profile_name: str = "SIMULATED_NANO_RESEARCH_PROFILE"
    initial_capital: Positive = Decimal("10.00")
    volumes: VolumeConstraints = Field(
        default_factory=lambda: VolumeConstraints(
            minimum=Decimal("0.0001"),
            step=Decimal("0.0001"),
            maximum=Decimal("1"),
        )
    )
    contract_size: Positive = Decimal("100000")
    point: Positive = Decimal("0.00001")
    margin_per_lot: Positive = Decimal("3333.333333333333")
    max_risk_pct: Decimal = Field(Decimal("0.50"), gt=0, le=100, allow_inf_nan=False)
    execution_latency_ms: int = Field(0, ge=0)
    max_open_positions: int = Field(1, ge=1, le=1)
    convert_usd_pnl_to_eur: bool = True

    def money(self, quote_currency_amount: Decimal, conversion_price: Decimal) -> Decimal:
        if self.convert_usd_pnl_to_eur:
            if conversion_price <= 0:
                raise ValueError("conversion price must be positive")
            return quote_currency_amount / conversion_price
        return quote_currency_amount

    def margin(self, volume: Decimal) -> Decimal:
        return self.margin_per_lot * volume

    def profile(self) -> SimulatedBrokerProfile:
        return SimulatedBrokerProfile(
            name=self.profile_name,
            effective_leverage=self.contract_size / self.margin_per_lot,
            volume_minimum=self.volumes.minimum,
            volume_step=self.volumes.step,
            volume_maximum=self.volumes.maximum,
            contract_size=self.contract_size,
            point=self.point,
            margin_per_lot=self.margin_per_lot,
            max_risk_pct=self.max_risk_pct,
        )

    def validate_latency(self) -> None:
        if self.execution_latency_ms not in (0, 25, 50, 100, 250):
            raise ValueError("execution latency must be one of 0, 25, 50, 100, 250 ms")


class SimulatedBrokerProfile(Model):
    name: str
    kind: Literal["SIMULATED"] = "SIMULATED"
    is_real_broker_capability: Literal[False] = False
    disclaimer: str = (
        "Research assumptions only; this is not a capability of MetaQuotes-Demo "
        "or evidence that any real broker supports nano-lots."
    )
    effective_leverage: Positive
    volume_minimum: Positive
    volume_step: Positive
    volume_maximum: Positive
    contract_size: Positive
    point: Positive
    margin_per_lot: Positive
    max_risk_pct: Positive
