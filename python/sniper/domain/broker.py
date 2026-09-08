"""Sanitized terminal observations; no login, server, name or credentials."""

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Self

from pydantic import Field, model_validator

from sniper.config import Model, NonNegative, Positive
from sniper.data.time import as_utc

Finite = Annotated[Decimal, Field(allow_inf_nan=False)]


class VolumeConstraints(Model):
    minimum: Positive
    step: Positive
    maximum: Positive

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.maximum < self.minimum:
            raise ValueError("maximum volume is smaller than minimum volume")
        return self

    @property
    def nano_lot_compliant(self) -> bool:
        return self.minimum <= Decimal("0.0001") and self.step <= Decimal("0.0001")


class AccountSnapshot(Model):
    currency: str = Field(min_length=1)
    equity: Finite
    free_margin: Finite
    trade_allowed: bool
    expert_allowed: bool


class SymbolSnapshot(Model):
    name: str
    currency_base: str
    currency_profit: str
    volumes: VolumeConstraints
    point: Positive
    digits: int = Field(ge=0, le=12)
    tick_size: Positive
    tick_value: Positive
    contract_size: Positive
    stops_level_points: NonNegative
    full_trading_allowed: bool

    @property
    def pip_size(self) -> Decimal:
        # Display convention for EUR/USD only. Never used as a monetary pip value.
        if (self.currency_base, self.currency_profit) != ("EUR", "USD"):
            raise ValueError("SNIPER V1 accepts EUR/USD only")
        if self.digits not in (4, 5) or self.point != Decimal(10) ** -self.digits:
            raise ValueError("unsupported EUR/USD quote precision")
        return self.point * (10 if self.digits == 5 else 1)


class Quote(Model):
    bid: Positive
    ask: Positive
    timestamp_utc: datetime
    timestamp_server: datetime | None = None
    timestamp_local: datetime | None = None
    source_time_msc: int | None = None
    time_basis: str = "MT5 Python API documented UTC contract"

    @model_validator(mode="after")
    def validate_quote(self) -> Self:
        if self.ask < self.bid:
            raise ValueError("crossed Bid/Ask quote")
        if self.timestamp_utc.utcoffset() is None:
            raise ValueError("quote timestamp must be timezone-aware")
        for representation in (self.timestamp_server, self.timestamp_local):
            if representation is not None and as_utc(representation) != as_utc(self.timestamp_utc):
                raise ValueError("timestamp representations refer to different instants")
        return self


class BrokerSnapshot(Model):
    account: AccountSnapshot
    symbol: SymbolSnapshot
    quote: Quote
