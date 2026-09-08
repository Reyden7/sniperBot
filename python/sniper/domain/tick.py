"""Validated EUR/USD market ticks with UTC as their canonical instant."""

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any, Self

from pydantic import Field, model_validator

from sniper.config import Model, NonNegative, Positive
from sniper.data.time import as_utc

Finite = Annotated[Decimal, Field(allow_inf_nan=False)]


class Tick(Model):
    symbol: str = "EURUSD"
    timestamp_utc: datetime
    timestamp_server: datetime | None = None
    timestamp_local: datetime
    source_time_msc: int = Field(ge=0)
    bid: Positive
    ask: Positive
    last: NonNegative = Decimal(0)
    volume: NonNegative = Decimal(0)
    volume_real: NonNegative = Decimal(0)
    flags: int = Field(ge=0)
    point: Positive
    time_basis: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_tick(self) -> Self:
        if self.symbol != "EURUSD":
            raise ValueError("Phase B supports EURUSD only")
        if self.ask < self.bid:
            raise ValueError("crossed Bid/Ask tick")
        utc = as_utc(self.timestamp_utc)
        if utc != self.timestamp_utc or self.timestamp_utc.tzname() != "UTC":
            raise ValueError("timestamp_utc must use UTC explicitly")
        for representation in (self.timestamp_server, self.timestamp_local):
            if representation is not None and as_utc(representation) != utc:
                raise ValueError("timestamp representations refer to different instants")
        return self

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal:
        return self.ask - self.bid

    @property
    def spread_points(self) -> Decimal:
        return self.spread / self.point

    @property
    def identity(self) -> tuple[Any, ...]:
        """Exact source identity; timestamps alone are not unique in MT5."""
        return (
            self.source_time_msc,
            self.bid,
            self.ask,
            self.last,
            self.volume,
            self.volume_real,
            self.flags,
        )

    def parquet_row(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp_utc": self.timestamp_utc,
            "timestamp_server": self.timestamp_server,
            "timestamp_local": self.timestamp_local,
            "source_time_msc": self.source_time_msc,
            "bid": float(self.bid),
            "ask": float(self.ask),
            "last": float(self.last),
            "volume": float(self.volume),
            "volume_real": float(self.volume_real),
            "flags": self.flags,
            "mid": float(self.mid),
            "spread": float(self.spread),
            "spread_points": float(self.spread_points),
            "time_basis": self.time_basis,
        }
