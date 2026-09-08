"""Canonical bars built incrementally from ticks, then from M1 bars."""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Literal, Self

from pydantic import Field, model_validator

from sniper.config import Model, NonNegative, Positive
from sniper.data.time import as_utc

Timeframe = Literal["M1", "M5", "M15"]


class Bar(Model):
    symbol: str = "EURUSD"
    timeframe: Timeframe
    open_time_utc: datetime
    open: Positive
    high: Positive
    low: Positive
    close: Positive
    tick_volume: int = Field(ge=1)
    real_volume: NonNegative = Decimal(0)
    spread: NonNegative
    is_complete: bool = False

    @model_validator(mode="after")
    def validate_bar(self) -> Self:
        if self.symbol != "EURUSD":
            raise ValueError("Phase B supports EURUSD only")
        utc = as_utc(self.open_time_utc)
        if utc != self.open_time_utc or self.open_time_utc.utcoffset() != timedelta(0):
            raise ValueError("open_time_utc must use UTC explicitly")
        minutes = {"M1": 1, "M5": 5, "M15": 15}[self.timeframe]
        if utc.second or utc.microsecond or utc.minute % minutes:
            raise ValueError("bar open time is not aligned to its timeframe")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("invalid OHLC envelope")
        return self
