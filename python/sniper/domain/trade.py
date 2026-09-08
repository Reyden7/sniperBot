"""Auditable research-trading records. They cannot transmit live orders."""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from sniper.config import Model, NonNegative, Positive


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> Decimal:
        return Decimal(1) if self == Side.BUY else Decimal(-1)


class OrderType(StrEnum):
    MARKET_BUY = "MARKET_BUY"
    MARKET_SELL = "MARKET_SELL"
    CLOSE = "CLOSE"


class ExitReason(StrEnum):
    MANUAL = "MANUAL"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    END_OF_DATA = "END_OF_DATA"


class RejectionReason(StrEnum):
    VOLUME_TOO_SMALL = "REJECT_VOLUME_TOO_SMALL"
    INVALID_VOLUME_STEP = "REJECT_INVALID_VOLUME_STEP"
    INSUFFICIENT_MARGIN = "REJECT_INSUFFICIENT_MARGIN"
    RISK_LIMIT = "REJECT_RISK_LIMIT"
    MAX_OPEN_POSITIONS = "REJECT_MAX_OPEN_POSITIONS"
    NO_EXECUTION_TICK = "REJECT_NO_EXECUTION_TICK"


class MarketTick(Model):
    timestamp_utc: datetime
    bid: Positive
    ask: Positive

    @model_validator(mode="after")
    def validate_tick(self) -> Self:
        if self.timestamp_utc.utcoffset() != timedelta(0):
            raise ValueError("backtest ticks must use canonical UTC")
        if self.ask < self.bid:
            raise ValueError("crossed Bid/Ask tick")
        return self

    @property
    def spread(self) -> Decimal:
        return self.ask - self.bid

    @property
    def mid(self) -> Decimal:
        return (self.ask + self.bid) / 2

    def entry_price(self, side: Side) -> Decimal:
        return self.ask if side == Side.BUY else self.bid

    def exit_price(self, side: Side) -> Decimal:
        return self.bid if side == Side.BUY else self.ask


class RejectedOrder(Model):
    sequence: int = Field(ge=0)
    order_type: OrderType
    requested_at: datetime
    reason: RejectionReason
    detail: str


class Trade(Model):
    trade_id: str
    entry_sequence: int = Field(ge=1)
    exit_sequence: int = Field(ge=1)
    side: Side
    signal_timestamp: datetime
    requested_at: datetime
    requested_price: Positive
    theoretical_entry_at: datetime
    execution_quote_price: Positive
    entry_slippage_price: Decimal
    executed_at: datetime
    executed_price: Positive
    requested_exit_at: datetime
    requested_exit_price: Positive
    theoretical_exit_at: datetime
    exit_execution_quote_price: Positive
    exit_slippage_price: Decimal
    exited_at: datetime
    exit_price: Positive
    volume: Positive
    stop_loss: Positive
    take_profit: Positive | None
    margin: NonNegative
    gross_pnl: Decimal
    execution_pnl: Decimal
    spread_cost: Decimal
    commission: NonNegative
    slippage: Decimal
    net_pnl: Decimal
    exit_reason: ExitReason
    spread_entry: NonNegative
    spread_exit: NonNegative
    balance_before: Decimal
    balance_after: Decimal
    execution_latency_ms: int = Field(ge=0)
