"""Binance Spot research risk and daily-performance engines (no execution)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum

from sniper.binance.filters import floor_to_step
from sniper.binance.models import SymbolRules

TARGET_DAILY_NET_RETURN = Decimal("1.00")


class DailyState(StrEnum):
    ACTIVE = "ACTIVE"
    DAILY_TARGET_REACHED = "DAILY_TARGET_REACHED"
    DAILY_LOSS_LIMIT_REACHED = "DAILY_LOSS_LIMIT_REACHED"
    COOLDOWN = "COOLDOWN"
    DISABLED = "DISABLED"


@dataclass(frozen=True)
class RiskDecision:
    accepted: bool
    quantity: Decimal
    notional: Decimal
    estimated_loss: Decimal
    risk_pct: Decimal
    reason: str


class BinanceRiskEngine:
    """Only authority allowed to size hypothetical Spot research positions."""

    def __init__(
        self,
        *,
        normal_risk_pct: Decimal = Decimal("0.20"),
        maximum_risk_pct: Decimal = Decimal("0.35"),
    ):
        if not Decimal(0) < normal_risk_pct <= maximum_risk_pct:
            raise ValueError("invalid Binance risk percentages")
        self.normal_risk_pct = normal_risk_pct
        self.maximum_risk_pct = maximum_risk_pct

    def size(
        self,
        *,
        equity: Decimal,
        available_cash: Decimal,
        entry_price: Decimal,
        stop_price: Decimal,
        rules: SymbolRules,
        round_trip_fee_rate: Decimal,
        round_trip_slippage_rate: Decimal,
    ) -> RiskDecision:
        if equity <= 0 or available_cash <= 0 or not Decimal(0) < stop_price < entry_price:
            return RiskDecision(False, Decimal(0), Decimal(0), Decimal(0), Decimal(0), "INVALID")
        loss_rate = (
            (entry_price - stop_price) / entry_price
            + round_trip_fee_rate
            + round_trip_slippage_rate
        )
        risk_budget = equity * self.normal_risk_pct / Decimal(100)
        quantity_by_risk = risk_budget / (entry_price * loss_rate)
        quantity_by_cash = available_cash / entry_price
        quantity = floor_to_step(
            min(quantity_by_risk, quantity_by_cash, rules.market_maximum_quantity),
            rules.market_quantity_step,
        )
        notional = quantity * entry_price
        if quantity < rules.market_minimum_quantity or notional < rules.minimum_notional:
            return RiskDecision(
                False,
                Decimal(0),
                notional,
                quantity * entry_price * loss_rate,
                Decimal(0),
                "MINIMUM_ORDER_INCOMPATIBLE_WITH_RISK_BUDGET",
            )
        estimated_loss = quantity * entry_price * loss_rate
        risk_pct = estimated_loss / equity * Decimal(100)
        if risk_pct > self.maximum_risk_pct:
            return RiskDecision(
                False, quantity, notional, estimated_loss, risk_pct, "MAXIMUM_RISK_EXCEEDED"
            )
        return RiskDecision(True, quantity, notional, estimated_loss, risk_pct, "ACCEPT")


class DailyPerformanceEngine:
    """Enforce UTC daily target, loss limit, trade count and loss cooldown."""

    def __init__(
        self,
        *,
        target_return_pct: Decimal = TARGET_DAILY_NET_RETURN,
        loss_limit_pct: Decimal = Decimal("-1.00"),
        maximum_trades: int = 6,
        maximum_consecutive_losses: int = 3,
    ):
        self.target_return_pct = target_return_pct
        self.loss_limit_pct = loss_limit_pct
        self.maximum_trades = maximum_trades
        self.maximum_consecutive_losses = maximum_consecutive_losses
        self.day: date | None = None
        self.start_equity = Decimal(0)
        self.realized_pnl = Decimal(0)
        self.fees = Decimal(0)
        self.trades_count = 0
        self.wins = 0
        self.losses = 0
        self.consecutive_losses = 0

    def synchronize(self, day: date, equity: Decimal) -> None:
        if self.day == day:
            return
        self.day = day
        self.start_equity = equity
        self.realized_pnl = Decimal(0)
        self.fees = Decimal(0)
        self.trades_count = 0
        self.wins = 0
        self.losses = 0
        self.consecutive_losses = 0

    @property
    def net_return_pct(self) -> Decimal:
        if self.start_equity <= 0:
            return Decimal(0)
        return self.realized_pnl / self.start_equity * Decimal(100)

    @property
    def state(self) -> DailyState:
        if self.net_return_pct >= self.target_return_pct:
            return DailyState.DAILY_TARGET_REACHED
        if self.net_return_pct <= self.loss_limit_pct:
            return DailyState.DAILY_LOSS_LIMIT_REACHED
        if self.consecutive_losses >= self.maximum_consecutive_losses:
            return DailyState.COOLDOWN
        if self.trades_count >= self.maximum_trades:
            return DailyState.DISABLED
        return DailyState.ACTIVE

    def record(self, net_pnl: Decimal, fee: Decimal) -> None:
        self.realized_pnl += net_pnl
        self.fees += fee
        self.trades_count += 1
        if net_pnl > 0:
            self.wins += 1
            self.consecutive_losses = 0
        else:
            self.losses += 1
            self.consecutive_losses += 1
