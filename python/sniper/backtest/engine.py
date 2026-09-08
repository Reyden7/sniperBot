"""Deterministic event-driven EUR/USD tick backtester."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Literal, Protocol

from sniper.backtest.execution_model import (
    BrokerSimulationConfig,
    CommissionModel,
    NoCommission,
    SimulatedBrokerProfile,
)
from sniper.backtest.metrics import BacktestMetrics, calculate_metrics
from sniper.backtest.slippage import NoSlippage, SlippageModel
from sniper.config import Model
from sniper.domain.trade import (
    ExitReason,
    MarketTick,
    OrderType,
    RejectedOrder,
    RejectionReason,
    Side,
    Trade,
)


class Strategy(Protocol):
    name: str

    def on_tick(self, engine: BacktestEngine, tick: MarketTick) -> None: ...


class AccountSnapshot(Model):
    balance: Decimal
    equity: Decimal
    used_margin: Decimal
    free_margin: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal


class BacktestResult(Model):
    mode: Literal["BACKTEST"] = "BACKTEST"
    live_trading_enabled: Literal[False] = False
    strategy: str
    symbol: str = "EURUSD"
    broker_profile: SimulatedBrokerProfile
    price_semantics: dict[str, str]
    ticks_processed: int
    metrics: BacktestMetrics
    account: AccountSnapshot
    trades: tuple[Trade, ...]
    rejections: tuple[RejectedOrder, ...]


@dataclass
class _Account:
    balance: Decimal
    equity: Decimal
    used_margin: Decimal = Decimal(0)
    free_margin: Decimal = Decimal(0)
    realized_pnl: Decimal = Decimal(0)
    unrealized_pnl: Decimal = Decimal(0)


@dataclass(frozen=True)
class _OrderRequest:
    sequence: int
    order_type: OrderType
    side: Side | None
    volume: Decimal
    stop_loss: Decimal | None
    take_profit: Decimal | None
    signal_timestamp: datetime
    requested_at: datetime
    requested_price: Decimal
    theoretical_execution_at: datetime
    exit_reason: ExitReason | None = None


@dataclass
class _OpenPosition:
    trade_id: str
    entry_sequence: int
    side: Side
    volume: Decimal
    stop_loss: Decimal
    take_profit: Decimal | None
    margin: Decimal
    signal_timestamp: datetime
    requested_at: datetime
    requested_price: Decimal
    theoretical_entry_at: datetime
    executed_at: datetime
    entry_quote_price: Decimal
    executed_price: Decimal
    entry_mid: Decimal
    spread_entry: Decimal
    entry_commission: Decimal
    balance_before: Decimal


class BacktestEngine:
    def __init__(
        self,
        config: BrokerSimulationConfig | None = None,
        *,
        slippage: SlippageModel | None = None,
        commission: CommissionModel | None = None,
    ) -> None:
        self.config = config or BrokerSimulationConfig()
        self.config.validate_latency()
        self.slippage = slippage or NoSlippage()
        self.commission = commission or NoCommission()
        capital = self.config.initial_capital
        self.account = _Account(balance=capital, equity=capital, free_margin=capital)
        self.position: _OpenPosition | None = None
        self.current_tick: MarketTick | None = None
        self.trades: list[Trade] = []
        self.rejections: list[RejectedOrder] = []
        self._pending: list[_OrderRequest] = []
        self._sequence = 0
        self._trade_sequence = 0
        self._equity_curve: list[Decimal] = [capital]
        self._last_timestamp: datetime | None = None
        self._ticks_processed = 0

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _reject(
        self, order_type: OrderType, requested_at: datetime, reason: RejectionReason, detail: str
    ) -> None:
        self.rejections.append(
            RejectedOrder(
                sequence=self._next_sequence(),
                order_type=order_type,
                requested_at=requested_at,
                reason=reason,
                detail=detail,
            )
        )

    def _require_tick(self) -> MarketTick:
        if self.current_tick is None:
            raise RuntimeError("orders can only be requested while processing a tick")
        return self.current_tick

    def open_market_position(
        self,
        side: Side,
        volume: Decimal,
        stop_loss: Decimal,
        take_profit: Decimal | None = None,
        *,
        signal_timestamp: datetime | None = None,
    ) -> int | None:
        """Queue a simulated market request using only the currently visible tick."""
        tick = self._require_tick()
        order_type = OrderType.MARKET_BUY if side == Side.BUY else OrderType.MARKET_SELL
        if self.position is not None or any(req.side is not None for req in self._pending):
            self._reject(
                order_type,
                tick.timestamp_utc,
                RejectionReason.MAX_OPEN_POSITIONS,
                "SNIPER V1 allows exactly one open or pending position",
            )
            return None
        requested = tick.entry_price(side)
        if volume < self.config.volumes.minimum:
            self._reject(
                order_type,
                tick.timestamp_utc,
                RejectionReason.VOLUME_TOO_SMALL,
                "volume is below broker minimum",
            )
            return None
        if volume > self.config.volumes.maximum or (
            (volume - self.config.volumes.minimum) % self.config.volumes.step != 0
        ):
            self._reject(
                order_type,
                tick.timestamp_utc,
                RejectionReason.INVALID_VOLUME_STEP,
                "volume is outside the broker grid",
            )
            return None
        if (
            stop_loss <= 0
            or (side == Side.BUY and stop_loss >= requested)
            or (side == Side.SELL and stop_loss <= requested)
        ):
            self._reject(
                order_type,
                tick.timestamp_utc,
                RejectionReason.RISK_LIMIT,
                "a protective stop must be on the loss side of the request price",
            )
            return None
        if take_profit is not None and (
            take_profit <= 0
            or (side == Side.BUY and take_profit <= requested)
            or (side == Side.SELL and take_profit >= requested)
        ):
            self._reject(
                order_type,
                tick.timestamp_utc,
                RejectionReason.RISK_LIMIT,
                "take profit must be on the gain side of the request price",
            )
            return None
        sequence = self._next_sequence()
        due = tick.timestamp_utc + timedelta(milliseconds=self.config.execution_latency_ms)
        self._pending.append(
            _OrderRequest(
                sequence=sequence,
                order_type=order_type,
                side=side,
                volume=volume,
                stop_loss=stop_loss,
                take_profit=take_profit,
                signal_timestamp=signal_timestamp or tick.timestamp_utc,
                requested_at=tick.timestamp_utc,
                requested_price=requested,
                theoretical_execution_at=due,
            )
        )
        return sequence

    def close_position(
        self,
        reason: ExitReason = ExitReason.MANUAL,
        *,
        signal_timestamp: datetime | None = None,
    ) -> int | None:
        tick = self._require_tick()
        if self.position is None:
            return None
        if any(req.order_type == OrderType.CLOSE for req in self._pending):
            return None
        sequence = self._next_sequence()
        due = tick.timestamp_utc + timedelta(milliseconds=self.config.execution_latency_ms)
        self._pending.append(
            _OrderRequest(
                sequence=sequence,
                order_type=OrderType.CLOSE,
                side=None,
                volume=self.position.volume,
                stop_loss=None,
                take_profit=None,
                signal_timestamp=signal_timestamp or tick.timestamp_utc,
                requested_at=tick.timestamp_utc,
                requested_price=tick.exit_price(self.position.side),
                theoretical_execution_at=due,
                exit_reason=reason,
            )
        )
        return sequence

    def _money(self, raw_quote_amount: Decimal, tick: MarketTick) -> Decimal:
        return self.config.money(raw_quote_amount, tick.mid)

    def _execute_open(self, request: _OrderRequest, tick: MarketTick) -> None:
        assert request.side is not None and request.stop_loss is not None
        side = request.side
        quote_price = tick.entry_price(side)
        actual = self.slippage.apply(quote_price, self.config.point, side, entry=True)
        if (side == Side.BUY and request.stop_loss >= actual) or (
            side == Side.SELL and request.stop_loss <= actual
        ):
            self._reject(
                request.order_type,
                request.requested_at,
                RejectionReason.RISK_LIMIT,
                "latency/slippage moved execution beyond the protective stop",
            )
            return
        margin = self.config.margin(request.volume)
        entry_commission = self.commission.per_side(request.volume)
        if margin + entry_commission > self.account.free_margin:
            self._reject(
                request.order_type,
                request.requested_at,
                RejectionReason.INSUFFICIENT_MARGIN,
                "required margin and entry commission exceed free margin",
            )
            return
        units = self.config.contract_size * request.volume
        stop_loss_quote = max(Decimal(0), -side.sign * (request.stop_loss - actual) * units)
        worst_slippage_quote = self.slippage.maximum_adverse_points * self.config.point * units
        estimated_loss = self._money(stop_loss_quote + worst_slippage_quote, tick)
        estimated_loss += 2 * self.commission.per_side(request.volume)
        if estimated_loss > self.account.equity * self.config.max_risk_pct / 100:
            self._reject(
                request.order_type,
                request.requested_at,
                RejectionReason.RISK_LIMIT,
                "stop, conservative slippage and round-trip commission exceed risk budget",
            )
            return
        self._trade_sequence += 1
        self.account.balance -= entry_commission
        self.account.used_margin = margin
        self.position = _OpenPosition(
            trade_id=f"T{self._trade_sequence:06d}",
            entry_sequence=request.sequence,
            side=side,
            volume=request.volume,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            margin=margin,
            signal_timestamp=request.signal_timestamp,
            requested_at=request.requested_at,
            requested_price=request.requested_price,
            theoretical_entry_at=request.theoretical_execution_at,
            executed_at=tick.timestamp_utc,
            entry_quote_price=quote_price,
            executed_price=actual,
            entry_mid=tick.mid,
            spread_entry=tick.spread,
            entry_commission=entry_commission,
            balance_before=self.account.balance + entry_commission,
        )
        self._mark_to_market(tick)

    def _execute_close(self, request: _OrderRequest, tick: MarketTick) -> None:
        if self.position is None:
            return
        self._close_now(
            tick=tick,
            requested_at=request.requested_at,
            requested_price=request.requested_price,
            theoretical_exit_at=request.theoretical_execution_at,
            reason=request.exit_reason or ExitReason.MANUAL,
            exit_sequence=request.sequence,
        )

    def _close_now(
        self,
        *,
        tick: MarketTick,
        requested_at: datetime,
        requested_price: Decimal,
        theoretical_exit_at: datetime,
        reason: ExitReason,
        exit_sequence: int,
    ) -> None:
        position = self.position
        if position is None:
            return
        quote_exit = tick.exit_price(position.side)
        actual_exit = self.slippage.apply(quote_exit, self.config.point, position.side, entry=False)
        units = self.config.contract_size * position.volume
        sign = position.side.sign
        conversion = tick.mid
        gross_quote = sign * (tick.mid - position.entry_mid) * units
        quoted_pnl = sign * (quote_exit - position.entry_quote_price) * units
        execution_pnl = sign * (actual_exit - position.executed_price) * units
        gross = self.config.money(gross_quote, conversion)
        spread_cost = self.config.money(gross_quote - quoted_pnl, conversion)
        slippage_cost = self.config.money(quoted_pnl - execution_pnl, conversion)
        exit_commission = self.commission.per_side(position.volume)
        total_commission = position.entry_commission + exit_commission
        execution_money = self.config.money(execution_pnl, conversion)
        net = execution_money - total_commission
        self.account.balance += execution_money - exit_commission
        self.account.realized_pnl += net
        self.account.used_margin = Decimal(0)
        self.account.unrealized_pnl = Decimal(0)
        self.account.equity = self.account.balance
        self.account.free_margin = self.account.equity
        self.trades.append(
            Trade(
                trade_id=position.trade_id,
                entry_sequence=position.entry_sequence,
                exit_sequence=exit_sequence,
                side=position.side,
                signal_timestamp=position.signal_timestamp,
                requested_at=position.requested_at,
                requested_price=position.requested_price,
                theoretical_entry_at=position.theoretical_entry_at,
                execution_quote_price=position.entry_quote_price,
                entry_slippage_price=position.executed_price - position.entry_quote_price,
                executed_at=position.executed_at,
                executed_price=position.executed_price,
                requested_exit_at=requested_at,
                requested_exit_price=requested_price,
                theoretical_exit_at=theoretical_exit_at,
                exit_execution_quote_price=quote_exit,
                exit_slippage_price=actual_exit - quote_exit,
                exited_at=tick.timestamp_utc,
                exit_price=actual_exit,
                volume=position.volume,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                margin=position.margin,
                gross_pnl=gross,
                execution_pnl=execution_money,
                spread_cost=spread_cost,
                commission=total_commission,
                slippage=slippage_cost,
                net_pnl=net,
                exit_reason=reason,
                spread_entry=position.spread_entry,
                spread_exit=tick.spread,
                balance_before=position.balance_before,
                balance_after=self.account.balance,
                execution_latency_ms=self.config.execution_latency_ms,
            )
        )
        self.position = None

    def _execute_due(self, tick: MarketTick) -> None:
        due = sorted(
            (
                request
                for request in self._pending
                if request.theoretical_execution_at <= tick.timestamp_utc
            ),
            key=lambda request: (request.theoretical_execution_at, request.sequence),
        )
        self._pending = [request for request in self._pending if request not in due]
        for request in due:
            if request.order_type == OrderType.CLOSE:
                self._execute_close(request, tick)
            elif self.position is None:
                self._execute_open(request, tick)
            else:
                self._reject(
                    request.order_type,
                    request.requested_at,
                    RejectionReason.MAX_OPEN_POSITIONS,
                    "a position was already open at execution time",
                )

    def _check_protection(self, tick: MarketTick) -> None:
        position = self.position
        if position is None:
            return
        exit_quote = tick.exit_price(position.side)
        stop_hit = (
            exit_quote <= position.stop_loss
            if position.side == Side.BUY
            else exit_quote >= position.stop_loss
        )
        target_hit = position.take_profit is not None and (
            exit_quote >= position.take_profit
            if position.side == Side.BUY
            else exit_quote <= position.take_profit
        )
        if stop_hit or target_hit:
            self._pending = [
                request for request in self._pending if request.order_type != OrderType.CLOSE
            ]
            self._close_now(
                tick=tick,
                requested_at=tick.timestamp_utc,
                requested_price=exit_quote,
                theoretical_exit_at=tick.timestamp_utc,
                reason=ExitReason.STOP_LOSS if stop_hit else ExitReason.TAKE_PROFIT,
                exit_sequence=self._next_sequence(),
            )

    def _mark_to_market(self, tick: MarketTick) -> None:
        if self.position is None:
            self.account.unrealized_pnl = Decimal(0)
        else:
            raw = (
                self.position.side.sign
                * (tick.exit_price(self.position.side) - self.position.executed_price)
                * self.config.contract_size
                * self.position.volume
            )
            self.account.unrealized_pnl = self._money(raw, tick)
        self.account.equity = self.account.balance + self.account.unrealized_pnl
        self.account.free_margin = self.account.equity - self.account.used_margin

    def _snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            balance=self.account.balance,
            equity=self.account.equity,
            used_margin=self.account.used_margin,
            free_margin=self.account.free_margin,
            realized_pnl=self.account.realized_pnl,
            unrealized_pnl=self.account.unrealized_pnl,
        )

    def run(self, ticks: Iterable[MarketTick], strategy: Strategy) -> BacktestResult:
        last_tick = None
        for tick in ticks:
            if self._last_timestamp is not None and tick.timestamp_utc < self._last_timestamp:
                raise ValueError("ticks are not in chronological order")
            self._last_timestamp = tick.timestamp_utc
            self.current_tick = tick
            last_tick = tick
            self._ticks_processed += 1
            self._execute_due(tick)
            self._check_protection(tick)
            strategy.on_tick(self, tick)
            self._execute_due(tick)
            self._check_protection(tick)
            self._mark_to_market(tick)
            self._equity_curve.append(self.account.equity)
        if last_tick is not None and self.position is not None:
            self._close_now(
                tick=last_tick,
                requested_at=last_tick.timestamp_utc,
                requested_price=last_tick.exit_price(self.position.side),
                theoretical_exit_at=last_tick.timestamp_utc,
                reason=ExitReason.END_OF_DATA,
                exit_sequence=self._next_sequence(),
            )
            self._equity_curve.append(self.account.equity)
        for request in self._pending:
            self._reject(
                request.order_type,
                request.requested_at,
                RejectionReason.NO_EXECUTION_TICK,
                "no tick exists at or after theoretical execution time",
            )
        self._pending.clear()
        self.current_tick = None
        metrics = calculate_metrics(
            self.config.initial_capital,
            self.account.equity,
            self.trades,
            self._equity_curve,
        )
        return BacktestResult(
            strategy=strategy.name,
            broker_profile=self.config.profile(),
            price_semantics={
                "requested_price": "executable-side quote visible on the signal/request tick",
                "execution_quote_price": "ASK for BUY or BID for SELL on the actual entry tick",
                "entry_slippage_price": "executed_price minus execution_quote_price",
                "executed_price": "entry quote on actual tick after slippage",
                "requested_exit_price": "executable-side quote visible on the close-request tick",
                "exit_execution_quote_price": "BID for BUY or ASK for SELL on actual exit tick",
                "exit_slippage_price": "exit_price minus exit_execution_quote_price",
                "exit_price": "exit quote on actual tick after slippage",
            },
            ticks_processed=self._ticks_processed,
            metrics=metrics,
            account=self._snapshot(),
            trades=tuple(self.trades),
            rejections=tuple(self.rejections),
        )


class DummyStrategy:
    """TEST ONLY: open once, then close after a fixed number of observed ticks."""

    name = "DummyStrategy(TEST_ONLY)"

    def __init__(
        self,
        *,
        side: Side = Side.BUY,
        volume: Decimal = Decimal("0.0001"),
        hold_ticks: int = 10,
        stop_points: Decimal = Decimal(50),
        take_profit_points: Decimal = Decimal(50),
    ) -> None:
        if hold_ticks < 1:
            raise ValueError("hold_ticks must be positive")
        self.side = side
        self.volume = volume
        self.hold_ticks = hold_ticks
        self.stop_points = stop_points
        self.take_profit_points = take_profit_points
        self._seen = 0
        self._requested = False
        self._close_requested = False

    def on_tick(self, engine: BacktestEngine, tick: MarketTick) -> None:
        self._seen += 1
        if not self._requested:
            entry = tick.entry_price(self.side)
            stop = entry - self.side.sign * self.stop_points * engine.config.point
            target = entry + self.side.sign * self.take_profit_points * engine.config.point
            engine.open_market_position(self.side, self.volume, stop, target)
            self._requested = True
        elif (
            engine.position is not None
            and not self._close_requested
            and self._seen >= self.hold_ticks
        ):
            engine.close_position()
            self._close_requested = True
