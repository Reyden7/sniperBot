from datetime import UTC, datetime, timedelta
from decimal import Decimal as D

import pytest
from hypothesis import given
from hypothesis import strategies as st
from sniper.backtest.engine import BacktestEngine
from sniper.backtest.execution_model import (
    BrokerSimulationConfig,
    MinimumCommission,
    NoCommission,
    PerLotCommission,
)
from sniper.backtest.slippage import FixedSlippage, RandomSlippage
from sniper.domain.broker import VolumeConstraints
from sniper.domain.trade import ExitReason, MarketTick, RejectionReason, Side

START = datetime(2026, 9, 8, 10, tzinfo=UTC)


def market_tick(milliseconds: int, bid: int | str, ask: int | str) -> MarketTick:
    return MarketTick(
        timestamp_utc=START + timedelta(milliseconds=milliseconds),
        bid=D(str(bid)),
        ask=D(str(ask)),
    )


def broker(
    *,
    capital: D = D(10000),
    latency: int = 0,
    margin_per_lot: D = D(10),
    max_risk_pct: D = D(100),
    minimum: D = D(1),
    step: D = D(1),
) -> BrokerSimulationConfig:
    return BrokerSimulationConfig(
        initial_capital=capital,
        volumes=VolumeConstraints(minimum=minimum, step=step, maximum=D(10)),
        contract_size=D(100),
        point=D(1),
        margin_per_lot=margin_per_lot,
        max_risk_pct=max_risk_pct,
        execution_latency_ms=latency,
        convert_usd_pnl_to_eur=False,
    )


class OpenCloseStrategy:
    name = "TEST_OPEN_CLOSE"

    def __init__(self, side=Side.BUY, volume=D(1)):
        self.side = side
        self.volume = volume
        self.seen = 0

    def on_tick(self, engine, tick):
        self.seen += 1
        if self.seen == 1:
            stop = D(50) if self.side == Side.BUY else D(150)
            target = D(150) if self.side == Side.BUY else D(50)
            engine.open_market_position(self.side, self.volume, stop, target)
        elif engine.position is not None:
            engine.close_position()


class OpenOnlyStrategy:
    name = "TEST_OPEN_ONLY"

    def __init__(self, side, stop, target=None, volume=D(1)):
        self.side = side
        self.stop = D(stop)
        self.target = D(target) if target is not None else None
        self.volume = volume
        self.done = False

    def on_tick(self, engine, tick):
        if not self.done:
            engine.open_market_position(self.side, self.volume, self.stop, self.target)
            self.done = True


class DoubleOpenStrategy:
    name = "TEST_DOUBLE_OPEN"

    def on_tick(self, engine, tick):
        if engine.position is None:
            engine.open_market_position(Side.BUY, D(1), D(50), D(150))
            engine.open_market_position(Side.SELL, D(1), D(150), D(50))


def test_buy_uses_ask_to_open_and_bid_to_close_and_pays_spread():
    result = BacktestEngine(broker()).run(
        [market_tick(0, 100, 102), market_tick(1, 105, 107)], OpenCloseStrategy(Side.BUY)
    )
    trade = result.trades[0]
    assert trade.executed_price == 102
    assert trade.exit_price == 105
    assert trade.gross_pnl == 500
    assert trade.spread_cost == 200
    assert trade.slippage == 0
    assert trade.net_pnl == 300


def test_sell_uses_bid_to_open_and_ask_to_close_and_pays_spread():
    result = BacktestEngine(broker()).run(
        [market_tick(0, 100, 102), market_tick(1, 93, 95)], OpenCloseStrategy(Side.SELL)
    )
    trade = result.trades[0]
    assert trade.executed_price == 100
    assert trade.exit_price == 95
    assert trade.gross_pnl == 700
    assert trade.spread_cost == 200
    assert trade.net_pnl == 500


def test_spread_slippage_commission_and_balance_reconcile_exactly():
    engine = BacktestEngine(
        broker(), slippage=FixedSlippage(D(1)), commission=PerLotCommission(D(10))
    )
    result = engine.run([market_tick(0, 100, 102), market_tick(1, 105, 107)], OpenCloseStrategy())
    trade = result.trades[0]
    assert trade.execution_quote_price == 102
    assert trade.entry_slippage_price == 1
    assert trade.executed_price == 103
    assert trade.exit_execution_quote_price == 105
    assert trade.exit_slippage_price == -1
    assert trade.exit_price == 104
    assert trade.gross_pnl == 500
    assert trade.execution_pnl == 100
    assert trade.spread_cost == 200
    assert trade.slippage == 200
    assert trade.commission == 20
    assert trade.net_pnl == 80
    assert trade.gross_pnl - trade.spread_cost - trade.slippage - trade.commission == 80
    direct_execution_pnl = (
        trade.side.sign
        * (trade.exit_price - trade.executed_price)
        * engine.config.contract_size
        * trade.volume
    )
    assert direct_execution_pnl == trade.execution_pnl
    assert direct_execution_pnl - trade.commission == trade.net_pnl
    assert trade.balance_after == trade.balance_before + trade.net_pnl == 10080
    assert result.account.balance == result.account.equity == 10080
    assert result.account.used_margin == result.account.unrealized_pnl == 0
    assert result.account.free_margin == 10080


@pytest.mark.parametrize("latency", [0, 25, 50, 100, 250])
def test_latency_uses_first_tick_at_or_after_theoretical_execution(latency):
    ticks = [
        market_tick(0, 100, 102),
        market_tick(max(0, latency - 1), 101, 103),
        market_tick(latency, 102, 104),
    ]
    result = BacktestEngine(broker(latency=latency)).run(ticks, OpenOnlyStrategy(Side.BUY, 50))
    trade = result.trades[0]
    assert trade.theoretical_entry_at == START + timedelta(milliseconds=latency)
    assert trade.executed_at == START + timedelta(milliseconds=latency)
    assert trade.execution_quote_price == (102 if latency == 0 else 104)


def test_latency_skips_nonexistent_in_between_tick_without_lookahead():
    result = BacktestEngine(broker(latency=25)).run(
        [market_tick(0, 100, 102), market_tick(20, 101, 103), market_tick(30, 102, 104)],
        OpenOnlyStrategy(Side.BUY, 50),
    )
    trade = result.trades[0]
    assert trade.requested_price == 102
    assert trade.theoretical_entry_at == START + timedelta(milliseconds=25)
    assert trade.executed_at == START + timedelta(milliseconds=30)
    assert trade.executed_price == 104


@pytest.mark.parametrize(
    "side,stop,target,trigger_bid,trigger_ask,reason,expected_exit",
    [
        (Side.BUY, 99, 120, 99, 101, ExitReason.STOP_LOSS, 99),
        (Side.BUY, 90, 105, 105, 107, ExitReason.TAKE_PROFIT, 105),
        (Side.SELL, 103, 80, 101, 103, ExitReason.STOP_LOSS, 103),
        (Side.SELL, 120, 97, 95, 97, ExitReason.TAKE_PROFIT, 97),
    ],
)
def test_stops_and_targets_use_the_exit_side_of_each_tick(
    side, stop, target, trigger_bid, trigger_ask, reason, expected_exit
):
    result = BacktestEngine(broker()).run(
        [market_tick(0, 100, 102), market_tick(1, trigger_bid, trigger_ask)],
        OpenOnlyStrategy(side, stop, target),
    )
    trade = result.trades[0]
    assert trade.exit_reason == reason
    assert trade.exit_price == expected_exit


@pytest.mark.parametrize(
    "config,volume,commission,reason",
    [
        (
            broker(minimum=D(1), step=D(1)),
            D("0.5"),
            NoCommission(),
            RejectionReason.VOLUME_TOO_SMALL,
        ),
        (
            broker(minimum=D(1), step=D("0.5")),
            D("1.25"),
            NoCommission(),
            RejectionReason.INVALID_VOLUME_STEP,
        ),
        (
            broker(capital=D(10), margin_per_lot=D(20)),
            D(1),
            NoCommission(),
            RejectionReason.INSUFFICIENT_MARGIN,
        ),
        (
            broker(capital=D(10), margin_per_lot=D(1), max_risk_pct=D("0.5")),
            D(1),
            MinimumCommission(D(0), D("0.10")),
            RejectionReason.RISK_LIMIT,
        ),
    ],
)
def test_broker_rejections_are_explicit(config, volume, commission, reason):
    result = BacktestEngine(config, commission=commission).run(
        [market_tick(0, 100, 102)], OpenOnlyStrategy(Side.BUY, 99, volume=volume)
    )
    assert result.trades == ()
    assert result.rejections[0].reason == reason


def test_pending_request_without_future_tick_is_rejected_not_backfilled():
    result = BacktestEngine(broker(latency=250)).run(
        [market_tick(0, 100, 102)], OpenOnlyStrategy(Side.BUY, 50)
    )
    assert result.trades == ()
    assert result.rejections[0].reason == RejectionReason.NO_EXECUTION_TICK


def test_v1_never_allows_more_than_one_open_or_pending_position():
    result = BacktestEngine(broker()).run([market_tick(0, 100, 102)], DoubleOpenStrategy())
    assert len(result.trades) == 1
    assert result.rejections[0].reason == RejectionReason.MAX_OPEN_POSITIONS


def test_out_of_order_input_is_rejected():
    with pytest.raises(ValueError, match="chronological"):
        BacktestEngine(broker()).run(
            [market_tick(1, 100, 102), market_tick(0, 100, 102)], OpenCloseStrategy()
        )


def test_same_random_seed_produces_identical_trade():
    ticks = [market_tick(0, 100, 102), market_tick(1, 105, 107)]
    first = BacktestEngine(broker(), slippage=RandomSlippage(3, 99)).run(ticks, OpenCloseStrategy())
    second = BacktestEngine(broker(), slippage=RandomSlippage(3, 99)).run(
        ticks, OpenCloseStrategy()
    )
    assert first.model_dump() == second.model_dump()


@given(
    move=st.integers(-20, 20),
    entry_spread=st.integers(0, 5),
    exit_spread=st.integers(0, 5),
    slippage_points=st.integers(-2, 2),
)
def test_financial_decomposition_invariant(move, entry_spread, exit_spread, slippage_points):
    entry = market_tick(0, 100, 100 + entry_spread)
    exit_tick = market_tick(1, 100 + move, 100 + move + exit_spread)
    result = BacktestEngine(broker(), slippage=FixedSlippage(D(slippage_points))).run(
        [entry, exit_tick], OpenCloseStrategy()
    )
    trade = result.trades[0]
    assert trade.net_pnl == trade.gross_pnl - trade.spread_cost - trade.slippage
    assert trade.balance_after == trade.balance_before + trade.net_pnl
