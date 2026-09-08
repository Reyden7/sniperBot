from datetime import UTC, datetime
from decimal import Decimal as D

import polars as pl
from sniper.backtest.engine import BacktestEngine
from sniper.backtest.execution_model import BrokerSimulationConfig, PerLotCommission
from sniper.backtest.slippage import FixedSlippage
from sniper.data.backtest_source import load_parquet_ticks
from sniper.domain.broker import VolumeConstraints
from sniper.domain.trade import Side

from tests.unit.test_backtest_engine import OpenCloseStrategy


def write_partition(root, day, rows):
    path = root / "ticks" / "symbol=EURUSD" / f"date={day}" / "part-000.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(path)


def test_two_partition_golden_backtest_is_chronological_and_exact(tmp_path):
    write_partition(
        tmp_path,
        "2026-09-08",
        [
            {
                "timestamp_utc": datetime(2026, 9, 8, 23, 59, 59, 900000, tzinfo=UTC),
                "source_time_msc": 2,
                "bid": 101.0,
                "ask": 103.0,
            },
            {
                "timestamp_utc": datetime(2026, 9, 8, 23, 59, 59, 800000, tzinfo=UTC),
                "source_time_msc": 1,
                "bid": 100.0,
                "ask": 102.0,
            },
        ],
    )
    write_partition(
        tmp_path,
        "2026-09-09",
        [
            {
                "timestamp_utc": datetime(2026, 9, 9, 0, 0, 0, tzinfo=UTC),
                "source_time_msc": 3,
                "bid": 105.0,
                "ask": 107.0,
            }
        ],
    )
    ticks = load_parquet_ticks(
        tmp_path,
        "EURUSD",
        datetime(2026, 9, 8, 23, 59, 59, 800000, tzinfo=UTC),
        datetime(2026, 9, 9, 0, 0, 1, tzinfo=UTC),
    )
    assert [tick.bid for tick in ticks] == [D(100), D(101), D(105)]
    config = BrokerSimulationConfig(
        initial_capital=D(10000),
        volumes=VolumeConstraints(minimum=D(1), step=D(1), maximum=D(1)),
        contract_size=D(100),
        point=D(1),
        margin_per_lot=D(10),
        max_risk_pct=D(100),
        convert_usd_pnl_to_eur=False,
    )
    result = BacktestEngine(
        config, slippage=FixedSlippage(D(1)), commission=PerLotCommission(D(10))
    ).run([ticks[0], ticks[-1]], OpenCloseStrategy(Side.BUY))
    trade = result.trades[0]
    assert trade.model_dump(mode="json") == {
        "trade_id": "T000001",
        "entry_sequence": 1,
        "exit_sequence": 2,
        "side": "BUY",
        "signal_timestamp": "2026-09-08T23:59:59.800000Z",
        "requested_at": "2026-09-08T23:59:59.800000Z",
        "requested_price": "102.0",
        "theoretical_entry_at": "2026-09-08T23:59:59.800000Z",
        "execution_quote_price": "102.0",
        "entry_slippage_price": "1.0",
        "executed_at": "2026-09-08T23:59:59.800000Z",
        "executed_price": "103.0",
        "requested_exit_at": "2026-09-09T00:00:00Z",
        "requested_exit_price": "105.0",
        "theoretical_exit_at": "2026-09-09T00:00:00Z",
        "exit_execution_quote_price": "105.0",
        "exit_slippage_price": "-1.0",
        "exited_at": "2026-09-09T00:00:00Z",
        "exit_price": "104.0",
        "volume": "1",
        "stop_loss": "50",
        "take_profit": "150",
        "margin": "10",
        "gross_pnl": "500.0",
        "execution_pnl": "100.0",
        "spread_cost": "200.0",
        "commission": "20",
        "slippage": "200.0",
        "net_pnl": "80.0",
        "exit_reason": "MANUAL",
        "spread_entry": "2.0",
        "spread_exit": "2.0",
        "balance_before": "10000",
        "balance_after": "10080.0",
        "execution_latency_ms": 0,
    }
