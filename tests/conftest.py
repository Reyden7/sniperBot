"""Synthetic MT5 data only. No tests access the real terminal or SDK."""

import os
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest
from sniper.config import BrokerCheckConfig, CommissionModel

NOW = datetime(2026, 9, 8, 10, 0, tzinfo=UTC)
D = Decimal


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("SNIPER_") or key == "LIVE_TRADING_ENABLED":
            monkeypatch.delenv(key)
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    monkeypatch.chdir(tmp_path)  # Ignore a developer's real .env.


class FakeMT5:
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    SYMBOL_TRADE_MODE_FULL = 4
    COPY_TICKS_ALL = 0

    def __init__(self):
        self.calls = []
        self.fail = None
        self.terminal = SimpleNamespace(connected=True)
        self.account = SimpleNamespace(
            currency="EUR",
            equity=10.0,
            margin_free=10.0,
            trade_allowed=True,
            trade_expert=True,
            login=999999,
            name="PRIVATE_IDENTITY",
            server="PRIVATE_SERVER",
        )
        self.info = SimpleNamespace(
            visible=True,
            currency_base="EUR",
            currency_profit="USD",
            volume_min=0.0001,
            volume_step=0.0001,
            volume_max=1.0,
            point=0.00001,
            digits=5,
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
            trade_contract_size=100000.0,
            trade_stops_level=0,
            trade_mode=4,
        )
        self.tick = SimpleNamespace(bid=1.10000, ask=1.10002, time_msc=int(NOW.timestamp() * 1000))
        self.margin_factor = D(1000)
        self.sell_factor = D(1)
        self.profit_override = None
        self.margin_override = None
        self.tick_rows = []

    def __getattr__(self, name):
        raise AssertionError(f"Unexpected SDK access: {name}")

    def initialize(self, *args, **kwargs):
        self.calls.append(("initialize", args, kwargs))
        return self.fail != "initialize"

    def shutdown(self):
        self.calls.append(("shutdown",))

    def last_error(self):
        return (-6, "PRIVATE_IDENTITY PRIVATE_SERVER")

    def terminal_info(self):
        self.calls.append(("terminal_info",))
        return None if self.fail == "terminal_info" else self.terminal

    def account_info(self):
        self.calls.append(("account_info",))
        return None if self.fail == "account_info" else self.account

    def symbol_info(self, symbol):
        self.calls.append(("symbol_info", symbol))
        return None if self.fail == "symbol_info" else self.info

    def symbol_select(self, symbol, enabled):
        self.calls.append(("symbol_select", symbol, enabled))
        return self.fail != "symbol_select"

    def symbol_info_tick(self, symbol):
        self.calls.append(("symbol_info_tick", symbol))
        return None if self.fail == "symbol_info_tick" else self.tick

    def copy_ticks_range(self, symbol, start, end, flags):
        self.calls.append(("copy_ticks_range", symbol, start, end, flags))
        if self.fail == "copy_ticks_range":
            return None
        return self.tick_rows

    def order_calc_profit(self, action, symbol, volume, entry, exit_price):
        self.calls.append(("order_calc_profit", action, symbol, volume, entry, exit_price))
        if self.fail == "order_calc_profit":
            return None
        if self.profit_override is not None:
            return self.profit_override
        move = D(str(exit_price)) - D(str(entry))
        multiplier = D(1) if action == self.ORDER_TYPE_BUY else -self.sell_factor
        return float(move * D(str(volume)) * D(100000) * multiplier)

    def order_calc_margin(self, action, symbol, volume, entry):
        self.calls.append(("order_calc_margin", action, symbol, volume, entry))
        if self.fail == "order_calc_margin":
            return None
        if self.margin_override is not None:
            return self.margin_override
        return float(D(str(volume)) * self.margin_factor)


@pytest.fixture
def api():
    return FakeMT5()


@pytest.fixture
def policy():
    return BrokerCheckConfig(
        commission=CommissionModel(
            currency="EUR",
            per_lot_per_side=D(2),
            minimum_per_side=D(0),
            source="SYNTHETIC TEST TARIFF",
        ),
        max_spread_points=D(5),
        max_round_trip_cost_pct=D("0.10"),
        round_trip_slippage_points=D(1),
        legally_accessible_from_france=True,
        scalping_allowed=True,
        historical_ticks_available=True,
        terms_source="SYNTHETIC TEST CONDITIONS",
    )
