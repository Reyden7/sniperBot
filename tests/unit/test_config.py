from decimal import Decimal as D

import pytest
from pydantic import ValidationError
from sniper.config import BrokerCheckConfig, CommissionModel, Settings


def test_default_mode_and_live_disabled():
    settings = Settings()
    assert settings.mode == "PAPER"
    assert settings.live_trading_enabled is False
    assert settings.broker.risk_per_trade_pct == D("0.25")
    assert settings.broker.max_spread_points is None


@pytest.mark.parametrize("key,value", [("LIVE_TRADING_ENABLED", "true"), ("SNIPER_MODE", "LIVE")])
def test_live_is_rejected_even_when_requested(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize(
    "values",
    [
        {"stop_pips": []},
        {"stop_pips": [0]},
        {"stop_pips": [3, 3]},
        {"stop_pips": ["NaN"]},
        {"max_spread_points": -1},
        {"max_spread_points": "Infinity"},
        {"max_round_trip_cost_pct": 0},
        {"round_trip_slippage_points": -1},
        {"risk_per_trade_pct": ".51"},
        {"hard_max_risk_pct": 1},
        {"scalping_allowed": True},
        {"schema_version": 2},
        {"unexpected": True},
    ],
)
def test_policy_rejects_invalid_values(values):
    with pytest.raises(ValidationError):
        BrokerCheckConfig.model_validate(values)


def test_environment_nested_config(monkeypatch):
    monkeypatch.setenv("SNIPER_BROKER__MAX_SPREAD_POINTS", "12")
    assert Settings().broker.max_spread_points == 12


def test_commission_requires_source_and_nonnegative_tariff():
    with pytest.raises(ValidationError):
        CommissionModel(currency="EUR", per_lot_per_side=D(-1), minimum_per_side=D(0), source="")
