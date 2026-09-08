from decimal import Decimal as D

import pytest
from conftest import NOW
from pydantic import ValidationError
from sniper.data.mt5_client import MT5Client, MT5Error
from sniper.risk.broker_compatibility import validate_broker_for_capital


def test_adapter_call_allowlist_and_cleanup(api, policy):
    api.info.visible = False
    with MT5Client(api=api, path="synthetic-terminal.exe", timeout_ms=1234) as client:
        validate_broker_for_capital(client, config=policy, now=NOW)
    assert api.calls[0] == ("initialize", ("synthetic-terminal.exe",), {"timeout": 1234})
    assert ("symbol_select", "EURUSD", True) in api.calls
    assert api.calls[-1] == ("shutdown",)
    assert {call[0] for call in api.calls} <= {
        "initialize",
        "terminal_info",
        "account_info",
        "symbol_info",
        "symbol_select",
        "symbol_info_tick",
        "order_calc_margin",
        "order_calc_profit",
        "shutdown",
    }
    assert not hasattr(MT5Client, "order_send")


@pytest.mark.parametrize(
    "failure",
    [
        "initialize",
        "terminal_info",
        "account_info",
        "symbol_info",
        "symbol_select",
        "symbol_info_tick",
        "order_calc_margin",
        "order_calc_profit",
    ],
)
def test_sdk_failures_are_explicit_and_always_shutdown(api, policy, failure):
    api.fail = failure
    api.info.visible = False
    with pytest.raises(MT5Error, match=failure) as caught:
        with MT5Client(api=api) as client:
            validate_broker_for_capital(client, config=policy, now=NOW)
    assert "PRIVATE" not in str(caught.value)
    assert api.calls[-1] == ("shutdown",)


def test_disconnected_terminal_rejected(api):
    api.terminal.connected = False
    with pytest.raises(MT5Error, match="disconnected"), MT5Client(api=api):
        pytest.fail("disconnected terminal entered")
    assert api.calls[-1] == ("shutdown",)


@pytest.mark.parametrize(
    "owner,field,value",
    [
        ("tick", "bid", 0),
        ("tick", "ask", 1.0),
        ("tick", "bid", float("nan")),
        ("info", "volume_step", 0),
        ("info", "trade_tick_size", 0),
        ("info", "trade_tick_value", float("inf")),
        ("info", "trade_contract_size", -1),
        ("account", "margin_free", float("nan")),
    ],
)
def test_invalid_observations_fail_closed(api, owner, field, value):
    setattr(getattr(api, owner), field, value)
    with pytest.raises((MT5Error, ValidationError)), MT5Client(api=api) as client:
        client.snapshot("EURUSD")
    assert api.calls[-1] == ("shutdown",)


@pytest.mark.parametrize(
    "field,value",
    [
        ("profit_override", float("nan")),
        ("profit_override", 1),
        ("margin_override", float("inf")),
        ("margin_override", -1),
        ("profit_override", 0),
    ],
)
def test_bad_calculation_outputs_never_produce_compatible_report(api, policy, field, value):
    setattr(api, field, value)
    with pytest.raises(MT5Error), MT5Client(api=api) as client:
        validate_broker_for_capital(client, config=policy, now=NOW)


def test_connection_required(api):
    with pytest.raises(MT5Error, match="with block"):
        MT5Client(api=api).snapshot("EURUSD")


def test_other_pairs_are_rejected_but_eurusd_suffix_is_supported(api, policy):
    with MT5Client(api=api) as client:
        result = validate_broker_for_capital(client, symbol="EURUSD.nano", config=policy, now=NOW)
        assert result.snapshot.symbol.name == "EURUSD.nano"
        api.info.currency_base = "GBP"
        with pytest.raises(ValueError, match="EUR/USD only"):
            validate_broker_for_capital(client, symbol="GBPUSD", config=policy, now=NOW)


@pytest.mark.parametrize("capital", [D(0), D(-1), D("Infinity"), D("NaN")])
def test_invalid_capital_fails_before_sdk_read(api, capital):
    with MT5Client(api=api) as client:
        with pytest.raises(ValueError, match="capital"):
            validate_broker_for_capital(client, capital_eur=capital)
    assert ("account_info",) not in api.calls
