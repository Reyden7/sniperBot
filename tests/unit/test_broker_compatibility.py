from decimal import Decimal as D

import pytest
from conftest import NOW
from sniper.config import BrokerCheckConfig
from sniper.data.mt5_client import MT5Client
from sniper.risk.broker_compatibility import Verdict, validate_broker_for_capital


def report(api, policy):
    with MT5Client(api=api) as client:
        return validate_broker_for_capital(client, config=policy, now=NOW)


def test_nano_lot_boundary_is_compatible(api, policy):
    result = report(api, policy)
    assert result.verdict == Verdict.COMPATIBLE
    assert result.challenge_status == Verdict.COMPATIBLE
    assert result.nano_lot_compliant
    assert result.blockers == ()
    assert result.live_trading_enabled is False


@pytest.mark.parametrize("field", ["volume_min", "volume_step"])
@pytest.mark.parametrize("value", [0.00010001, 0.001, 0.01])
def test_rejects_each_non_nano_volume_property(api, policy, field, value):
    setattr(api.info, field, value)
    result = report(api, policy)
    assert not result.nano_lot_compliant
    assert result.verdict == Verdict.TOO_COARSE
    assert result.challenge_status == Verdict.INCOMPATIBLE_FOR_10_EUR


def test_unknown_costs_and_conditions_never_mean_zero(api):
    result = report(api, BrokerCheckConfig())
    assert result.verdict == Verdict.INCOMPATIBLE_FOR_10_EUR
    assert result.round_trip_commission is None
    assert result.minimum_round_trip_cost is None
    assert result.minimum_risk_pct is None
    assert all(row.total_loss is None for row in result.stops)
    assert {
        "COMMISSION_UNKNOWN",
        "SCALPING_UNKNOWN",
        "SLIPPAGE_UNCALIBRATED",
        "SPREAD_LIMIT_UNCALIBRATED",
        "COST_LIMIT_UNCALIBRATED",
    } <= set(result.blockers)


@pytest.mark.parametrize("currency", ["USD", "USC", "EUC"])
def test_currency_mismatch_preserves_currency_and_blocks_percentages(api, policy, currency):
    api.account.currency = currency
    result = report(api, policy)
    assert result.verdict == Verdict.INCOMPATIBLE_FOR_10_EUR
    assert result.snapshot.account.currency == currency
    assert all(row.risk_pct is None for row in result.stops)
    assert "ACCOUNT_CURRENCY_NOT_EUR_NO_VERIFIED_CONVERSION" in result.blockers
    assert "COMMISSION_CURRENCY_MISMATCH" in result.blockers


@pytest.mark.parametrize(
    "margin,free,buffer", [(100001, 10, 0), (1000, 0.09, 0), (1000, 0.1, 0.01)]
)
def test_margin_limits_use_capital_free_margin_and_buffer(api, policy, margin, free, buffer):
    api.margin_factor = D(str(margin))
    api.account.margin_free = free
    api.account.equity = 1000  # A large connected account must not expand challenge capital.
    result = report(api, policy.model_copy(update={"minimum_margin_buffer": D(str(buffer))}))
    assert result.verdict == Verdict.INSUFFICIENT_MARGIN


def test_equity_below_requested_capital_is_not_compatible(api, policy):
    api.account.equity = 1
    assert report(api, policy).verdict == Verdict.INSUFFICIENT_MARGIN


def test_fixed_minimum_commission_is_charged_twice_and_rejected(api, policy):
    commission = policy.commission.model_copy(update={"minimum_per_side": D("0.02")})
    result = report(api, policy.model_copy(update={"commission": commission}))
    assert result.round_trip_commission == D("0.04")
    assert "ROUND_TRIP_COST_EXCEEDS_BUDGET" in result.blockers
    assert result.challenge_status == Verdict.INCOMPATIBLE_FOR_10_EUR


def test_cost_threshold_has_own_verdict(api, policy):
    result = report(api, policy.model_copy(update={"max_round_trip_cost_pct": D("0.001")}))
    assert result.verdict == Verdict.COST_TOO_HIGH


def test_spread_limit_boundary(api, policy):
    result = report(api, policy.model_copy(update={"max_spread_points": D(2)}))
    assert result.verdict == Verdict.COMPATIBLE
    result = report(api, policy.model_copy(update={"max_spread_points": D("1.99")}))
    assert result.verdict == Verdict.SPREAD_TOO_HIGH


def test_stale_quote_blocks_compatibility(api, policy):
    api.tick.time_msc -= 61000
    assert "STALE_QUOTE" in report(api, policy).blockers


def test_future_offset_alone_is_unverified_not_clock_desynchronization(api, policy):
    api.tick.time_msc += 6000
    result = report(api, policy)
    assert result.blockers == ()
    assert result.time_diagnostic.status == "TIME_REFERENCE_UNVERIFIED"
    assert result.warnings == ("TIME_REFERENCE_UNVERIFIED",)


@pytest.mark.parametrize(
    "owner,field,reason",
    [
        ("account", "trade_allowed", "ACCOUNT_TRADING_NOT_ALLOWED"),
        ("account", "trade_expert", "EXPERT_ADVISORS_NOT_ALLOWED"),
        ("info", "trade_mode", "SYMBOL_NOT_FULLY_TRADABLE"),
    ],
)
def test_terminal_permissions_are_respected(api, policy, owner, field, reason):
    setattr(getattr(api, owner), field, 0)
    assert reason in report(api, policy).blockers


@pytest.mark.parametrize(
    "field,reason",
    [
        ("scalping_allowed", "SCALPING_NOT_ALLOWED"),
        ("legally_accessible_from_france", "LEGAL_ACCESS_FRANCE_NOT_ALLOWED"),
        ("historical_ticks_available", "HISTORICAL_TICKS_NOT_ALLOWED"),
    ],
)
def test_contractual_requirements_are_respected(api, policy, field, reason):
    assert reason in report(api, policy.model_copy(update={field: False})).blockers


def test_stops_below_broker_minimum_are_excluded(api, policy):
    api.info.trade_stops_level = 50
    result = report(api, policy)
    assert all(not row.broker_stop_valid for row in result.stops if row.stop_pips == 3)
    assert result.minimum_risk_pct == D("0.057")  # 5 pips + spread + tariff + slippage.
    api.info.trade_stops_level = 101
    assert "NO_CONFIGURED_STOP_MEETS_BROKER_MINIMUM" in report(api, policy).blockers


def test_stop_prices_round_outward_to_actual_tick_size(api, policy):
    api.info.trade_tick_size = 0.00003
    result = report(api, policy)
    for row in result.stops:
        assert row.stop_price % D("0.00003") == 0
        distance = row.stop_pips * D("0.0001")
        if row.side == "BUY":
            assert row.stop_price <= D("1.1") - distance
        else:
            assert row.stop_price >= D("1.10002") + distance


def test_worst_direction_controls_gate(api, policy):
    api.sell_factor = D(10)
    result = report(api, policy)
    assert result.verdict == Verdict.TOO_COARSE
    assert result.minimum_risk_pct > policy.risk_per_trade_pct


def test_four_digit_quotes_use_same_eurusd_pip_convention(api, policy):
    api.info.point = 0.0001
    api.info.digits = 4
    api.info.trade_tick_size = 0.0001
    api.tick.ask = 1.1001
    result = report(api, policy)
    assert result.spread_points == 1
    assert result.stops[0].stop_price == D("1.0997")


def test_no_sensitive_account_fields_in_json(api, policy):
    serialized = report(api, policy).model_dump_json()
    for private in ["999999", "PRIVATE_IDENTITY", "PRIVATE_SERVER", '"login"', '"server"']:
        assert private not in serialized


def test_zero_commission_requires_explicit_model(api, policy):
    commission = policy.commission.model_copy(update={"per_lot_per_side": D(0)})
    result = report(api, policy.model_copy(update={"commission": commission}))
    assert result.round_trip_commission == 0
    assert result.verdict == Verdict.COMPATIBLE
