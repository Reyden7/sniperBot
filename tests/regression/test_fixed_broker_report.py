from decimal import Decimal as D

from conftest import NOW
from sniper.data.mt5_client import MT5Client
from sniper.risk.broker_compatibility import validate_broker_for_capital


def test_fixed_synthetic_report_amounts(api, policy):
    # Contract=100000 and min volume=.0001 yield 10 account units / price unit
    # in this synthetic oracle. This is intentionally not a real broker model.
    with MT5Client(api=api) as client:
        result = validate_broker_for_capital(client, config=policy, now=NOW)
    assert result.spread_points == D(2)
    assert result.spread_cost == D("0.0002")
    assert result.round_trip_commission == D("0.0004")
    assert result.slippage_cost == D("0.0001")
    assert result.minimum_round_trip_cost == D("0.0006")
    assert result.estimated_round_trip_cost == D("0.0007")
    assert result.margin_buy == result.margin_sell == D("0.1")
    assert result.minimum_risk_pct == D("0.037")
    assert [
        (str(row.side), row.stop_pips, row.stop_price, row.price_loss, row.total_loss, row.risk_pct)
        for row in result.stops
    ] == [
        ("BUY", D(3), D("1.0997"), D("0.0032"), D("0.0037"), D("0.037")),
        ("BUY", D(5), D("1.0995"), D("0.0052"), D("0.0057"), D("0.057")),
        ("BUY", D(10), D("1.0990"), D("0.0102"), D("0.0107"), D("0.107")),
        ("SELL", D(3), D("1.10032"), D("0.0032"), D("0.0037"), D("0.037")),
        ("SELL", D(5), D("1.10052"), D("0.0052"), D("0.0057"), D("0.057")),
        ("SELL", D(10), D("1.10102"), D("0.0102"), D("0.0107"), D("0.107")),
    ]
