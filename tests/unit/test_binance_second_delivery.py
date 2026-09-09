from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sniper.binance.filters import parse_symbol_rules
from sniper.binance.opportunity import EconomicTradeFilter, OpportunityEngine
from sniper.binance.qualification import replay_spot
from sniper.binance.risk_engine import BinanceRiskEngine, DailyPerformanceEngine, DailyState
from sniper.binance.strategy_engine import (
    BinanceStrategyEngine,
    MarketRegime,
    ResearchBar,
    SetupType,
    StrategySignal,
    aggregate_bars,
)


def _bar(index: int, close: float = 100.0) -> ResearchBar:
    timestamp = datetime(2026, 6, 10, tzinfo=UTC) + timedelta(minutes=index)
    return ResearchBar(
        symbol="BTCEUR",
        timestamp_utc=timestamp,
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=10,
        quote_volume=1000,
        trades_count=10,
        taker_buy_quote_volume=550,
    )


def _rules():
    return parse_symbol_rules(
        {
            "symbol": "BTCEUR",
            "status": "TRADING",
            "baseAsset": "BTC",
            "quoteAsset": "EUR",
            "isSpotTradingAllowed": True,
            "permissions": ["SPOT"],
            "orderTypes": ["MARKET"],
            "filters": [
                {
                    "filterType": "PRICE_FILTER",
                    "minPrice": "0.01",
                    "maxPrice": "1000000",
                    "tickSize": "0.01",
                },
                {
                    "filterType": "LOT_SIZE",
                    "minQty": "0.001",
                    "maxQty": "100",
                    "stepSize": "0.001",
                },
                {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
            ],
        }
    )


def test_timeframe_aggregation_requires_complete_utc_groups():
    bars = [_bar(index, 100 + index / 10) for index in range(6)]
    aggregated = aggregate_bars(bars, 5)
    assert len(aggregated) == 1
    assert aggregated[0].timestamp_utc == bars[0].timestamp_utc
    assert aggregated[0].close == bars[4].close


def test_future_data_does_not_change_past_strategy_signals():
    bars = [_bar(index, 100 + index * 0.01) for index in range(600)]
    engine = BinanceStrategyEngine()
    _, original = engine.generate(bars)
    changed = bars.copy()
    changed[-1] = _bar(599, 10000)
    _, mutated = engine.generate(changed)
    cutoff = bars[-2].timestamp_utc
    assert [item for item in original if item.timestamp_utc <= cutoff] == [
        item for item in mutated if item.timestamp_utc <= cutoff
    ]


def test_economic_filter_and_opportunity_ranking_do_not_size_positions():
    signal = StrategySignal(
        "BTCEUR",
        datetime(2026, 6, 10, 1, tzinfo=UTC),
        "BUY",
        SetupType.MOMENTUM_PULLBACK,
        MarketRegime.TREND_UP,
        100,
        99,
        101,
        1,
        0.7,
        ("TEST",),
    )
    ranked = OpportunityEngine(EconomicTradeFilter(Decimal("0.02"))).rank(
        [(signal, _bar(60), Decimal("2"))],
        fee_rate=Decimal("0.001"),
        slippage_bps_per_side=Decimal("1"),
    )
    assert ranked[0].accepted is True
    assert ranked[0].expected_gross_move_pct == Decimal("1.00")
    assert ranked[0].estimated_cost_pct == Decimal("0.24")


def test_risk_engine_is_only_sizer_and_enforces_binance_minimum():
    decision = BinanceRiskEngine().size(
        equity=Decimal("500"),
        available_cash=Decimal("500"),
        entry_price=Decimal("100"),
        stop_price=Decimal("99"),
        rules=_rules(),
        round_trip_fee_rate=Decimal("0.002"),
        round_trip_slippage_rate=Decimal("0.0002"),
    )
    assert decision.accepted is True
    assert decision.risk_pct <= Decimal("0.35")
    assert decision.quantity == Decimal("0.819")


def test_daily_engine_target_loss_cooldown_and_utc_reset():
    engine = DailyPerformanceEngine()
    engine.synchronize(datetime(2026, 6, 10, tzinfo=UTC).date(), Decimal("100"))
    for _ in range(3):
        engine.record(Decimal("-0.1"), Decimal("0.01"))
    assert engine.state == DailyState.COOLDOWN
    engine.synchronize(datetime(2026, 6, 11, tzinfo=UTC).date(), Decimal("99.7"))
    assert engine.state == DailyState.ACTIVE
    engine.record(Decimal("1"), Decimal("0.01"))
    assert engine.state == DailyState.DAILY_TARGET_REACHED


def test_replay_uses_executable_prices_and_crosschecks_net_pnl():
    start = datetime(2026, 6, 10, tzinfo=UTC)
    signal = StrategySignal(
        "BTCEUR",
        start,
        "BUY",
        SetupType.BREAKOUT_RETEST,
        MarketRegime.TREND_UP,
        100,
        99,
        102,
        1,
        0.8,
        ("TEST",),
    )
    bars = [
        ResearchBar("BTCEUR", start, 100, 103, 99.5, 102, 1, 100, 1, 50),
        ResearchBar("BTCEUR", start + timedelta(minutes=5), 102, 103, 101, 102, 1, 100, 1, 50),
    ]
    metrics, trades = replay_spot(
        bars_by_symbol={"BTCEUR": bars},
        signals_by_symbol={"BTCEUR": [signal]},
        rules_by_symbol={"BTCEUR": _rules()},
        spreads_bps={"BTCEUR": Decimal("2")},
        capital=Decimal("500"),
        fee_rate=Decimal("0.001"),
        scenario="BASE",
        start_utc=start,
        end_utc=start + timedelta(days=1),
    )
    assert len(trades) == 1
    assert trades[0].entry_executed > trades[0].entry_reference
    assert trades[0].exit_executed < trades[0].exit_reference
    assert metrics.pnl_crosscheck_passed is True
    assert metrics.one_position_invariant is True
