from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError
from sniper.binance.forward_paper import (
    ForwardPaperState,
    PaperExecutionEngine,
    concentration_audit,
)
from sniper.binance.market_data import combined_stream_url, normalize_kline
from sniper.binance.models import CanonicalBookTicker, SymbolRules
from sniper.binance.settings import BinanceSettings
from sniper.binance.strategy_engine import ResearchBar
from sniper.binance.v2_market import V2PairMarketProfile
from sniper.binance.v3_cross_sectional import (
    RankedAsset,
    RelativeStrengthFeature,
    evaluate_v3_timestamp,
)

NOW = datetime(2026, 9, 10, 12, tzinfo=UTC)


def rules() -> SymbolRules:
    return SymbolRules(
        symbol="TESTUSDT",
        status="TRADING",
        base_asset="TEST",
        quote_asset="USDT",
        spot_trading_allowed=True,
        permissions=("SPOT",),
        order_types=("MARKET",),
        tick_size=Decimal("0.01"),
        minimum_price=Decimal("0.01"),
        maximum_price=Decimal("100000"),
        minimum_quantity=Decimal("0.001"),
        maximum_quantity=Decimal("10000"),
        quantity_step=Decimal("0.001"),
        market_minimum_quantity=Decimal("0.001"),
        market_maximum_quantity=Decimal("10000"),
        market_quantity_step=Decimal("0.001"),
        minimum_notional=Decimal("10"),
        filters_raw=(),
    )


def profile() -> V2PairMarketProfile:
    return V2PairMarketProfile(
        rank=1,
        symbol="TESTUSDT",
        base_asset="TEST",
        quote_asset="USDT",
        category="ECONOMICALLY_ATTRACTIVE",
        maker_fee=Decimal("0.001"),
        taker_fee=Decimal("0.001"),
        fee_source="BINANCE_ACCOUNT_API",
        spread_average_pct=Decimal("0.02"),
        spread_p50_pct=Decimal("0.02"),
        spread_p95_pct=Decimal("0.03"),
        slippage_pct_by_notional={"500": Decimal("0.02")},
        quote_volume_24h=Decimal("10000000"),
        trades_count_24h=100000,
        top20_bid_depth_quote=Decimal("100000"),
        top20_ask_depth_quote=Decimal("100000"),
        volatility_pct_by_horizon={"1h": Decimal("1")},
        expected_move_pct_by_horizon={"1h": Decimal("1")},
        taker_taker_round_trip_cost_pct=Decimal("0.24"),
        maker_taker_round_trip_cost_pct=Decimal("0.14"),
        movement_to_cost_ratio_by_horizon={"1h": Decimal("4")},
        expected_net_edge_pct_by_horizon={"1h": Decimal("0.76")},
        maximum_strategic_ratio=Decimal("4"),
        liquidity_sufficient=True,
        spread_acceptable=True,
        slippage_acceptable=True,
        tick_size=Decimal("0.01"),
        quantity_step=Decimal("0.001"),
        minimum_notional=Decimal("10"),
    )


def ranked() -> RankedAsset:
    feature = RelativeStrengthFeature(
        symbol="TESTUSDT",
        timestamp_utc=NOW,
        close=100,
        atr=1,
        return_15m=0.01,
        return_30m=0.01,
        return_1h=0.02,
        return_4h=0.03,
        return_24h=0.04,
        volume_relative=1.5,
        trade_activity_relative=1.5,
        atr_percentile=0.8,
        trend_efficiency=0.8,
        expected_move_pct=1.0,
    )
    return RankedAsset(
        feature=feature,
        relative_return_vs_btc=0.01,
        relative_return_vs_eth=0.01,
        relative_return_vs_universe=0.01,
        score=1,
        rank=1,
    )


def book(bid: str, ask: str) -> CanonicalBookTicker:
    bid_value, ask_value = Decimal(bid), Decimal(ask)
    midpoint = (bid_value + ask_value) / 2
    return CanonicalBookTicker(
        symbol="TESTUSDT",
        timestamp_utc=NOW,
        exchange_event_id=1,
        bid_price=bid_value,
        bid_quantity=Decimal("100"),
        ask_price=ask_value,
        ask_quantity=Decimal("100"),
        spread=ask_value - bid_value,
        spread_bps=(ask_value - bid_value) / midpoint * Decimal(10000),
    )


def test_paper_engine_uses_observed_book_and_closes_without_an_order_client():
    state = ForwardPaperState(
        started_at_utc=NOW,
        updated_at_utc=NOW,
        fee_rates={"TESTUSDT": Decimal("0.001")},
    )
    engine = PaperExecutionEngine(state, {"TESTUSDT": profile()}, {"TESTUSDT": rules()})
    confirmation = ResearchBar(
        symbol="TESTUSDT",
        timestamp_utc=NOW,
        open=100,
        high=102,
        low=99,
        close=101,
        volume=100,
        quote_volume=10000,
        trades_count=100,
        taker_buy_quote_volume=5000,
    )
    evaluation = evaluate_v3_timestamp(
        feature_time=NOW,
        regime="MARKET_RISK_ON",
        ranked=[ranked()],
        profiles={"TESTUSDT": profile()},
        confirmation_bars={"TESTUSDT": confirmation},
        leader_count=3,
    )
    opened = engine.process_signal(
        evaluation=evaluation,
        data_ready_time=NOW + timedelta(minutes=5),
        decision_time=NOW + timedelta(minutes=5),
        books={"TESTUSDT": book("99.9", "100.1")},
    )
    assert opened is state.position
    assert state.position is not None
    assert state.position.simulated_fill > state.position.ask
    closed = engine.mark(NOW + timedelta(minutes=10), {"TESTUSDT": book("104", "104.1")})
    assert closed is not None
    assert closed.exit_reason == "TARGET"
    assert closed.commission > 0
    assert closed.spread_cost > 0
    audit = concentration_audit(state.trades)
    assert audit["largest_symbol"] == "TESTUSDT"
    assert audit["NET_PNL_CONCENTRATION_BY_SYMBOL"] == {"TESTUSDT": Decimal(1)}


def test_forward_stream_archives_hourly_klines_and_normalizes_them():
    url = combined_stream_url("wss://stream.binance.com:9443", ("BTCUSDT",))
    assert "btcusdt@kline_1h" in url
    event = normalize_kline(
        {
            "s": "BTCUSDT",
            "k": {
                "i": "1h",
                "t": 1789041600000,
                "T": 1789045199999,
                "o": "100",
                "h": "102",
                "l": "99",
                "c": "101",
                "v": "10",
                "q": "1000",
                "n": 5,
                "V": "6",
                "Q": "600",
                "x": True,
            },
        }
    )
    assert event.interval == "1h"


def test_paper_settings_fail_closed_on_any_internal_execution_permission():
    with pytest.raises(ValidationError, match="LIVE trading is unavailable"):
        BinanceSettings(
            _env_file=None,
            trading_mode="PAPER",
            live_trading_enabled=False,
            internal_execution_permission=True,
        )
