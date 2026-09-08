from datetime import UTC, datetime, timedelta
from decimal import Decimal as D

from hypothesis import given
from hypothesis import strategies as st
from sniper.domain.bar import Bar
from sniper.domain.trade import MarketTick
from sniper.features.engine import FeatureEngine
from sniper.features.patterns import candle_features
from sniper.features.structure import breakout_retest
from sniper.features.volatility import atr, range_regime

BASE = datetime(2026, 9, 8, tzinfo=UTC)


def bar(timeframe, index, minutes, close, *, open_price=None, high=None, low=None):
    close = D(str(close))
    open_price = D(str(open_price)) if open_price is not None else close - D("0.00005")
    high = D(str(high)) if high is not None else max(open_price, close) + D("0.00005")
    low = D(str(low)) if low is not None else min(open_price, close) - D("0.00005")
    return Bar(
        timeframe=timeframe,
        open_time_utc=BASE + timedelta(minutes=index * minutes),
        open=open_price,
        high=high,
        low=low,
        close=close,
        tick_volume=10,
        real_volume=0,
        spread=D("0.00002"),
        is_complete=True,
    )


def trending_bars(timeframe, count, minutes, *, down=False, start_index=0):
    direction = D(-1) if down else D(1)
    return [
        bar(
            timeframe,
            index + start_index,
            minutes,
            D("1.1000") + direction * index * D("0.0001"),
        )
        for index in range(count)
    ]


def ticks(count=10, *, down=False, future=False):
    direction = D(-1) if down else D(1)
    start = BASE + timedelta(hours=1, minutes=59, seconds=59)
    result = []
    for index in range(count):
        instant = start + timedelta(milliseconds=index * 100)
        if future:
            instant += timedelta(hours=1)
        bid = D("1.1000") + direction * D(index * index) * D("0.000001")
        result.append(MarketTick(timestamp_utc=instant, bid=bid, ask=bid + D("0.00001")))
    return result


def full_inputs(*, down=False):
    return {
        "M15": trending_bars("M15", 14, 15, down=down, start_index=-6),
        "M5": trending_bars("M5", 24, 5, down=down),
        "M1": trending_bars("M1", 120, 1, down=down),
    }


def test_feature_engine_computes_all_four_horizons():
    snapshot = FeatureEngine().compute(
        bars=full_inputs(), ticks=ticks(), as_of_utc=BASE + timedelta(hours=2)
    )
    assert snapshot.m15.ema_slope > 0
    assert snapshot.m15.structure == "HH_HL"
    assert snapshot.m15.regime == "TREND"
    assert snapshot.m5.ema_short > snapshot.m5.ema_long
    assert snapshot.m5.momentum > 0
    assert snapshot.m5.support < snapshot.m5.resistance
    assert snapshot.m1.momentum_1 > 0
    assert snapshot.m1.momentum_3 > 0
    assert snapshot.m1.momentum_5 > 0
    assert snapshot.ticks.uptick_ratio == 1
    assert snapshot.ticks.price_acceleration > 0
    assert snapshot.ticks.current_spread_points == 1


def test_bullish_breakout_retest_uses_only_prior_resistance():
    history = [
        bar(
            "M5",
            index,
            5,
            "1.1000",
            open_price="1.0998",
            high="1.1010",
            low="1.0990",
        )
        for index in range(5)
    ]
    breakout = bar("M5", 5, 5, "1.1020", open_price="1.1005", high="1.1021", low="1.1004")
    retest = bar("M5", 6, 5, "1.1012", open_price="1.1015", high="1.1018", low="1.1009")
    assert breakout_retest(history + [breakout, retest]) == "BULLISH"


def test_compression_and_expansion_are_atr_relative():
    normal = [bar("M1", index, 1, "1.1000", high="1.1005", low="1.0995") for index in range(13)]
    compressed = bar("M1", 13, 1, "1.1000", high="1.10005", low="1.09995")
    compressed_bars = normal + [compressed]
    ratio, is_compressed, is_expansion = range_regime(compressed_bars, atr(compressed_bars))
    assert ratio < D("0.60")
    assert is_compressed is True
    assert is_expansion is False
    expanded = bar("M1", 13, 1, "1.1000", high="1.1015", low="1.0985")
    expanded_bars = normal + [expanded]
    ratio, is_compressed, is_expansion = range_regime(expanded_bars, atr(expanded_bars))
    assert ratio > D("1.50")
    assert is_compressed is False
    assert is_expansion is True


def test_future_bars_and_ticks_do_not_change_point_in_time_features():
    as_of = BASE + timedelta(hours=2)
    bars = full_inputs()
    baseline = FeatureEngine().compute(bars=bars, ticks=ticks(), as_of_utc=as_of)
    with_future = {key: list(values) for key, values in bars.items()}
    with_future["M15"].append(bar("M15", 9, 15, D("9")))
    with_future["M5"].append(bar("M5", 25, 5, D("9")))
    with_future["M1"].append(bar("M1", 121, 1, D("9")))
    observed = FeatureEngine().compute(
        bars=with_future,
        ticks=ticks() + ticks(future=True),
        as_of_utc=as_of,
    )
    assert observed == baseline


def test_engulfing_hammer_and_shooting_star_features():
    bearish = bar("M1", 0, 1, "1.1000", open_price="1.1010", high="1.1011", low="1.0999")
    bullish = bar("M1", 1, 1, "1.1011", open_price="1.0999", high="1.1012", low="1.0998")
    assert candle_features([bearish, bullish])[3].engulfing == "BULLISH"
    hammer = bar("M1", 2, 1, "1.1005", open_price="1.1004", high="1.10051", low="1.1000")
    assert candle_features([hammer])[3].hammer is True
    star = bar("M1", 3, 1, "1.1004", open_price="1.1005", high="1.1010", low="1.10039")
    assert candle_features([star])[3].shooting_star is True


@given(st.lists(st.integers(-10, 10), min_size=3, max_size=30))
def test_tick_feature_invariants(moves):
    start = BASE + timedelta(hours=2) - timedelta(seconds=2)
    generated = []
    price = D("1.1000")
    for index, move in enumerate(moves):
        price += D(move) * D("0.000001")
        generated.append(
            MarketTick(
                timestamp_utc=start + timedelta(milliseconds=index * 10),
                bid=price,
                ask=price + D("0.00001"),
            )
        )
    snapshot = FeatureEngine().compute(
        bars=full_inputs(), ticks=generated, as_of_utc=BASE + timedelta(hours=2)
    )
    assert snapshot.ticks.rate_1s <= snapshot.ticks.rate_5s <= snapshot.ticks.rate_15s
    assert snapshot.ticks.uptick_ratio + snapshot.ticks.downtick_ratio == 1
    assert snapshot.ticks.current_spread_points >= 0
