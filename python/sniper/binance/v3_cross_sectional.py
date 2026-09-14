"""Frozen causal cross-sectional relative-strength research for Binance Spot V3."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Literal

import numpy as np
import polars as pl

from sniper.binance.filters import floor_to_step
from sniper.binance.models import SymbolRules
from sniper.binance.qualification import CryptoTrade, ScenarioMetrics, _scenario_metrics
from sniper.binance.risk_engine import BinanceRiskEngine
from sniper.binance.strategy_engine import ResearchBar, _atr, aggregate_bars
from sniper.binance.v2_market import V2PairMarketProfile

MarketRegime = Literal["MARKET_RISK_ON", "MARKET_NEUTRAL", "MARKET_RISK_OFF"]


@dataclass(frozen=True)
class RelativeStrengthFeature:
    symbol: str
    timestamp_utc: datetime
    close: float
    atr: float
    return_15m: float
    return_30m: float
    return_1h: float
    return_4h: float
    return_24h: float
    volume_relative: float
    trade_activity_relative: float
    atr_percentile: float
    trend_efficiency: float
    expected_move_pct: float


@dataclass(frozen=True)
class RankedAsset:
    feature: RelativeStrengthFeature
    relative_return_vs_btc: float
    relative_return_vs_eth: float
    relative_return_vs_universe: float
    score: float
    rank: int


V3_ENTRY_GATE_NAMES = (
    "RELATIVE_STRENGTH_VS_BTC",
    "RELATIVE_STRENGTH_VS_UNIVERSE",
    "MOMENTUM_1H",
    "MOMENTUM_4H",
    "VOLUME",
    "LIQUIDITY",
    "MOVEMENT_COST_RATIO",
    "EXPECTED_MOVE",
    "M5_CONFIRMATION",
)


@dataclass(frozen=True)
class V3CandidateEvaluation:
    symbol: str
    rank: int
    feature: RelativeStrengthFeature
    gates: tuple[tuple[str, bool], ...]
    first_rejection_reason: str | None
    tradable: bool


@dataclass(frozen=True)
class V3TimestampEvaluation:
    """Frozen causal decision output shared by replay and forward paper."""

    feature_time: datetime
    signal_time: datetime
    market_regime: MarketRegime
    ranking: tuple[str, ...]
    top3: tuple[str, ...]
    features: tuple[RelativeStrengthFeature, ...]
    candidates: tuple[V3CandidateEvaluation, ...]
    first_rejection_reason: str | None
    signal: str | None


def evaluate_v3_timestamp(
    *,
    feature_time: datetime,
    regime: MarketRegime,
    ranked: list[RankedAsset],
    profiles: dict[str, V2PairMarketProfile],
    confirmation_bars: dict[str, ResearchBar],
    leader_count: int = 3,
) -> V3TimestampEvaluation:
    """Evaluate the immutable V3 gates once the following M5 bar has closed."""
    evaluations: list[V3CandidateEvaluation] = []
    for item in ranked[:leader_count]:
        feature = item.feature
        profile = profiles.get(feature.symbol)
        bar = confirmation_bars.get(feature.symbol)
        gate_values = (
            item.relative_return_vs_btc > 0,
            item.relative_return_vs_universe > 0,
            feature.return_1h > 0,
            feature.return_4h >= 0,
            feature.volume_relative >= 1,
            bool(profile and profile.liquidity_sufficient),
            bool(profile and profile.maximum_strategic_ratio >= Decimal(3)),
            feature.expected_move_pct >= 0.60,
            bool(bar and bar.open > 0 and bar.close > bar.open),
        )
        gates = tuple(zip(V3_ENTRY_GATE_NAMES, gate_values, strict=True))
        first_rejection = next((name for name, passed in gates if not passed), None)
        evaluations.append(
            V3CandidateEvaluation(
                symbol=feature.symbol,
                rank=item.rank,
                feature=feature,
                gates=gates,
                first_rejection_reason=first_rejection,
                tradable=first_rejection is None,
            )
        )
    signal = next((item.symbol for item in evaluations if item.tradable), None)
    regime_rejection = None if regime == "MARKET_RISK_ON" else regime
    first_rejection = regime_rejection
    if first_rejection is None and signal is None:
        first_rejection = next(
            (item.first_rejection_reason for item in evaluations if item.first_rejection_reason),
            "NO_ELIGIBLE_CANDIDATE",
        )
    return V3TimestampEvaluation(
        feature_time=feature_time,
        signal_time=feature_time + timedelta(minutes=5),
        market_regime=regime,
        ranking=tuple(item.feature.symbol for item in ranked),
        top3=tuple(item.feature.symbol for item in ranked[:3]),
        features=tuple(item.feature for item in ranked),
        candidates=tuple(evaluations),
        first_rejection_reason=first_rejection,
        signal=signal if regime == "MARKET_RISK_ON" else None,
    )


@dataclass
class _Position:
    symbol: str
    entry_time: datetime
    quantity: Decimal
    entry_reference: Decimal
    entry_executed: Decimal
    entry_fee: Decimal
    stop: Decimal
    target: Decimal
    maximum: Decimal
    minimum: Decimal
    turnover: Decimal


def _features(
    m1: list[ResearchBar],
) -> tuple[list[ResearchBar], dict[datetime, RelativeStrengthFeature]]:
    m5 = aggregate_bars(m1, 5)
    if len(m5) < 300:
        return m5, {}
    close = np.asarray([bar.close for bar in m5], dtype=float)
    volume = np.asarray([bar.quote_volume for bar in m5], dtype=float)
    activity = np.asarray([bar.trades_count for bar in m5], dtype=float)
    atr = _atr(m5)
    atr_pct = atr / close * 100
    volume_median = pl.Series(volume).shift(1).rolling_median(window_size=288).to_numpy()
    activity_median = pl.Series(activity).shift(1).rolling_median(window_size=288).to_numpy()
    output: dict[datetime, RelativeStrengthFeature] = {}
    for index in range(288, len(m5) - 1):
        path = float(np.abs(np.diff(close[index - 48 : index + 1])).sum())
        efficiency = abs(close[index] - close[index - 48]) / path if path > 0 else 0.0
        atr_history = atr_pct[index - 288 : index]
        percentile = float(np.mean(atr_history <= atr_pct[index]))
        output[m5[index].timestamp_utc + timedelta(minutes=5)] = RelativeStrengthFeature(
            symbol=m5[index].symbol,
            timestamp_utc=m5[index].timestamp_utc + timedelta(minutes=5),
            close=close[index],
            atr=atr[index],
            return_15m=close[index] / close[index - 3] - 1,
            return_30m=close[index] / close[index - 6] - 1,
            return_1h=close[index] / close[index - 12] - 1,
            return_4h=close[index] / close[index - 48] - 1,
            return_24h=close[index] / close[index - 288] - 1,
            volume_relative=(volume[index] / volume_median[index])
            if volume_median[index] > 0
            else 0.0,
            trade_activity_relative=(activity[index] / activity_median[index])
            if activity_median[index] > 0
            else 0.0,
            atr_percentile=percentile,
            trend_efficiency=efficiency,
            expected_move_pct=atr_pct[index] * 2,
        )
    return m5, output


def _features_reference(
    m1: list[ResearchBar],
) -> tuple[list[ResearchBar], dict[datetime, RelativeStrengthFeature]]:
    """Pre-vectorization reference retained solely for equivalence audits."""
    m5 = aggregate_bars(m1, 5)
    if len(m5) < 300:
        return m5, {}
    close = np.asarray([bar.close for bar in m5], dtype=float)
    volume = np.asarray([bar.quote_volume for bar in m5], dtype=float)
    activity = np.asarray([bar.trades_count for bar in m5], dtype=float)
    atr = _atr(m5)
    atr_pct = atr / close * 100
    output: dict[datetime, RelativeStrengthFeature] = {}
    for index in range(288, len(m5) - 1):
        path = float(np.abs(np.diff(close[index - 48 : index + 1])).sum())
        efficiency = abs(close[index] - close[index - 48]) / path if path > 0 else 0.0
        atr_history = atr_pct[index - 288 : index]
        volume_median = float(np.median(volume[index - 288 : index]))
        activity_median = float(np.median(activity[index - 288 : index]))
        output[m5[index].timestamp_utc + timedelta(minutes=5)] = RelativeStrengthFeature(
            symbol=m5[index].symbol,
            timestamp_utc=m5[index].timestamp_utc + timedelta(minutes=5),
            close=close[index],
            atr=atr[index],
            return_15m=close[index] / close[index - 3] - 1,
            return_30m=close[index] / close[index - 6] - 1,
            return_1h=close[index] / close[index - 12] - 1,
            return_4h=close[index] / close[index - 48] - 1,
            return_24h=close[index] / close[index - 288] - 1,
            volume_relative=volume[index] / volume_median if volume_median > 0 else 0.0,
            trade_activity_relative=(
                activity[index] / activity_median if activity_median > 0 else 0.0
            ),
            atr_percentile=float(np.mean(atr_history <= atr_pct[index])),
            trend_efficiency=efficiency,
            expected_move_pct=atr_pct[index] * 2,
        )
    return m5, output


def _percentile_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    denominator = max(len(values) - 1, 1)
    for rank, index in enumerate(order):
        ranks[index] = rank / denominator
    return ranks


def rank_cross_section(
    features: list[RelativeStrengthFeature],
    btc: RelativeStrengthFeature,
    eth: RelativeStrengthFeature,
) -> tuple[MarketRegime, list[RankedAsset]]:
    universe_4h = float(np.mean([item.return_4h for item in features]))
    breadth = float(np.mean([item.return_4h > 0 for item in features]))
    if btc.return_4h > 0 and eth.return_4h > 0 and breadth >= 0.55:
        regime: MarketRegime = "MARKET_RISK_ON"
    elif btc.return_4h < 0 and eth.return_4h < 0 and breadth <= 0.45:
        regime = "MARKET_RISK_OFF"
    else:
        regime = "MARKET_NEUTRAL"
    components = (
        [item.return_1h for item in features],
        [item.return_4h for item in features],
        [item.return_24h for item in features],
        [item.return_4h - btc.return_4h for item in features],
        [item.return_4h - universe_4h for item in features],
        [item.volume_relative for item in features],
        [item.trend_efficiency for item in features],
    )
    percentiles = [_percentile_ranks(values) for values in components]
    weights = (0.25, 0.25, 0.15, 0.15, 0.10, 0.05, 0.05)
    ranked = []
    for index, feature in enumerate(features):
        score = sum(
            weight * values[index] for weight, values in zip(weights, percentiles, strict=True)
        )
        ranked.append(
            RankedAsset(
                feature=feature,
                relative_return_vs_btc=feature.return_4h - btc.return_4h,
                relative_return_vs_eth=feature.return_4h - eth.return_4h,
                relative_return_vs_universe=feature.return_4h - universe_4h,
                score=score,
                rank=1,
            )
        )
    ranked.sort(key=lambda item: (-item.score, item.feature.symbol))
    return regime, [
        RankedAsset(**{**item.__dict__, "rank": rank}) for rank, item in enumerate(ranked, 1)
    ]


def _entry_signature(
    regime: MarketRegime,
    ranked: list[RankedAsset],
    profiles: dict[str, V2PairMarketProfile],
    leader_count: int,
) -> tuple[str, ...]:
    if regime != "MARKET_RISK_ON":
        return ()
    accepted = []
    for ranked_asset in ranked[:leader_count]:
        feature = ranked_asset.feature
        profile = profiles.get(feature.symbol)
        if profile is None:
            continue
        gates = (
            ranked_asset.relative_return_vs_btc > 0,
            ranked_asset.relative_return_vs_universe > 0,
            feature.return_1h > 0,
            feature.return_4h >= 0,
            feature.volume_relative >= 1,
            profile.liquidity_sufficient,
            profile.maximum_strategic_ratio >= Decimal(3),
            feature.expected_move_pct >= 0.60,
        )
        if all(gates):
            accepted.append(feature.symbol)
    return tuple(accepted)


def audit_vectorization_equivalence(
    m1_by_symbol: dict[str, list[ResearchBar]],
    profiles: dict[str, V2PairMarketProfile],
    *,
    sample_size: int = 32,
) -> dict[str, float | int | bool]:
    """Compare old loops and vectorized medians on fixed causal timestamps."""
    vector = {}
    reference = {}
    maximum_difference = 0.0
    fields = (
        "close",
        "atr",
        "return_15m",
        "return_30m",
        "return_1h",
        "return_4h",
        "return_24h",
        "volume_relative",
        "trade_activity_relative",
        "atr_percentile",
        "trend_efficiency",
        "expected_move_pct",
    )
    for symbol, rows in sorted(m1_by_symbol.items()):
        _, vector[symbol] = _features(rows)
        _, reference[symbol] = _features_reference(rows)
        common = sorted(set(vector[symbol]) & set(reference[symbol]))
        step = max(len(common) // max(sample_size, 1), 1)
        for timestamp in common[::step][:sample_size]:
            vector_feature = vector[symbol][timestamp]
            reference_feature = reference[symbol][timestamp]
            for field in fields:
                difference = abs(
                    float(getattr(vector_feature, field)) - float(getattr(reference_feature, field))
                )
                maximum_difference = max(maximum_difference, difference)
    timeline = sorted(set(vector.get("BTCUSDT", {})) & set(vector.get("ETHUSDT", {})))
    step = max(len(timeline) // max(sample_size, 1), 1)
    sample = timeline[::step][:sample_size]
    ranking_identity = True
    signal_identity = True
    for timestamp in sample:
        vector_features = [
            mapping[timestamp] for _, mapping in sorted(vector.items()) if timestamp in mapping
        ]
        reference_features = [
            mapping[timestamp] for _, mapping in sorted(reference.items()) if timestamp in mapping
        ]
        vector_regime, vector_rank = rank_cross_section(
            vector_features, vector["BTCUSDT"][timestamp], vector["ETHUSDT"][timestamp]
        )
        reference_reg, reference_rank = rank_cross_section(
            reference_features,
            reference["BTCUSDT"][timestamp],
            reference["ETHUSDT"][timestamp],
        )
        if vector_regime != reference_reg or [item.feature.symbol for item in vector_rank] != [
            item.feature.symbol for item in reference_rank
        ]:
            ranking_identity = False
        for leader_count in (1, 3, 5):
            if _entry_signature(
                vector_regime, vector_rank, profiles, leader_count
            ) != _entry_signature(reference_reg, reference_rank, profiles, leader_count):
                signal_identity = False
    return {
        "sample_timestamps": len(sample),
        "symbols": len(m1_by_symbol),
        "max_abs_feature_difference": maximum_difference,
        "ranking_identity": ranking_identity,
        "signal_identity": signal_identity,
        "passed": maximum_difference <= 1e-12 and ranking_identity and signal_identity,
    }


def build_cross_sections(
    m1_by_symbol: dict[str, list[ResearchBar]],
) -> tuple[
    dict[str, list[ResearchBar]],
    dict[datetime, tuple[MarketRegime, list[RankedAsset]]],
    dict[str, int],
]:
    bars_by_symbol: dict[str, list[ResearchBar]] = {}
    features_by_symbol: dict[str, dict[datetime, RelativeStrengthFeature]] = {}
    for symbol, rows in m1_by_symbol.items():
        bars_by_symbol[symbol], features_by_symbol[symbol] = _features(rows)
    cross_sections, regime_counts = assemble_cross_sections(features_by_symbol)
    return bars_by_symbol, cross_sections, regime_counts


def build_symbol_features(
    m1: list[ResearchBar],
) -> tuple[list[ResearchBar], dict[datetime, RelativeStrengthFeature]]:
    """Aggregate one symbol and release its M1 caller-side before the next symbol."""
    return _features(m1)


def assemble_cross_sections(
    features_by_symbol: dict[str, dict[datetime, RelativeStrengthFeature]],
) -> tuple[dict[datetime, tuple[MarketRegime, list[RankedAsset]]], dict[str, int]]:
    """Assemble cross-sections from already-computed, causally identical features."""
    btc_features = features_by_symbol.get("BTCUSDT", {})
    eth_features = features_by_symbol.get("ETHUSDT", {})
    timeline = sorted(set(btc_features) & set(eth_features))
    cross_sections = {}
    regime_counts: Counter[str] = Counter()
    for timestamp in timeline:
        available = [
            mapping[timestamp] for mapping in features_by_symbol.values() if timestamp in mapping
        ]
        if len(available) < 5:
            continue
        regime, ranked = rank_cross_section(
            available, btc_features[timestamp], eth_features[timestamp]
        )
        cross_sections[timestamp] = (regime, ranked)
        regime_counts[regime] += 1
    return cross_sections, dict(regime_counts)


def _d(value: float | int | str) -> Decimal:
    return Decimal(str(value))


def replay_relative_strength(
    *,
    bars_by_symbol: dict[str, list[ResearchBar]],
    cross_sections: dict[datetime, tuple[MarketRegime, list[RankedAsset]]],
    profiles: dict[str, V2PairMarketProfile],
    rules: dict[str, SymbolRules],
    leader_count: int,
    timeout_hours: int,
    start_utc: datetime,
    end_utc: datetime,
    stress: bool = False,
) -> tuple[ScenarioMetrics, list[CryptoTrade], Decimal]:
    bar_maps = {
        symbol: {bar.timestamp_utc: bar for bar in bars} for symbol, bars in bars_by_symbol.items()
    }
    timeline = [time for time in sorted(cross_sections) if start_utc <= time < end_utc]
    capital = Decimal("500")
    equity = capital
    equity_curve = [equity]
    position: _Position | None = None
    trades: list[CryptoTrade] = []
    daily_start: dict[date, Decimal] = {}
    daily_pnl: dict[date, Decimal] = defaultdict(Decimal)
    daily_entries: Counter[date] = Counter()
    rejections: Counter[str] = Counter()
    last_exit: datetime | None = None
    turnover = Decimal(0)
    risk_engine = BinanceRiskEngine()

    def execution_costs(symbol: str) -> tuple[Decimal, Decimal, Decimal]:
        profile = profiles[symbol]
        multiplier = Decimal(2) if stress else Decimal(1)
        fee = profile.taker_fee * (Decimal("1.5") if stress else Decimal(1))
        half_spread = profile.spread_p50_pct / Decimal(200) * multiplier
        slip = (
            profile.slippage_pct_by_notional["500"]
            / Decimal(200)
            * (Decimal(3) if stress else Decimal(1))
        )
        return fee, half_spread, slip

    def close(time: datetime, reference: Decimal, reason: str) -> None:
        nonlocal position, equity, last_exit, turnover
        if position is None:
            return
        fee, half_spread, slip = execution_costs(position.symbol)
        executed = reference * (Decimal(1) - half_spread) * (Decimal(1) - slip)
        exit_fee = position.quantity * executed * fee
        commission = position.entry_fee + exit_fee
        market_pnl = position.quantity * (reference - position.entry_reference)
        spread_cost = position.quantity * (
            position.entry_reference * half_spread + reference * half_spread
        )
        slippage_cost = position.quantity * (
            position.entry_reference * (Decimal(1) + half_spread) * slip
            + reference * (Decimal(1) - half_spread) * slip
        )
        net = position.quantity * (executed - position.entry_executed) - commission
        turnover += position.turnover + position.quantity * executed
        trades.append(
            CryptoTrade(
                symbol=position.symbol,
                setup_type=f"TOP{leader_count}_RS_{timeout_hours}H",
                entry_time_utc=position.entry_time,
                exit_time_utc=time,
                quantity=position.quantity,
                entry_reference=position.entry_reference,
                entry_executed=position.entry_executed,
                exit_reference=reference,
                exit_executed=executed,
                stop_price=position.stop,
                target_price=position.target,
                exit_reason=reason,
                market_pnl=market_pnl,
                spread_cost=spread_cost,
                slippage_cost=slippage_cost,
                commission=commission,
                net_pnl=net,
                net_pnl_crosscheck=market_pnl - spread_cost - slippage_cost - commission,
                mfe_pct=(position.maximum - position.entry_reference)
                / position.entry_reference
                * Decimal(100),
                mae_pct=(position.minimum - position.entry_reference)
                / position.entry_reference
                * Decimal(100),
            )
        )
        equity += net
        equity_curve.append(equity)
        daily_pnl[time.date()] += net
        last_exit = time
        position = None

    for feature_time in timeline:
        regime, ranked = cross_sections[feature_time]
        confirmation = {
            symbol: mapping[feature_time]
            for symbol, mapping in bar_maps.items()
            if feature_time in mapping
        }
        evaluation = evaluate_v3_timestamp(
            feature_time=feature_time,
            regime=regime,
            ranked=ranked,
            profiles=profiles,
            confirmation_bars=confirmation,
            leader_count=leader_count,
        )
        timestamp = evaluation.signal_time
        daily_start.setdefault(timestamp.date(), equity)
        leaders = {item.feature.symbol for item in ranked[:leader_count]}
        ranked_map = {item.feature.symbol: item for item in ranked}
        if position is not None:
            bar = bar_maps[position.symbol].get(feature_time)
            ranked_position = ranked_map.get(position.symbol)
            if bar is not None:
                position.maximum = max(position.maximum, _d(bar.high))
                position.minimum = min(position.minimum, _d(bar.low))
                if _d(bar.low) <= position.stop:
                    close(timestamp, position.stop, "STRUCTURAL_STOP")
                elif _d(bar.high) >= position.target:
                    close(timestamp, position.target, "TARGET")
                elif timestamp - position.entry_time >= timedelta(hours=timeout_hours):
                    close(timestamp, _d(bar.open), "TIMEOUT")
                elif position.symbol not in leaders:
                    close(timestamp, _d(bar.open), "LEADER_LOST")
                elif ranked_position is None or ranked_position.feature.return_1h <= 0:
                    close(timestamp, _d(bar.open), "MOMENTUM_REVERSED")
        if position is not None or regime != "MARKET_RISK_ON":
            continue
        start_equity = daily_start[timestamp.date()]
        realized = daily_pnl[timestamp.date()]
        daily_return = realized / start_equity * Decimal(100) if start_equity > 0 else Decimal(0)
        if not Decimal("-1") < daily_return < Decimal("1"):
            rejections["DAILY_RISK_LIMIT"] += 1
            continue
        if daily_entries[timestamp.date()] >= 3:
            rejections["MAX_3_TRADES_DAY"] += 1
            continue
        if last_exit is not None and timestamp < last_exit + timedelta(minutes=30):
            rejections["COOLDOWN_30M"] += 1
            continue
        candidate = next((item for item in evaluation.candidates if item.tradable), None)
        if candidate is None:
            rejections[evaluation.first_rejection_reason or "ENTRY_GATES"] += 1
            continue
        selected = ranked_map[candidate.symbol]
        bar = confirmation[candidate.symbol]
        symbol = selected.feature.symbol
        fee, half_spread, slip = execution_costs(symbol)
        reference = _d(bar.close)
        executed = reference * (Decimal(1) + half_spread) * (Decimal(1) + slip)
        stop_distance = max(_d(selected.feature.atr) * Decimal("1.5"), reference * Decimal("0.003"))
        stop = floor_to_step(reference - stop_distance, rules[symbol].tick_size)
        target = floor_to_step(
            reference + max(stop_distance * Decimal(2), reference * Decimal("0.006")),
            rules[symbol].tick_size,
        )
        risk = risk_engine.size(
            equity=equity,
            available_cash=equity,
            entry_price=executed,
            stop_price=stop,
            rules=rules[symbol],
            round_trip_fee_rate=fee * Decimal(2),
            round_trip_slippage_rate=slip * Decimal(2) + half_spread * Decimal(2),
        )
        if not risk.accepted:
            rejections[risk.reason] += 1
            continue
        quantity = risk.quantity
        notional = risk.notional
        entry_fee = notional * fee
        position = _Position(
            symbol=symbol,
            entry_time=timestamp,
            quantity=quantity,
            entry_reference=reference,
            entry_executed=executed,
            entry_fee=entry_fee,
            stop=stop,
            target=target,
            maximum=reference,
            minimum=reference,
            turnover=notional,
        )
        daily_entries[timestamp.date()] += 1
    if position is not None:
        final_bar = bar_maps[position.symbol].get(timeline[-1])
        if final_bar is not None:
            close(timeline[-1] + timedelta(minutes=5), _d(final_bar.close), "END_OF_DATA")
    daily_returns = {
        day: pnl / daily_start[day] * Decimal(100)
        for day, pnl in daily_pnl.items()
        if daily_start[day] > 0
    }
    metrics = _scenario_metrics(
        scenario="STRESS" if stress else "BASE",
        capital=capital,
        trades=trades,
        equity_curve=equity_curve,
        start_utc=start_utc,
        end_utc=end_utc,
        daily_returns=daily_returns,
        rejection_counts=rejections,
    )
    return metrics, trades, turnover


def theoretical_short_oracle(
    cross_sections: dict[datetime, tuple[MarketRegime, list[RankedAsset]]],
    bars_by_symbol: dict[str, list[ResearchBar]],
) -> dict[str, Decimal | int | bool]:
    bar_maps = {
        symbol: {bar.timestamp_utc: bar for bar in bars} for symbol, bars in bars_by_symbol.items()
    }
    returns = []
    for timestamp, (regime, ranked) in cross_sections.items():
        if regime != "MARKET_RISK_OFF" or not ranked:
            continue
        weakest = ranked[-1].feature.symbol
        now = bar_maps.get(weakest, {}).get(timestamp)
        later = bar_maps.get(weakest, {}).get(timestamp + timedelta(hours=1))
        if now is not None and later is not None and now.open > 0:
            returns.append(-(later.open / now.open - 1) * 100)
    expectancy = _d(float(np.mean(returns))) if returns else Decimal(0)
    return {
        "observations": len(returns),
        "gross_short_oracle_expectancy_pct": expectancy,
        "LONG_ONLY_CONSTRAINT_MATERIAL": len(returns) >= 30 and expectancy >= Decimal("0.10"),
    }
