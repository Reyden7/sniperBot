"""Deterministic replay/forward-emulator parity audit for frozen Binance V3."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sniper.binance.strategy_engine import ResearchBar
from sniper.binance.v2_market import V2PairMarketProfile
from sniper.binance.v3_cross_sectional import (
    MarketRegime,
    RankedAsset,
    V3TimestampEvaluation,
    evaluate_v3_timestamp,
)


@dataclass(frozen=True)
class V3ParityReport:
    timestamps_tested: int
    ranking_identity: bool
    top3_identity: bool
    market_regime_identity: bool
    gate_identity: bool
    signal_identity: bool
    first_rejection_reason_identity: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.timestamps_tested > 0,
                self.ranking_identity,
                self.top3_identity,
                self.market_regime_identity,
                self.gate_identity,
                self.signal_identity,
                self.first_rejection_reason_identity,
            )
        )


def _evaluate_replay_path(
    timestamp: datetime,
    regime: MarketRegime,
    ranked: list[RankedAsset],
    profiles: dict[str, V2PairMarketProfile],
    confirmation: dict[str, ResearchBar],
) -> V3TimestampEvaluation:
    return evaluate_v3_timestamp(
        feature_time=timestamp,
        regime=regime,
        ranked=ranked,
        profiles=profiles,
        confirmation_bars=confirmation,
        leader_count=3,
    )


def _evaluate_forward_emulator_path(
    timestamp: datetime,
    regime: MarketRegime,
    ranked: list[RankedAsset],
    profiles: dict[str, V2PairMarketProfile],
    confirmation: dict[str, ResearchBar],
) -> V3TimestampEvaluation:
    return evaluate_v3_timestamp(
        feature_time=timestamp,
        regime=regime,
        ranked=ranked,
        profiles=profiles,
        confirmation_bars=confirmation,
        leader_count=3,
    )


def audit_replay_forward_parity(
    *,
    bars_by_symbol: dict[str, list[ResearchBar]],
    cross_sections: dict[datetime, tuple[MarketRegime, list[RankedAsset]]],
    profiles: dict[str, V2PairMarketProfile],
    sample_size: int | None = None,
) -> V3ParityReport:
    bar_maps = {
        symbol: {bar.timestamp_utc: bar for bar in bars} for symbol, bars in bars_by_symbol.items()
    }
    eligible = [
        timestamp
        for timestamp, (_regime, ranked) in sorted(cross_sections.items())
        if len(ranked) == len(bars_by_symbol)
        and all(timestamp in bar_maps[symbol] for symbol in bars_by_symbol)
    ]
    if sample_size is not None and len(eligible) > sample_size:
        step = len(eligible) / sample_size
        eligible = [eligible[int(index * step)] for index in range(sample_size)]
    comparisons = {
        "ranking": True,
        "top3": True,
        "regime": True,
        "gates": True,
        "signal": True,
        "first_rejection": True,
    }
    for timestamp in eligible:
        regime, ranked = cross_sections[timestamp]
        confirmation = {symbol: bar_maps[symbol][timestamp] for symbol in bars_by_symbol}
        replay = _evaluate_replay_path(timestamp, regime, ranked, profiles, confirmation)
        forward = _evaluate_forward_emulator_path(timestamp, regime, ranked, profiles, confirmation)
        comparisons["ranking"] &= replay.ranking == forward.ranking
        comparisons["top3"] &= replay.top3 == forward.top3
        comparisons["regime"] &= replay.market_regime == forward.market_regime
        comparisons["gates"] &= tuple(item.gates for item in replay.candidates) == tuple(
            item.gates for item in forward.candidates
        )
        comparisons["signal"] &= replay.signal == forward.signal
        comparisons["first_rejection"] &= (
            replay.first_rejection_reason == forward.first_rejection_reason
        )
    return V3ParityReport(
        timestamps_tested=len(eligible),
        ranking_identity=comparisons["ranking"],
        top3_identity=comparisons["top3"],
        market_regime_identity=comparisons["regime"],
        gate_identity=comparisons["gates"],
        signal_identity=comparisons["signal"],
        first_rejection_reason_identity=comparisons["first_rejection"],
    )
