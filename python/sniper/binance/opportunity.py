"""Economic filtering and ranking for deterministic Binance Spot signals."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from sniper.binance.strategy_engine import ResearchBar, StrategySignal


@dataclass(frozen=True)
class OpportunityCandidate:
    signal: StrategySignal
    entry_bar: ResearchBar
    expected_gross_move_pct: Decimal
    estimated_cost_pct: Decimal
    expected_net_edge_pct: Decimal
    confidence_adjusted_edge: Decimal
    accepted: bool
    reason: str


class EconomicTradeFilter:
    """Require positive expected edge beyond a separately configured safety buffer."""

    def __init__(self, uncertainty_buffer_pct: Decimal):
        if uncertainty_buffer_pct < 0:
            raise ValueError("uncertainty buffer cannot be negative")
        self.uncertainty_buffer_pct = uncertainty_buffer_pct

    def evaluate(
        self,
        *,
        signal: StrategySignal,
        entry_bar: ResearchBar,
        spread_bps: Decimal,
        fee_rate: Decimal,
        slippage_bps_per_side: Decimal,
    ) -> OpportunityCandidate:
        entry = Decimal(str(entry_bar.open))
        target_distance = Decimal(str(signal.target_reference - signal.entry_reference))
        gross = target_distance / entry * Decimal(100)
        commission = fee_rate * Decimal(200)
        spread = spread_bps / Decimal(100)
        slippage = slippage_bps_per_side / Decimal(50)
        costs = commission + spread + slippage
        net = gross - costs
        accepted = net > self.uncertainty_buffer_pct
        return OpportunityCandidate(
            signal=signal,
            entry_bar=entry_bar,
            expected_gross_move_pct=gross,
            estimated_cost_pct=costs,
            expected_net_edge_pct=net,
            confidence_adjusted_edge=net * Decimal(str(signal.confidence)),
            accepted=accepted,
            reason="POSITIVE_NET_EDGE_ABOVE_BUFFER" if accepted else "INSUFFICIENT_NET_EDGE",
        )


class OpportunityEngine:
    """Select at most one best economic opportunity; never choose position size."""

    def __init__(self, economic_filter: EconomicTradeFilter):
        self.economic_filter = economic_filter

    def rank(
        self,
        items: list[tuple[StrategySignal, ResearchBar, Decimal]],
        *,
        fee_rate: Decimal,
        slippage_bps_per_side: Decimal,
    ) -> list[OpportunityCandidate]:
        candidates = [
            self.economic_filter.evaluate(
                signal=signal,
                entry_bar=bar,
                spread_bps=spread,
                fee_rate=fee_rate,
                slippage_bps_per_side=slippage_bps_per_side,
            )
            for signal, bar, spread in items
        ]
        return sorted(
            candidates,
            key=lambda item: item.confidence_adjusted_edge,
            reverse=True,
        )
