"""Chronological, strictly out-of-sample evaluation of fixed V1 signals."""

from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from decimal import Decimal

from pydantic import Field

from sniper.config import Model, NonNegative, Positive
from sniper.data.time import as_utc
from sniper.domain.bar import Bar, Timeframe
from sniper.domain.evaluation import (
    DirectionOutcome,
    EvaluationOutcomes,
    HorizonReturns,
    MonotonicDiagnostics,
    ScoreBucket,
    SignalEvaluationReport,
    SignalObservation,
)
from sniper.domain.signal import Bias
from sniper.domain.trade import MarketTick
from sniper.features.engine import FeatureEngine
from sniper.strategy.filters import EvaluationContext
from sniper.strategy.scorer import SignalEngine

HORIZONS = (5, 10, 30, 60, 180)
BUCKETS = ((50, 59), (60, 69), (70, 79), (80, 89), (90, 94), (95, 100))


class SignalEvaluationConfig(Model):
    interval_seconds: int = Field(default=60, gt=0)
    max_horizon_tick_delay_seconds: Positive = Decimal(2)
    point: Positive = Decimal("0.00001")
    slippage_points_per_side: NonNegative = Decimal(1)
    commission_points_round_trip: NonNegative = Decimal(0)
    minimum_target_cost_ratio: Positive = Decimal(3)


def _session(timestamp: datetime) -> str:
    hour = timestamp.hour
    if 13 <= hour < 16:
        return "LONDON_NEW_YORK"
    if 7 <= hour < 13:
        return "LONDON"
    if 16 <= hour < 21:
        return "NEW_YORK"
    if 0 <= hour < 7:
        return "ASIA"
    return "OFF_HOURS"


def _direction_outcome(
    ticks: list[MarketTick],
    timestamps: list[datetime],
    current: MarketTick,
    timestamp: datetime,
    point: Decimal,
    max_delay_seconds: Decimal,
    *,
    long: bool,
) -> DirectionOutcome:
    returns: dict[int, Decimal] = {}
    for horizon in HORIZONS:
        index = bisect_left(timestamps, timestamp + timedelta(seconds=horizon))
        if index >= len(ticks):
            raise ValueError("future tick horizon is incomplete")
        future = ticks[index]
        target = timestamp + timedelta(seconds=horizon)
        delay = Decimal(str((future.timestamp_utc - target).total_seconds()))
        if delay > max_delay_seconds:
            raise ValueError("future tick horizon exceeds delay tolerance")
        returns[horizon] = (
            (future.bid - current.ask) / point if long else (current.bid - future.ask) / point
        )
    end_index = bisect_right(timestamps, timestamp + timedelta(seconds=30))
    start_index = bisect_right(timestamps, timestamp)
    window = ticks[start_index:end_index]
    if not window:
        raise ValueError("30-second excursion window contains no future tick")
    if long:
        favorable = (max(tick.bid for tick in window) - current.ask) / point
        adverse = (current.ask - min(tick.bid for tick in window)) / point
    else:
        favorable = (current.bid - min(tick.ask for tick in window)) / point
        adverse = (max(tick.ask for tick in window) - current.bid) / point
    return DirectionOutcome(
        returns_points=HorizonReturns(
            return_after_5s=returns[5],
            return_after_10s=returns[10],
            return_after_30s=returns[30],
            return_after_60s=returns[60],
            return_after_180s=returns[180],
        ),
        mfe_30s_points=max(Decimal(0), favorable),
        mae_30s_points=max(Decimal(0), adverse),
    )


def _mean(values: list[Decimal]) -> Decimal | None:
    return sum(values, Decimal(0)) / len(values) if values else None


def _non_decreasing(values: list[Decimal]) -> bool:
    return all(left <= right for left, right in zip(values, values[1:], strict=False))


def _buckets(observations: list[SignalObservation]) -> tuple[ScoreBucket, ...]:
    result = []
    for minimum, maximum in BUCKETS:
        rows = [
            row
            for row in observations
            if minimum <= row.score <= maximum and row.outcomes is not None
        ]
        directional_returns: list[Decimal] = []
        mfes: list[Decimal] = []
        maes: list[Decimal] = []
        costs: list[Decimal] = []
        ratios: list[Decimal] = []
        cost_shares: list[Decimal] = []
        net_returns: list[Decimal] = []
        for row in rows:
            assert row.outcomes is not None
            outcome = row.outcomes.long if row.trigger_direction == Bias.BUY else row.outcomes.short
            return_30s = outcome.returns_points.return_after_30s
            incremental_cost = row.expected_total_execution_cost_points - row.spread_points
            directional_returns.append(return_30s)
            mfes.append(outcome.mfe_30s_points)
            maes.append(outcome.mae_30s_points)
            costs.append(row.expected_total_execution_cost_points)
            net_returns.append(return_30s - incremental_cost)
            if row.target_cost_ratio is not None:
                ratios.append(row.target_cost_ratio)
                cost_shares.append(Decimal(100) / row.target_cost_ratio)
        average_ratio = _mean(ratios)
        result.append(
            ScoreBucket(
                label=f"{minimum}-{maximum}",
                score_min=minimum,
                score_max=maximum,
                observations=len(rows),
                win_direction_30s_pct=(
                    Decimal(100)
                    * Decimal(sum(value > 0 for value in directional_returns))
                    / len(directional_returns)
                    if directional_returns
                    else None
                ),
                average_mfe_30s_points=_mean(mfes),
                average_mae_30s_points=_mean(maes),
                average_potential_net_return_30s_points=_mean(net_returns),
                average_total_execution_cost_points=_mean(costs),
                average_target_cost_ratio=average_ratio,
                target_cost_share_pct=_mean(cost_shares),
            )
        )
    return tuple(result)


def _diagnostics(buckets: tuple[ScoreBucket, ...]) -> MonotonicDiagnostics:
    eligible = [bucket for bucket in buckets if bucket.observations > 0]
    if len(eligible) < 2:
        return MonotonicDiagnostics(
            eligible_buckets=len(eligible),
            win_rate_non_decreasing=None,
            mfe_non_decreasing=None,
            mae_non_increasing=None,
            expectancy_non_decreasing=None,
            conclusion="INSUFFICIENT_BUCKETS",
        )
    wins = [bucket.win_direction_30s_pct for bucket in eligible]
    mfes = [bucket.average_mfe_30s_points for bucket in eligible]
    maes = [bucket.average_mae_30s_points for bucket in eligible]
    expectancy = [bucket.average_potential_net_return_30s_points for bucket in eligible]
    assert all(value is not None for value in wins + mfes + maes + expectancy)
    win_ok = _non_decreasing([value for value in wins if value is not None])
    mfe_ok = _non_decreasing([value for value in mfes if value is not None])
    mae_ok = _non_decreasing([-value for value in maes if value is not None])
    expectancy_ok = _non_decreasing([value for value in expectancy if value is not None])
    return MonotonicDiagnostics(
        eligible_buckets=len(eligible),
        win_rate_non_decreasing=win_ok,
        mfe_non_decreasing=mfe_ok,
        mae_non_increasing=mae_ok,
        expectancy_non_decreasing=expectancy_ok,
        conclusion=(
            "MONOTONIC_RELATIONSHIP_OBSERVED"
            if all((win_ok, mfe_ok, mae_ok, expectancy_ok))
            else "MONOTONIC_RELATIONSHIP_NOT_OBSERVED"
        ),
    )


class SignalEvaluator:
    def __init__(self, config: SignalEvaluationConfig | None = None) -> None:
        self.config = config or SignalEvaluationConfig()
        self.features = FeatureEngine(point=self.config.point)
        self.signals = SignalEngine()

    def run(
        self,
        *,
        bars: dict[Timeframe, list[Bar]],
        ticks: list[MarketTick],
        start_utc: datetime,
        end_utc: datetime,
        context: EvaluationContext,
        symbol: str = "EURUSD",
    ) -> SignalEvaluationReport:
        start, end = as_utc(start_utc), as_utc(end_utc)
        ordered_ticks = sorted(ticks, key=lambda tick: tick.timestamp_utc)
        if symbol != "EURUSD" or start >= end or not ordered_ticks:
            raise ValueError("expected EURUSD, ticks and a nonempty UTC range")
        timestamps = [tick.timestamp_utc for tick in ordered_ticks]
        timeframes: tuple[Timeframe, ...] = ("M15", "M5", "M1")
        ordered_bars = {
            timeframe: sorted(bars.get(timeframe, []), key=lambda bar: bar.open_time_utc)
            for timeframe in timeframes
        }
        bar_times = {
            timeframe: [bar.open_time_utc for bar in values]
            for timeframe, values in ordered_bars.items()
        }
        observations: list[SignalObservation] = []
        cursor = start
        latest_signal_time = end - timedelta(seconds=max(HORIZONS))
        while cursor <= latest_signal_time:
            current_index = bisect_right(timestamps, cursor) - 1
            if current_index < 0:
                cursor += timedelta(seconds=self.config.interval_seconds)
                continue
            current = ordered_ticks[current_index]
            tick_start = bisect_left(timestamps, cursor - timedelta(seconds=60))
            visible_ticks = ordered_ticks[tick_start : current_index + 1]
            if not visible_ticks:
                visible_ticks = [current]
            visible_bars: dict[Timeframe, list[Bar]] = {}
            for timeframe in timeframes:
                values = ordered_bars[timeframe]
                cutoff = bisect_right(bar_times.get(timeframe, []), cursor)
                visible_bars[timeframe] = values[max(0, cutoff - 64) : cutoff]
            try:
                snapshot = self.features.compute(
                    bars=visible_bars,
                    ticks=visible_ticks,
                    as_of_utc=cursor,
                    symbol=symbol,
                )
            except ValueError:
                cursor += timedelta(seconds=self.config.interval_seconds)
                continue
            decision = self.signals.evaluate(snapshot, context, point=self.config.point)
            total_cost = (
                snapshot.ticks.current_spread_points
                + self.config.slippage_points_per_side * 2
                + self.config.commission_points_round_trip
            )
            target_ratio = (
                Decimal(decision.proposed_target_distance_points) / total_cost
                if decision.proposed_target_distance_points is not None and total_cost > 0
                else None
            )
            outcomes = None
            outcome_status = "SIGNAL_BLOCKED"
            if not decision.blockers:
                try:
                    outcomes = EvaluationOutcomes(
                        long=_direction_outcome(
                            ordered_ticks,
                            timestamps,
                            current,
                            cursor,
                            self.config.point,
                            self.config.max_horizon_tick_delay_seconds,
                            long=True,
                        ),
                        short=_direction_outcome(
                            ordered_ticks,
                            timestamps,
                            current,
                            cursor,
                            self.config.point,
                            self.config.max_horizon_tick_delay_seconds,
                            long=False,
                        ),
                    )
                    outcome_status = "COMPLETE"
                except ValueError:
                    outcome_status = "FUTURE_DATA_INCOMPLETE"
            observations.append(
                SignalObservation(
                    timestamp_utc=cursor,
                    score=decision.score,
                    tier=decision.tier,
                    side=decision.side,
                    market_bias=decision.market_bias,
                    trigger_direction=decision.trigger_direction,
                    components=decision.components,
                    blockers=decision.blockers,
                    current_quote_timestamp_utc=snapshot.ticks.last_tick_timestamp_utc,
                    current_quote_age_seconds=snapshot.ticks.last_tick_age_seconds,
                    current_bid=current.bid,
                    current_ask=current.ask,
                    spread_points=snapshot.ticks.current_spread_points,
                    m1_atr_points=snapshot.m1.atr / self.config.point,
                    session=_session(cursor),
                    expected_total_execution_cost_points=total_cost,
                    target_cost_ratio=target_ratio,
                    meets_minimum_target_cost_ratio=(
                        target_ratio >= self.config.minimum_target_cost_ratio
                        if target_ratio is not None
                        else None
                    ),
                    outcome_status=outcome_status,
                    outcomes=outcomes,
                )
            )
            cursor += timedelta(seconds=self.config.interval_seconds)
        score_buckets = _buckets(observations)
        return SignalEvaluationReport(
            requested_start_utc=start,
            requested_end_utc=end,
            interval_seconds=self.config.interval_seconds,
            max_horizon_tick_delay_seconds=self.config.max_horizon_tick_delay_seconds,
            point=self.config.point,
            slippage_points_per_side=self.config.slippage_points_per_side,
            commission_points_round_trip=self.config.commission_points_round_trip,
            minimum_target_cost_ratio=self.config.minimum_target_cost_ratio,
            executable_price_semantics=(
                "LONG uses current ASK to future BID; SHORT uses current BID to future ASK. "
                "Spread is embedded once in returns; potential net return additionally subtracts "
                "round-trip slippage and commission. Future ticks are evaluator-only."
            ),
            session_definition=(
                "Fixed UTC research buckets: ASIA 00:00-07:00, LONDON 07:00-13:00, "
                "LONDON_NEW_YORK 13:00-16:00, NEW_YORK 16:00-21:00, OFF_HOURS otherwise. "
                "These labels do not claim DST-aware exchange-session verification."
            ),
            recorded_observations=len(observations),
            unblocked_observations=sum(not row.blockers for row in observations),
            complete_outcome_observations=sum(row.outcomes is not None for row in observations),
            observations=tuple(observations),
            score_buckets=score_buckets,
            monotonic_diagnostics=_diagnostics(score_buckets),
        )
