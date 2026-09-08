"""Deterministic metrics derived only from the event-time equity curve and closed trades."""

from decimal import Decimal

from sniper.config import Model, NonNegative
from sniper.domain.trade import Trade


class BacktestMetrics(Model):
    starting_equity: Decimal
    ending_equity: Decimal
    net_return_pct: Decimal
    gross_pnl: Decimal
    net_pnl: Decimal
    total_spread_cost: NonNegative
    total_slippage: Decimal
    total_commission: NonNegative
    trades_count: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal
    average_win: Decimal
    average_loss: Decimal
    expectancy: Decimal
    profit_factor: Decimal | None
    max_drawdown_pct: NonNegative
    max_drawdown_value: NonNegative
    max_consecutive_losses: int


def calculate_metrics(
    starting_equity: Decimal, ending_equity: Decimal, trades: list[Trade], equity: list[Decimal]
) -> BacktestMetrics:
    wins = [trade.net_pnl for trade in trades if trade.net_pnl > 0]
    losses = [trade.net_pnl for trade in trades if trade.net_pnl < 0]
    count = len(trades)
    net = sum((trade.net_pnl for trade in trades), Decimal(0))
    gross = sum((trade.gross_pnl for trade in trades), Decimal(0))
    peak = starting_equity
    max_drawdown = Decimal(0)
    max_drawdown_pct = Decimal(0)
    for value in equity:
        peak = max(peak, value)
        drawdown = peak - value
        max_drawdown = max(max_drawdown, drawdown)
        if peak > 0:
            max_drawdown_pct = max(max_drawdown_pct, drawdown / peak * 100)
    streak = 0
    max_streak = 0
    for trade in trades:
        streak = streak + 1 if trade.net_pnl < 0 else 0
        max_streak = max(max_streak, streak)
    total_wins = sum(wins, Decimal(0))
    total_losses = -sum(losses, Decimal(0))
    return BacktestMetrics(
        starting_equity=starting_equity,
        ending_equity=ending_equity,
        net_return_pct=(ending_equity - starting_equity) / starting_equity * 100,
        gross_pnl=gross,
        net_pnl=net,
        total_spread_cost=sum((trade.spread_cost for trade in trades), Decimal(0)),
        total_slippage=sum((trade.slippage for trade in trades), Decimal(0)),
        total_commission=sum((trade.commission for trade in trades), Decimal(0)),
        trades_count=count,
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=Decimal(len(wins)) / count * 100 if count else Decimal(0),
        average_win=total_wins / len(wins) if wins else Decimal(0),
        average_loss=-total_losses / len(losses) if losses else Decimal(0),
        expectancy=net / count if count else Decimal(0),
        profit_factor=(total_wins / total_losses if total_wins else Decimal(0))
        if total_losses
        else None,
        max_drawdown_pct=max_drawdown_pct,
        max_drawdown_value=max_drawdown,
        max_consecutive_losses=max_streak,
    )
