"""Forward-only paper execution for the frozen Binance V3 primary strategy."""

from __future__ import annotations

import json
import os
import shutil
import signal
import sys
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from statistics import median
from time import sleep
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from sniper.binance.client import BinanceApiError, BinanceReadOnlyClient
from sniper.binance.costs import fee_schedules_from_trade_fee
from sniper.binance.filters import (
    floor_to_step,
    operational_symbol_is_unambiguous,
    parse_symbol_rules,
)
from sniper.binance.historical import load_historical_m1
from sniper.binance.m1_continuity import repair_m1_continuity
from sniper.binance.models import CanonicalBookTicker, SymbolRules
from sniper.binance.risk_engine import BinanceRiskEngine
from sniper.binance.service import collect_binance_market_data
from sniper.binance.settings import BinanceSettings
from sniper.binance.storage import BinanceParquetStore
from sniper.binance.strategy_engine import ResearchBar, bars_from_frame
from sniper.binance.v2_market import V2PairMarketProfile
from sniper.binance.v3_cross_sectional import (
    V3TimestampEvaluation,
    assemble_cross_sections,
    build_symbol_features,
    evaluate_v3_timestamp,
)
from sniper.config import Model

PAPER_PROTOCOL = "BINANCE_V3_FORWARD_PAPER_FROZEN_2026_09_10"
PAPER_CONFIGURATION = "TOP3_RS_2H"
PAPER_CAPITAL = Decimal("500")
MINIMUM_FORWARD_START = datetime(2026, 9, 10, tzinfo=UTC)
PRE_PARITY_EPOCH: Literal["PRE_PARITY_INFRASTRUCTURE_INVALID"] = "PRE_PARITY_INFRASTRUCTURE_INVALID"
VALIDATED_PARITY_EPOCH: Literal["FORWARD_PAPER_V3_PARITY_VALIDATED"] = (
    "FORWARD_PAPER_V3_PARITY_VALIDATED"
)


class PaperOrder(Model):
    order_id: str
    signal_time: datetime
    decision_time: datetime
    data_ready_time: datetime | None = None
    decision_latency_seconds: float | None = Field(default=None, ge=0)
    entry_time: datetime
    symbol: str
    reference_price: Decimal
    simulated_fill: Decimal
    bid: Decimal
    ask: Decimal
    spread: Decimal = Field(ge=0)
    estimated_slippage: Decimal = Field(ge=0)
    commission: Decimal = Field(ge=0)
    quantity: Decimal = Field(gt=0)
    notional: Decimal = Field(gt=0)
    stop: Decimal = Field(gt=0)
    target: Decimal = Field(gt=0)
    exit_time: datetime | None = None
    exit: Decimal | None = None
    exit_reason: str | None = None
    gross_pnl: Decimal | None = None
    net_pnl: Decimal | None = None
    spread_cost: Decimal = Field(default=Decimal(0), ge=0)
    slippage_cost: Decimal = Field(default=Decimal(0), ge=0)
    mfe: Decimal = Decimal(0)
    mae: Decimal = Decimal(0)


class MutableModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PaperPosition(MutableModel):
    order_id: str
    signal_time: datetime
    decision_time: datetime
    data_ready_time: datetime | None = None
    decision_latency_seconds: float | None = Field(default=None, ge=0)
    entry_time: datetime
    symbol: str
    reference_price: Decimal
    simulated_fill: Decimal
    bid: Decimal
    ask: Decimal
    entry_spread_cost: Decimal = Field(ge=0)
    entry_slippage_cost: Decimal = Field(ge=0)
    entry_commission: Decimal = Field(ge=0)
    quantity: Decimal = Field(gt=0)
    notional: Decimal = Field(gt=0)
    stop: Decimal = Field(gt=0)
    target: Decimal = Field(gt=0)
    maximum_reference: Decimal = Field(gt=0)
    minimum_reference: Decimal = Field(gt=0)


class ForwardPaperState(MutableModel):
    protocol_id: Literal["BINANCE_V3_FORWARD_PAPER_FROZEN_2026_09_10"] = (
        "BINANCE_V3_FORWARD_PAPER_FROZEN_2026_09_10"
    )
    configuration: Literal["TOP3_RS_2H"] = "TOP3_RS_2H"
    epoch: Literal["PRE_PARITY_INFRASTRUCTURE_INVALID", "FORWARD_PAPER_V3_PARITY_VALIDATED"] = (
        PRE_PARITY_EPOCH
    )
    paper_validation_start_utc: datetime | None = None
    qualification_excludes_before_utc: datetime | None = None
    started_at_utc: datetime
    updated_at_utc: datetime
    starting_equity: Decimal = PAPER_CAPITAL
    equity: Decimal = PAPER_CAPITAL
    peak_equity: Decimal = PAPER_CAPITAL
    max_drawdown_pct: Decimal = Field(default=Decimal(0), ge=0)
    position: PaperPosition | None = None
    trades: tuple[PaperOrder, ...] = ()
    daily_starting_equity: dict[str, Decimal] = Field(default_factory=dict)
    daily_entries: dict[str, int] = Field(default_factory=dict)
    last_exit_time: datetime | None = None
    last_processed_signal_time: datetime | None = None
    benchmark_start_prices: dict[str, Decimal] = Field(default_factory=dict)
    benchmark_latest_prices: dict[str, Decimal] = Field(default_factory=dict)
    fee_rates: dict[str, Decimal] = Field(default_factory=dict)
    fee_source: Literal["BINANCE_ACCOUNT_API"] = "BINANCE_ACCOUNT_API"
    fee_snapshot_at_utc: datetime | None = None
    problems: tuple[str, ...] = ()
    last_data_complete: bool | None = None
    m1_coverage_pct: float | None = Field(default=None, ge=0, le=100)


class ForwardPaperStatus(Model):
    protocol_id: str = PAPER_PROTOCOL
    configuration: str = PAPER_CONFIGURATION
    generated_at_utc: datetime
    started_at_utc: datetime
    calendar_days: int
    running: bool
    pid: int | None
    starting_equity: Decimal
    ending_equity: Decimal
    net_return_pct: Decimal
    trades: int
    wins: int
    losses: int
    fees: Decimal
    spread_cost: Decimal
    slippage: Decimal
    max_drawdown_pct: Decimal
    net_expectancy: Decimal
    profit_factor: Decimal | None
    positive_weeks: int
    active_weeks: int
    days_at_least_one_pct: int
    average_daily_return_pct: Decimal
    median_daily_return_pct: Decimal
    stress_equivalent_net_expectancy: Decimal
    benchmarks: dict[str, Decimal]
    concentration_audit: dict[str, Any]
    promusdt_diagnostic: dict[str, Decimal]
    position: PaperPosition | None
    last_processed_signal_time: datetime | None
    verdict: Literal[
        "BINANCE_V3_PAPER_FAILED",
        "BINANCE_V3_PAPER_INSUFFICIENT_EVIDENCE",
        "BINANCE_V3_PAPER_PASSED",
    ]
    trading_mode: Literal["PAPER"] = "PAPER"
    live_trading_enabled: Literal[False] = False
    internal_execution_permission: Literal[False] = False
    order_endpoints_present: Literal[False] = False
    futures_enabled: Literal[False] = False
    margin_enabled: Literal[False] = False


class DailyPaperReport(Model):
    protocol_id: str = PAPER_PROTOCOL
    configuration: str = PAPER_CONFIGURATION
    date_utc: date
    generated_at_utc: datetime
    starting_equity: Decimal
    ending_equity: Decimal
    net_return: Decimal
    trades: int
    wins: int
    losses: int
    fees: Decimal
    spread_cost: Decimal
    slippage: Decimal
    drawdown: Decimal = Field(ge=0)
    target_1pct_reached: bool
    live_trading_enabled: Literal[False] = False
    internal_execution_permission: Literal[False] = False
    order_endpoints_present: Literal[False] = False


def _atomic_json(path: Path, payload: BaseModel | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        payload.model_dump_json(indent=2)
        if isinstance(payload, BaseModel)
        else json.dumps(payload, indent=2)
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    def encode(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.astimezone(UTC).isoformat()
        if isinstance(value, Decimal):
            return str(value)
        item = getattr(value, "item", None)
        if callable(item):
            return item()
        return str(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":"), default=encode) + "\n")


def _m5_boundary(value: datetime) -> datetime:
    value = value.astimezone(UTC)
    return value.replace(minute=value.minute - value.minute % 5, second=0, microsecond=0)


def archive_pre_parity_and_create_epoch(
    data_root: Path, validation_start_utc: datetime
) -> ForwardPaperState:
    """Preserve the invalid epoch, then atomically create a fresh official state."""
    paper_root = data_root / "binance" / "paper" / "v3"
    state_path = paper_root / "state.json"
    if not state_path.exists():
        raise ValueError("no pre-parity forward state to archive")
    old_state = ForwardPaperState.model_validate_json(state_path.read_text(encoding="utf-8"))
    if old_state.epoch == VALIDATED_PARITY_EPOCH:
        raise ValueError("validated parity epoch already exists")
    stamp = validation_start_utc.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive = paper_root / "epochs" / PRE_PARITY_EPOCH / stamp
    archive.mkdir(parents=True, exist_ok=False)
    for name in ("state.json", "status.json", "runner-errors.jsonl", "events.jsonl"):
        source = paper_root / name
        if source.exists():
            shutil.copy2(source, archive / name)
    reports = data_root / "binance" / "reports" / "forward-paper-v3"
    if reports.exists():
        shutil.copytree(reports, archive / "reports")
    _atomic_json(
        archive / "manifest.json",
        {
            "epoch": PRE_PARITY_EPOCH,
            "started_at_utc": old_state.started_at_utc.isoformat(),
            "ended_at_utc": validation_start_utc.astimezone(UTC).isoformat(),
            "qualification_eligible": False,
            "preserved_trade_count": len(old_state.trades),
        },
    )
    new_state = ForwardPaperState(
        epoch=VALIDATED_PARITY_EPOCH,
        started_at_utc=validation_start_utc.astimezone(UTC),
        paper_validation_start_utc=validation_start_utc.astimezone(UTC),
        qualification_excludes_before_utc=validation_start_utc.astimezone(UTC),
        updated_at_utc=validation_start_utc.astimezone(UTC),
        starting_equity=PAPER_CAPITAL,
        equity=PAPER_CAPITAL,
        peak_equity=PAPER_CAPITAL,
    )
    _atomic_json(state_path, new_state)
    return new_state


def load_frozen_forward_inputs(
    data_root: Path,
) -> tuple[tuple[str, ...], dict[str, V2PairMarketProfile]]:
    path = data_root / "binance" / "reports" / "binance-v3-universe-freeze.json"
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("protocol_id") != "BINANCE_V3_CROSS_SECTIONAL_FROZEN_2026_09_10":
        raise ValueError("V3 frozen universe protocol mismatch")
    if frozen.get("primary_configuration") != PAPER_CONFIGURATION:
        raise ValueError("V3 primary configuration mismatch")
    symbols = tuple(str(item) for item in frozen["symbols"])
    if len(symbols) != 30 or len(set(symbols)) != 30:
        raise ValueError("V3 frozen universe must contain exactly 30 unique symbols")
    if not {"BTCUSDT", "ETHUSDT", "PROMUSDT"} <= set(symbols):
        raise ValueError("V3 mandatory benchmark/PROM symbols missing")
    if "牛来USDT" in symbols or "WLDUSDTT" in symbols:
        raise ValueError("excluded symbol present in V3 frozen universe")
    profiles = {
        item.symbol: item
        for item in (V2PairMarketProfile.model_validate(raw) for raw in frozen["profiles"])
    }
    if tuple(profiles) != symbols:
        raise ValueError("V3 frozen profile order/identity mismatch")
    return symbols, profiles


def _closed(order: PaperOrder) -> bool:
    return order.exit_time is not None and order.net_pnl is not None


def concentration_audit(trades: tuple[PaperOrder, ...]) -> dict[str, Any]:
    closed = [trade for trade in trades if _closed(trade)]
    total = sum((trade.net_pnl or Decimal(0) for trade in closed), Decimal(0))
    by_symbol: dict[str, Decimal] = defaultdict(Decimal)
    for trade in closed:
        by_symbol[trade.symbol] += trade.net_pnl or Decimal(0)
    if total <= 0 or not closed:
        return {
            "largest_symbol": None,
            "largest_symbol_share_of_total_net_pnl": Decimal(0),
            "largest_trade_share_of_total_net_pnl": Decimal(0),
            "profit_without_largest_symbol": total,
            "profit_without_largest_trade": total,
            "NET_PNL_CONCENTRATION_BY_SYMBOL": {},
        }
    largest_symbol, largest_symbol_pnl = max(by_symbol.items(), key=lambda item: item[1])
    largest_trade = max((trade.net_pnl or Decimal(0) for trade in closed), default=Decimal(0))
    return {
        "largest_symbol": largest_symbol,
        "largest_symbol_share_of_total_net_pnl": largest_symbol_pnl / total,
        "largest_trade_share_of_total_net_pnl": largest_trade / total,
        "profit_without_largest_symbol": total - largest_symbol_pnl,
        "profit_without_largest_trade": total - largest_trade,
        "NET_PNL_CONCENTRATION_BY_SYMBOL": {
            symbol: pnl / total for symbol, pnl in sorted(by_symbol.items())
        },
    }


class PaperExecutionEngine:
    """Simulate frozen TOP3_RS_2H fills; this class has no Binance client."""

    def __init__(
        self,
        state: ForwardPaperState,
        profiles: dict[str, V2PairMarketProfile],
        rules: dict[str, SymbolRules],
        risk_engine: BinanceRiskEngine | None = None,
    ):
        self.state = state
        self.profiles = profiles
        self.rules = rules
        self.risk_engine = risk_engine or BinanceRiskEngine()
        self.last_rejection_reason: str | None = None

    def _slippage_rate(self, symbol: str) -> Decimal:
        return self.profiles[symbol].slippage_pct_by_notional["500"] / Decimal(200)

    def _fee_rate(self, symbol: str) -> Decimal:
        try:
            return self.state.fee_rates[symbol]
        except KeyError as exc:
            raise ValueError(f"missing current account fee for {symbol}") from exc

    def _daily_realized(self, day: date) -> Decimal:
        return sum(
            (
                trade.net_pnl or Decimal(0)
                for trade in self.state.trades
                if trade.exit_time is not None and trade.exit_time.date() == day
            ),
            Decimal(0),
        )

    def _daily_enabled(self, now: datetime) -> bool:
        key = now.date().isoformat()
        start = self.state.daily_starting_equity.setdefault(key, self.state.equity)
        realized = self._daily_realized(now.date())
        daily_return = realized / start * Decimal(100) if start > 0 else Decimal(0)
        return (
            Decimal("-1") < daily_return < Decimal("1") and self.state.daily_entries.get(key, 0) < 3
        )

    def _close(self, now: datetime, book: CanonicalBookTicker, reason: str) -> PaperOrder:
        position = self.state.position
        if position is None:
            raise RuntimeError("paper close without an open position")
        midpoint = (book.bid_price + book.ask_price) / Decimal(2)
        slippage_rate = self._slippage_rate(position.symbol)
        exit_fill = book.bid_price * (Decimal(1) - slippage_rate)
        exit_commission = position.quantity * exit_fill * self._fee_rate(position.symbol)
        gross = position.quantity * (midpoint - position.reference_price)
        spread_cost = position.entry_spread_cost + position.quantity * (midpoint - book.bid_price)
        slippage_cost = position.entry_slippage_cost + position.quantity * (
            book.bid_price * slippage_rate
        )
        commission = position.entry_commission + exit_commission
        net = position.quantity * (exit_fill - position.simulated_fill) - commission
        order = PaperOrder(
            order_id=position.order_id,
            signal_time=position.signal_time,
            decision_time=position.decision_time,
            data_ready_time=position.data_ready_time,
            decision_latency_seconds=position.decision_latency_seconds,
            entry_time=position.entry_time,
            symbol=position.symbol,
            reference_price=position.reference_price,
            simulated_fill=position.simulated_fill,
            bid=position.bid,
            ask=position.ask,
            spread=position.ask - position.bid,
            estimated_slippage=position.entry_slippage_cost
            + position.quantity * (book.bid_price * slippage_rate),
            commission=commission,
            quantity=position.quantity,
            notional=position.notional,
            stop=position.stop,
            target=position.target,
            exit_time=now,
            exit=exit_fill,
            exit_reason=reason,
            gross_pnl=gross,
            net_pnl=net,
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            mfe=(position.maximum_reference - position.reference_price)
            / position.reference_price
            * Decimal(100),
            mae=(position.minimum_reference - position.reference_price)
            / position.reference_price
            * Decimal(100),
        )
        self.state.trades = (*self.state.trades, order)
        self.state.equity += net
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)
        if self.state.peak_equity > 0:
            self.state.max_drawdown_pct = max(
                self.state.max_drawdown_pct,
                (self.state.peak_equity - self.state.equity)
                / self.state.peak_equity
                * Decimal(100),
            )
        self.state.position = None
        self.state.last_exit_time = now
        return order

    def mark(self, now: datetime, books: dict[str, CanonicalBookTicker]) -> PaperOrder | None:
        position = self.state.position
        if position is None or position.symbol not in books:
            return None
        book = books[position.symbol]
        midpoint = (book.bid_price + book.ask_price) / Decimal(2)
        position.maximum_reference = max(position.maximum_reference, midpoint)
        position.minimum_reference = min(position.minimum_reference, midpoint)
        if book.bid_price <= position.stop:
            return self._close(now, book, "STRUCTURAL_STOP")
        if book.bid_price >= position.target:
            return self._close(now, book, "TARGET")
        if now - position.entry_time >= timedelta(hours=2):
            return self._close(now, book, "TIMEOUT")
        return None

    def process_signal(
        self,
        *,
        evaluation: V3TimestampEvaluation,
        data_ready_time: datetime,
        decision_time: datetime,
        books: dict[str, CanonicalBookTicker],
    ) -> PaperOrder | PaperPosition | None:
        self.last_rejection_reason = None
        signal_time = evaluation.signal_time
        leaders = set(evaluation.top3)
        feature_map = {item.symbol: item for item in evaluation.features}
        position = self.state.position
        if position is not None and position.symbol in books:
            current = feature_map.get(position.symbol)
            if position.symbol not in leaders:
                return self._close(decision_time, books[position.symbol], "LEADER_LOST")
            if current is None or current.return_1h <= 0:
                return self._close(decision_time, books[position.symbol], "MOMENTUM_REVERSED")
        if self.state.position is not None:
            self.last_rejection_reason = "POSITION_ALREADY_OPEN"
            return None
        if evaluation.market_regime != "MARKET_RISK_ON":
            self.last_rejection_reason = evaluation.market_regime
            return None
        if not self._daily_enabled(decision_time):
            self.last_rejection_reason = "DAILY_RISK_LIMIT"
            return None
        if self.state.last_exit_time is not None and (
            decision_time < self.state.last_exit_time + timedelta(minutes=30)
        ):
            self.last_rejection_reason = "COOLDOWN_30M"
            return None
        if evaluation.signal is None:
            self.last_rejection_reason = evaluation.first_rejection_reason
            return None
        symbol = evaluation.signal
        if symbol not in books:
            self.last_rejection_reason = "BOOK_TICKER_UNAVAILABLE"
            return None
        selected = feature_map[symbol]
        book = books[symbol]
        midpoint = (book.bid_price + book.ask_price) / Decimal(2)
        slippage_rate = self._slippage_rate(symbol)
        fill = book.ask_price * (Decimal(1) + slippage_rate)
        stop_distance = max(
            Decimal(str(selected.atr)) * Decimal("1.5"),
            midpoint * Decimal("0.003"),
        )
        rule = self.rules[symbol]
        stop = floor_to_step(midpoint - stop_distance, rule.tick_size)
        target = floor_to_step(
            midpoint + max(stop_distance * Decimal(2), midpoint * Decimal("0.006")),
            rule.tick_size,
        )
        risk = self.risk_engine.size(
            equity=self.state.equity,
            available_cash=self.state.equity,
            entry_price=fill,
            stop_price=stop,
            rules=rule,
            round_trip_fee_rate=self._fee_rate(symbol) * Decimal(2),
            round_trip_slippage_rate=slippage_rate * Decimal(2)
            + (book.ask_price - book.bid_price) / midpoint,
        )
        if not risk.accepted:
            self.last_rejection_reason = risk.reason
            return None
        entry_commission = risk.quantity * fill * self._fee_rate(symbol)
        entry_spread_cost = risk.quantity * (book.ask_price - midpoint)
        entry_slippage_cost = risk.quantity * book.ask_price * slippage_rate
        order_id = f"{signal_time.strftime('%Y%m%dT%H%M%S')}-{symbol}"
        position = PaperPosition(
            order_id=order_id,
            signal_time=signal_time,
            decision_time=decision_time,
            data_ready_time=data_ready_time,
            decision_latency_seconds=(decision_time - signal_time).total_seconds(),
            entry_time=decision_time,
            symbol=symbol,
            reference_price=midpoint,
            simulated_fill=fill,
            bid=book.bid_price,
            ask=book.ask_price,
            entry_spread_cost=entry_spread_cost,
            entry_slippage_cost=entry_slippage_cost,
            entry_commission=entry_commission,
            quantity=risk.quantity,
            notional=risk.quantity * fill,
            stop=stop,
            target=target,
            maximum_reference=midpoint,
            minimum_reference=midpoint,
        )
        self.state.position = position
        key = decision_time.date().isoformat()
        self.state.daily_entries[key] = self.state.daily_entries.get(key, 0) + 1
        return position


def _load_recent_m1(
    data_root: Path,
    symbols: tuple[str, ...],
    start_utc: datetime,
    end_utc: datetime,
) -> dict[str, list[ResearchBar]]:
    frames = [load_historical_m1(data_root, symbols, start_utc, end_utc)]
    live_root = data_root / "binance" / "normalized" / "klines"
    paths = [str(path) for path in live_root.glob("date=*/part-*.parquet")]
    if paths:
        decimals = (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        )
        live = (
            pl.scan_parquet(paths)
            .filter(
                pl.col("symbol").is_in(symbols)
                & (pl.col("interval") == "1m")
                & pl.col("closed")
                & (pl.col("timestamp_utc") >= start_utc)
                & (pl.col("timestamp_utc") < end_utc)
            )
            .with_columns(pl.col(column).cast(pl.Float64) for column in decimals)
            .collect()
        )
        frames.append(live)
    frame = (
        pl.concat(frames, how="vertical_relaxed")
        .unique(subset=["symbol", "timestamp_utc"], keep="last")
        .sort(["timestamp_utc", "symbol"])
    )
    return bars_from_frame(frame)


def _latest_books(
    client: BinanceReadOnlyClient, symbols: tuple[str, ...]
) -> dict[str, CanonicalBookTicker]:
    now = datetime.now(UTC)
    wanted = set(symbols)
    output: dict[str, CanonicalBookTicker] = {}
    for index, item in enumerate(client.book_tickers()):
        symbol = str(item["symbol"])
        if symbol not in wanted:
            continue
        bid = Decimal(str(item["bidPrice"]))
        ask = Decimal(str(item["askPrice"]))
        if bid <= 0 or ask < bid:
            continue
        midpoint = (bid + ask) / Decimal(2)
        output[symbol] = CanonicalBookTicker(
            symbol=symbol,
            timestamp_utc=now,
            exchange_event_id=int(now.timestamp() * 1_000_000) + index,
            bid_price=bid,
            bid_quantity=Decimal(str(item["bidQty"])),
            ask_price=ask,
            ask_quantity=Decimal(str(item["askQty"])),
            spread=ask - bid,
            spread_bps=(ask - bid) / midpoint * Decimal(10000),
        )
    return output


def _status(state: ForwardPaperState, *, running: bool, pid: int | None) -> ForwardPaperStatus:
    closed = [trade for trade in state.trades if _closed(trade)]
    wins = [trade.net_pnl or Decimal(0) for trade in closed if (trade.net_pnl or 0) > 0]
    losses = [-(trade.net_pnl or Decimal(0)) for trade in closed if (trade.net_pnl or 0) < 0]
    net = sum((trade.net_pnl or Decimal(0) for trade in closed), Decimal(0))
    by_day: dict[date, Decimal] = defaultdict(Decimal)
    by_week: dict[str, Decimal] = defaultdict(Decimal)
    for trade in closed:
        assert trade.exit_time is not None
        by_day[trade.exit_time.date()] += trade.net_pnl or Decimal(0)
        by_week[trade.exit_time.strftime("%G-W%V")] += trade.net_pnl or Decimal(0)
    calendar_days = max((datetime.now(UTC).date() - state.started_at_utc.date()).days + 1, 1)
    daily_returns = []
    for offset in range(calendar_days):
        day = state.started_at_utc.date() + timedelta(days=offset)
        start = state.daily_starting_equity.get(day.isoformat(), state.starting_equity)
        daily_returns.append(by_day.get(day, Decimal(0)) / start * Decimal(100))
    gross = sum((trade.gross_pnl or Decimal(0) for trade in closed), Decimal(0))
    stress_net = sum(
        (
            (trade.gross_pnl or Decimal(0))
            - trade.commission * Decimal("1.5")
            - trade.spread_cost * Decimal(2)
            - trade.slippage_cost * Decimal(3)
            for trade in closed
        ),
        Decimal(0),
    )
    btc_start = state.benchmark_start_prices.get("BTCUSDT", Decimal(0))
    btc_latest = state.benchmark_latest_prices.get("BTCUSDT", Decimal(0))
    hold_returns = [
        (latest / state.benchmark_start_prices[symbol] - 1) * Decimal(100)
        for symbol, latest in state.benchmark_latest_prices.items()
        if state.benchmark_start_prices.get(symbol, Decimal(0)) > 0
    ]
    benchmarks = {
        "CASH": Decimal(0),
        "BUY_AND_HOLD_BTC": (btc_latest / btc_start - 1) * Decimal(100)
        if btc_start > 0
        else Decimal(0),
        "BUY_AND_HOLD_EQUAL_WEIGHT_UNIVERSE": sum(hold_returns, Decimal(0))
        / Decimal(max(len(hold_returns), 1)),
    }
    prom = sum(
        (trade.net_pnl or Decimal(0) for trade in closed if trade.symbol == "PROMUSDT"),
        Decimal(0),
    )
    enough = calendar_days >= 30 and len(closed) >= 60
    profit_factor = sum(wins, Decimal(0)) / sum(losses, Decimal(0)) if losses else None
    positive_week_ratio = Decimal(sum(value > 0 for value in by_week.values())) / Decimal(
        max(len(by_week), 1)
    )
    passes = (
        net > 0
        and gross > 0
        and profit_factor is not None
        and profit_factor >= Decimal("1.20")
        and state.max_drawdown_pct <= Decimal(10)
        and positive_week_ratio >= Decimal("0.60")
        and stress_net >= 0
    )
    verdict: Literal[
        "BINANCE_V3_PAPER_FAILED",
        "BINANCE_V3_PAPER_INSUFFICIENT_EVIDENCE",
        "BINANCE_V3_PAPER_PASSED",
    ]
    if not enough:
        verdict = "BINANCE_V3_PAPER_INSUFFICIENT_EVIDENCE"
    else:
        verdict = "BINANCE_V3_PAPER_PASSED" if passes else "BINANCE_V3_PAPER_FAILED"
    return ForwardPaperStatus(
        generated_at_utc=datetime.now(UTC),
        started_at_utc=state.started_at_utc,
        calendar_days=calendar_days,
        running=running,
        pid=pid,
        starting_equity=state.starting_equity,
        ending_equity=state.equity,
        net_return_pct=net / state.starting_equity * Decimal(100),
        trades=len(closed),
        wins=len(wins),
        losses=len(losses),
        fees=sum((trade.commission for trade in closed), Decimal(0)),
        spread_cost=sum((trade.spread_cost for trade in closed), Decimal(0)),
        slippage=sum((trade.slippage_cost for trade in closed), Decimal(0)),
        max_drawdown_pct=state.max_drawdown_pct,
        net_expectancy=net / Decimal(max(len(closed), 1)),
        profit_factor=profit_factor,
        positive_weeks=sum(value > 0 for value in by_week.values()),
        active_weeks=len(by_week),
        days_at_least_one_pct=sum(value >= Decimal(1) for value in daily_returns),
        average_daily_return_pct=sum(daily_returns, Decimal(0)) / Decimal(calendar_days),
        median_daily_return_pct=Decimal(str(median(daily_returns))),
        stress_equivalent_net_expectancy=stress_net / Decimal(max(len(closed), 1)),
        benchmarks=benchmarks,
        concentration_audit=concentration_audit(state.trades),
        promusdt_diagnostic={
            "net_pnl_with_promusdt": net,
            "promusdt_net_pnl": prom,
            "net_pnl_without_promusdt": net - prom,
        },
        position=state.position,
        last_processed_signal_time=state.last_processed_signal_time,
        verdict=verdict,
    )


def _daily_report(state: ForwardPaperState, day: date) -> DailyPaperReport:
    trades = [
        trade
        for trade in state.trades
        if trade.exit_time is not None and trade.exit_time.date() == day and _closed(trade)
    ]
    start = state.daily_starting_equity.get(day.isoformat(), state.equity)
    net = sum((trade.net_pnl or Decimal(0) for trade in trades), Decimal(0))
    equity = start
    peak = start
    drawdown = Decimal(0)
    for trade in sorted(trades, key=lambda item: item.exit_time or item.entry_time):
        equity += trade.net_pnl or Decimal(0)
        peak = max(peak, equity)
        if peak > 0:
            drawdown = max(drawdown, (peak - equity) / peak * Decimal(100))
    net_return = net / start * Decimal(100) if start > 0 else Decimal(0)
    return DailyPaperReport(
        date_utc=day,
        generated_at_utc=datetime.now(UTC),
        starting_equity=start,
        ending_equity=start + net,
        net_return=net_return,
        trades=len(trades),
        wins=sum((trade.net_pnl or Decimal(0)) > 0 for trade in trades),
        losses=sum((trade.net_pnl or Decimal(0)) <= 0 for trade in trades),
        fees=sum((trade.commission for trade in trades), Decimal(0)),
        spread_cost=sum((trade.spread_cost for trade in trades), Decimal(0)),
        slippage=sum((trade.slippage_cost for trade in trades), Decimal(0)),
        drawdown=drawdown,
        target_1pct_reached=net_return >= Decimal(1),
    )


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        process = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not process:
            return False
        ctypes.windll.kernel32.CloseHandle(process)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class ForwardPaperRunner:
    """Collect genuine Spot observations and advance one resumable paper portfolio."""

    def __init__(
        self,
        client: BinanceReadOnlyClient,
        settings: BinanceSettings,
        data_root: Path,
        *,
        cycle_seconds: float = 60,
        progress: Callable[[str], None] | None = None,
    ):
        if settings.trading_mode != "PAPER":
            raise ValueError("FORWARD_PAPER_V3 requires TRADING_MODE=PAPER")
        if not settings.authenticated or settings.execution_permitted:
            raise ValueError("authenticated non-executable Binance settings required")
        if cycle_seconds < 5:
            raise ValueError("paper cycle must be at least five seconds")
        self.client = client
        self.settings = settings
        self.data_root = data_root
        self.cycle_seconds = cycle_seconds
        self.progress = progress or (lambda _message: None)
        self.paper_root = data_root / "binance" / "paper" / "v3"
        self.state_path = self.paper_root / "state.json"
        self.status_path = self.paper_root / "status.json"
        self.lock_path = self.paper_root / "runner.lock"
        self.error_path = self.paper_root / "runner-errors.jsonl"
        self.events_path = self.paper_root / "events.jsonl"
        self.stop_requested = False
        self.current_stage = "INITIALIZING"
        self.symbols, self.profiles = load_frozen_forward_inputs(data_root)
        self.rules: dict[str, SymbolRules] = {}

    def _load_or_create_state(self) -> ForwardPaperState:
        if self.state_path.exists():
            state = ForwardPaperState.model_validate_json(
                self.state_path.read_text(encoding="utf-8")
            )
            if state.started_at_utc < MINIMUM_FORWARD_START:
                raise ValueError("paper state predates the frozen forward start")
            if state.epoch != VALIDATED_PARITY_EPOCH:
                raise ValueError("pre-parity state cannot be resumed as an official epoch")
            return state
        now = datetime.now(UTC)
        if now < MINIMUM_FORWARD_START:
            raise ValueError("forward paper cannot start before 2026-09-10 UTC")
        raise ValueError("validated paper state must be explicitly initialized after parity tests")

    def _acquire_lock(self) -> None:
        self.paper_root.mkdir(parents=True, exist_ok=True)
        if self.lock_path.exists():
            try:
                existing = int(self.lock_path.read_text(encoding="ascii").strip())
            except ValueError:
                existing = -1
            if _pid_running(existing):
                raise ValueError(f"FORWARD_PAPER_V3 already running with pid {existing}")
            self.lock_path.unlink()
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(str(os.getpid()))

    def _release_lock(self) -> None:
        if self.lock_path.exists() and self.lock_path.read_text(encoding="ascii").strip() == str(
            os.getpid()
        ):
            self.lock_path.unlink()

    def _refresh_exchange_state(self, state: ForwardPaperState) -> None:
        exchange = self.client.exchange_info()
        by_symbol = {str(item["symbol"]): item for item in exchange.get("symbols", [])}
        rules = {}
        for symbol in self.symbols:
            if symbol not in by_symbol:
                raise ValueError(f"frozen symbol unavailable from exchangeInfo: {symbol}")
            rule = parse_symbol_rules(by_symbol[symbol])
            if not operational_symbol_is_unambiguous(rule) or rule.status != "TRADING":
                raise ValueError(f"frozen symbol is not unambiguous TRADING Spot: {symbol}")
            rules[symbol] = rule
        self.rules = rules
        fees = fee_schedules_from_trade_fee(self.client.trade_fees())
        missing = set(self.symbols) - fees.keys()
        if missing:
            raise ValueError(f"missing current Binance account fees for {sorted(missing)}")
        state.fee_rates = {symbol: fees[symbol].taker_rate for symbol in self.symbols}
        state.fee_snapshot_at_utc = datetime.now(UTC)

    def _archive_depth(self) -> None:
        store = BinanceParquetStore(self.data_root)
        now = datetime.now(UTC)
        records = []
        for symbol in self.symbols:
            payload = self.client.depth(symbol, 20)
            records.append(
                {
                    "timestamp_utc": now,
                    "symbol": symbol,
                    "last_update_id": int(payload["lastUpdateId"]),
                    "payload_json": json.dumps(payload, separators=(",", ":")),
                }
            )
        store.write(
            "raw",
            "depth",
            records,
            key_fields=("symbol", "last_update_id"),
        )

    def _write_reports(self, state: ForwardPaperState, running: bool) -> ForwardPaperStatus:
        pid = os.getpid() if running else None
        status = _status(state, running=running, pid=pid)
        _atomic_json(self.state_path, state)
        _atomic_json(self.status_path, status)
        daily_path = (
            self.data_root
            / "binance"
            / "reports"
            / "forward-paper-v3"
            / "daily"
            / f"{datetime.now(UTC).date().isoformat()}.json"
        )
        _atomic_json(daily_path, _daily_report(state, datetime.now(UTC).date()))
        return status

    def bootstrap(self, state: ForwardPaperState) -> None:
        self._refresh_exchange_state(state)
        now = datetime.now(UTC).replace(second=0, microsecond=0)
        continuity = repair_m1_continuity(
            self.client,
            self.data_root,
            self.symbols,
            now - timedelta(hours=30),
            now,
        )
        state.m1_coverage_pct = continuity.coverage_pct
        state.last_data_complete = continuity.complete
        if not continuity.complete:
            raise ValueError("M1 continuity repair did not reach 100 percent")
        books = _latest_books(self.client, self.symbols)
        if set(books) != set(self.symbols):
            raise ValueError("incomplete initial Binance bookTicker snapshot")
        if not state.benchmark_start_prices:
            state.benchmark_start_prices = {
                symbol: (book.bid_price + book.ask_price) / Decimal(2)
                for symbol, book in books.items()
            }
        state.benchmark_latest_prices = {
            symbol: (book.bid_price + book.ask_price) / Decimal(2) for symbol, book in books.items()
        }
        state.updated_at_utc = datetime.now(UTC)
        self._write_reports(state, running=True)

    def _append_decision_event(
        self,
        *,
        signal_time: datetime,
        data_complete: bool,
        data_ready_time: datetime | None,
        decision_time: datetime,
        evaluation: V3TimestampEvaluation | None,
        first_rejection_reason: str | None,
        tradable: bool,
    ) -> None:
        _append_jsonl(
            self.events_path,
            {
                "event_type": "M5_DECISION" if data_complete else "SKIPPED_DATA_QUALITY",
                "timestamp": signal_time,
                "data_complete": data_complete,
                "data_watermark": "DATA_COMPLETE" if data_complete else "DATA_INCOMPLETE",
                "market_regime": evaluation.market_regime if evaluation else None,
                "ranking": list(evaluation.ranking) if evaluation else [],
                "top3": list(evaluation.top3) if evaluation else [],
                "features": {
                    feature.symbol: {
                        key: value
                        for key, value in feature.__dict__.items()
                        if key not in {"symbol", "timestamp_utc"}
                    }
                    for feature in (evaluation.features if evaluation else ())
                },
                "candidates": [
                    {
                        "symbol": candidate.symbol,
                        "rank": candidate.rank,
                        "gates": dict(candidate.gates),
                        "first_rejection_reason": candidate.first_rejection_reason,
                        "tradable": candidate.tradable,
                    }
                    for candidate in (evaluation.candidates if evaluation else ())
                ],
                "first_rejection_reason": first_rejection_reason,
                "tradable": tradable,
                "signal": evaluation.signal if evaluation else None,
                "signal_time": signal_time,
                "data_ready_time": data_ready_time,
                "decision_time": decision_time,
                "decision_latency_seconds": max((decision_time - signal_time).total_seconds(), 0.0),
            },
        )

    def _append_execution_event(
        self,
        event_type: Literal["ENTRY", "EXIT"],
        payload: PaperOrder | PaperPosition,
        book: CanonicalBookTicker,
    ) -> None:
        if isinstance(payload, PaperOrder):
            timestamp = payload.exit_time
            price = payload.exit
            slippage = payload.slippage_cost
            commission = payload.commission
            reason = payload.exit_reason
            gross_pnl = payload.gross_pnl
            net_pnl = payload.net_pnl
        else:
            timestamp = payload.entry_time
            price = payload.simulated_fill
            slippage = payload.entry_slippage_cost
            commission = payload.entry_commission
            reason = PAPER_CONFIGURATION
            gross_pnl = None
            net_pnl = None
        _append_jsonl(
            self.events_path,
            {
                "event_type": event_type,
                "timestamp": timestamp,
                "symbol": payload.symbol,
                "price": price,
                "bid": book.bid_price,
                "ask": book.ask_price,
                "spread": book.ask_price - book.bid_price,
                "slippage": slippage,
                "commission": commission,
                "quantity": payload.quantity,
                "reason": reason,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "signal_time": payload.signal_time,
                "decision_time": payload.decision_time,
                "decision_latency_seconds": payload.decision_latency_seconds,
            },
        )

    def run_cycle(
        self, state: ForwardPaperState, *, signal_time: datetime | None = None
    ) -> ForwardPaperStatus:
        self.current_stage = "REFRESH_ACCOUNT_FEES"
        if (
            state.fee_snapshot_at_utc is None
            or state.fee_snapshot_at_utc.date() < datetime.now(UTC).date()
        ):
            self._refresh_exchange_state(state)
        now = datetime.now(UTC)
        target = signal_time or _m5_boundary(now)
        if target > _m5_boundary(now):
            raise ValueError("cannot process an M5 timestamp before its close")
        if (
            state.last_processed_signal_time is not None
            and target <= state.last_processed_signal_time
        ):
            state.updated_at_utc = now
            return self._write_reports(state, running=True)
        if (now - target).total_seconds() > 60:
            self._append_decision_event(
                signal_time=target,
                data_complete=False,
                data_ready_time=None,
                decision_time=now,
                evaluation=None,
                first_rejection_reason="RUNNER_INTERRUPTION_NO_RETROACTIVE_CATCHUP",
                tradable=False,
            )
            state.last_processed_signal_time = target
            state.last_data_complete = False
            state.updated_at_utc = now
            return self._write_reports(state, running=True)

        self.current_stage = "REPAIR_AND_VERIFY_M1_CONTINUITY"
        continuity = repair_m1_continuity(
            self.client,
            self.data_root,
            self.symbols,
            target - timedelta(hours=30),
            target,
            observed_at_utc=now,
        )
        state.m1_coverage_pct = continuity.coverage_pct
        data_complete = continuity.complete
        evaluation: V3TimestampEvaluation | None = None
        data_ready_time: datetime | None = None
        first_rejection: str | None = None
        output: PaperOrder | PaperPosition | None = None
        books: dict[str, CanonicalBookTicker] = {}
        if data_complete:
            self.current_stage = "LOAD_CAUSAL_HISTORY"
            m1 = _load_recent_m1(
                self.data_root,
                self.symbols,
                target - timedelta(hours=30),
                target,
            )
            self.current_stage = "BUILD_FROZEN_FEATURES"
            features = {}
            m5 = {}
            for symbol in self.symbols:
                bars, mapping = build_symbol_features(m1.get(symbol, []))
                features[symbol] = mapping
                m5[symbol] = {bar.timestamp_utc: bar for bar in bars}
            self.current_stage = "RANK_FROZEN_UNIVERSE"
            cross_sections, _counts = assemble_cross_sections(features)
            feature_time = target - timedelta(minutes=5)
            cross_section = cross_sections.get(feature_time)
            confirmation = {
                symbol: mapping[feature_time]
                for symbol, mapping in m5.items()
                if feature_time in mapping
            }
            data_complete = cross_section is not None and len(confirmation) == len(self.symbols)
            if data_complete and cross_section is not None:
                regime, ranked = cross_section
                data_complete = len(ranked) == len(self.symbols)
            if data_complete and cross_section is not None:
                regime, ranked = cross_section
                evaluation = evaluate_v3_timestamp(
                    feature_time=feature_time,
                    regime=regime,
                    ranked=ranked,
                    profiles=self.profiles,
                    confirmation_bars=confirmation,
                    leader_count=3,
                )
                data_ready_time = datetime.now(UTC)
                self.current_stage = "READ_BOOK_TICKERS"
                books = _latest_books(self.client, self.symbols)
                decision_time = datetime.now(UTC)
                if set(books) == set(self.symbols):
                    state.benchmark_latest_prices.update(
                        {
                            symbol: (book.bid_price + book.ask_price) / Decimal(2)
                            for symbol, book in books.items()
                        }
                    )
                    engine = PaperExecutionEngine(state, self.profiles, self.rules)
                    self.current_stage = "MARK_OPEN_POSITION"
                    closed = engine.mark(decision_time, books)
                    if closed is not None:
                        self._append_execution_event("EXIT", closed, books[closed.symbol])
                    self.current_stage = "PROCESS_CURRENT_CAUSAL_SIGNAL"
                    output = engine.process_signal(
                        evaluation=evaluation,
                        data_ready_time=data_ready_time,
                        decision_time=decision_time,
                        books=books,
                    )
                    first_rejection = (
                        engine.last_rejection_reason or evaluation.first_rejection_reason
                    )
                    if isinstance(output, PaperPosition):
                        self._append_execution_event("ENTRY", output, books[output.symbol])
                    elif isinstance(output, PaperOrder):
                        self._append_execution_event("EXIT", output, books[output.symbol])
                else:
                    data_complete = False
                    first_rejection = "INCOMPLETE_BOOK_TICKER_SNAPSHOT"
            else:
                first_rejection = "SKIPPED_DATA_QUALITY"
        else:
            first_rejection = "SKIPPED_DATA_QUALITY"

        decision_time = datetime.now(UTC)
        self._append_decision_event(
            signal_time=target,
            data_complete=data_complete,
            data_ready_time=data_ready_time,
            decision_time=decision_time,
            evaluation=evaluation,
            first_rejection_reason=first_rejection,
            tradable=isinstance(output, PaperPosition),
        )
        state.last_processed_signal_time = target
        state.last_data_complete = data_complete
        state.updated_at_utc = decision_time
        self.current_stage = "WRITE_CHECKPOINT_AND_REPORTS"
        status = self._write_reports(state, running=True)

        self.current_stage = "COLLECT_MARKET_DATA"
        collect_binance_market_data(
            self.client,
            self.settings,
            self.data_root,
            self.symbols,
            websocket_duration_seconds=self.cycle_seconds,
            kline_limit=10,
        )
        self.current_stage = "ARCHIVE_DEPTH"
        self._archive_depth()
        return status

    def _record_technical_error(
        self,
        state: ForwardPaperState,
        exc: Exception,
        consecutive_errors: int,
    ) -> None:
        occurred = datetime.now(UTC)
        record: dict[str, Any] = {
            "timestamp_utc": occurred.isoformat(),
            "stage": self.current_stage,
            "exception_type": type(exc).__name__,
            "consecutive_errors": consecutive_errors,
        }
        if isinstance(exc, BinanceApiError):
            record.update({"http_status": exc.status, "binance_code": exc.code})
        self.error_path.parent.mkdir(parents=True, exist_ok=True)
        with self.error_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        diagnostic = f"{occurred.isoformat()}:{self.current_stage}:{type(exc).__name__}"
        state.problems = (*state.problems[-49:], diagnostic)
        self.progress(
            f"FORWARD_PAPER_V3_TECHNICAL_RETRY stage={self.current_stage} "
            f"type={type(exc).__name__} consecutive={consecutive_errors}"
        )

    def run_forever(self) -> None:
        self._acquire_lock()
        try:
            state = self._load_or_create_state()

            def stop_handler(_signum: int, _frame: Any) -> None:
                self.stop_requested = True

            signal.signal(signal.SIGTERM, stop_handler)
            signal.signal(signal.SIGINT, stop_handler)
            self.bootstrap(state)
            self.progress(f"FORWARD_PAPER_V3_STARTED pid={os.getpid()}")
            consecutive_errors = 0
            next_signal_time = (
                state.last_processed_signal_time + timedelta(minutes=5)
                if state.last_processed_signal_time is not None
                else _m5_boundary(datetime.now(UTC)) + timedelta(minutes=5)
            )
            while not self.stop_requested:
                while datetime.now(UTC) < next_signal_time + timedelta(seconds=2):
                    if self.stop_requested:
                        break
                    remaining = (
                        next_signal_time + timedelta(seconds=2) - datetime.now(UTC)
                    ).total_seconds()
                    sleep(min(max(remaining, 0.05), 30.0))
                if self.stop_requested:
                    break
                try:
                    status = self.run_cycle(state, signal_time=next_signal_time)
                    self.progress(
                        f"FORWARD_PAPER_V3_CYCLE trades={status.trades} "
                        f"equity={status.ending_equity}"
                    )
                    consecutive_errors = 0
                    next_signal_time += timedelta(minutes=5)
                except (
                    BinanceApiError,
                    OSError,
                    KeyError,
                    TypeError,
                    pl.exceptions.PolarsError,
                ) as exc:
                    consecutive_errors += 1
                    self._record_technical_error(state, exc, consecutive_errors)
                    sleep(min(60.0, float(2 ** min(consecutive_errors, 5))))
                    continue
            self._write_reports(state, running=False)
        finally:
            self._release_lock()


def read_forward_paper_status(data_root: Path) -> ForwardPaperStatus:
    root = data_root / "binance" / "paper" / "v3"
    status_path = root / "status.json"
    if not status_path.exists():
        raise ValueError("FORWARD_PAPER_V3 has not been started")
    status = ForwardPaperStatus.model_validate_json(status_path.read_text(encoding="utf-8"))
    lock_path = root / "runner.lock"
    pid: int | None = None
    if lock_path.exists():
        try:
            pid = int(lock_path.read_text(encoding="ascii").strip())
        except ValueError:
            pid = None
    return status.model_copy(update={"running": bool(pid and _pid_running(pid)), "pid": pid})
