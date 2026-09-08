"""Small read-only SDK boundary. No order transmission method is exposed."""

import importlib
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
from types import TracebackType
from typing import Any, Protocol, Self

from sniper.data.time import TimePolicy, as_utc
from sniper.domain.broker import (
    AccountSnapshot,
    BrokerSnapshot,
    Quote,
    SymbolSnapshot,
    VolumeConstraints,
)
from sniper.domain.trade import Side


class MT5Error(RuntimeError):
    """Safe operational error that does not expose terminal identity information."""


class BrokerReader(Protocol):
    def snapshot(self, symbol: str) -> BrokerSnapshot: ...

    def profit(
        self, side: Side, symbol: str, volume: Decimal, entry: Decimal, exit_price: Decimal
    ) -> Decimal: ...

    def margin(self, side: Side, symbol: str, volume: Decimal, entry: Decimal) -> Decimal: ...


def number(value: Any, operation: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (ValueError, ArithmeticError) as exc:
        raise MT5Error(f"{operation}: invalid numeric result") from exc
    if not result.is_finite():
        raise MT5Error(f"{operation}: non-finite result")
    return result


class MT5Client:
    def __init__(
        self,
        *,
        path: str | None = None,
        timeout_ms: int = 10000,
        api: Any = None,
        time_policy: TimePolicy | None = None,
    ) -> None:
        self._api = api
        self._path = path
        self._timeout_ms = timeout_ms
        self._connected = False
        self.time_policy = time_policy or TimePolicy()

    def _error(self, operation: str) -> MT5Error:
        # SDK error descriptions may contain account/server details. Keep only the code.
        code = self._api.last_error()[0]
        return MT5Error(f"{operation} failed (MT5 code {code})")

    def __enter__(self) -> Self:
        if self._connected:
            raise MT5Error("MT5Client is already connected")
        if self._api is None:
            try:
                self._api = importlib.import_module("MetaTrader5")
            except (ImportError, OSError) as exc:
                raise MT5Error(
                    "MetaTrader5 unavailable; use Windows and uv sync --locked --extra mt5"
                ) from exc
        try:
            args = (self._path,) if self._path else ()
            if not self._api.initialize(*args, timeout=self._timeout_ms):
                raise self._error("initialize")
            self._connected = True
            terminal = self._api.terminal_info()
            if terminal is None:
                raise self._error("terminal_info")
            if not terminal.connected:
                raise MT5Error("MT5 terminal is disconnected")
        except Exception:
            self._api.shutdown()
            self._connected = False
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._api.shutdown()
        self._connected = False

    def _require_connection(self) -> None:
        if not self._connected:
            raise MT5Error("MT5Client must be used inside a with block")

    def snapshot(self, symbol: str) -> BrokerSnapshot:
        self._require_connection()
        account = self._api.account_info()
        if account is None:
            raise self._error("account_info")
        info = self._api.symbol_info(symbol)
        if info is None:
            raise self._error("symbol_info")
        if not info.visible:
            # Subscribe in Market Watch, without modifying account/trading permissions.
            if not self._api.symbol_select(symbol, True):
                raise self._error("symbol_select")
        tick = self._api.symbol_info_tick(symbol)
        if tick is None:
            raise self._error("symbol_info_tick")
        timestamp_utc = self.time_policy.decode(int(tick.time_msc))
        server, local = self.time_policy.representations(timestamp_utc)
        return BrokerSnapshot(
            account=AccountSnapshot(
                currency=account.currency,
                equity=number(account.equity, "equity"),
                free_margin=number(account.margin_free, "margin_free"),
                trade_allowed=account.trade_allowed,
                expert_allowed=account.trade_expert,
            ),
            symbol=SymbolSnapshot(
                name=symbol,
                currency_base=info.currency_base,
                currency_profit=info.currency_profit,
                volumes=VolumeConstraints(
                    minimum=number(info.volume_min, "volume_min"),
                    step=number(info.volume_step, "volume_step"),
                    maximum=number(info.volume_max, "volume_max"),
                ),
                point=number(info.point, "point"),
                digits=info.digits,
                tick_size=number(info.trade_tick_size, "trade_tick_size"),
                tick_value=number(info.trade_tick_value, "trade_tick_value"),
                contract_size=number(info.trade_contract_size, "trade_contract_size"),
                stops_level_points=number(info.trade_stops_level, "trade_stops_level"),
                full_trading_allowed=info.trade_mode == self._api.SYMBOL_TRADE_MODE_FULL,
            ),
            quote=Quote(
                bid=number(tick.bid, "bid"),
                ask=number(tick.ask, "ask"),
                timestamp_utc=timestamp_utc,
                timestamp_server=server,
                timestamp_local=local,
                source_time_msc=int(tick.time_msc),
                time_basis=self.time_policy.basis_evidence,
            ),
        )

    def tick_metadata(self, symbol: str) -> Decimal:
        """Collect EURUSD data independently of account capital/live eligibility."""
        self._require_connection()
        if symbol != "EURUSD":
            raise ValueError("Phase B supports EURUSD only")
        info = self._api.symbol_info(symbol)
        if info is None:
            raise self._error("symbol_info")
        if (info.currency_base, info.currency_profit) != ("EUR", "USD"):
            raise ValueError("symbol must be EUR/USD")
        if not info.visible and not self._api.symbol_select(symbol, True):
            raise self._error("symbol_select")
        point = number(info.point, "point")
        if point <= 0:
            raise MT5Error("invalid symbol point")
        return point

    def copy_ticks_range(
        self, symbol: str, start_utc: datetime, end_utc: datetime, *, max_ticks: int = 2_000_000
    ) -> Iterator[dict[str, Any]]:
        """Fetch all SDK ticks, including its inclusive upper endpoint.

        Collector filters to [start, end). Never paginate by timestamp + 1ms:
        multiple distinct ticks can share the same millisecond.
        """
        self._require_connection()
        if symbol != "EURUSD" or as_utc(start_utc) >= as_utc(end_utc):
            raise ValueError("expected EURUSD and a nonempty UTC range")
        start = self.time_policy.encode_query(start_utc)
        end = self.time_policy.encode_query(end_utc)
        if start >= end:
            raise ValueError("source range crosses an ambiguous server clock transition")
        rows = self._api.copy_ticks_range(symbol, start, end, self._api.COPY_TICKS_ALL)
        if rows is None:
            raise self._error("copy_ticks_range")
        if len(rows) > max_ticks:
            raise MT5Error("tick chunk limit exceeded; reduce --chunk-minutes")
        fields = ("time", "time_msc", "bid", "ask", "last", "volume", "volume_real", "flags")
        for row in rows:
            yield {field: row[field] for field in fields}

    def _action(self, side: Side) -> int:
        return int(self._api.ORDER_TYPE_BUY if side == Side.BUY else self._api.ORDER_TYPE_SELL)

    def profit(
        self, side: Side, symbol: str, volume: Decimal, entry: Decimal, exit_price: Decimal
    ) -> Decimal:
        self._require_connection()
        value = self._api.order_calc_profit(
            self._action(side), symbol, float(volume), float(entry), float(exit_price)
        )
        if value is None:
            raise self._error("order_calc_profit")
        return number(value, "order_calc_profit")

    def margin(self, side: Side, symbol: str, volume: Decimal, entry: Decimal) -> Decimal:
        self._require_connection()
        value = self._api.order_calc_margin(self._action(side), symbol, float(volume), float(entry))
        if value is None:
            raise self._error("order_calc_margin")
        result = number(value, "order_calc_margin")
        if result < 0:
            raise MT5Error("order_calc_margin: negative margin")
        return result
