"""Canonical Binance Spot market-data normalization and combined WebSocket collection."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from decimal import Decimal
from time import monotonic
from typing import Any

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from sniper.binance.models import CanonicalAggTrade, CanonicalBookTicker, CanonicalKline

CanonicalEvent = CanonicalAggTrade | CanonicalBookTicker | CanonicalKline


def _utc_from_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, tz=UTC)


def normalize_book_ticker(
    payload: dict[str, Any], received_at_utc: datetime
) -> CanonicalBookTicker:
    bid = Decimal(str(payload["b"]))
    ask = Decimal(str(payload["a"]))
    if ask < bid:
        raise ValueError("Binance book ticker ask is below bid")
    midpoint = (ask + bid) / 2
    spread = ask - bid
    return CanonicalBookTicker(
        symbol=str(payload["s"]),
        timestamp_utc=received_at_utc.astimezone(UTC),
        exchange_event_id=int(payload["u"]),
        bid_price=bid,
        bid_quantity=Decimal(str(payload["B"])),
        ask_price=ask,
        ask_quantity=Decimal(str(payload["A"])),
        spread=spread,
        spread_bps=spread / midpoint * Decimal(10000),
    )


def normalize_agg_trade(payload: dict[str, Any]) -> CanonicalAggTrade:
    price = Decimal(str(payload["p"]))
    quantity = Decimal(str(payload["q"]))
    buyer_is_maker = bool(payload["m"])
    return CanonicalAggTrade(
        symbol=str(payload["s"]),
        timestamp_utc=_utc_from_ms(int(payload["T"])),
        exchange_event_id=int(payload["a"]),
        price=price,
        quantity=quantity,
        quote_quantity=price * quantity,
        buyer_is_maker=buyer_is_maker,
        taker_side="SELL" if buyer_is_maker else "BUY",
    )


def normalize_rest_agg_trade(symbol: str, payload: dict[str, Any]) -> CanonicalAggTrade:
    """REST aggTrades omit ``s``; inject the requested symbol canonically."""
    return normalize_agg_trade({**payload, "s": symbol})


def normalize_kline(payload: dict[str, Any]) -> CanonicalKline:
    kline = payload["k"]
    interval = str(kline["i"])
    if interval not in {"1m", "5m", "15m", "1h"}:
        raise ValueError("unsupported kline interval")
    return CanonicalKline(
        symbol=str(payload["s"]),
        timestamp_utc=_utc_from_ms(int(kline["t"])),
        close_time_utc=_utc_from_ms(int(kline["T"])),
        interval=interval,
        open=Decimal(str(kline["o"])),
        high=Decimal(str(kline["h"])),
        low=Decimal(str(kline["l"])),
        close=Decimal(str(kline["c"])),
        volume=Decimal(str(kline["v"])),
        quote_volume=Decimal(str(kline["q"])),
        trades_count=int(kline["n"]),
        taker_buy_base_volume=Decimal(str(kline["V"])),
        taker_buy_quote_volume=Decimal(str(kline["Q"])),
        closed=bool(kline["x"]),
    )


def normalize_rest_kline(symbol: str, interval: str, row: list[Any]) -> CanonicalKline:
    if interval not in {"1m", "5m", "15m", "1h"}:
        raise ValueError("unsupported kline interval")
    return CanonicalKline(
        symbol=symbol,
        timestamp_utc=_utc_from_ms(int(row[0])),
        close_time_utc=_utc_from_ms(int(row[6])),
        interval=interval,
        open=Decimal(str(row[1])),
        high=Decimal(str(row[2])),
        low=Decimal(str(row[3])),
        close=Decimal(str(row[4])),
        volume=Decimal(str(row[5])),
        quote_volume=Decimal(str(row[7])),
        trades_count=int(row[8]),
        taker_buy_base_volume=Decimal(str(row[9])),
        taker_buy_quote_volume=Decimal(str(row[10])),
        closed=int(row[6]) < int(datetime.now(UTC).timestamp() * 1000),
    )


def normalize_stream_payload(payload: dict[str, Any], received_at_utc: datetime) -> CanonicalEvent:
    event = payload.get("e")
    if event == "aggTrade":
        return normalize_agg_trade(payload)
    if event == "kline":
        return normalize_kline(payload)
    if "u" in payload and {"b", "B", "a", "A", "s"} <= payload.keys():
        return normalize_book_ticker(payload, received_at_utc)
    raise ValueError("unsupported Binance WebSocket market event")


def combined_stream_url(ws_base_url: str, symbols: Iterable[str]) -> str:
    streams = []
    for symbol in symbols:
        lower = symbol.lower()
        streams.extend(
            [
                f"{lower}@aggTrade",
                f"{lower}@bookTicker",
                f"{lower}@kline_1m",
                f"{lower}@kline_5m",
                f"{lower}@kline_15m",
                f"{lower}@kline_1h",
            ]
        )
    if not streams:
        raise ValueError("at least one symbol is required")
    return f"{ws_base_url.rstrip('/')}" + "/stream?streams=" + "/".join(streams)


async def collect_combined_stream(
    ws_base_url: str,
    symbols: tuple[str, ...],
    duration_seconds: float,
) -> tuple[list[dict[str, Any]], list[CanonicalEvent], list[str]]:
    """Collect a bounded combined stream, reconnecting without fabricating data."""
    if duration_seconds < 0:
        raise ValueError("duration_seconds must be nonnegative")
    if duration_seconds == 0:
        return [], [], []
    url = combined_stream_url(ws_base_url, symbols)
    deadline = monotonic() + duration_seconds
    raw_records: list[dict[str, Any]] = []
    events: list[CanonicalEvent] = []
    problems: list[str] = []
    while monotonic() < deadline:
        try:
            async with connect(
                url,
                open_timeout=min(10.0, max(1.0, duration_seconds)),
                ping_interval=20,
                ping_timeout=60,
                max_queue=4096,
            ) as websocket:
                while (remaining := deadline - monotonic()) > 0:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=remaining)
                    except TimeoutError:
                        break
                    received = datetime.now(UTC)
                    envelope = json.loads(message)
                    payload = dict(envelope.get("data", envelope))
                    stream = str(envelope.get("stream", ""))
                    raw_records.append(
                        {
                            "timestamp_utc": received,
                            "stream": stream,
                            "symbol": str(payload.get("s", "")),
                            "payload_json": json.dumps(payload, separators=(",", ":")),
                        }
                    )
                    try:
                        events.append(normalize_stream_payload(payload, received))
                    except (KeyError, TypeError, ValueError) as exc:
                        problems.append(f"NORMALIZATION_REJECTED:{type(exc).__name__}")
        except (ConnectionClosed, OSError, TimeoutError) as exc:
            problems.append(f"WEBSOCKET_RECONNECT:{type(exc).__name__}")
            if deadline - monotonic() > 0.25:
                await asyncio.sleep(0.25)
    return raw_records, events, problems
