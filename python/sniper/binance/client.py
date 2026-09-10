"""Official Binance Spot REST adapter limited to read-only endpoints."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from sniper.binance.settings import BinanceSettings


class BinanceApiError(RuntimeError):
    """Sanitized Binance connectivity or API failure."""

    def __init__(self, message: str, *, status: int | None = None, code: int | None = None):
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class TransportResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class HttpTransport(Protocol):
    def get(self, url: str, headers: Mapping[str, str], timeout: float) -> TransportResponse: ...


class UrllibTransport:
    """Tiny dependency-free HTTPS transport with explicit timeout."""

    def get(self, url: str, headers: Mapping[str, str], timeout: float) -> TransportResponse:
        request = Request(url, headers=dict(headers), method="GET")  # noqa: S310 - URL is configured
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 - HTTPS enforced
                return TransportResponse(
                    status=int(response.status),
                    headers=dict(response.headers.items()),
                    body=response.read(),
                )
        except HTTPError as exc:
            return TransportResponse(
                status=exc.code,
                headers=dict(exc.headers.items()) if exc.headers else {},
                body=exc.read(),
            )
        except (URLError, TimeoutError, OSError) as exc:
            raise BinanceApiError("Binance HTTPS connection failed") from exc


class BinanceReadOnlyClient:
    """Read public/account state; this class deliberately exposes no order method."""

    READ_ONLY_PATHS = frozenset(
        {
            "/api/v3/ping",
            "/api/v3/time",
            "/api/v3/exchangeInfo",
            "/api/v3/ticker/bookTicker",
            "/api/v3/ticker/24hr",
            "/api/v3/klines",
            "/api/v3/aggTrades",
            "/api/v3/depth",
            "/api/v3/account",
            "/api/v3/account/commission",
            "/sapi/v1/asset/tradeFee",
        }
    )

    def __init__(
        self,
        settings: BinanceSettings,
        transport: HttpTransport | None = None,
        *,
        sleep: Any = time.sleep,
    ):
        self.settings = settings
        self.transport = transport or UrllibTransport()
        self._sleep = sleep
        self._clock_offset_ms = 0
        self._last_clock_sync_monotonic: float | None = None

    def _decode(self, response: TransportResponse) -> Any:
        try:
            payload = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BinanceApiError("Binance returned an invalid JSON response") from exc
        if response.status >= 400:
            code = payload.get("code") if isinstance(payload, dict) else None
            message = (
                payload.get("msg", "Binance API request failed")
                if isinstance(payload, dict)
                else "Binance API request failed"
            )
            raise BinanceApiError(str(message)[:240], status=response.status, code=code)
        return payload

    def _get(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        *,
        signed: bool = False,
        market_data: bool = True,
    ) -> Any:
        if path not in self.READ_ONLY_PATHS:
            raise BinanceApiError("Endpoint is not permitted by the read-only adapter")
        query_params = {key: value for key, value in (params or {}).items() if value is not None}
        headers = {"User-Agent": "SNIPER/0.16.0 read-only"}
        if signed:
            if not self.settings.authenticated:
                raise BinanceApiError("Binance account credentials are not configured")
            query_params["timestamp"] = int(time.time() * 1000) + self._clock_offset_ms
            query_params["recvWindow"] = self.settings.recv_window_ms
            unsigned_query = urlencode(sorted(query_params.items()))
            secret = self.settings.api_secret
            assert secret is not None
            signature = hmac.new(
                secret.get_secret_value().encode(),
                unsigned_query.encode(),
                hashlib.sha256,
            ).hexdigest()
            query_params["signature"] = signature
            api_key = self.settings.api_key
            assert api_key is not None
            headers["X-MBX-APIKEY"] = api_key.get_secret_value()
            market_data = False
        query = urlencode(sorted(query_params.items()))
        base_url = (
            self.settings.market_data_base_url if market_data else self.settings.rest_base_url
        ).rstrip("/")
        url = f"{base_url}{path}" + (f"?{query}" if query else "")
        for attempt in range(3):
            response = self.transport.get(url, headers, self.settings.request_timeout_seconds)
            if response.status not in (418, 429):
                return self._decode(response)
            if attempt == 2:
                return self._decode(response)
            retry_after = float(response.headers.get("Retry-After", "1"))
            self._sleep(min(max(retry_after, 0.0), 30.0))
        raise AssertionError("unreachable")

    def ping(self) -> None:
        self._get("/api/v3/ping")

    def server_time(self) -> int:
        payload = self._get("/api/v3/time")
        return int(payload["serverTime"])

    def synchronize_clock(self) -> int:
        before = int(time.time() * 1000)
        server = self.server_time()
        after = int(time.time() * 1000)
        midpoint = (before + after) // 2
        self._clock_offset_ms = server - midpoint
        self._last_clock_sync_monotonic = time.monotonic()
        return self._clock_offset_ms

    def _ensure_clock_synchronized(self) -> None:
        if (
            self._last_clock_sync_monotonic is None
            or time.monotonic() - self._last_clock_sync_monotonic > 30
        ):
            self.synchronize_clock()

    def exchange_info(self) -> dict[str, Any]:
        return dict(self._get("/api/v3/exchangeInfo"))

    def book_tickers(self) -> list[dict[str, Any]]:
        payload = self._get("/api/v3/ticker/bookTicker")
        return [dict(item) for item in payload]

    def ticker_24h(self) -> list[dict[str, Any]]:
        payload = self._get("/api/v3/ticker/24hr")
        return [dict(item) for item in payload]

    def klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 200,
        *,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> list[list[Any]]:
        if interval not in {"1m", "5m", "15m", "30m", "1h", "4h"}:
            raise ValueError("unsupported Binance Spot kline interval")
        if not 1 <= limit <= 1000:
            raise ValueError("kline limit must be between 1 and 1000")
        payload = self._get(
            "/api/v3/klines",
            {
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
                "startTime": start_time_ms,
                "endTime": end_time_ms,
            },
        )
        return [list(item) for item in payload]

    def aggregate_trades(self, symbol: str, limit: int = 100) -> list[dict[str, Any]]:
        payload = self._get("/api/v3/aggTrades", {"symbol": symbol, "limit": limit})
        return [dict(item) for item in payload]

    def depth(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        if limit not in {5, 10, 20, 50, 100, 500, 1000, 5000}:
            raise ValueError("unsupported Binance depth limit")
        return dict(self._get("/api/v3/depth", {"symbol": symbol, "limit": limit}))

    def account_information(self) -> dict[str, Any]:
        self._ensure_clock_synchronized()
        return dict(self._get("/api/v3/account", signed=True, market_data=False))

    def trade_fees(self, symbol: str | None = None) -> list[dict[str, Any]]:
        self._ensure_clock_synchronized()
        payload = self._get(
            "/sapi/v1/asset/tradeFee",
            {"symbol": symbol},
            signed=True,
            market_data=False,
        )
        return [dict(item) for item in payload]

    def commission_rates(self, symbol: str) -> dict[str, Any]:
        self._ensure_clock_synchronized()
        return dict(
            self._get(
                "/api/v3/account/commission",
                {"symbol": symbol},
                signed=True,
                market_data=False,
            )
        )
