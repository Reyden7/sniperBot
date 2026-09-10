from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError
from sniper.binance.client import BinanceApiError, BinanceReadOnlyClient, TransportResponse
from sniper.binance.costs import (
    BinanceCostModel,
    fallback_fee_schedule,
    fee_schedules_from_trade_fee,
)
from sniper.binance.filters import (
    floor_to_step,
    minimum_order_quantity,
    parse_symbol_rules,
    validate_order,
)
from sniper.binance.market_data import (
    combined_stream_url,
    normalize_agg_trade,
    normalize_book_ticker,
    normalize_kline,
    normalize_rest_agg_trade,
)
from sniper.binance.models import MakerFillSimulation
from sniper.binance.service import account_permission_status
from sniper.binance.settings import BinanceSettings
from sniper.binance.storage import BinanceParquetStore
from sniper.binance.universe import CryptoUniverseScanner, LowCostCryptoUniverseScanner


def symbol_payload(symbol: str = "BTCUSDC", base: str = "BTC", quote: str = "USDC"):
    return {
        "symbol": symbol,
        "status": "TRADING",
        "baseAsset": base,
        "quoteAsset": quote,
        "isSpotTradingAllowed": True,
        "permissions": ["SPOT"],
        "orderTypes": ["LIMIT", "MARKET"],
        "filters": [
            {
                "filterType": "PRICE_FILTER",
                "minPrice": "0.01",
                "maxPrice": "1000000",
                "tickSize": "0.01",
            },
            {
                "filterType": "LOT_SIZE",
                "minQty": "0.00001",
                "maxQty": "100",
                "stepSize": "0.00001",
            },
            {
                "filterType": "MARKET_LOT_SIZE",
                "minQty": "0",
                "maxQty": "100",
                "stepSize": "0",
            },
            {"filterType": "NOTIONAL", "minNotional": "10", "maxNotional": "500000"},
        ],
    }


def test_binance_settings_are_read_only_and_require_complete_credentials(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "LIVE")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    with pytest.raises(ValidationError, match="LIVE"):
        BinanceSettings(_env_file=None)
    monkeypatch.setenv("TRADING_MODE", "READ_ONLY")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    monkeypatch.setenv("BINANCE_API_KEY", "only-key")
    with pytest.raises(ValidationError, match="BINANCE_API_SECRET"):
        BinanceSettings(_env_file=None)


def test_low_cost_universe_defaults_cover_requested_assets_and_quotes():
    settings = BinanceSettings(_env_file=None)
    requested = {"DOGE", "SHIB", "PEPE", "POL", "ADA", "TRX", "LINK", "AVAX", "SUI", "XLM"}
    assert requested <= set(settings.preferred_base_assets)
    assert settings.eligible_quote_assets == ("EUR", "USDT", "USDC", "FDUSD")


def test_account_capabilities_are_never_reported_as_api_key_permissions():
    status = account_permission_status({"canTrade": True, "canWithdraw": True})
    assert status["ACCOUNT_CAN_TRADE"] is True
    assert status["ACCOUNT_CAN_WITHDRAW"] is True
    assert status["API_KEY_TRADING_PERMISSION_CONFIRMED"] is None
    assert status["API_KEY_WITHDRAW_PERMISSION_CONFIRMED"] is None
    assert status["API_KEY_PERMISSION_STATUS"] == "REQUIRES_MANUAL_BINANCE_UI_CONFIRMATION"


def test_dynamic_filters_round_quantity_and_enforce_notional():
    rules = parse_symbol_rules(symbol_payload())
    assert rules.market_quantity_step == Decimal("0.00001")
    assert floor_to_step(Decimal("0.001234"), rules.quantity_step) == Decimal("0.00123")
    quantity = minimum_order_quantity(rules, Decimal("50000"))
    assert quantity == Decimal("0.0002")
    valid, reasons = validate_order(
        rules, quantity=quantity, price=Decimal("50000.00"), market=True
    )
    assert valid is True
    assert reasons == ()
    valid, reasons = validate_order(
        rules, quantity=Decimal("0.00011"), price=Decimal("50000.00"), market=True
    )
    assert valid is False
    assert "BELOW_MINIMUM_NOTIONAL" in reasons


def test_binance_cost_model_uses_spread_two_fees_and_two_sided_slippage():
    settings = BinanceSettings(_env_file=None)
    fee = fallback_fee_schedule("BTCUSDC", settings)
    estimate = BinanceCostModel(
        fee,
        slippage_bps_per_side=Decimal("1"),
        uncertainty_buffer_bps=Decimal("2"),
    ).estimate(
        expected_gross_move_pct=Decimal("0.50"),
        bid=Decimal("99.90"),
        ask=Decimal("100.10"),
    )
    assert estimate.commission_pct == Decimal("0.200")
    assert estimate.spread_pct == Decimal("0.200")
    assert estimate.expected_slippage_pct == Decimal("0.02")
    assert estimate.estimated_cost_pct == Decimal("0.420")
    assert estimate.expected_net_edge_pct == Decimal("0.080")
    assert estimate.acceptable is True
    assert estimate.fee_source == "CONFIGURED_FALLBACK"


def test_cost_model_compares_taker_taker_with_maker_taker():
    settings = BinanceSettings(_env_file=None)
    comparison = BinanceCostModel(
        fallback_fee_schedule("DOGEFDUSD", settings),
        slippage_bps_per_side=Decimal("1"),
        uncertainty_buffer_bps=Decimal("2"),
    ).compare_execution_modes(
        expected_gross_move_pct=Decimal("0.50"),
        bid=Decimal("99.90"),
        ask=Decimal("100.10"),
    )
    assert comparison.taker_taker.estimated_cost_pct == Decimal("0.420")
    assert comparison.maker_taker.estimated_cost_pct == Decimal("0.310")
    assert (
        comparison.maker_taker.expected_net_edge_pct > comparison.taker_taker.expected_net_edge_pct
    )


def test_account_trade_fee_payload_is_used_without_zero_fee_assumption():
    schedules = fee_schedules_from_trade_fee(
        [
            {
                "symbol": "BTCUSDC",
                "makerCommission": "0.00075",
                "takerCommission": "0.00100",
            }
        ]
    )
    assert schedules["BTCUSDC"].maker_rate == Decimal("0.00075")
    assert schedules["BTCUSDC"].is_account_specific is True


def test_market_events_are_utc_and_book_uses_executable_spread():
    received = datetime(2026, 9, 9, 12, tzinfo=UTC)
    book = normalize_book_ticker(
        {"u": 4, "s": "BTCUSDC", "b": "99.9", "B": "2", "a": "100.1", "A": "3"},
        received,
    )
    trade = normalize_agg_trade(
        {"a": 7, "s": "BTCUSDC", "p": "100", "q": "0.2", "T": 1788955200000, "m": False}
    )
    kline = normalize_kline(
        {
            "e": "kline",
            "s": "BTCUSDC",
            "k": {
                "t": 1788955200000,
                "T": 1788955499999,
                "i": "5m",
                "o": "99",
                "h": "101",
                "l": "98",
                "c": "100",
                "v": "10",
                "q": "1000",
                "n": 20,
                "V": "6",
                "Q": "600",
                "x": True,
            },
        }
    )
    assert book.spread == Decimal("0.2")
    assert trade.taker_side == "BUY"
    assert trade.timestamp_utc.tzinfo is UTC
    assert kline.interval == "5m"
    assert kline.timestamp_utc.tzinfo is UTC
    rest_trade = normalize_rest_agg_trade(
        "ETHUSDC", {"a": 8, "p": "2000", "q": "0.1", "T": 1788955200000, "m": True}
    )
    assert rest_trade.symbol == "ETHUSDC"
    assert rest_trade.taker_side == "SELL"


def test_combined_websocket_url_contains_all_required_streams():
    url = combined_stream_url("wss://stream.binance.com:9443", ["BTCUSDC"])
    assert "btcusdc@aggTrade" in url
    assert "btcusdc@bookTicker" in url
    assert "btcusdc@kline_1m" in url
    assert "btcusdc@kline_5m" in url
    assert "btcusdc@kline_15m" in url


def test_parquet_storage_is_idempotent_and_deduplicates_events(tmp_path: Path):
    store = BinanceParquetStore(tmp_path)
    records = [
        {"timestamp_utc": datetime(2026, 9, 9, tzinfo=UTC), "symbol": "BTCUSDC", "id": 1},
        {"timestamp_utc": datetime(2026, 9, 9, tzinfo=UTC), "symbol": "BTCUSDC", "id": 1},
    ]
    first = store.write("normalized", "trades", records, key_fields=("symbol", "id"))
    second = store.write("normalized", "trades", records, key_fields=("symbol", "id"))
    assert first.written == 1
    assert first.duplicates_skipped == 1
    assert second.written == 0
    assert second.duplicates_skipped == 2


class SequenceTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.urls = []
        self.headers = []

    def get(self, url, headers, timeout):
        self.urls.append(url)
        self.headers.append(headers)
        return self.responses.pop(0)


def test_rest_adapter_retries_rate_limit_and_refuses_order_path():
    transport = SequenceTransport(
        [
            TransportResponse(429, {"Retry-After": "0"}, b'{"code":-1003,"msg":"limit"}'),
            TransportResponse(200, {}, b"{}"),
        ]
    )
    client = BinanceReadOnlyClient(BinanceSettings(_env_file=None), transport, sleep=lambda _: None)
    client.ping()
    assert len(transport.urls) == 2
    with pytest.raises(BinanceApiError, match="not permitted"):
        client._get("/api/v3/order")


def test_signed_account_read_uses_hmac_and_api_key_without_order_access(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "public-test-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "private-test-secret")
    transport = SequenceTransport(
        [
            TransportResponse(200, {}, b'{"serverTime":1788955200000}'),
            TransportResponse(
                200,
                {},
                b'{"accountType":"SPOT","canTrade":true,"balances":[]}',
            ),
        ]
    )
    client = BinanceReadOnlyClient(BinanceSettings(_env_file=None), transport)
    account = client.account_information()
    assert account["accountType"] == "SPOT"
    assert transport.headers[-1]["X-MBX-APIKEY"] == "public-test-key"
    assert "signature=" in transport.urls[-1]
    assert "private-test-secret" not in transport.urls[-1]


class FakeMarketClient:
    def exchange_info(self):
        return {
            "symbols": [
                symbol_payload("BTCUSDC", "BTC", "USDC"),
                symbol_payload("BTCXYZ", "BTC", "XYZ"),
                symbol_payload("ETHUSDC", "ETH", "USDC"),
            ]
        }

    def book_tickers(self):
        return [
            {
                "symbol": symbol,
                "bidPrice": "99.9",
                "bidQty": "10",
                "askPrice": "100.1",
                "askQty": "10",
            }
            for symbol in ("BTCUSDC", "BTCXYZ", "ETHUSDC")
        ]

    def ticker_24h(self):
        volumes = {"BTCUSDC": "1000000", "BTCXYZ": "10", "ETHUSDC": "500000"}
        return [
            {"symbol": symbol, "volume": volume, "quoteVolume": volume, "count": 1000}
            for symbol, volume in volumes.items()
        ]

    def klines(self, symbol, interval, limit):
        return [
            [
                index,
                "100",
                "101",
                "99",
                str(100 + index / 100),
                "10",
                index + 1,
                "1000",
                20,
                "6",
                "600",
                "0",
            ]
            for index in range(100)
        ]

    def depth(self, symbol, limit):
        return {"bids": [["99.9", "10"]], "asks": [["100.1", "10"]]}


def test_universe_selects_most_liquid_quote_dynamically_without_account():
    scanner = CryptoUniverseScanner(FakeMarketClient(), BinanceSettings(_env_file=None))
    candidates, quotes, account, problems = scanner.scan()
    assert [item.symbol for item in candidates] == ["BTCUSDC", "ETHUSDC"]
    assert quotes == ("USDC",)
    assert account is None
    assert all(item.fee_source == "CONFIGURED_FALLBACK" for item in candidates)
    assert all(item.available_quote_balance is None for item in candidates)
    assert all(item.tick_size == Decimal("0.01") for item in candidates)
    assert "CONFIGURED_NONZERO_FEE_FALLBACK_USED" in problems
    assert all(item.move_to_cost_ratio_taker > 0 for item in candidates)
    assert all(item.preferred_execution_mode == "TAKER_ENTRY_TAKER_EXIT" for item in candidates)
    assert all(item.maker_fill_simulation_credible is False for item in candidates)


def test_maker_mode_requires_credible_queue_aware_fill_simulation():
    evidence = MakerFillSimulation(
        symbol="BTCUSDC",
        observations=200,
        fill_probability=Decimal("0.70"),
        median_time_to_fill_ms=Decimal("250"),
        adverse_selection_pct=Decimal("0.01"),
        queue_position_modeled=True,
        post_only_modeled=True,
    )
    candidates, _, _, _ = LowCostCryptoUniverseScanner(
        FakeMarketClient(),
        BinanceSettings(_env_file=None),
        maker_fill_simulations={"BTCUSDC": evidence},
    ).scan()
    by_symbol = {item.symbol: item for item in candidates}
    assert by_symbol["BTCUSDC"].maker_fill_simulation_credible is True
    assert by_symbol["BTCUSDC"].preferred_execution_mode == "MAKER_ENTRY_TAKER_EXIT"
    assert by_symbol["ETHUSDC"].preferred_execution_mode == "TAKER_ENTRY_TAKER_EXIT"


class FakeAuthenticatedMultiQuoteClient(FakeMarketClient):
    def exchange_info(self):
        return {
            "symbols": [
                symbol_payload("BTCEUR", "BTC", "EUR"),
                symbol_payload("BTCUSDC", "BTC", "USDC"),
            ]
        }

    def book_tickers(self):
        return [
            {
                "symbol": symbol,
                "bidPrice": "99.9",
                "bidQty": "10",
                "askPrice": "100.1",
                "askQty": "10",
            }
            for symbol in ("BTCEUR", "BTCUSDC")
        ]

    def ticker_24h(self):
        return [
            {"symbol": symbol, "volume": "10000", "quoteVolume": "1000000", "count": 1000}
            for symbol in ("BTCEUR", "BTCUSDC")
        ]

    def account_information(self):
        return {
            "accountType": "SPOT",
            "canTrade": True,
            "canWithdraw": True,
            "balances": [
                {"asset": "EUR", "free": "100", "locked": "0"},
                {"asset": "USDC", "free": "0", "locked": "0"},
            ],
        }

    def trade_fees(self):
        return [
            {
                "symbol": symbol,
                "makerCommission": "0" if symbol == "BTCUSDC" else "0.001",
                "takerCommission": "0.001",
            }
            for symbol in ("BTCEUR", "BTCUSDC")
        ]

    def commission_rates(self, symbol):
        return {
            "symbol": symbol,
            "standardCommission": {"maker": "0.001", "taker": "0.001"},
            "specialCommission": {"maker": "0", "taker": "0"},
            "taxCommission": {"maker": "0", "taker": "0"},
            "discount": {
                "enabledForAccount": True,
                "enabledForSymbol": symbol == "BTCUSDC",
                "discountAsset": "BNB",
                "discount": "0.25",
            },
        }


class FakeExpandedUniverseClient(FakeMarketClient):
    symbols = ("DOGEEUR", "DOGEUSDT", "DOGEUSDC", "DOGEFDUSD", "NEARUSDT", "PENNYUSDT")

    def exchange_info(self):
        return {
            "symbols": [
                symbol_payload(symbol, "DOGE", symbol.removeprefix("DOGE"))
                for symbol in self.symbols[:4]
            ]
            + [
                symbol_payload("NEARUSDT", "NEAR", "USDT"),
                symbol_payload("PENNYUSDT", "PENNY", "USDT"),
            ]
        }

    def book_tickers(self):
        return [
            {
                "symbol": symbol,
                "bidPrice": "0.0999",
                "bidQty": "100000",
                "askPrice": "0.1001",
                "askQty": "100000",
            }
            for symbol in self.symbols
        ]

    def ticker_24h(self):
        return [
            {
                "symbol": symbol,
                "volume": "10000000",
                "quoteVolume": "100" if symbol == "PENNYUSDT" else "1000000",
                "count": 1000,
            }
            for symbol in self.symbols
        ]

    def depth(self, symbol, limit):
        return {"bids": [["0.0999", "100000"]], "asks": [["0.1001", "100000"]]}


def test_scanner_compares_all_quotes_and_adds_other_liquid_spot_assets():
    candidates, quotes, _, _ = LowCostCryptoUniverseScanner(
        FakeExpandedUniverseClient(), BinanceSettings(_env_file=None)
    ).scan()
    symbols = {item.symbol for item in candidates}
    assert {"DOGEEUR", "DOGEUSDT", "DOGEUSDC", "DOGEFDUSD", "NEARUSDT"} <= symbols
    assert "PENNYUSDT" not in symbols
    assert quotes == ("EUR", "FDUSD", "USDC", "USDT")


def test_authenticated_universe_keeps_all_quotes_and_reports_balance_compatibility(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "test-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "test-secret")
    settings = BinanceSettings(_env_file=None)
    candidates, quotes, account, problems = CryptoUniverseScanner(
        FakeAuthenticatedMultiQuoteClient(), settings
    ).scan()
    by_symbol = {item.symbol: item for item in candidates}
    assert quotes == ("EUR", "USDC")
    assert account is not None
    assert by_symbol["BTCEUR"].available_quote_balance == Decimal("100")
    assert by_symbol["BTCEUR"].compatible_with_available_balance is True
    assert by_symbol["BTCUSDC"].available_quote_balance is None
    assert by_symbol["BTCUSDC"].compatible_with_available_balance is False
    assert all(item.fee_source == "BINANCE_ACCOUNT_API" for item in candidates)
    assert by_symbol["BTCUSDC"].maker_fee == Decimal(0)
    assert by_symbol["BTCUSDC"].special_pricing_visible is True
    assert by_symbol["BTCUSDC"].commission_details["discount"]["discountAsset"] == "BNB"
    assert "CONFIGURED_NONZERO_FEE_FALLBACK_USED" not in problems
