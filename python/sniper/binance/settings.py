"""Fail-closed configuration for the Binance Spot pivot."""

from decimal import Decimal
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BinanceSettings(BaseSettings):
    """Environment-only Binance settings; LIVE is forbidden in this delivery."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    trading_mode: Literal["READ_ONLY", "PAPER", "TESTNET", "LIVE"] = Field(
        "READ_ONLY", validation_alias="TRADING_MODE"
    )
    live_trading_enabled: bool = Field(False, validation_alias="LIVE_TRADING_ENABLED")
    internal_execution_permission: bool = Field(
        False, validation_alias="INTERNAL_EXECUTION_PERMISSION"
    )
    api_key: SecretStr | None = Field(None, validation_alias="BINANCE_API_KEY")
    api_secret: SecretStr | None = Field(None, validation_alias="BINANCE_API_SECRET")
    rest_base_url: str = Field("https://api.binance.com", validation_alias="BINANCE_REST_BASE_URL")
    market_data_base_url: str = Field(
        "https://data-api.binance.vision",
        validation_alias="BINANCE_MARKET_DATA_BASE_URL",
    )
    ws_base_url: str = Field(
        "wss://stream.binance.com:9443", validation_alias="BINANCE_WS_BASE_URL"
    )
    request_timeout_seconds: float = Field(
        10.0, gt=0, le=60, validation_alias="BINANCE_REQUEST_TIMEOUT_SECONDS"
    )
    recv_window_ms: int = Field(5000, ge=1000, le=60000, validation_alias="BINANCE_RECV_WINDOW_MS")
    fallback_maker_fee_rate: Decimal = Field(
        Decimal("0.001"),
        gt=0,
        le=Decimal("0.02"),
        validation_alias="BINANCE_FALLBACK_MAKER_FEE_RATE",
    )
    fallback_taker_fee_rate: Decimal = Field(
        Decimal("0.001"),
        gt=0,
        le=Decimal("0.02"),
        validation_alias="BINANCE_FALLBACK_TAKER_FEE_RATE",
    )
    slippage_bps_per_side: Decimal = Field(
        Decimal("1"), ge=0, validation_alias="BINANCE_SLIPPAGE_BPS_PER_SIDE"
    )
    uncertainty_buffer_bps: Decimal = Field(
        Decimal("2"), ge=0, validation_alias="BINANCE_UNCERTAINTY_BUFFER_BPS"
    )
    preferred_base_assets: tuple[str, ...] = (
        "BTC",
        "ETH",
        "SOL",
        "XRP",
        "BNB",
        "DOGE",
        "SHIB",
        "PEPE",
        "POL",
        "ADA",
        "TRX",
        "LINK",
        "AVAX",
        "SUI",
        "XLM",
    )
    eligible_quote_assets: tuple[str, ...] = ("EUR", "USDT", "USDC", "FDUSD")
    minimum_quote_volume_24h: Decimal = Field(
        Decimal("500000"),
        ge=0,
        validation_alias="BINANCE_MIN_QUOTE_VOLUME_24H",
    )
    minimum_top20_depth_quote: Decimal = Field(
        Decimal("500"),
        ge=0,
        validation_alias="BINANCE_MIN_TOP20_DEPTH_QUOTE",
    )
    slippage_reference_notional: Decimal = Field(
        Decimal("500"),
        gt=0,
        validation_alias="BINANCE_SLIPPAGE_REFERENCE_NOTIONAL",
    )
    maximum_diagnostic_pairs: int = Field(
        100,
        ge=1,
        le=500,
        validation_alias="BINANCE_MAX_DIAGNOSTIC_PAIRS",
    )
    maker_fill_minimum_observations: int = Field(
        100,
        ge=1,
        validation_alias="BINANCE_MAKER_FILL_MIN_OBSERVATIONS",
    )
    maker_fill_minimum_probability: Decimal = Field(
        Decimal("0.50"),
        ge=0,
        le=1,
        validation_alias="BINANCE_MAKER_FILL_MIN_PROBABILITY",
    )
    v2_minimum_move_to_cost_ratio: Decimal = Field(
        Decimal("3.0"),
        gt=0,
        validation_alias="BINANCE_V2_MIN_MOVE_TO_COST_RATIO",
    )
    v2_maximum_spread_p95_bps: Decimal = Field(
        Decimal("20"),
        gt=0,
        validation_alias="BINANCE_V2_MAX_SPREAD_P95_BPS",
    )
    v2_maximum_slippage_500_bps: Decimal = Field(
        Decimal("10"),
        gt=0,
        validation_alias="BINANCE_V2_MAX_SLIPPAGE_500_BPS",
    )
    starting_capital_eur: Decimal | None = Field(
        None, gt=0, validation_alias="STARTING_CAPITAL_EUR"
    )

    @model_validator(mode="after")
    def fail_closed(self) -> Self:
        if (self.api_key is None) != (self.api_secret is None):
            raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET must be provided together")
        if (
            self.trading_mode == "LIVE"
            or self.live_trading_enabled
            or self.internal_execution_permission
        ):
            raise ValueError("LIVE trading is unavailable in the first Binance delivery")
        if not self.rest_base_url.startswith("https://"):
            raise ValueError("BINANCE_REST_BASE_URL must use HTTPS")
        if not self.market_data_base_url.startswith("https://"):
            raise ValueError("BINANCE_MARKET_DATA_BASE_URL must use HTTPS")
        if not self.ws_base_url.startswith("wss://"):
            raise ValueError("BINANCE_WS_BASE_URL must use WSS")
        return self

    @property
    def authenticated(self) -> bool:
        return self.api_key is not None and self.api_secret is not None

    @property
    def execution_permitted(self) -> bool:
        return False
