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
    preferred_base_assets: tuple[str, ...] = ("BTC", "ETH", "SOL", "BNB", "XRP")
    eligible_quote_assets: tuple[str, ...] = ("EUR", "USDC", "USDT")
    starting_capital_eur: Decimal | None = Field(
        None, gt=0, validation_alias="STARTING_CAPITAL_EUR"
    )

    @model_validator(mode="after")
    def fail_closed(self) -> Self:
        if (self.api_key is None) != (self.api_secret is None):
            raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET must be provided together")
        if self.trading_mode == "LIVE" or self.live_trading_enabled:
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
