"""Versioned, fail-closed configuration for the first mission."""

from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Positive = Annotated[Decimal, Field(gt=0, allow_inf_nan=False)]
NonNegative = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CommissionModel(Model):
    """Verified tariff in account currency, charged on each side of a round trip."""

    currency: str = Field(min_length=3)
    per_lot_per_side: NonNegative
    minimum_per_side: NonNegative
    source: str = Field(min_length=1)

    def round_trip(self, volume: Decimal) -> Decimal:
        return 2 * max(self.per_lot_per_side * volume, self.minimum_per_side)


class BrokerCheckConfig(Model):
    schema_version: Literal[1] = 1
    stop_pips: tuple[Positive, ...] = (Decimal(3), Decimal(5), Decimal(10))
    risk_per_trade_pct: Positive = Decimal("0.25")
    hard_max_risk_pct: Annotated[Decimal, Field(gt=0, le=Decimal("0.50"))] = Decimal("0.50")
    minimum_margin_buffer: NonNegative = Decimal(0)
    max_spread_points: Positive | None = None
    max_round_trip_cost_pct: Positive | None = None
    round_trip_slippage_points: NonNegative | None = None
    commission: CommissionModel | None = None
    legally_accessible_from_france: bool | None = None
    scalping_allowed: bool | None = None
    historical_ticks_available: bool | None = None
    terms_source: str | None = None
    max_tick_age_seconds: Positive = Decimal(60)
    max_future_tick_seconds: NonNegative = Decimal(5)

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if not self.stop_pips or len(set(self.stop_pips)) != len(self.stop_pips):
            raise ValueError("stop_pips must be nonempty and unique")
        if self.risk_per_trade_pct > self.hard_max_risk_pct:
            raise ValueError("risk_per_trade_pct exceeds hard_max_risk_pct")
        if any(
            value is not None
            for value in (
                self.legally_accessible_from_france,
                self.scalping_allowed,
                self.historical_ticks_available,
            )
        ) and not (self.terms_source and self.terms_source.strip()):
            raise ValueError("verified broker terms require terms_source")
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SNIPER_", env_nested_delimiter="__", env_file=".env", extra="ignore"
    )

    mode: Literal["PAPER", "BACKTEST"] = "PAPER"
    live_trading_enabled: bool = Field(False, validation_alias="LIVE_TRADING_ENABLED")
    mt5_path: str | None = None
    mt5_timeout_ms: int = Field(10000, ge=1000, le=60000)
    broker: BrokerCheckConfig = Field(default_factory=BrokerCheckConfig)

    @field_validator("live_trading_enabled")
    @classmethod
    def reject_live(cls, enabled: bool) -> bool:
        if enabled:
            raise ValueError("live trading is unavailable in mission 1")
        return False
