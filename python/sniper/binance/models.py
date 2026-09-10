"""Canonical exchange-independent crypto market-data and compatibility models."""

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import Field, model_validator

from sniper.config import Model


class SymbolRules(Model):
    symbol: str
    status: str
    base_asset: str
    quote_asset: str
    spot_trading_allowed: bool
    permissions: tuple[str, ...]
    order_types: tuple[str, ...]
    tick_size: Decimal = Field(gt=0)
    minimum_price: Decimal = Field(ge=0)
    maximum_price: Decimal = Field(ge=0)
    minimum_quantity: Decimal = Field(gt=0)
    maximum_quantity: Decimal = Field(gt=0)
    quantity_step: Decimal = Field(gt=0)
    market_minimum_quantity: Decimal = Field(gt=0)
    market_maximum_quantity: Decimal = Field(gt=0)
    market_quantity_step: Decimal = Field(gt=0)
    minimum_notional: Decimal = Field(ge=0)
    maximum_notional: Decimal | None = Field(default=None, gt=0)
    filters_raw: tuple[dict[str, Any], ...]


class FeeSchedule(Model):
    symbol: str
    maker_rate: Decimal = Field(ge=0, le=Decimal("0.02"))
    taker_rate: Decimal = Field(ge=0, le=Decimal("0.02"))
    source: Literal["BINANCE_ACCOUNT_API", "ACCOUNT_COMMISSION_RATE", "CONFIGURED_FALLBACK"]
    is_account_specific: bool
    commission_details: dict[str, Any] = Field(default_factory=dict)
    special_pricing_visible: bool = False


class MakerFillSimulation(Model):
    """External queue-aware evidence required before treating a maker entry as usable."""

    symbol: str
    observations: int = Field(ge=0)
    fill_probability: Decimal = Field(ge=0, le=1)
    median_time_to_fill_ms: Decimal = Field(ge=0)
    adverse_selection_pct: Decimal = Field(ge=0)
    queue_position_modeled: bool
    post_only_modeled: bool


class TradablePairCostProfile(Model):
    """Comparable Spot-pair economics, independent of a token's nominal price."""

    symbol: str
    base_asset: str
    quote_asset: str
    maker_fee: Decimal = Field(ge=0)
    taker_fee: Decimal = Field(ge=0)
    spread_pct: Decimal = Field(ge=0)
    estimated_slippage_pct: Decimal = Field(ge=0)
    maker_taker_round_trip_cost_pct: Decimal = Field(ge=0)
    taker_taker_round_trip_cost_pct: Decimal = Field(ge=0)
    quote_volume_24h: Decimal = Field(ge=0)
    depth: Decimal = Field(ge=0)
    expected_move_pct: Decimal = Field(ge=0)
    move_to_cost_ratio_maker: Decimal = Field(ge=0)
    move_to_cost_ratio_taker: Decimal = Field(ge=0)
    expected_net_edge_taker_pct: Decimal
    expected_net_edge_maker_pct: Decimal
    maker_fill_simulation_credible: bool = False
    preferred_execution_mode: Literal["TAKER_ENTRY_TAKER_EXIT", "MAKER_ENTRY_TAKER_EXIT"] | None = (
        None
    )
    trade_allowed: bool = False

    @property
    def expected_net_edge_taker(self) -> Decimal:
        return self.expected_net_edge_taker_pct

    @property
    def expected_net_edge_maker(self) -> Decimal:
        return self.expected_net_edge_maker_pct


class CanonicalBookTicker(Model):
    event_type: Literal["BOOK_TICKER"] = "BOOK_TICKER"
    symbol: str
    timestamp_utc: datetime
    exchange_event_id: int
    bid_price: Decimal = Field(gt=0)
    bid_quantity: Decimal = Field(ge=0)
    ask_price: Decimal = Field(gt=0)
    ask_quantity: Decimal = Field(ge=0)
    spread: Decimal = Field(ge=0)
    spread_bps: Decimal = Field(ge=0)


class CanonicalAggTrade(Model):
    event_type: Literal["AGG_TRADE"] = "AGG_TRADE"
    symbol: str
    timestamp_utc: datetime
    exchange_event_id: int
    price: Decimal = Field(gt=0)
    quantity: Decimal = Field(gt=0)
    quote_quantity: Decimal = Field(gt=0)
    buyer_is_maker: bool
    taker_side: Literal["BUY", "SELL"]


class CanonicalKline(Model):
    event_type: Literal["KLINE"] = "KLINE"
    symbol: str
    timestamp_utc: datetime
    close_time_utc: datetime
    interval: Literal["1m", "5m", "15m"]
    open: Decimal = Field(gt=0)
    high: Decimal = Field(gt=0)
    low: Decimal = Field(gt=0)
    close: Decimal = Field(gt=0)
    volume: Decimal = Field(ge=0)
    quote_volume: Decimal = Field(ge=0)
    trades_count: int = Field(ge=0)
    taker_buy_base_volume: Decimal = Field(ge=0)
    taker_buy_quote_volume: Decimal = Field(ge=0)
    closed: bool


class CostEstimate(Model):
    symbol: str
    expected_gross_move_pct: Decimal
    entry_fee_pct: Decimal = Field(ge=0)
    exit_fee_pct: Decimal = Field(ge=0)
    spread_pct: Decimal = Field(ge=0)
    expected_slippage_pct: Decimal = Field(ge=0)
    commission_pct: Decimal = Field(ge=0)
    estimated_cost_pct: Decimal = Field(ge=0)
    uncertainty_buffer_pct: Decimal = Field(ge=0)
    expected_net_edge_pct: Decimal
    acceptable: bool
    fee_source: str


class UniverseCandidate(TradablePairCostProfile):
    symbol: str
    base_asset: str
    quote_asset: str
    status: str
    rank: int = Field(ge=1)
    opportunity_score: Decimal
    expected_net_edge_pct: Decimal
    expected_gross_move_pct: Decimal
    estimated_cost_pct: Decimal
    fee_source: str
    maker_fee_rate: Decimal
    taker_fee_rate: Decimal
    commission_details: dict[str, Any] = Field(default_factory=dict)
    special_pricing_visible: bool = False
    bid: Decimal
    ask: Decimal
    spread_bps: Decimal
    quote_volume_24h: Decimal
    trades_count_24h: int
    realized_volatility_m5_pct: Decimal
    top20_bid_depth_quote: Decimal
    top20_ask_depth_quote: Decimal
    minimum_quantity: Decimal
    quantity_step: Decimal
    minimum_notional: Decimal
    minimum_order_at_ask_quote: Decimal
    tick_size: Decimal | None = Field(default=None, gt=0)
    available_quote_balance: Decimal | None = None
    compatible_with_available_balance: bool | None = None
    ranking_components: dict[str, Decimal] = Field(default_factory=dict)
    reasons: tuple[str, ...]

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_cost_profile(cls, value: Any) -> Any:
        """Read frozen V1 candidates without inventing an executable maker mode."""
        if not isinstance(value, dict) or "maker_fee" in value:
            return value
        migrated = dict(value)

        def decimal(key: str) -> Decimal:
            return Decimal(str(migrated[key]))

        maker_fee = decimal("maker_fee_rate")
        taker_fee = decimal("taker_fee_rate")
        spread_pct = decimal("spread_bps") / Decimal(100)
        taker_cost = decimal("estimated_cost_pct")
        slippage_pct = max(taker_cost - spread_pct - taker_fee * Decimal(200), Decimal(0))
        maker_cost = (
            spread_pct / Decimal(2)
            + slippage_pct / Decimal(2)
            + (maker_fee + taker_fee) * Decimal(100)
        )
        expected_move = decimal("expected_gross_move_pct")
        ratio_floor = Decimal("0.000000000001")
        migrated.update(
            {
                "maker_fee": maker_fee,
                "taker_fee": taker_fee,
                "spread_pct": spread_pct,
                "estimated_slippage_pct": slippage_pct,
                "maker_taker_round_trip_cost_pct": maker_cost,
                "taker_taker_round_trip_cost_pct": taker_cost,
                "depth": min(
                    decimal("top20_bid_depth_quote"),
                    decimal("top20_ask_depth_quote"),
                ),
                "expected_move_pct": expected_move,
                "move_to_cost_ratio_maker": expected_move / max(maker_cost, ratio_floor),
                "move_to_cost_ratio_taker": expected_move / max(taker_cost, ratio_floor),
                "expected_net_edge_taker_pct": decimal("expected_net_edge_pct"),
                "expected_net_edge_maker_pct": expected_move - maker_cost,
                "maker_fill_simulation_credible": False,
                "preferred_execution_mode": None,
                "trade_allowed": False,
            }
        )
        return migrated


class BinanceCheckReport(Model):
    mode: str
    live_trading_enabled: Literal[False] = False
    order_endpoints_present: Literal[False] = False
    connection: dict[str, Any]
    account: dict[str, Any]
    quote_assets_available: tuple[str, ...]
    symbols: tuple[UniverseCandidate, ...]
    problems: tuple[str, ...]
    generated_at_utc: datetime


class CollectionReport(Model):
    mode: str
    live_trading_enabled: Literal[False] = False
    symbols: tuple[str, ...]
    intervals: tuple[str, ...]
    websocket_duration_seconds: float = Field(ge=0)
    raw_records_written: int = Field(ge=0)
    normalized_records_written: int = Field(ge=0)
    duplicates_skipped: int = Field(ge=0)
    records_by_type: dict[str, int]
    data_quality: dict[str, Any]
    output_files: tuple[str, ...]
    started_at_utc: datetime
    ended_at_utc: datetime
    problems: tuple[str, ...]
