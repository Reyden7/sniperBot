"""Dynamic Binance Spot filter parsing and exact Decimal compatibility checks."""

from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Any

from sniper.binance.models import SymbolRules


class BinanceFilterError(ValueError):
    """Raised when exchange metadata is missing a required Spot filter."""


def _decimal(item: dict[str, Any], key: str, default: str = "0") -> Decimal:
    return Decimal(str(item.get(key, default)))


def _active_market_lot(
    market_filter: dict[str, Any] | None, lot_filter: dict[str, Any]
) -> dict[str, Any]:
    if market_filter is None:
        return lot_filter
    if _decimal(market_filter, "minQty") <= 0 or _decimal(market_filter, "stepSize") <= 0:
        return lot_filter
    return market_filter


def parse_symbol_rules(symbol: dict[str, Any]) -> SymbolRules:
    """Parse all required filters from one live ``exchangeInfo`` symbol."""
    filters = {item["filterType"]: item for item in symbol.get("filters", [])}
    missing = {"PRICE_FILTER", "LOT_SIZE"} - filters.keys()
    if missing:
        raise BinanceFilterError(f"missing required filters: {sorted(missing)}")
    price_filter = filters["PRICE_FILTER"]
    lot_filter = filters["LOT_SIZE"]
    market_filter = _active_market_lot(filters.get("MARKET_LOT_SIZE"), lot_filter)
    notional_filter = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {}
    minimum_notional = _decimal(notional_filter, "minNotional")
    maximum_notional_raw = _decimal(notional_filter, "maxNotional")
    permissions = tuple(str(item) for item in symbol.get("permissions", []))
    if not permissions:
        permission_sets = symbol.get("permissionSets", [])
        permissions = tuple(
            sorted({str(item) for permission_set in permission_sets for item in permission_set})
        )
    return SymbolRules(
        symbol=str(symbol["symbol"]),
        status=str(symbol.get("status", "UNKNOWN")),
        base_asset=str(symbol["baseAsset"]),
        quote_asset=str(symbol["quoteAsset"]),
        spot_trading_allowed=bool(symbol.get("isSpotTradingAllowed", "SPOT" in permissions)),
        permissions=permissions,
        order_types=tuple(str(item) for item in symbol.get("orderTypes", [])),
        tick_size=_decimal(price_filter, "tickSize"),
        minimum_price=_decimal(price_filter, "minPrice"),
        maximum_price=_decimal(price_filter, "maxPrice"),
        minimum_quantity=_decimal(lot_filter, "minQty"),
        maximum_quantity=_decimal(lot_filter, "maxQty"),
        quantity_step=_decimal(lot_filter, "stepSize"),
        market_minimum_quantity=_decimal(market_filter, "minQty"),
        market_maximum_quantity=_decimal(market_filter, "maxQty"),
        market_quantity_step=_decimal(market_filter, "stepSize"),
        minimum_notional=minimum_notional,
        maximum_notional=maximum_notional_raw if maximum_notional_raw > 0 else None,
        filters_raw=tuple(dict(item) for item in symbol.get("filters", [])),
    )


def floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        raise BinanceFilterError("step must be positive")
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def ceil_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        raise BinanceFilterError("step must be positive")
    return (value / step).to_integral_value(rounding=ROUND_UP) * step


def minimum_order_quantity(rules: SymbolRules, ask: Decimal, *, market: bool = True) -> Decimal:
    """Return the smallest valid quantity satisfying quantity and notional filters."""
    if ask <= 0:
        raise BinanceFilterError("ask must be positive")
    minimum = rules.market_minimum_quantity if market else rules.minimum_quantity
    maximum = rules.market_maximum_quantity if market else rules.maximum_quantity
    step = rules.market_quantity_step if market else rules.quantity_step
    notional_quantity = rules.minimum_notional / ask if rules.minimum_notional > 0 else Decimal(0)
    quantity = ceil_to_step(max(minimum, notional_quantity), step)
    if quantity > maximum:
        raise BinanceFilterError("minimum order exceeds maximum quantity")
    return quantity


def validate_order(
    rules: SymbolRules,
    *,
    quantity: Decimal,
    price: Decimal,
    market: bool = True,
) -> tuple[bool, tuple[str, ...]]:
    """Validate a hypothetical order without submitting anything."""
    reasons = []
    minimum = rules.market_minimum_quantity if market else rules.minimum_quantity
    maximum = rules.market_maximum_quantity if market else rules.maximum_quantity
    step = rules.market_quantity_step if market else rules.quantity_step
    if rules.status != "TRADING" or not rules.spot_trading_allowed:
        reasons.append("SYMBOL_NOT_SPOT_TRADABLE")
    if quantity < minimum:
        reasons.append("BELOW_MINIMUM_QUANTITY")
    if quantity > maximum:
        reasons.append("ABOVE_MAXIMUM_QUANTITY")
    if quantity != floor_to_step(quantity, step):
        reasons.append("INVALID_QUANTITY_STEP")
    if price < rules.minimum_price or (rules.maximum_price > 0 and price > rules.maximum_price):
        reasons.append("PRICE_OUT_OF_RANGE")
    if price != floor_to_step(price, rules.tick_size):
        reasons.append("INVALID_PRICE_TICK")
    notional = quantity * price
    if notional < rules.minimum_notional:
        reasons.append("BELOW_MINIMUM_NOTIONAL")
    if rules.maximum_notional is not None and notional > rules.maximum_notional:
        reasons.append("ABOVE_MAXIMUM_NOTIONAL")
    return not reasons, tuple(reasons)
