"""Dynamic liquid-crypto universe discovery for Binance Spot."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import numpy as np

from sniper.binance.client import BinanceApiError, BinanceReadOnlyClient
from sniper.binance.costs import (
    BinanceCostModel,
    fallback_fee_schedule,
    fee_schedule_from_account,
    fee_schedules_from_trade_fee,
)
from sniper.binance.filters import BinanceFilterError, minimum_order_quantity, parse_symbol_rules
from sniper.binance.models import FeeSchedule, UniverseCandidate
from sniper.binance.settings import BinanceSettings


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _market_statistics(rows: list[list[Any]]) -> tuple[Decimal, Decimal]:
    closed = np.asarray([float(row[4]) for row in rows], dtype=float)
    high = np.asarray([float(row[2]) for row in rows], dtype=float)
    low = np.asarray([float(row[3]) for row in rows], dtype=float)
    if len(closed) < 3 or np.any(closed <= 0):
        return Decimal(0), Decimal(0)
    returns = np.diff(np.log(closed)) * 100
    realized_volatility = Decimal(str(float(np.std(returns, ddof=1))))
    move_proxy = Decimal(str(float(np.median((high - low) / closed * 100))))
    return realized_volatility, move_proxy


def _depth_quote(payload: dict[str, Any]) -> tuple[Decimal, Decimal]:
    bid_depth = sum(
        (_decimal(price) * _decimal(quantity) for price, quantity in payload["bids"]),
        Decimal(0),
    )
    ask_depth = sum(
        (_decimal(price) * _decimal(quantity) for price, quantity in payload["asks"]),
        Decimal(0),
    )
    return bid_depth, ask_depth


def _account_balances(account: dict[str, Any] | None) -> dict[str, Decimal]:
    if account is None:
        return {}
    return {
        str(item["asset"]): _decimal(item["free"])
        for item in account.get("balances", [])
        if _decimal(item["free"]) > 0
    }


def resolve_fee_schedules(
    client: BinanceReadOnlyClient,
    settings: BinanceSettings,
    symbols: list[str],
    account: dict[str, Any] | None,
) -> tuple[dict[str, FeeSchedule], list[str]]:
    """Prefer actual account fees and make every fallback explicit and nonzero."""
    problems: list[str] = []
    schedules: dict[str, FeeSchedule] = {}
    if settings.authenticated:
        try:
            schedules = fee_schedules_from_trade_fee(client.trade_fees())
        except (BinanceApiError, KeyError, ValueError) as exc:
            problems.append(f"ACCOUNT_TRADE_FEE_UNAVAILABLE:{type(exc).__name__}")
            if account is not None and account.get("commissionRates"):
                for symbol in symbols:
                    schedules[symbol] = fee_schedule_from_account(symbol, account)
    for symbol in symbols:
        if symbol not in schedules:
            schedules[symbol] = fallback_fee_schedule(symbol, settings)
    if any(not item.is_account_specific for item in schedules.values()):
        problems.append("CONFIGURED_NONZERO_FEE_FALLBACK_USED")
    return schedules, problems


class CryptoUniverseScanner:
    """Rank all eligible Spot pairs without assuming that any symbol exists."""

    def __init__(self, client: BinanceReadOnlyClient, settings: BinanceSettings):
        self.client = client
        self.settings = settings

    def scan(
        self,
    ) -> tuple[list[UniverseCandidate], tuple[str, ...], dict[str, Any] | None, list[str]]:
        problems: list[str] = []
        exchange = self.client.exchange_info()
        book_map = {item["symbol"]: item for item in self.client.book_tickers()}
        ticker_map = {item["symbol"]: item for item in self.client.ticker_24h()}
        account = None
        if self.settings.authenticated:
            try:
                account = self.client.account_information()
            except BinanceApiError as exc:
                problems.append(f"ACCOUNT_UNAVAILABLE:{type(exc).__name__}")
        rules_by_symbol = {}
        for raw_symbol in exchange.get("symbols", []):
            try:
                rules = parse_symbol_rules(raw_symbol)
            except BinanceFilterError, KeyError, ValueError:
                continue
            if (
                rules.base_asset in self.settings.preferred_base_assets
                and rules.quote_asset in self.settings.eligible_quote_assets
                and rules.status == "TRADING"
                and rules.spot_trading_allowed
                and rules.symbol in book_map
                and rules.symbol in ticker_map
            ):
                rules_by_symbol[rules.symbol] = rules
        quote_assets = tuple(sorted({rules.quote_asset for rules in rules_by_symbol.values()}))
        # Keep every real eligible symbol. Account capability flags are informational and
        # are deliberately not interpreted as API-key permissions.
        shortlisted = []
        for base_asset in self.settings.preferred_base_assets:
            matching = [
                rules for rules in rules_by_symbol.values() if rules.base_asset == base_asset
            ]
            if matching:
                shortlisted.extend(matching)
            else:
                problems.append(f"NO_TRADABLE_PAIR:{base_asset}")
        fee_schedules, fee_problems = resolve_fee_schedules(
            self.client,
            self.settings,
            [rules.symbol for rules in shortlisted],
            account,
        )
        problems.extend(fee_problems)
        balances = _account_balances(account)
        candidates = []
        for rules in shortlisted:
            book = book_map[rules.symbol]
            ticker = ticker_map[rules.symbol]
            bid = _decimal(book["bidPrice"])
            ask = _decimal(book["askPrice"])
            if bid <= 0 or ask <= 0 or ask < bid:
                problems.append(f"INVALID_BOOK:{rules.symbol}")
                continue
            try:
                klines = self.client.klines(rules.symbol, "5m", 100)
                volatility, expected_move = _market_statistics(klines)
                bid_depth, ask_depth = _depth_quote(self.client.depth(rules.symbol, 20))
                minimum_quantity = minimum_order_quantity(rules, ask, market=True)
            except (BinanceApiError, BinanceFilterError, KeyError, ValueError) as exc:
                problems.append(f"MARKET_DIAGNOSTIC_FAILED:{rules.symbol}:{type(exc).__name__}")
                continue
            minimum_order = minimum_quantity * ask
            available = balances.get(rules.quote_asset)
            compatible = (
                None
                if account is None
                else bool(available is not None and available >= minimum_order)
            )
            fee = fee_schedules[rules.symbol]
            cost = BinanceCostModel(
                fee,
                slippage_bps_per_side=self.settings.slippage_bps_per_side,
                uncertainty_buffer_bps=self.settings.uncertainty_buffer_bps,
            ).estimate(expected_gross_move_pct=expected_move, bid=bid, ask=ask)
            spread_bps = (ask - bid) / ((ask + bid) / 2) * Decimal(10000)
            reasons = ["SPOT_TRADING", "DYNAMIC_EXCHANGE_FILTERS", "M5_VOLATILITY_PROXY"]
            reasons.append(
                "EDGE_PROXY_ABOVE_BUFFER" if cost.acceptable else "EDGE_PROXY_NOT_ABOVE_BUFFER"
            )
            if compatible is False:
                reasons.append("AVAILABLE_BALANCE_BELOW_MINIMUM_ORDER")
            quote_volume = _decimal(ticker["quoteVolume"])
            depth_total = bid_depth + ask_depth
            capital_score = Decimal(1) if compatible else Decimal(0)
            liquidity_score = Decimal(str(np.log10(float(max(quote_volume, Decimal(1))))))
            depth_score = Decimal(str(np.log10(float(max(depth_total, Decimal(1))))))
            cost_score = -cost.estimated_cost_pct
            volatility_score = min(volatility, Decimal(1))
            ranking_score = (
                liquidity_score
                + depth_score / Decimal(2)
                + cost_score * Decimal(10)
                + volatility_score
                + capital_score * Decimal(2)
            )
            candidates.append(
                UniverseCandidate(
                    symbol=rules.symbol,
                    base_asset=rules.base_asset,
                    quote_asset=rules.quote_asset,
                    status=rules.status,
                    rank=1,
                    opportunity_score=ranking_score,
                    expected_net_edge_pct=cost.expected_net_edge_pct,
                    expected_gross_move_pct=expected_move,
                    estimated_cost_pct=cost.estimated_cost_pct,
                    fee_source=fee.source,
                    maker_fee_rate=fee.maker_rate,
                    taker_fee_rate=fee.taker_rate,
                    bid=bid,
                    ask=ask,
                    spread_bps=spread_bps,
                    quote_volume_24h=quote_volume,
                    trades_count_24h=int(ticker.get("count", 0)),
                    realized_volatility_m5_pct=volatility,
                    top20_bid_depth_quote=bid_depth,
                    top20_ask_depth_quote=ask_depth,
                    minimum_quantity=minimum_quantity,
                    quantity_step=rules.market_quantity_step,
                    minimum_notional=rules.minimum_notional,
                    minimum_order_at_ask_quote=minimum_order,
                    tick_size=rules.tick_size,
                    available_quote_balance=available,
                    compatible_with_available_balance=compatible,
                    ranking_components={
                        "liquidity": liquidity_score,
                        "depth": depth_score,
                        "cost": cost_score,
                        "volatility": volatility_score,
                        "capital_compatibility": capital_score,
                    },
                    reasons=tuple(reasons),
                )
            )
        candidates.sort(key=lambda item: item.opportunity_score, reverse=True)
        ranked = [item.model_copy(update={"rank": rank}) for rank, item in enumerate(candidates, 1)]
        if not ranked:
            problems.append("NO_ELIGIBLE_CRYPTO_SYMBOL")
        return ranked, quote_assets, account, problems
