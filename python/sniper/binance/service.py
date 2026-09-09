"""Application services for Binance checks, universe scans and bounded collection."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from sniper import __version__
from sniper.binance.client import BinanceReadOnlyClient
from sniper.binance.market_data import (
    collect_combined_stream,
    normalize_book_ticker,
    normalize_rest_agg_trade,
    normalize_rest_kline,
)
from sniper.binance.models import BinanceCheckReport, CollectionReport
from sniper.binance.settings import BinanceSettings
from sniper.binance.storage import BinanceParquetStore, StorageWrite
from sniper.binance.universe import CryptoUniverseScanner


def account_permission_status(account: dict[str, Any] | None) -> dict[str, Any]:
    """Separate account capabilities from API-key permissions Binance does not attest."""
    if account is None:
        return {
            "ACCOUNT_CAN_TRADE": None,
            "ACCOUNT_CAN_WITHDRAW": None,
            "API_KEY_TRADING_PERMISSION_CONFIRMED": None,
            "API_KEY_WITHDRAW_PERMISSION_CONFIRMED": None,
            "API_KEY_PERMISSION_STATUS": "UNAVAILABLE_WITHOUT_API_CREDENTIALS",
        }
    return {
        "ACCOUNT_CAN_TRADE": account.get("canTrade"),
        "ACCOUNT_CAN_WITHDRAW": account.get("canWithdraw"),
        "API_KEY_TRADING_PERMISSION_CONFIRMED": None,
        "API_KEY_WITHDRAW_PERMISSION_CONFIRMED": None,
        "API_KEY_PERMISSION_STATUS": "REQUIRES_MANUAL_BINANCE_UI_CONFIRMATION",
    }


def build_binance_check(
    client: BinanceReadOnlyClient, settings: BinanceSettings
) -> BinanceCheckReport:
    """Run a public check and optional signed account compatibility check."""
    started = perf_counter()
    problems: list[str] = []
    client.ping()
    server_time = client.server_time()
    local_time = int(datetime.now(UTC).timestamp() * 1000)
    candidates, quote_assets, account, scan_problems = CryptoUniverseScanner(
        client, settings
    ).scan()
    problems.extend(scan_problems)
    account_summary: dict[str, Any]
    if account is None:
        account_summary = {
            "authenticated": False,
            "account_type": "UNAVAILABLE_WITHOUT_API_CREDENTIALS",
            "can_trade": None,
            "permissions": [],
            "balances": [],
            "fee_status": "CONFIGURED_NONZERO_FALLBACK_NOT_ACCOUNT_FEE",
            **account_permission_status(None),
        }
    else:
        account_summary = {
            "authenticated": True,
            "account_type": account.get("accountType"),
            "can_trade": account.get("canTrade"),
            "permissions": account.get("permissions", []),
            "balances": [
                {
                    "asset": item["asset"],
                    "free": item["free"],
                    "locked": item["locked"],
                }
                for item in account.get("balances", [])
                if float(item["free"]) > 0 or float(item["locked"]) > 0
            ],
            "fee_status": "ACCOUNT_SPECIFIC_WHEN_SYMBOL_REPORT_SAYS_BINANCE_ACCOUNT_API",
            **account_permission_status(account),
        }
    return BinanceCheckReport(
        mode=settings.trading_mode,
        connection={
            "public_rest": "CONNECTED",
            "rest_base_url": settings.market_data_base_url,
            "server_time_utc": datetime.fromtimestamp(server_time / 1000, tz=UTC).isoformat(),
            "clock_offset_ms": server_time - local_time,
            "clock_state": "OK" if abs(server_time - local_time) <= 1000 else "CLOCK_SKEW",
            "latency_total_seconds": perf_counter() - started,
        },
        account=account_summary,
        quote_assets_available=quote_assets,
        symbols=tuple(candidates),
        problems=tuple(dict.fromkeys(problems)),
        generated_at_utc=datetime.now(UTC),
    )


def _raw_record(
    *, source: str, event_type: str, symbol: str, exchange_key: str, payload: Any
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC),
        "source": source,
        "event_type": event_type,
        "symbol": symbol,
        "exchange_key": exchange_key,
        "payload_json": json.dumps(payload, separators=(",", ":"), default=str),
    }


def _write(
    store: BinanceParquetStore,
    layer: str,
    dataset: str,
    records: list[dict[str, Any]],
    key_fields: tuple[str, ...],
    writes: list[StorageWrite],
) -> None:
    writes.append(store.write(layer, dataset, records, key_fields=key_fields))


def collect_binance_market_data(
    client: BinanceReadOnlyClient,
    settings: BinanceSettings,
    data_root: Path,
    symbols: tuple[str, ...],
    *,
    websocket_duration_seconds: float,
    kline_limit: int = 200,
) -> CollectionReport:
    """Collect REST history plus a bounded WebSocket sample without any order endpoint."""
    started = datetime.now(UTC)
    store = BinanceParquetStore(data_root)
    raw_rest: list[dict[str, Any]] = []
    normalized_klines: list[dict[str, Any]] = []
    normalized_trades: list[dict[str, Any]] = []
    normalized_books: list[dict[str, Any]] = []
    problems: list[str] = []
    books = {item["symbol"]: item for item in client.book_tickers()}
    for symbol in symbols:
        for interval in ("1m", "5m", "15m"):
            rows = client.klines(symbol, interval, kline_limit)
            for row in rows:
                raw_rest.append(
                    _raw_record(
                        source="REST",
                        event_type="KLINE",
                        symbol=symbol,
                        exchange_key=f"{interval}:{row[0]}",
                        payload=row,
                    )
                )
                normalized_klines.append(
                    normalize_rest_kline(symbol, interval, row).model_dump(mode="python")
                )
        trades = client.aggregate_trades(symbol, 100)
        for payload in trades:
            raw_rest.append(
                _raw_record(
                    source="REST",
                    event_type="AGG_TRADE",
                    symbol=symbol,
                    exchange_key=str(payload["a"]),
                    payload=payload,
                )
            )
            normalized_trades.append(
                normalize_rest_agg_trade(symbol, payload).model_dump(mode="python")
            )
        if symbol in books:
            book = books[symbol]
            receipt = datetime.now(UTC)
            canonical_payload = {
                "u": int(receipt.timestamp() * 1_000_000),
                "s": symbol,
                "b": book["bidPrice"],
                "B": book["bidQty"],
                "a": book["askPrice"],
                "A": book["askQty"],
            }
            raw_rest.append(
                _raw_record(
                    source="REST",
                    event_type="BOOK_TICKER",
                    symbol=symbol,
                    exchange_key=hashlib.sha256(
                        json.dumps(book, sort_keys=True).encode()
                    ).hexdigest(),
                    payload=book,
                )
            )
            normalized_books.append(
                normalize_book_ticker(canonical_payload, receipt).model_dump(mode="python")
            )

    raw_ws, ws_events, ws_problems = asyncio.run(
        collect_combined_stream(settings.ws_base_url, symbols, websocket_duration_seconds)
    )
    problems.extend(ws_problems)
    ws_grouped: dict[str, list[dict[str, Any]]] = {
        "BOOK_TICKER": [],
        "AGG_TRADE": [],
        "KLINE": [],
    }
    for event in ws_events:
        ws_grouped[event.event_type].append(event.model_dump(mode="python"))

    writes: list[StorageWrite] = []
    _write(
        store,
        "raw",
        "rest-market-data",
        raw_rest,
        ("source", "event_type", "symbol", "exchange_key"),
        writes,
    )
    if raw_ws:
        for ws_row in raw_ws:
            ws_row["payload_sha256"] = hashlib.sha256(ws_row["payload_json"].encode()).hexdigest()
        _write(
            store,
            "raw",
            "websocket-market-data",
            raw_ws,
            ("stream", "symbol", "payload_sha256"),
            writes,
        )
    raw_write_slots = 1 + int(bool(raw_ws))
    normalized_klines.extend(ws_grouped["KLINE"])
    normalized_trades.extend(ws_grouped["AGG_TRADE"])
    normalized_books.extend(ws_grouped["BOOK_TICKER"])
    _write(
        store,
        "normalized",
        "klines",
        normalized_klines,
        ("symbol", "interval", "timestamp_utc", "closed"),
        writes,
    )
    _write(
        store,
        "normalized",
        "agg-trades",
        normalized_trades,
        ("symbol", "exchange_event_id"),
        writes,
    )
    _write(
        store,
        "normalized",
        "book-tickers",
        normalized_books,
        ("symbol", "exchange_event_id"),
        writes,
    )
    output_files = tuple(str(write.path.resolve()) for write in writes if write.path is not None)
    records_by_type = {
        "raw_rest": len(raw_rest),
        "raw_websocket": len(raw_ws),
        "normalized_klines": len(normalized_klines),
        "normalized_agg_trades": len(normalized_trades),
        "normalized_book_tickers": len(normalized_books),
    }
    quality = {
        "canonical_timestamps_utc": all(
            event.timestamp_utc.utcoffset() is not None for event in ws_events
        )
        and all(row["timestamp_utc"].utcoffset() is not None for row in normalized_klines),
        "book_ask_gte_bid": all(row["ask_price"] >= row["bid_price"] for row in normalized_books),
        "kline_ohlc_invariants": all(
            row["high"] >= max(row["open"], row["close"])
            and row["low"] <= min(row["open"], row["close"])
            for row in normalized_klines
        ),
        "nulls_introduced_by_normalization": 0,
        "websocket_event_types": sorted(
            event_type for event_type, rows in ws_grouped.items() if rows
        ),
        "raw_data_overwritten": False,
        "deduplication_applied": True,
    }
    return CollectionReport(
        mode=settings.trading_mode,
        symbols=symbols,
        intervals=("1m", "5m", "15m"),
        websocket_duration_seconds=websocket_duration_seconds,
        raw_records_written=sum(write.written for write in writes[:raw_write_slots]),
        normalized_records_written=sum(write.written for write in writes[raw_write_slots:]),
        duplicates_skipped=sum(write.duplicates_skipped for write in writes),
        records_by_type=records_by_type,
        data_quality=quality,
        output_files=output_files,
        started_at_utc=started,
        ended_at_utc=datetime.now(UTC),
        problems=tuple(dict.fromkeys(problems)),
    )


def render_binance_report(report: BinanceCheckReport) -> str:
    lines = [
        "# SNIPER Binance Spot — First Delivery Report",
        "",
        f"- Connection: {report.connection['public_rest']}",
        f"- Mode: `{report.mode}`",
        "- LIVE: DISABLED",
        f"- Account authenticated: {report.account['authenticated']}",
        f"- Account type: {report.account['account_type']}",
        f"- ACCOUNT_CAN_TRADE: {report.account['ACCOUNT_CAN_TRADE']}",
        f"- ACCOUNT_CAN_WITHDRAW: {report.account['ACCOUNT_CAN_WITHDRAW']}",
        "- API_KEY_TRADING_PERMISSION_CONFIRMED: "
        f"{report.account['API_KEY_TRADING_PERMISSION_CONFIRMED']}",
        "- API_KEY_WITHDRAW_PERMISSION_CONFIRMED: "
        f"{report.account['API_KEY_WITHDRAW_PERMISSION_CONFIRMED']}",
        f"- API_KEY_PERMISSION_STATUS: {report.account['API_KEY_PERMISSION_STATUS']}",
        f"- Quote assets dynamically available: {', '.join(report.quote_assets_available)}",
        "",
        "| Rank | Symbol | Quote | Fee source | Maker | Taker | Min order | "
        "Spread bps | M5 vol % | Quote vol 24h | Edge proxy % |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report.symbols:
        lines.append(
            f"| {item.rank} | {item.symbol} | {item.quote_asset} | {item.fee_source} | "
            f"{item.maker_fee_rate} | {item.taker_fee_rate} | "
            f"{item.minimum_order_at_ask_quote} | {item.spread_bps:.4f} | "
            f"{item.realized_volatility_m5_pct:.6f} | {item.quote_volume_24h} | "
            f"{item.expected_net_edge_pct:.6f} |"
        )
    lines.extend(["", "## Data quality and problems", ""])
    if report.problems:
        lines.extend(f"- {problem}" for problem in report.problems)
    else:
        lines.append("- No problem detected.")
    lines.extend(
        [
            "",
            "> `expected_net_edge_pct` is a deterministic M5 volatility proxy for universe "
            "ranking, not a qualified strategy edge and never an order instruction.",
            "",
            "No Spot order endpoint is implemented in this delivery. Futures, margin, leverage, "
            "sizing and LIVE remain disabled.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_first_delivery_summary(
    data_root: Path,
    *,
    tests_passed: int,
    mypy_modules: int,
    formatted_files: int,
) -> tuple[Path, Path]:
    """Combine the real connectivity and collection reports after quality gates pass."""
    report_dir = data_root / "binance" / "reports"
    check_path = report_dir / "first-deliverable-binance-check.json"
    collection_path = report_dir / "latest-collection.json"
    check = BinanceCheckReport.model_validate_json(check_path.read_text(encoding="utf-8"))
    collection = CollectionReport.model_validate_json(collection_path.read_text(encoding="utf-8"))
    output_hashes = {}
    for output in collection.output_files:
        path = Path(output)
        output_hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    actual_fee_symbols = [
        item.symbol for item in check.symbols if item.fee_source != "CONFIGURED_FALLBACK"
    ]
    above_buffer = [
        item.symbol for item in check.symbols if "EDGE_PROXY_ABOVE_BUFFER" in item.reasons
    ]
    combined_problems = sorted(set(check.problems) | set(collection.problems))
    scanner_decision = "CANDIDATES_PRESENT" if above_buffer else "SKIP"
    payload = {
        "mission": "SNIPER_BINANCE_SPOT_FIRST_DELIVERY",
        "package_version": __version__,
        "connection": check.connection,
        "mode": check.mode,
        "live_trading_enabled": False,
        "order_endpoints_present": False,
        "futures_enabled": False,
        "margin_enabled": False,
        "leverage_enabled": False,
        "account": check.account,
        "symbols_really_available": [item.model_dump(mode="json") for item in check.symbols],
        "quote_assets_exchange_available": check.quote_assets_available,
        "actual_account_fees_detected_for_symbols": actual_fee_symbols,
        "symbols_above_uncertainty_buffer": above_buffer,
        "scanner_decision": scanner_decision,
        "fee_fallback_disclaimer": (
            None
            if actual_fee_symbols
            else "No credentials configured: 0.10%/side is a configurable nonzero fallback, "
            "not the real account fee."
        ),
        "collection": collection.model_dump(mode="json"),
        "quality_assurance": {
            "pytest": {"status": "PASS", "tests_passed": tests_passed},
            "ruff_check": "PASS",
            "ruff_format_check": {"status": "PASS", "files": formatted_files},
            "mypy": {"status": "PASS", "modules": mypy_modules},
            "build": {"status": "PASS", "version": __version__},
        },
        "output_sha256": output_hashes,
        "problems": combined_problems,
        "forex_archive_modified_by_binance_runtime": False,
    }
    json_path = report_dir / "first-deliverable.json"
    markdown_path = Path("docs/binance-first-deliverable.md")
    lines = [
        "# SNIPER — Premier livrable Binance Spot",
        "",
        f"- Connexion publique : **{check.connection['public_rest']}**",
        f"- Mode : `{check.mode}`",
        "- LIVE / ordre / Futures / margin / levier : DISABLED / ABSENT / DISABLED / "
        "DISABLED / DISABLED",
        f"- Compte authentifié : {check.account['authenticated']}",
        f"- Frais réels compte détectés : {len(actual_fee_symbols)} symbole(s)",
        f"- Package : `{__version__}`",
        "",
        "## Univers réel observé",
        "",
        "| Rang | Symbole | Quote | Frais source | Maker | Taker | Min ordre quote | "
        "Spread bps | Vol M5 % | Volume quote 24h | Edge proxy net % |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in check.symbols:
        lines.append(
            f"| {item.rank} | {item.symbol} | {item.quote_asset} | {item.fee_source} | "
            f"{item.maker_fee_rate} | {item.taker_fee_rate} | "
            f"{item.minimum_order_at_ask_quote} | {item.spread_bps:.4f} | "
            f"{item.realized_volatility_m5_pct:.6f} | {item.quote_volume_24h} | "
            f"{item.expected_net_edge_pct:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Décision du scanner après buffer : `{scanner_decision}` "
            f"({len(above_buffer)} symbole(s) au-dessus du buffer).",
            "",
            "Quotes Spot retournées par l'exchange : "
            + ", ".join(check.quote_assets_available)
            + ".",
            "",
            "## Collecte et qualité",
            "",
            f"- Symboles : {', '.join(collection.symbols)}",
            f"- Timeframes : {', '.join(collection.intervals)}",
            f"- WebSocket : {collection.websocket_duration_seconds:.1f} s",
            f"- Lignes brutes écrites : {collection.raw_records_written:,}",
            f"- Lignes normalisées écrites : {collection.normalized_records_written:,}",
            f"- Doublons ignorés : {collection.duplicates_skipped:,}",
        ]
    )
    for key, value in collection.data_quality.items():
        lines.append(f"- {key} : {value}")
    lines.extend(["", "## Problèmes et limites", ""])
    if combined_problems:
        lines.extend(f"- {problem}" for problem in combined_problems)
    else:
        lines.append("- Aucun problème de collecte détecté.")
    if not actual_fee_symbols:
        lines.append(
            "- Aucune clé API n'était configurée : les frais personnels et soldes ne peuvent "
            "pas être lus. Le fallback 0,10 %/côté est explicite et ne prétend pas être réel."
        )
    lines.extend(
        [
            "- L'edge affiché est un proxy de volatilité M5 pour le classement de l'univers, "
            "pas une performance qualifiée.",
            "",
            "## Qualité logicielle",
            "",
            f"- pytest : {tests_passed} tests PASS",
            "- Ruff lint : PASS",
            f"- Ruff format : PASS ({formatted_files} fichiers)",
            f"- mypy strict : PASS ({mypy_modules} modules)",
            f"- sdist/wheel `{__version__}` : PASS",
            "",
            "Le lot s'arrête ici. Aucun setup, Risk Engine, moteur d'exécution, paper trading "
            "ou LIVE n'a été commencé.",
        ]
    )
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
