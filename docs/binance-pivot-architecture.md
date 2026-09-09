# SNIPER — Architecture du pivot Binance Spot

## Périmètre du premier livrable

Le runtime actif est un pipeline de données et de compatibilité **Binance Spot**. Il ne
contient aucun endpoint de création, modification ou annulation d'ordre. Les modes
Futures, margin, levier et LIVE sont interdits par configuration. `READ_ONLY` est la
valeur par défaut ; `LIVE_TRADING_ENABLED=true` ou `TRADING_MODE=LIVE` provoque un refus.

Les recherches EURUSD A–D.14 restent présentes comme archive historique. Aucun module
Binance ne les importe et aucune conclusion Forex n'influence la sélection crypto.

## Flux du premier livrable

```text
Official Binance Spot REST ──> exchangeInfo / account / fees / market snapshots
                 │
                 ├──> dynamic SymbolRules (PRICE_FILTER, LOT_SIZE,
                 │    MARKET_LOT_SIZE, MIN_NOTIONAL/NOTIONAL)
                 │
Official Spot WebSocket ─────> aggTrade / bookTicker / kline 1m,5m,15m
                 │
                 v
Canonical UTC models ────────> raw immutable Parquet parts
                 │             normalized deduplicated Parquet parts
                 v
CryptoUniverseScanner ───────> one most-liquid quote per priority base
                 │             spread, depth, volume, volatility, min order
                 v
BinanceCostModel ────────────> explicit cost and net-edge ranking proxy
```

Le choix de la quote n'utilise aucun suffixe codé en dur. Pour une même base, le scanner
compare le volume 24 h exprimé dans l'actif de base, seule unité directement comparable
entre ses paires. Toutes les quotes Spot réellement retournées sont conservées dans le
rapport. En présence d'un compte authentifié, la compatibilité tient aussi compte du
solde disponible et de `canTrade`.

## Adaptateur REST en lecture seule

`BinanceReadOnlyClient` applique une liste blanche fermée : ping, heure serveur,
`exchangeInfo`, book ticker, ticker 24 h, klines, trades agrégés, profondeur, compte et
commissions. Toute autre route, notamment `/api/v3/order`, est refusée avant le réseau.

Les appels signés emploient HMAC-SHA256, synchronisent l'horloge, ajoutent `recvWindow`
et lisent les secrets uniquement depuis l'environnement. Les réponses 418/429 suivent
`Retry-After` avec tentatives bornées. Aucun secret ou URL signée n'entre dans une erreur.

Références officielles :

- [Binance Spot REST API](https://developers.binance.com/en/docs/products/spot/rest-api)
- [Binance Spot account and commission API](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/ws-api/account)
- [Binance Spot WebSocket streams](https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams)
- [Binance Spot filters](https://developers.binance.com/docs/binance-spot-api-docs/filters)

## Coûts

Les frais maker/taker proviennent en priorité de l'API du compte. Si aucune clé n'est
configurée, un fallback non nul et configurable est utilisé avec la provenance
`CONFIGURED_FALLBACK`; il ne doit jamais être présenté comme le tarif réel du compte.

```text
EXPECTED_NET_EDGE
  = M5_EXPECTED_GROSS_MOVE_PROXY
  - OBSERVED_BID_ASK_SPREAD
  - TWO_SIDED_EXPECTED_SLIPPAGE
  - ENTRY_FEE
  - EXIT_FEE
```

Le buffer d'incertitude est séparé. Cette valeur sert seulement au classement de
l'univers dans ce lot : elle n'est ni une stratégie qualifiée ni une instruction d'ordre.

## Stockage

```text
data/binance/raw/{dataset}/date=YYYY-MM-DD/part-{content-hash}.parquet
data/binance/normalized/{dataset}/date=YYYY-MM-DD/part-{content-hash}.parquet
data/binance/reports/
```

Les fichiers bruts sont immuables et nommés par hash. Les écritures sont atomiques ; les
clés d'événement sont dédupliquées dans le batch et contre les parts existantes. Les
timestamps canoniques sont UTC. Les données dérivées futures iront dans
`data/binance/features`, sans modifier les données brutes.

## Étapes ultérieures non commencées

M15/M5/M1 Feature Engine, setups, OpportunityEngine qualifié, Risk Engine,
DailyPerformanceEngine, PositionManager, exécution Spot, backtester, paper/testnet et
qualification LIVE restent hors de ce premier livrable. Leur ajout ne pourra pas
réintroduire martingale, grid, averaging down, trade forcé ou prix mid d'exécution.
