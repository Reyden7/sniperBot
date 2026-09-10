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
LowCostCryptoUniverseScanner > all sufficiently-liquid EUR/USDT/USDC/FDUSD pairs
                 │             spread, walked depth, volume, movement, exchange filters
                 v
BinanceCostModel ────────────> taker/taker and maker/taker cost, ratio and net edge
```

Le choix de la quote n'utilise aucun suffixe préféré : toutes les paires `TRADING` contre
EUR, USDT, USDC et FDUSD sont comparées. BTC, ETH, SOL, XRP, BNB, DOGE, SHIB, PEPE, POL,
ADA, TRX, LINK, AVAX, SUI et XLM sont explicitement couverts, puis les autres bases
franchissant les seuils de liquidité rejoignent l'univers. Le prix nominal n'entre dans
aucun score. En présence d'un compte authentifié, la compatibilité tient aussi compte du
solde disponible et les détails standard/spéciaux/taxe/remise visibles sont conservés.

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
EXPECTED_NET_EDGE_TAKER
  = M5_EXPECTED_GROSS_MOVE_PROXY
  - OBSERVED_BID_ASK_SPREAD
  - TWO_SIDED_EXPECTED_SLIPPAGE
  - ENTRY_FEE
  - EXIT_FEE
```

`EXPECTED_NET_EDGE_MAKER` retranche une entrée maker, une sortie taker, un demi-spread,
le slippage de sortie et l'adverse selection mesurée. Il reste non exécutable sans une
simulation de fill externe ayant modélisé file d'attente et post-only sur un échantillon
suffisant. Une promotion n'est jamais codée en dur : seuls les frais courants renvoyés
par le compte, ou le fallback conservateur explicitement marqué, sont utilisés.

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

## Deuxième livrable de recherche

Le runtime contient désormais un moteur causal M15/M5/M1, les trois setups gelés
Momentum Pullback, Breakout Retest et Range Mean Reversion, `EconomicTradeFilter`,
`OpportunityEngine`, `BinanceRiskEngine`, `DailyPerformanceEngine` et un replay Spot
long-only. Les M5/M15 sont reconstruits depuis les M1 closes. Le Risk Engine reste la
seule autorité de sizing ; les soldes réels n'entrent pas dans la qualification.

Le backtester entre à Ask, sort à Bid, ajoute slippage et frais, applique les filtres
Binance, impose une position et six trades maximum, et recalcule le PnL depuis les prix
exécutés. Faute de Bid/Ask historique officiel complet, le spread du snapshot initial est
un proxy explicitement bloquant pour une qualification paper.

PositionManager, ordre Spot, paper/testnet et LIVE restent hors périmètre. Aucun ajout ne
peut réintroduire martingale, grid, averaging down, trade forcé ou prix mid d'exécution.
