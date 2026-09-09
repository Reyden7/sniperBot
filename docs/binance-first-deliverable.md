# SNIPER — Premier livrable Binance Spot

- Connexion publique : **CONNECTED**
- Mode : `READ_ONLY`
- LIVE / ordre / Futures / margin / levier : DISABLED / ABSENT / DISABLED / DISABLED / DISABLED
- Compte authentifié : False
- Frais réels compte détectés : 0 symbole(s)
- Package : `0.15.0`

## Univers réel observé

| Rang | Symbole | Quote | Frais source | Maker | Taker | Min ordre quote | Spread bps | Vol M5 % | Volume quote 24h | Edge proxy net % |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | XRPUSDT | USDT | CONFIGURED_FALLBACK | 0.001 | 0.001 | 5.1217200000000000 | 0.7029 | 0.203346 | 199852932.12932000 | 0.004061 |
| 2 | SOLUSDT | USDT | CONFIGURED_FALLBACK | 0.001 | 0.001 | 5.0768900000000000 | 0.9652 | 0.204413 | 198661711.38093000 | -0.016793 |
| 3 | ETHUSDT | USDT | CONFIGURED_FALLBACK | 0.001 | 0.001 | 5.2399620000000000 | 0.0401 | 0.169756 | 676587623.64350300 | -0.045476 |
| 4 | BNBUSDT | USDT | CONFIGURED_FALLBACK | 0.001 | 0.001 | 5.1908500000000000 | 0.1349 | 0.142635 | 109086832.50641000 | -0.063454 |
| 5 | BTCUSDT | USDT | CONFIGURED_FALLBACK | 0.001 | 0.001 | 5.5178459000000000 | 0.0013 | 0.150715 | 1094786318.73890940 | -0.078720 |

Décision du scanner après buffer : `SKIP` (0 symbole(s) au-dessus du buffer).

Quotes Spot retournées par l'exchange : ARS, BNB, BRL, BTC, ETH, EUR, FDUSD, IDR, JPY, MXN, RLUSD, TRY, U, USD, USD1, USDC, USDS, USDT.

## Collecte et qualité

- Symboles : BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT
- Timeframes : 1m, 5m, 15m
- WebSocket : 5.0 s
- Lignes brutes écrites : 810
- Lignes normalisées écrites : 796
- Doublons ignorés : 5,940
- canonical_timestamps_utc : True
- book_ask_gte_bid : True
- kline_ohlc_invariants : True
- nulls_introduced_by_normalization : 0
- websocket_event_types : ['AGG_TRADE', 'BOOK_TICKER', 'KLINE']
- raw_data_overwritten : False
- deduplication_applied : True

## Problèmes et limites

- CONFIGURED_NONZERO_FEE_FALLBACK_USED
- Aucune clé API n'était configurée : les frais personnels et soldes ne peuvent pas être lus. Le fallback 0,10 %/côté est explicite et ne prétend pas être réel.
- L'edge affiché est un proxy de volatilité M5 pour le classement de l'univers, pas une performance qualifiée.

## Qualité logicielle

- pytest : 254 tests PASS
- Ruff lint : PASS
- Ruff format : PASS (124 fichiers)
- mypy strict : PASS (76 modules)
- sdist/wheel `0.15.0` : PASS

Le lot s'arrête ici. Aucun setup, Risk Engine, moteur d'exécution, paper trading ou LIVE n'a été commencé.
