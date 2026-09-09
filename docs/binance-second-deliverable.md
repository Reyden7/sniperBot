# SNIPER — Deuxième livrable Binance Spot

- Verdict: `BINANCE_STRATEGY_INSUFFICIENT_EVIDENCE`
- Période UTC: 2026-06-10T00:00:00+00:00 → 2026-09-08T00:00:00+00:00
- Symboles: BTCEUR, ETHEUR, SOLEUR, XRPEUR
- M1 / M5 / signaux: 518,400 / 103,680 / 5,086
- Frais maker/taker: 0.001 / 0.001 (`BINANCE_ACCOUNT_API`)
- LIVE / ordres: DISABLED / ABSENT

## Résultats

| Capital EUR | Scénario | Trades | Expectancy EUR | PF net | Return % | DD % | Semaines actives | Semaines + | Jours >=1% | Jours négatifs | Sans trade |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | BASE | 392 | -0.06960280 | 0.2057848923444151665660234298 | -27.284298 | 27.358377 | 14 | 0.00% | 0 | 82 | 1 |
| 100 | STRESS | 275 | -0.08249857 | 0.1482432873336192693260144550 | -22.687106 | 22.687106 | 14 | 0.00% | 0 | 71 | 15 |
| 500 | BASE | 392 | -0.34814922 | 0.2067976055291448454007847184 | -27.294899 | 27.369183 | 14 | 0.00% | 0 | 82 | 1 |
| 500 | STRESS | 275 | -0.41280364 | 0.1496916814473481588065650201 | -22.704200 | 22.704200 | 14 | 0.00% | 0 | 71 | 15 |
| 1000 | BASE | 392 | -0.69644766 | 0.2067958642298158056638307626 | -27.300748 | 27.375047 | 14 | 0.00% | 0 | 82 | 1 |
| 1000 | STRESS | 275 | -0.82578974 | 0.1497116946975856423879461447 | -22.709218 | 22.709218 | 14 | 0.00% | 0 | 71 | 15 |

## Qualification

- FAIL — `net_expectancy_positive`
- FAIL — `net_profit_factor_gte_1_20`
- PASS — `minimum_150_trades`
- PASS — `minimum_8_active_weeks`
- FAIL — `positive_weeks_gte_60pct`
- PASS — `crypto_profit_concentration_lte_60pct`
- FAIL — `week_profit_concentration_lte_60pct`
- FAIL — `stress_expectancy_positive`
- FAIL — `stress_profit_factor_gte_1`
- PASS — `pnl_crosscheck`
- FAIL — `historical_bid_ask_coverage`

## Limites

- `NO_HISTORICAL_BID_ASK_OR_ORDER_BOOK_COVERAGE`
- `CURRENT_SPREAD_SNAPSHOT_USED_AS_EXECUTION_PROXY`
- `CURRENT_EXCHANGE_FILTERS_APPLIED_TO_HISTORICAL_REPLAY`
- `OHLC_SAME_BAR_AMBIGUITY_RESOLVED_STOP_FIRST`
- `SPOT_LONG_ONLY_NO_SHORT_SELLING`
- `REAL_ACCOUNT_BALANCE_NOT_USED_FOR_RESEARCH_SIZING`

Le solde réel n'a servi ni au calibrage ni au sizing du replay. Aucune conversion EUR/USDT, aucun ordre, aucune route LIVE et aucune optimisation post-résultats n'ont été effectués.
