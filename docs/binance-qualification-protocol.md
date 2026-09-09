# SNIPER Binance Spot V1 — protocole de qualification gelé

Protocol ID: `BINANCE_SPOT_V1_FROZEN_2026_09_09`

Ce document est écrit avant lecture des résultats du replay. Aucun paramètre ne peut
être ajusté à partir de ces résultats dans ce livrable.

## Périmètre

- Marché : Binance Spot, long-only ; aucune vente à découvert.
- Période : `2026-06-10T00:00:00Z` à `2026-09-08T00:00:00Z` (fin exclusive).
- Univers recherche : bases BTC/ETH/SOL/BNB/XRP, paires `EUR` réellement `TRADING`,
  volume quote 24 h >= 100 000 EUR, spread <= 20 bps et profondeur top-20 >= 5 000 EUR
  de chaque côté au snapshot initial.
- L'univers et les spreads de ce snapshot initial sont gelés dans
  `qualification-universe-freeze.json`; une variation ultérieure de profondeur ne peut
  ajouter ou retirer un symbole au replay enregistré.
- Scanner opérationnel : toutes les paires réellement `TRADING` contre EUR/USDC/USDT.
- Capital simulé : 100, 500 et 1000 EUR, indépendant du solde réel.
- Position simultanée : une seule.
- Maximum : six trades/jour UTC ; risque normal 0,20 %, plafond 0,35 %.
- Arrêt nouvelles entrées : +1,00 % net/jour, -1,00 % net/jour ou trois pertes
  consécutives. Aucun trade n'est forcé.

## Données et causalité

Les M5 et M15 sont reconstruits exclusivement depuis les M1 closes. Un signal M5 à T
utilise seulement les M1 dont la clôture est inférieure ou égale à T et le dernier M15
entièrement clos. L'entrée est évaluée à l'open M5 suivant. Un gap supérieur à 0,25 ATR
annule le signal. Les tests imposent qu'altérer une bougie future ne change aucun signal
antérieur.

La REST Spot officielle fournit l'historique OHLCV, mais pas un historique complet de
best Bid/Ask ni de carnet. Le replay utilise donc le spread observé au moment du snapshot
de qualification, ce qui est explicitement une approximation. Si aucune série Bid/Ask
historique couvrant la période n'est disponible, le maximum autorisé est
`BINANCE_STRATEGY_INSUFFICIENT_EVIDENCE`, même si les métriques numériques franchissent
les autres seuils.

## Règles fixes

M15 : EMA 9/21, pente et trend-efficiency sur 12 barres ; tendance si efficiency >= 0,35
et écart EMA normalisé >= 0,0005, sinon RANGE.

M5 :

- `MOMENTUM_PULLBACK` : M15 haussier, EMA 9 > EMA 21, pullback sur EMA 9, reprise et
  volume >= 0,8 fois la médiane 20 ; target 1,5 ATR.
- `BREAKOUT_RETEST` : cassure préalable du plus haut 20, volume >= 1,2 fois la médiane,
  retest dans 0,15 %, reprise au-dessus ; target 2 ATR.
- `RANGE_MEAN_REVERSION` : M15 RANGE, clôture préalable sous moyenne 20 - 2 écarts-types,
  puis réentrée haussière ; target moyenne 20.

M1 : momentum trois bougies strictement positif. La mean reversion est interdite en
régime tendanciel. Chaque entrée possède stop et target avant sizing.

## Coûts préenregistrés

BASE : frais taker réels du snapshot compte `BINANCE_ACCOUNT_API` (0,001 par côté),
spread snapshot une fois aller-retour, slippage 1 bp/côté, buffer 2 bps.

STRESS : frais 1,5 fois BASE, spread doublé, slippage 3 bps/côté, buffer 5 bps.

Le PnL est décomposé en market PnL, spread, slippage, commission et net PnL. Un test
croisé recalcule le net directement depuis les prix exécutés.

## Exécution simulée

BUY entre à Ask plus slippage ; une sortie Spot vend à Bid moins slippage. Si stop et
target sont tous deux touchés dans la même M5, le stop est choisi. Timeout : 60 minutes.
Les filtres Binance courants (notional, quantity step, min/max quantity) sont appliqués.
Il s'agit d'une simulation de recherche et non d'une capacité future garantie.

## Qualification

Le scénario principal est 500 EUR BASE. Tous les critères sont requis : expectancy nette
positive, profit factor net >= 1,20, au moins 150 trades, au moins huit semaines actives,
au moins 60 % de semaines positives, au plus 60 % du profit positif issu d'une crypto ou
d'une semaine, résistance STRESS (expectancy > 0 et profit factor >= 1), PnL cross-check
valide et couverture Bid/Ask historique suffisante.

Verdict unique :

- données/trades/semaines ou couverture d'exécution insuffisants :
  `BINANCE_STRATEGY_INSUFFICIENT_EVIDENCE` ;
- effectif suffisant mais au moins un critère économique échoue :
  `BINANCE_STRATEGY_REJECTED` ;
- tous les critères franchis : `BINANCE_STRATEGY_QUALIFIED_FOR_PAPER`.

LIVE, routes d'ordre, Futures, margin et levier restent interdits.
