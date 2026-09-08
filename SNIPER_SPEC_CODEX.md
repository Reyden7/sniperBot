# SNIPER
## Cahier des charges technique pour Codex

**Version :** 0.2  
**Date :** 8 septembre 2026  
**Objet :** robot autonome de scalping Forex EUR/USD, M15/M5/M1/tick, avec capital initial strict de 10 EUR et broker nano-lots.

> Ce document est une spécification de développement et de validation. Il ne promet aucun rendement. La cible de +1 %/jour est un objectif expérimental moyen, jamais une obligation quotidienne. Le système doit prioriser la survie du capital et refuser de trader lorsque le risque ne peut pas être respecté.

---

# 1. Instructions principales à Codex

Construire **SNIPER** comme un système de trading algorithmique déterministe, testable et auditable.

Principes obligatoires :

1. **Ne jamais envoyer d'ordre réel par défaut.** Le mode par défaut est `BACKTEST` ou `PAPER`.
2. Le passage en réel doit nécessiter une activation explicite : `LIVE_TRADING_ENABLED=true` + confirmation côté MetaTrader.
3. **La boucle critique live doit être en MQL5**, directement dans MetaTrader 5.
4. Python sert à l'acquisition historique, la recherche, l'analyse, le backtest de recherche, l'optimisation, les rapports et le dashboard.
5. L'EA MQL5 ne doit effectuer aucun calcul lourd, accès réseau non nécessaire, ML, grosse allocation mémoire ou boucle lente dans `OnTick()`.
6. Toute décision doit être journalisée : données utilisées, score, motifs, taille de position, stop, take profit, spread, marge estimée et résultat.
7. Le robot doit pouvoir **ne prendre aucun trade pendant des heures** si aucun setup ne passe les filtres.
8. Aucune martingale, aucun doublement après perte, aucun revenge trading algorithmique.
9. Tous les paramètres doivent être configurables et versionnés.
10. Les calculs de risque doivent utiliser les propriétés réelles du symbole et la devise réelle du compte, pas des constantes de pip codées en dur.

---

# 2. Objectif du projet

Créer un robot de scalping Forex autonome capable de :

- trader **EUR/USD uniquement en V1** ;
- analyser les horizons **M15, M5, M1 et les ticks** ;
- identifier uniquement des setups de haute qualité ;
- choisir BUY, SELL ou WAIT ;
- dimensionner la position selon le capital et le stop ;
- intégrer spread, commission, slippage, marge et contraintes du broker ;
- placer et gérer SL/TP ;q
- couper automatiquement le trading en cas d'anomalie ;
- fonctionner en backtest, paper trading puis éventuellement en réel ;
- produire un historique totalement reproductible et analysable.

## Objectif de performance

Cible expérimentale :

```text
average_daily_return_target = +1.0 %
```

Cette cible ne doit **jamais** provoquer une augmentation du risque pour « rattraper » une journée négative.

La fonction d'objectif du système doit chercher à **maximiser le rendement net ajusté du risque**, pas le rendement brut à n'importe quel prix :

```text
maximize(
    net_return_after_costs
    - drawdown_penalty
    - ruin_risk_penalty
    - instability_penalty
)
```

La priorité reste : survivre, conserver un edge net reproductible, puis laisser la capitalisation composée faire croître le compte.

Le comportement correct est :

```text
pas de setup valide -> 0 trade
perte journalière max atteinte -> STOP
objectif journalier déjà atteint -> risque inchangé ou réduction du risque
journée négative -> aucune augmentation du risque le lendemain
```

---

# 3. Contrainte fondamentale : capital initial de 10 EUR

Capital initial strict :

```text
INITIAL_CAPITAL_EUR = 10.00
```

Cette contrainte influence toute l'architecture.

Le problème principal n'est pas la vitesse du robot mais la **granularité minimale des positions du broker**.

## Règle broker obligatoire pour le challenge 10 EUR

SNIPER V1 ne doit être utilisé qu'avec un broker répondant **simultanément** aux critères suivants :

- accessible légalement depuis la France ;
- MetaTrader 5 disponible ;
- Expert Advisors autorisés ;
- scalping autorisé ;
- **`SYMBOL_VOLUME_MIN <= 0.0001 lot`** sur EUR/USD ;
- **`SYMBOL_VOLUME_STEP <= 0.0001 lot`** sur EUR/USD ;
- modèle de commission compatible avec les nano-lots et **sans commission minimale fixe disproportionnée** ;
- spread EUR/USD suffisamment faible pour que l'edge net reste positif ;
- exécution et slippage mesurables ;
- accès à un historique de ticks exploitable.

Le volume `0.0001` n'est pas une taille de position imposée. Il s'agit de la **granularité minimale recherchée** afin que le Risk Engine puisse ajuster précisément la taille du trade à un compte de 10 EUR.

Un broker qui ne respecte pas ces critères doit être classé `INCOMPATIBLE_FOR_10_EUR`, même s'il est utilisable avec un capital supérieur.

Au démarrage, SNIPER doit récupérer au minimum :

- `SYMBOL_VOLUME_MIN`
- `SYMBOL_VOLUME_STEP`
- `SYMBOL_TRADE_TICK_SIZE`
- `SYMBOL_TRADE_TICK_VALUE`
- `SYMBOL_TRADE_CONTRACT_SIZE`
- spread Bid/Ask réel
- devise du compte
- marge requise

Le robot doit calculer le risque avec `OrderCalcProfit` / `OrderCalcMargin` côté MQL5 ou leurs équivalents Python pendant les outils de recherche.

## Exemple de problème de granularité

Pour EUR/USD, une approximation classique donne environ 10 USD/pip pour 1 lot standard.

- 0.01 lot : environ 0.10 USD/pip ; avec un stop de 5 pips, risque voisin de 0.50 USD, soit environ 5 % d'un compte de 10 EUR.
- 0.001 lot : environ 0.01 USD/pip ; avec un stop de 5 pips, risque voisin de 0.05 USD, soit environ 0.5 %.

Les valeurs exactes doivent toujours être demandées au terminal, car elles dépendent du symbole, du compte et du broker.

## Gate de compatibilité broker

Créer une fonction obligatoire :

```text
BrokerCompatibilityReport ValidateBrokerForCapital(10 EUR)
```

Le rapport doit afficher :

- volume minimum ;
- pas de volume ;
- marge nécessaire au volume minimum ;
- perte estimée au stop minimum raisonnable ;
- pourcentage minimal réellement risquable ;
- spread médian/actuel ;
- coût aller-retour estimé au volume envisagé ;
- éventuelle commission minimale fixe ;
- compatibilité EA/scalping ;
- verdict : `COMPATIBLE`, `TOO_COARSE`, `INSUFFICIENT_MARGIN`, `SPREAD_TOO_HIGH`, `COST_TOO_HIGH`, `INCOMPATIBLE_FOR_10_EUR`.

Si le volume minimum ne permet pas de respecter `MAX_RISK_PER_TRADE`, si `SYMBOL_VOLUME_MIN > 0.0001`, si `SYMBOL_VOLUME_STEP > 0.0001`, ou si les coûts minimums rendent le scalping non viable, le live est interdit pour le challenge 10 EUR.

---

# 4. Stack technique recommandée

## 4.1 Exécution live : MetaTrader 5 + MQL5

**Choix principal : MQL5 Expert Advisor.**

Raisons :

- accès direct aux ticks du terminal ;
- `OnTick()` est déclenché sur les nouvelles cotations ;
- accès natif aux propriétés du symbole et du compte ;
- ordres, SL/TP et événements de transaction natifs ;
- Strategy Tester intégré ;
- possibilité de tester sur ticks réels du broker ;
- moins de couches entre le signal et l'ordre qu'une architecture Python-only.

Important : MetaTrader précise qu'un nouvel événement `NewTick` n'est pas ajouté si un `NewTick` est déjà dans la file ou en cours de traitement. Donc `OnTick()` doit rester **extrêmement court**.

### Boucle MQL5 recommandée

```text
OnTick()
  -> lire Bid/Ask/tick courant
  -> mettre à jour l'état incrémental
  -> vérifier filtres rapides
  -> calculer/mettre à jour le score
  -> si seuil franchi : RiskEngine
  -> si autorisé : ExecutionEngine
  -> retour immédiat
```

Pas de :

- lecture massive de fichiers ;
- appel à un LLM ;
- entraînement ML ;
- requêtes HTTP lentes ;
- scans historiques massifs ;
- gros calculs de DataFrame.

## 4.2 Recherche / backtests / optimisation : Python 3.14.x stable

Version recommandée au démarrage du projet : **Python 3.14.7** ou dernière maintenance stable 3.14.x disponible.

Ne pas utiliser Python 3.15 RC en production tant que 3.15 final et les wheels de toutes les dépendances nécessaires ne sont pas stabilisés.

### Bibliothèques Python

Base :

```text
MetaTrader5
numpy
polars
duckdb
pyarrow
pydantic
pydantic-settings
pytest
hypothesis
rich
typer
```

Optionnelles :

```text
scipy
scikit-learn
optuna
plotly
fastapi
uvicorn
```

### Pourquoi Polars + Parquet + DuckDB

Les données tick deviennent volumineuses rapidement.

- **Parquet** : stockage colonnaire compressé des ticks et barres.
- **Polars** : traitement lazy/parallel, efficace pour transformations et features.
- **DuckDB** : requêtes SQL directement sur les fichiers Parquet sans obligation d'importer tout dans une base serveur.

PostgreSQL n'est pas nécessaire pour le MVP local.

## 4.3 Dashboard

Phase MVP : CLI + rapports HTML/Plotly.

Phase suivante :

```text
FastAPI + WebSocket
React + TypeScript + Vite
```

Le dashboard ne doit jamais se trouver sur le chemin critique d'exécution des ordres.

## 4.4 Gestion de l'environnement

Recommandation :

```text
Git
GitHub
uv (Python dependency/project manager)
pre-commit
ruff
mypy ou pyright
pytest
```

---

# 5. Architecture générale

```text
                          SNIPER

             +---------------------------+
             |   MetaTrader 5 Terminal   |
             +-------------+-------------+
                           |
                        TICKS
                           |
                 +---------v---------+
                 |   MQL5 EA LIVE     |
                 |                    |
                 | MarketState        |
                 | FeatureEngine      |
                 | SignalEngine       |
                 | RiskEngine         |
                 | ExecutionEngine    |
                 | KillSwitch         |
                 +---------+----------+
                           |
                      ORDERS / DEALS
                           |
                     Broker Server

Research plane (hors boucle critique) :

MT5 ticks -> Python Data Collector -> Parquet
                                  -> Polars Features
                                  -> Python Backtester
                                  -> Optimizer
                                  -> Walk-forward validation
                                  -> Reports
                                  -> paramètres validés -> MQL5 inputs
```

---

# 6. Structure du dépôt

```text
sniper/
|
+-- README.md
+-- .gitignore
+-- .env.example
+-- pyproject.toml
+-- uv.lock
+-- Makefile
+-- docs/
|   +-- architecture.md
|   +-- risk-model.md
|   +-- strategy-spec.md
|   +-- validation-protocol.md
|
+-- mt5/
|   +-- experts/
|   |   +-- SniperEA.mq5
|   +-- include/
|       +-- SniperTypes.mqh
|       +-- MarketState.mqh
|       +-- FeatureEngine.mqh
|       +-- SignalEngine.mqh
|       +-- RiskEngine.mqh
|       +-- ExecutionEngine.mqh
|       +-- KillSwitch.mqh
|       +-- Telemetry.mqh
|
+-- python/
|   +-- sniper/
|       +-- __init__.py
|       +-- config.py
|       +-- domain/
|       |   +-- tick.py
|       |   +-- bar.py
|       |   +-- signal.py
|       |   +-- trade.py
|       +-- data/
|       |   +-- mt5_client.py
|       |   +-- collector.py
|       |   +-- parquet_store.py
|       +-- features/
|       |   +-- momentum.py
|       |   +-- volatility.py
|       |   +-- structure.py
|       |   +-- patterns.py
|       +-- strategy/
|       |   +-- scorer.py
|       |   +-- filters.py
|       +-- risk/
|       |   +-- sizing.py
|       |   +-- broker_compatibility.py
|       |   +-- kill_switch.py
|       +-- backtest/
|       |   +-- engine.py
|       |   +-- execution_model.py
|       |   +-- slippage.py
|       |   +-- metrics.py
|       +-- optimization/
|       |   +-- walk_forward.py
|       |   +-- objective.py
|       +-- reports/
|           +-- report.py
|
+-- tests/
|   +-- unit/
|   +-- integration/
|   +-- regression/
|
+-- data/
    +-- raw/
    +-- curated/
    +-- reports/
```

---

# 7. Modèle de marché

Le robot ne « regarde » pas une image de graphique.

Il consomme les données numériques à l'origine du graphique.

## 7.1 Tick

```text
Tick {
    timestamp_utc
    timestamp_server
    symbol
    bid
    ask
    last
    spread_points
    flags
    volume
}
```

Calculer également :

```text
mid = (bid + ask) / 2
spread = ask - bid
```

## 7.2 Barres

Maintenir au minimum :

- M1
- M5
- M15

Chaque barre :

```text
Bar {
    open
    high
    low
    close
    tick_volume
    real_volume
    spread
    open_time
}
```

Les barres doivent être mises à jour incrémentalement, pas recalculées depuis tout l'historique à chaque tick.

---

# 8. Pipeline de décision multi-timeframe

```text
M15  -> contexte + structure
M5   -> tendance locale / momentum
M1   -> setup de scalping
Tick -> timing d'entrée, accélération et contrôle du spread
```

**EUR/USD est le seul symbole tradé en V1.** Les autres paires ne seront ajoutées qu'après validation robuste d'EUR/USD sur backtest, out-of-sample, walk-forward et paper trading.

Une entrée n'est autorisée que si les horizons ne se contredisent pas au-delà d'un seuil configurable.

---

# 9. Feature Engine

Commencer avec des features déterministes et explicables.

## 9.1 Prix / momentum

- variation N ticks ;
- variation 5 s / 15 s / 30 s / 60 s ;
- pente EMA courte / longue ;
- distance aux EMA ;
- RSI M1/M5 ;
- rate of change ;
- accélération du momentum.

## 9.2 Volatilité

- ATR M1/M5 ;
- range courant / ATR ;
- volatilité réalisée courte ;
- expansion/contraction ;
- vitesse de variation du spread.

## 9.3 Structure

- dernier swing high / swing low ;
- distance support/résistance ;
- breakout ;
- retest ;
- range ;
- compression ;
- tendance / régime latéral.

## 9.4 Candlesticks

Coder mathématiquement, sans vision :

- engulfing ;
- hammer ;
- shooting star ;
- doji ;
- inside bar ;
- outside bar.

## 9.5 Microstructure accessible

Selon les données broker :

- séquence bid/ask ;
- tick rate ;
- ratio ticks haussiers/baissiers ;
- accélération du tick rate ;
- spread instantané vs médiane ;
- profondeur de marché si disponible et suffisamment fiable.

---

# 10. Signal Engine : modèle « Sniper »

Chaque opportunité reçoit un score 0-100.

Exemple initial :

```text
Trend alignment M15/M5        0..20
Momentum M1/ticks             0..20
Structure / breakout          0..15
Volatility regime             0..10
Spread quality                0..15
Pattern quality               0..10
News/session risk             0..10
-----------------------------------
TOTAL                         0..100
```

Seuils initiaux :

```text
0..74   -> WAIT
75..84  -> WATCH
85..89  -> signal faible, paper/backtest seulement au début
90..94  -> setup valide
95..100 -> setup premium
```

Ces seuils sont des hypothèses initiales à valider, pas des vérités.

Le moteur doit retourner un objet explicable :

```text
SignalDecision {
    side: BUY | SELL | WAIT
    score
    reasons[]
    blockers[]
    proposed_stop
    proposed_target
    expected_cost
}
```

---

# 11. Risk Engine

Le Risk Engine possède un droit de veto absolu.

Même avec `score = 100`, il peut répondre `REJECT`.

## 11.1 Paramètres initiaux prudents

```text
TARGET_AVG_DAILY_RETURN   = 1.00 %   # objectif moyen, jamais une obligation
NORMAL_RISK_PER_TRADE     = 0.25 %
PREMIUM_RISK_PER_TRADE    = 0.35 %
HARD_MAX_RISK_PER_TRADE   = 0.50 %
SOFT_DAILY_PROFIT_TARGET   = 1.00 %
HARD_DAILY_PROFIT_STOP     = 1.50 %
MAX_DAILY_LOSS             = 1.50 %
MAX_CONSECUTIVE_LOSSES     = 3
MAX_OPEN_POSITIONS         = 1 au MVP
```

Règles associées :

- atteindre `SOFT_DAILY_PROFIT_TARGET` ne force jamais un arrêt, mais doit au minimum réduire le risque ou rendre les critères d'entrée plus stricts ;
- atteindre `HARD_DAILY_PROFIT_STOP` arrête les nouvelles entrées pour la session ;
- atteindre `MAX_DAILY_LOSS` arrête immédiatement les nouvelles entrées pour la session ;
- aucune augmentation de risque n'est autorisée pour atteindre un objectif de rendement.

Avec 10 EUR, la taille minimale du broker peut rendre même 0.25 % de risque impossible. Dans ce cas : **pas de trade réel**. Le broker doit permettre un volume minimum et un pas de volume de `0.0001 lot` ou moins pour être éligible au challenge SNIPER V1.

## 11.2 Dimensionnement

Pseudo-code :

```text
risk_budget = equity * target_risk_pct

for each allowed volume from SYMBOL_VOLUME_MIN by SYMBOL_VOLUME_STEP:
    estimated_loss = abs(OrderCalcProfit(side, symbol, volume, entry, stop))
    keep largest volume where estimated_loss <= risk_budget

if no valid volume:
    REJECT_INSUFFICIENT_GRANULARITY
```

Puis vérifier :

```text
required_margin = OrderCalcMargin(...)
free_margin_after_trade >= minimum_buffer
```

## 11.3 Interdictions absolues

- martingale ;
- augmentation du lot après une perte ;
- suppression/élargissement automatique du stop pour éviter une perte ;
- position sans stop en mode live ;
- ajout à une position perdante au MVP ;
- ouverture quand spread > seuil ;
- ouverture si perte journalière max atteinte.

---

# 12. Spread, slippage et coûts

Pour du scalping, les coûts font partie du signal.

Avant chaque entrée :

```text
expected_gross_edge
- spread
- commission
- expected_slippage
= expected_net_edge
```

Une entrée est rejetée si l'avantage net estimé est insuffisant.

Métriques obligatoires :

- spread à l'entrée ;
- spread à la sortie ;
- spread médian de la session ;
- slippage demandé vs exécuté ;
- commission ;
- résultat brut ;
- résultat net.

---

# 13. News filter et sessions

Phase 1 : filtre horaire simple, centré sur les plages les plus liquides pour EUR/USD. La **session Londres** et surtout le **chevauchement Londres/New York** sont les plages candidates prioritaires, mais leur avantage doit être démontré par les données avant d'être figé.

Phase 2 : utiliser le calendrier économique intégré à MetaTrader pour identifier les événements importants par devise.

Règle configurable :

```text
HIGH_IMPACT_NEWS_BLOCK_BEFORE = 15 min
HIGH_IMPACT_NEWS_BLOCK_AFTER  = 15 min
```

Au MVP, un événement `HIGH_IMPACT` concernant EUR ou USD bloque les nouvelles entrées dans cette fenêtre. Plus tard, son impact pourra être mesuré pour décider s'il faut conserver ou spécialiser ce filtre.

---

# 14. Kill Switch

Créer un composant indépendant.

Déclencheurs :

```text
max daily loss reached
max consecutive losses reached
spread anomaly
market data stale
terminal disconnected
trade server unavailable
margin below threshold
unexpected open position
execution error repeated
clock/time desynchronization
live configuration mismatch
```

États :

```text
ACTIVE
PAUSED
LOCKED_FOR_SESSION
EMERGENCY
```

Tout passage en `EMERGENCY` doit empêcher toute nouvelle entrée.

---

# 15. Execution Engine MQL5

Responsabilités :

- envoyer l'ordre ;
- attacher SL/TP ;
- vérifier le code retour ;
- journaliser requête et résultat ;
- suivre `OnTradeTransaction` ;
- réconcilier positions attendues / positions réelles ;
- empêcher les doubles ordres causés par deux ticks proches.

Utiliser un identifiant `magic number` dédié à SNIPER.

Ajouter un `trade_intent_id` logique dans les logs.

---

# 16. Contrainte de performance OnTick

Puisqu'un nouvel événement `NewTick` peut ne pas être ajouté lorsque le précédent est encore traité, l'EA doit viser un handler court.

Mesurer :

```text
ontick_duration_us
p50
p95
p99
max
```

Architecture :

- calculs incrémentaux ;
- buffers circulaires ;
- indicateurs pré-créés dans `OnInit` ;
- aucun historique massif dans `OnTick` ;
- telemetry bufferisée ;
- tâches non critiques déportées hors `OnTick`.

---

# 17. Data Collector Python

Utiliser l'intégration officielle MetaTrader5 pour :

- `copy_ticks_from`
- `copy_ticks_range`
- `copy_rates_*`
- `symbol_info`
- `symbol_info_tick`
- historique ordres/deals

Stockage recommandé :

```text
data/raw/ticks/symbol=EURUSD/year=2026/month=09/day=08/*.parquet
```

Colonnes horodatées en UTC et conservation de l'heure serveur quand utile.

Ne jamais mélanger naïvement heure locale, UTC et heure serveur.

---

# 18. Backtest Python

Le backtester Python est destiné à la recherche rapide et aux tests automatisés.

Il doit être **event-driven** au niveau tick pour les validations sérieuses.

Il doit modéliser :

- Bid/Ask ;
- spread variable ;
- commission ;
- slippage ;
- latency configurable ;
- taille de lot / pas minimum ;
- marge ;
- stop et target ;
- ordres rejetés ;
- sessions ;
- gaps ;
- données manquantes.

Ne pas utiliser un moteur OHLC simpliste comme validation finale d'une stratégie tick/scalping.

---

# 19. Validation finale MetaTrader Strategy Tester

Après recherche Python, toute stratégie candidate doit être portée dans l'EA MQL5 et validée dans le Strategy Tester.

Mode obligatoire pour la validation finale :

```text
Every tick based on real ticks
```

MetaTrader indique que ce mode utilise les ticks réels accumulés par le broker et fournit les conditions les plus proches du réel disponibles dans le testeur.

Tester également avec délai d'exécution simulé lorsque possible.

---

# 20. Protocole anti-overfitting

Séparer strictement :

```text
TRAIN / RESEARCH
VALIDATION
OUT-OF-SAMPLE TEST
FORWARD PAPER
```

Aucun changement de stratégie ne doit être justifié après consultation répétée du jeu final out-of-sample.

Utiliser walk-forward :

```text
train window -> validate next window
shift
train window -> validate next window
shift
...
```

Les paramètres doivent rester relativement stables entre fenêtres.

Un paramètre qui doit être ré-optimisé chaque semaine avec des valeurs totalement différentes est suspect.

---

# 21. Stress tests

Chaque stratégie candidate doit être testée avec :

- spread x1.25 ;
- spread x1.5 ;
- spread x2 ;
- slippage aléatoire ;
- délai d'exécution supérieur ;
- suppression aléatoire de certains trades favorables ;
- ordre des trades bootstrap/Monte-Carlo ;
- sessions de forte volatilité ;
- semaines calmes ;
- différentes années et régimes de marché.

Le but est de rechercher la **fragilité** de la stratégie, pas seulement son meilleur résultat.

---

# 22. Métriques à produire

Minimum :

```text
starting_equity
ending_equity
net_return_pct
average_daily_return_pct
median_daily_return_pct
max_drawdown_pct
max_drawdown_duration
profit_factor
expectancy_per_trade
win_rate
average_win
average_loss
payoff_ratio
trades_count
trades_per_day
max_consecutive_losses
average_spread
average_slippage
commission_total
cost_per_trade
exposure_time
recovery_factor
```

Afficher également la distribution des rendements journaliers, pas seulement leur moyenne.

---

# 23. Critères de promotion d'une stratégie

Une stratégie ne passe pas directement du backtest au live.

États :

```text
EXPERIMENTAL
BACKTEST_VALIDATED
OUT_OF_SAMPLE_VALIDATED
PAPER_VALIDATED
LIVE_ELIGIBLE
REJECTED
```

Une stratégie doit pouvoir être rejetée même si son rendement est élevé si :

- le drawdown est trop important ;
- les résultats reposent sur quelques trades extrêmes ;
- le spread détruit l'avantage ;
- le résultat disparaît avec slippage ;
- le volume minimum du broker impose trop de risque ;
- l'out-of-sample n'est pas profitable ;
- les paramètres sont instables.

---

# 24. Machine Learning : uniquement phase ultérieure

Ne pas commencer par un réseau de neurones ou un LLM.

MVP : stratégie déterministe + features explicables.

Si suffisamment de données sont collectées, ajouter plus tard un modèle qui estime :

```text
P(trade profitable | market state)
expected_return_after_costs
expected_adverse_excursion
```

Le ML ne doit pas contrôler directement le risque maximal.

Le Risk Engine reste déterministe et prioritaire.

---

# 25. Observabilité

Chaque décision doit produire un log structuré.

Exemple :

```json
{
  "timestamp": "2026-09-08T08:31:12.123Z",
  "symbol": "EURUSD",
  "decision": "WAIT",
  "score": 82.4,
  "spread_points": 3,
  "trend_score": 18,
  "momentum_score": 17,
  "structure_score": 12,
  "blockers": ["score_below_entry_threshold"]
}
```

Pour un ordre :

```text
signal_id
trade_intent_id
symbol
side
score
entry_requested
entry_executed
volume
stop
limit
risk_budget
estimated_loss_at_stop
margin_required
spread
slippage
commission
exit_reason
net_pnl
```

---

# 26. Configuration

Exemple de configuration de recherche :

```yaml
mode: PAPER
initial_capital: 10.0
account_currency: EUR
market: FOREX
symbols:
  - EURUSD

broker_requirements:
  require_mt5: true
  require_ea_allowed: true
  require_scalping_allowed: true
  max_symbol_volume_min_lot: 0.0001
  max_symbol_volume_step_lot: 0.0001
  reject_fixed_min_commission_if_disproportionate: true

target_average_daily_return_pct: 1.0
risk:
  normal_per_trade_pct: 0.25
  premium_per_trade_pct: 0.35
  hard_max_per_trade_pct: 0.50
  soft_daily_profit_target_pct: 1.0
  hard_daily_profit_stop_pct: 1.5
  max_daily_loss_pct: 1.5
  max_consecutive_losses: 3
  max_open_positions: 1

signal:
  watch_threshold: 75
  candidate_threshold: 85
  trade_threshold: 90
  premium_threshold: 95

timeframes:
  context: M15
  trend: M5
  setup: M1
  trigger: TICK

sessions:
  prioritize_london: true
  prioritize_london_new_york_overlap: true
  hardcode_session_edge: false  # doit être démontré par les données

execution:
  max_spread_points: null   # calibrer sur EURUSD et par session
  max_slippage_points: null # calibrer par mesure réelle

news:
  enabled: true
  currencies:
    - EUR
    - USD
  high_impact_only: true
  block_before_minutes: 15
  block_after_minutes: 15
```

Les seuils `null` doivent être calibrés à partir de données réelles et non inventés.

---

# 27. Tests automatisés

## Unit tests

Tester notamment :

- construction de bougies ;
- indicateurs incrémentaux ;
- détection des patterns ;
- calcul des scores ;
- sizing ;
- arrondi au `SYMBOL_VOLUME_STEP` ;
- refus si volume minimal trop risqué ;
- daily loss ;
- consecutive losses ;
- spread filters ;
- stop/target long et short ;
- PnL après coûts.

## Property-based tests

Utiliser Hypothesis pour des invariants :

```text
volume >= 0
risk_after_rounding <= hard_max_risk
no live order when live flag is false
no order without stop
kill switch => no new trade
```

## Regression tests

Conserver des petits jeux de ticks fixes avec résultats attendus.

Une modification du moteur ne doit pas silencieusement modifier les décisions historiques sans explication.

---

# 28. Sécurité

- aucun mot de passe broker dans Git ;
- `.env` ignoré ;
- `.env.example` sans secret ;
- secrets live idéalement gérés par le terminal MT5 ;
- logs sans identifiants sensibles ;
- mode live explicitement désactivé dans les tests ;
- ordre réel impossible depuis CI.

---

# 29. Roadmap d'implémentation

## Phase A - Fondation

- créer repository ;
- config Python ;
- tests ;
- connexion read-only à MT5 ;
- récupération propriétés compte/symboles ;
- rapport de compatibilité 10 EUR.

## Phase B - Données

- téléchargement ticks ;
- Parquet partitionné ;
- reconstruction M1/M5/M15 ;
- validation données.

## Phase C - Backtester

- event loop tick ;
- Bid/Ask ;
- coûts ;
- positions ;
- SL/TP ;
- métriques.

## Phase D - Features et signal

- momentum ;
- volatilité ;
- structure ;
- patterns ;
- scorer ;
- WAIT/BUY/SELL.

## Phase E - Risk Engine

- sizing exact ;
- marge ;
- lot min/step ;
- drawdown ;
- daily limits ;
- kill switch.

## Phase F - EA MQL5

- architecture modulaire ;
- `OnTick` court ;
- signal ;
- risk ;
- execution ;
- `OnTradeTransaction` ;
- logs.

## Phase G - Validation

- ticks réels Strategy Tester ;
- walk-forward ;
- out-of-sample ;
- stress tests ;
- paper trading.

## Phase H - Dashboard

- courbe equity ;
- PnL ;
- drawdown ;
- signaux ;
- rejets ;
- spread/slippage ;
- état KillSwitch.

---

# 30. Première mission à exécuter par Codex

Ne pas essayer d'implémenter tout le robot en une seule passe.

Commencer par ce lot :

## Task 1 - Scaffold

Créer la structure du dépôt décrite dans ce document.

## Task 2 - Python MT5 read-only adapter

Créer un client Python qui :

1. initialise MetaTrader 5 ;
2. lit le compte ;
3. lit EURUSD ;
4. récupère Bid/Ask ;
5. récupère les propriétés de volume/tick ;
6. vérifie explicitement `SYMBOL_VOLUME_MIN <= 0.0001` et `SYMBOL_VOLUME_STEP <= 0.0001` ;
7. calcule la marge du volume minimum ;
8. estime la perte au stop pour différents stops configurables ;
9. estime spread, commission et coût aller-retour minimal ;
10. produit `BrokerCompatibilityReport` pour 10 EUR ;
11. n'envoie **aucun ordre**.

## Task 3 - Tests

Ajouter les tests unitaires autour du rapport et du sizing.

## Task 4 - CLI

Commande souhaitée :

```bash
sniper broker-check --capital 10 --symbol EURUSD
```

Sortie exemple :

```text
SNIPER Broker Compatibility
Capital:             10.00 EUR
Symbol:              EURUSD
Min volume:          ...
Volume step:         ...
Nano-lot compliant:  YES/NO
Current spread:      ...
Commission model:    ...
Min round-trip cost: ...
Margin at min volume:...
Risk at 3 pip stop:  ... EUR (...%)
Risk at 5 pip stop:  ... EUR (...%)
Risk at 10 pip stop: ... EUR (...%)
Verdict:             ...
```

Ne pas inventer les valeurs : lire celles du terminal connecté.

---

# 31. Definition of Done du MVP

Le MVP est terminé quand :

1. le repository est reproductible ;
2. les ticks peuvent être collectés et rejoués ;
3. le backtester gère Bid/Ask et coûts ;
4. les décisions sont explicables ;
5. le sizing respecte les contraintes du broker ;
6. un broker dont `VOLUME_MIN` ou `VOLUME_STEP` dépasse `0.0001 lot`, ou dont les coûts rendent le challenge 10 EUR non viable, est explicitement refusé ;
7. l'EA compile sans warning critique ;
8. `OnTick` est mesuré ;
9. le Strategy Tester peut exécuter un test sur ticks réels ;
10. aucune fonctionnalité live n'est active par défaut ;
11. un rapport complet de performance est généré ;
12. les tests automatisés passent.

---

# 32. Décisions d'architecture à ne pas remettre en cause sans raison mesurée

- **Forex uniquement en V1.**
- **EUR/USD uniquement en V1.**
- **M15 -> M5 -> M1 -> ticks** comme pipeline de décision V1.
- **Broker nano-lots obligatoire : `VOLUME_MIN <= 0.0001` et `VOLUME_STEP <= 0.0001`.**
- **MQL5 pour l'exécution live.**
- **Python pour recherche/data/backtest/optimisation.**
- **Parquet + Polars + DuckDB pour les données locales.**
- Pas de vision artificielle pour « lire » les graphiques : utiliser les données numériques.
- Pas de ML au MVP.
- Pas de martingale.
- Pas d'obligation de trader.
- Cible de **+1 % moyen par journée tradée**, sans garantie ni obligation quotidienne.
- À +1 % journalier : réduire le risque ou durcir le filtre ; à +1.5 % : arrêter les nouvelles entrées de la session.
- Daily loss stop V1 : -1.5 %.
- Risk Engine prioritaire sur Signal Engine.
- Capital 10 EUR traité comme une contrainte système, pas comme un détail de configuration.

---

# 33. Références techniques

**R1 - MQL5 OnTick**  
https://www.mql5.com/en/docs/event_handlers/ontick

**R2 - MetaTrader 5 Strategy Tester**  
https://www.metatrader5.com/en/terminal/help/algotrading/testing

**R3 - MQL5 tests et ticks réels**  
https://www.mql5.com/en/docs/runtime/testing

**R4 - Python MetaTrader5 / copy_ticks_from**  
https://www.mql5.com/en/docs/python_metatrader5/mt5copyticksfrom_py

**R5 - Python MetaTrader5 / order_send**  
https://www.mql5.com/en/docs/python_metatrader5/mt5ordersend_py

**R6 - Propriétés du symbole : volume min/step, tick size/value**  
https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants

**R7 - MQL5 OrderCalcProfit**  
https://www.mql5.com/en/docs/trading/ordercalcprofit

**R8 - MQL5 OrderCalcMargin**  
https://www.mql5.com/en/docs/trading/ordercalcmargin

**R9 - Calendrier économique MQL5**  
https://www.mql5.com/en/docs/calendar

**R10 - Python 3.14.7**  
https://www.python.org/downloads/release/python-3147/

**R11 - DuckDB + Parquet**  
https://duckdb.org/docs/current/guides/file_formats/query_parquet

**R12 - Polars scan_parquet**  
https://docs.pola.rs/api/python/stable/reference/api/polars.scan_parquet.html

**R13 - ESMA, protections CFD retail**  
https://www.esma.europa.eu/press-news/esma-news/esma-agrees-prohibit-binary-options-and-restrict-cfds-protect-retail-investors

---

# 34. Note finale pour Codex

Le but n'est pas de fabriquer un robot qui multiplie les trades pour atteindre arbitrairement +1 % dans la journée. SNIPER V1 doit chercher **peu de trades EUR/USD à très forte conviction**, surtout pendant les plages de liquidité pertinentes, avec M15/M5/M1/ticks et des coûts de transaction explicitement intégrés.

Le but est de fabriquer un système mesurable qui :

```text
observe énormément
filtre énormément
trade rarement
risque peu
mesure tout
survit longtemps
```

La première réussite du projet n'est pas de gagner de l'argent. La première réussite est d'obtenir un moteur dont les résultats de backtest, de validation hors échantillon, de Strategy Tester et de paper trading racontent la même histoire.
