# Architecture des phases A à D.9

Perimetre : fondation read-only, diagnostic broker et donnees historiques EURUSD.

CLI Typer -> configuration validee -> contexte MT5Client -> observations
anonymisees -> validate_broker_for_capital -> BrokerCompatibilityReport -> texte/JSON.
La connexion est fermee meme en cas d'erreur. L'adaptateur utilise uniquement
initialize, terminal_info, account_info, symbol_info, symbol_select (abonnement
Market Watch), symbol_info_tick, order_calc_profit, order_calc_margin, last_error
et shutdown. Il ne change pas les autorisations de trading du terminal.

Les calculs monetaires viennent des equivalents Python de
[OrderCalcProfit](https://www.mql5.com/en/docs/python_metatrader5/mt5ordercalcprofit_py)
et [OrderCalcMargin](https://www.mql5.com/en/docs/python_metatrader5/mt5ordercalcmargin_py).
Ils sont exprimes dans la devise du compte ; la marge calculee ne tient pas compte
des positions ou ordres en cours. Le rapport compare donc aussi a la marge libre
observee et au capital du challenge. Ce diagnostic n'est pas une verification
d'execution d'un ordre.

Les donnees du compte sont reduites a la devise, equity, marge libre et drapeaux
de permission ; aucun nom, login, serveur ou secret n'entre dans le rapport.
Les resultats JSON conservent les observations, la politique versionnee, les
prix exacts des stops, les calculs par direction et tous les motifs de refus.

Les tests injectent un double de l'API ; ils ne lancent jamais un terminal.
L'import MetaTrader5 est retarde jusqu'a l'ouverture explicite du client.
Les modules Python ne peuvent pas envoyer d'ordre et l'EA refuse OnInit.
La collecte suit ce flux :

```text
MT5 copy_ticks_range -> normalisation UTC -> validation/deduplication exacte
                     -> Parquet ticks journalier
                     -> constructeur incremental M1
                     -> agregateurs incrementaux M5 et M15
                     -> Parquet barres journalier + rapport JSON
```

Seules les partitions journalieres touchees sont relues pour consolider une bougie
partielle entre deux collectes. Aucun historique complet n'est rescane a chaque tick.
Le backtester suit un second flux strictement hors-ligne :

```text
Parquet ticks -> tri UTC stable -> evenement MarketTick
              -> ordres en attente (latence + sequence)
              -> courtier simule (volume, marge, risque)
              -> execution ASK/BID + slippage + commission
              -> SL/TP tick -> compte/equity -> journal -> metriques
```

La strategie ne recoit qu'un tick courant. Elle n'a acces ni aux bougies futures ni au
tableau de ticks source. Les protections sont evaluees sur BID pour BUY et ASK pour
SELL.

Le plan d'analyse Phase D est separe du backtester et de tout sizing :

```text
Parquet barres completes + ticks visibles a T
  -> FeatureEngine(M15, M5, M1, ticks)
  -> filtres fail-closed(spread, session, news, fraicheur, warm-up)
  -> SignalEngine(barreme fixe 0..100)
  -> BUY / SELL / WAIT + raisons + blockers + distances
  -> sizing_authority=RISK_ENGINE_ONLY
```

Le Signal Engine expose separement le biais de marche M15/M5 et la direction du
trigger M1/ticks. La V1 refuse un trigger oppose a M5. L'evaluateur D.5 appelle ce
moteur uniquement avec les donnees disponibles a T, puis transmet la decision a un
calculateur separe qui peut seul lire les ticks futurs :

```text
Signal(T, donnees <= T) -> observation immuable
                        -> Evaluateur(ticks > T)
                        -> returns LONG/SHORT + MFE/MAE + couts
                        -> tranches de score + diagnostic monotone
```

Une fenetre future manquante est marquee `FUTURE_DATA_INCOMPLETE`; aucun rendement
n'est interpole ou invente. Les labels de session sont des buckets UTC de recherche,
explicitement non presentes comme une verification DST d'une place de marche.

La D.6 ne charge jamais le corpus tick complet. Elle lit une partition journaliere et
ses bordures, calcule une observation au premier tick de chaque minute UTC active,
ecrit cette observation en Parquet journalier puis agrege uniquement les observations.
Les features tick sont vectorisees selon les formules gelees et testees champ par champ
contre le Feature Engine de reference.

La D.7 réutilise ce flux journalier mais scanne le premier tick de chaque seconde UTC
active. Les features de barres sont calculées une fois par minute. Deux évaluations
maximales du Signal Engine inchangé donnent une borne supérieure du score : si elle
prouve que 90 est impossible, la seconde est écartée sans modifier la machine d'état.
Les franchissements restants sont évalués exactement et regroupés en épisodes 90/85.
Seul `FIRST_CROSSING` représente un instant tradable ; `PEAK_SCORE` est rétrospectif.

L'execution live MQL5, l'optimisation et le branchement au Risk Engine appartiennent
aux phases suivantes.

La D.8 conserve V1 comme baseline rejetée et découple la recherche :

```text
Features <= T -> OpportunityModel ─┐
Features <= T -> DirectionModel ───┼-> ExecutionGate -> CANDIDATE ou REJECT
Coûts simulés explicites ──────────┘
```

Les folds walk-forward 1–2 servent à comparer les modèles. Le fold 3 est
`INTERNAL_FREEZE_CHECK` et ne peut déclencher aucune adaptation. Le dataset HOLDOUT
vit sous une racine séparée scellée et n'entre dans aucun chemin d'évaluation D.8.

La D.9 remplace la cible directionnelle terminale par deux chemins de barrières :

```text
Features <= T -> LongOutcomeModel  ─┐
Features <= T -> ShortOutcomeModel ─┼-> MetaGate -> LONG / SHORT / SKIP
Bid/Ask futurs -> labels isolés ─────┘
```

Les labels futurs n'entrent jamais dans la matrice de features. Le MetaGate n'a aucune
autorité de sizing ou d'exécution.

La D.9A durcit uniquement la frontière temporelle TRAIN. Pour chaque configuration,
le dernier label TRAIN doit être entièrement observable avant VALIDATION :

```text
observation_timestamp + configuration.timeout_seconds <= validation_start
```

Les timeouts de purge sont B01=180 s, B02_PRIMARY=300 s, B03=600 s et B04=900 s.
La tolérance de 2 s sert à accepter la dernière quote disponible avant le timeout et
n'étend pas la fin du label. Le HOLDOUT n'entre dans aucun chemin de D.9A.

La D.10 conserve B02_PRIMARY et remplace le MetaGate binaire par une espérance à trois
issues :

```text
Features <= T -> P(TARGET), P(STOP), P(NEITHER) par side
TRAIN payoffs -> EV_BASE par side
Nested daily residuals -> safety buffer
EV_BASE - buffer > 0 -> LONG / SHORT / SKIP
```

Le payoff `NEITHER` est le PnL exécutable au timeout. Les estimations de payoff et le
buffer proviennent uniquement de TRAIN purgé. Le freeze check n'autorise aucune
adaptation, aucun sizing et aucun accès au HOLDOUT.

La D.11 réutilise exactement les modèles et folds D.10 pour produire un tableau OOF
side-level, puis sépare les diagnostics de ranking, calibration économique, changement
de régime et anatomie post-hoc. Ses quantiles sont descriptifs : aucun chemin ne les
convertit en seuil de trading. Les accès `data/holdout-v2` sont refusés avant lecture de
protocole, hash ou modèle. Le verdict D.11 n'a aucune autorité de promotion, de sizing
ou d'exécution.
