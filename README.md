# SNIPER

La mission active est désormais un scanner autonome multi-crypto **Binance Spot**, centré
sur M5 et l'espérance nette après frais. Le premier livrable reste strictement
`READ_ONLY` : Futures, margin, levier, sizing, endpoint d'ordre et LIVE sont absents.

Les recherches Forex EURUSD A à D.14 sont terminées et conservées comme archive en
lecture seule. Elles ne doivent plus être optimisées ni utilisées comme stratégie active.

Commandes du pivot Binance :

```powershell
uv sync --locked
uv run --locked sniper binance-check
uv run --locked sniper crypto-universe
uv run --locked sniper crypto-collect --duration-seconds 10 --data data
```

Sans clés, les endpoints publics fonctionnent et les frais sont un fallback configurable,
non nul et explicitement marqué comme non spécifique au compte. Avec
`BINANCE_API_KEY` et `BINANCE_API_SECRET`, `binance-check` lit le compte et les frais
réels via des endpoints signés de lecture. Les secrets ne sont jamais écrits.

Voir [l'architecture du pivot](docs/binance-pivot-architecture.md).

## Archive Forex EURUSD

Phases A à D.14 : fondation read-only, diagnostic broker 10 EUR, pipeline historique,
backtester tick event-driven, V2/V3 rejetés, D.10 sans preuve suffisante, D.11 instable,
D.12 favorable à la recherche, V4 D.13 non prête à geler et audit économique D.14.
Cette archive ne contient aucun ordre live.

Installation avec Python stable 3.14.x et uv :

```powershell
uv sync --locked --extra mt5
uv run --locked --extra mt5 sniper broker-check --capital 10 --symbol EURUSD
uv run --locked --extra mt5 sniper collect-ticks --start 2026-09-08T07:00:00Z --end 2026-09-08T08:00:00Z --output data
uv run --locked --extra mt5 sniper collect-history --symbol EURUSD --days 90 --output data
uv run --locked sniper backtest --symbol EURUSD --start 2026-09-08T07:46:00Z --end 2026-09-08T07:51:00Z --capital 10 --data data --json
uv run --locked sniper signal --symbol EURUSD --as-of 2026-09-08T08:00:00Z --data data --json
uv run --locked sniper evaluate-signals --symbol EURUSD --start 2026-06-01T00:00:00Z --end 2026-09-01T00:00:00Z --data data --json
uv run --locked sniper validate-edge --symbol EURUSD --start 2026-06-10T00:00:00Z --end 2026-09-08T00:00:00Z --data data
uv run --locked sniper research-v2 --data data
uv run --locked sniper research-v3 --data data
uv run --locked sniper research-d12 --data data --protocol docs/research-protocol-d12.yaml
uv run --locked sniper research-d13 --data data --protocol docs/research-protocol-d13.yaml
uv run --locked --extra mt5 sniper collect-holdout --output data/holdout-v2
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
```

MetaTrader 5 doit etre installe sous Windows et connecte au compte a examiner.
L'option `--terminal-path` permet de choisir son executable. Le package MT5 est
optionnel pour que les tests sur doubles fonctionnent sans terminal, y compris en CI.
`uv sync --locked` suffit pour ces tests. Python 3.14.6 est present sur la machine
de developpement ; le projet accepte les versions de maintenance stables 3.14.x.

Le mode par defaut est PAPER. Aucun chemin d'envoi d'ordre n'est implemente.
`LIVE_TRADING_ENABLED=true` et le mode LIVE sont refuses pour ce premier lot.
L'EA MQL5 est un squelette inerte qui refuse son initialisation.

## Rapport

```powershell
uv run --locked --extra mt5 sniper broker-check --capital 10 --symbol EURUSD --stop-pips 3 --stop-pips 5 --stop-pips 10
uv run --locked --extra mt5 sniper broker-check --config broker-check.json --json
```

`--capital` est toujours exprime en EUR. Les calculs MT5 sont exprimes dans la
devise reelle du compte. Pour ce premier lot, un compte non EUR produit un refus
explicite : aucune conversion ni assimilation d'un compte en cents n'est inventee.
Un nom broker avec suffixe est accepte si ses devises de base/profit sont EUR/USD.

Les commissions, autorisations contractuelles, disponibilite d'un historique
exploitable et seuils calibres ne sont pas deduits des seules proprietes MT5.
Sans ces informations, la sortie affiche UNKNOWN et les motifs du refus.
Le spread courant vient du Bid/Ask ; sa mediane reste inconnue dans ce lot.
Un verdict COMPATIBLE est un diagnostic de faisabilite, jamais une validation
de strategie ni une autorisation de trading reel.

Le fichier JSON optionnel est decrit dans [docs/risk-model.md](docs/risk-model.md).
Ne pas remplir ses valeurs avec des hypotheses presentees comme des mesures.
Codes de sortie : 0 compatible, 1 incompatible/incomplet, 2 configuration ou MT5
indisponible. `--json` produit aussi les erreurs operationnelles sous forme JSON.

## Collecte historique EURUSD

Les bornes sont des instants ISO 8601 obligatoirement timezone-aware. Elles sont
normalisees en UTC et la plage est `[start, end)`. La commande telecharge par blocs,
valide Bid/Ask et les valeurs numeriques, conserve les ticks distincts ayant la meme
milliseconde, supprime seulement les doublons exacts, ecrit des partitions Parquet
journalieres idempotentes et reconstruit incrementiellement M1 puis M5/M15.

```powershell
.venv\Scripts\sniper.exe collect-ticks `
  --start 2026-09-08T07:00:00Z `
  --end 2026-09-08T08:00:00Z `
  --output data `
  --terminal-path "C:\Program Files\MetaTrader 5\terminal64.exe" `
  --json
```

Les ticks sont dans `data/ticks/symbol=EURUSD/date=YYYY-MM-DD`, les bougies dans
`data/bars/timeframe=M1|M5|M15/symbol=EURUSD/date=YYYY-MM-DD`, et chaque diagnostic
JSON dans `data/reports`. Voir [docs/data-pipeline.md](docs/data-pipeline.md).

Pour une collecte massive, `collect-history --days 90` decoupe la plage en jours UTC.
Chaque jour termine recoit un manifeste de completion ; une relance saute uniquement
les tranches ayant ce manifeste valide. Une interruption avant le manifeste rejoue la
tranche, et les ecritures Parquet idempotentes evitent les doublons.

## Backtest tick par tick

La commande `backtest` lit directement les partitions Phase B dans l'ordre UTC. Elle
utilise exclusivement `DummyStrategy(TEST_ONLY)` : une ouverture connue suivie d'une
fermeture apres un nombre fixe de ticks. Elle ne constitue pas une strategie SNIPER et
n'est jamais optimisee.

```powershell
.venv\Scripts\sniper.exe backtest `
  --symbol EURUSD `
  --start 2026-09-08T07:46:00Z `
  --end 2026-09-08T07:51:00Z `
  --capital 10 `
  --data data `
  --execution-latency-ms 25 `
  --slippage-model fixed `
  --slippage-points 1 `
  --commission-per-lot-side 2 `
  --json
```

Le modele par defaut applique un slippage fixe defavorable d'un point. Les modeles
`none` et `random --seed ...` sont disponibles pour les tests. Voir
[docs/backtester.md](docs/backtester.md) et
[docs/validation-protocol.md](docs/validation-protocol.md).

Chaque rapport contient le profil broker `SIMULATED`, son levier effectif, sa grille de
volume et un avertissement indiquant que ces hypotheses ne prouvent aucune capacite de
MetaQuotes-Demo ou d'un broker reel.

## Features et signaux

```powershell
.venv\Scripts\sniper.exe signal `
  --symbol EURUSD `
  --as-of 2026-09-08T08:00:00Z `
  --data data `
  --max-spread-points 2 `
  --session-status allowed `
  --news-status clear `
  --json
```

Le moteur suit M15 contexte/regime, M5 tendance, M1 setup et ticks confirmation. La
sortie BUY/SELL/WAIT distingue `market_bias` de `trigger_direction` et contient le
detail des huit composantes, les raisons, blockers et distances proposees. Elle ne
contient aucun lot : `sizing_authority=RISK_ENGINE_ONLY`.
Les seuils 80/90/95 sont fixes et explicitement non optimises. Sans seuil de spread ou
etat session/news verifies, le moteur echoue en securite vers WAIT. Il force aussi WAIT
si le dernier tick a depasse `--max-tick-age-seconds`, si un warm-up M15/M5/M1 est
incomplet, ou si le trigger contredit la tendance M5. Un flux stale donne toujours une
composante `tick_confirmation=0`.

## Evaluation des signaux (Phase D.5)

```powershell
.venv\Scripts\sniper.exe evaluate-signals `
  --symbol EURUSD `
  --start 2026-06-01T00:00:00Z `
  --end 2026-09-01T00:00:00Z `
  --data data `
  --interval-seconds 60 `
  --max-spread-points 2 `
  --session-status allowed `
  --news-status clear `
  --slippage-points-per-side 1 `
  --json
```

Le rapport conserve chaque observation, les rendements LONG et SHORT a
5/10/30/60/180 secondes, MFE/MAE a 30 secondes, les couts attendus et le ratio
target/cout. Il agrege sans optimiser les tranches 50-59, 60-69, 70-79, 80-89,
90-94 et 95-100. Voir [docs/signal-evaluation.md](docs/signal-evaluation.md).

L'optimisation, le Risk Engine d'execution, la Phase E et le trading live restent hors
de ce perimetre.

## Validation d'edge (Phase D.6)

```powershell
.venv\Scripts\sniper.exe validate-edge `
  --symbol EURUSD `
  --start 2026-06-10T00:00:00Z `
  --end 2026-09-08T00:00:00Z `
  --data data
```

La politique fixe le premier tick de chaque minute UTC active, soit au plus une
observation par minute. Les ticks sont lus jour par jour avec 60 secondes de passe et
180 secondes de futur aux frontieres. Les observations detaillees sont partitionnees
dans `data/evaluations`; le JSON agrege et le rapport Markdown sont dans
`data/reports`. Les sources du Feature/Signal Engine sont controlees par empreinte
avant le scan. Voir [docs/edge-validation.md](docs/edge-validation.md).

## Découverte événementielle des candidats (Phase D.7)

```powershell
.venv\Scripts\sniper.exe discover-candidate-events `
  --symbol EURUSD `
  --start 2026-06-10T00:00:00Z `
  --end 2026-09-08T00:00:00Z `
  --data data
```

La D.7 scanne le premier tick de chaque seconde UTC active. Un épisode commence au
franchissement de 90 sans blocker, reste compatible avec sa direction, puis se termine
sous 90, sur blocker ou changement de direction. Après un épisode, le détecteur ne se
réarme qu'une fois sous 85. `FIRST_CROSSING` est le seul instant tradable ; `PEAK_SCORE`
est rétrospectif et uniquement diagnostique. L'orientation inverse est mesurée sans
modifier la stratégie. Voir [docs/event-discovery.md](docs/event-discovery.md).

## Research Engine V2 (Phase D.8)

V1 est `RESEARCH_REJECTED` et reste conservé avec ses hashes et rapports. V2 sépare
OpportunityModel, DirectionModel et ExecutionGate, sans sizing ni exécution. Le split
est strictement temporel : les folds 1–2 servent à la comparaison, le fold 3 est un
`INTERNAL_FREEZE_CHECK` immuable. Le rapport principal est fixé avant résultats à
300 s / BASE / 1,5×. Le HOLDOUT antérieur est collectable dans une racine scellée mais
n'est jamais évalué sans validation explicite du freeze. Voir
[docs/research-v2.md](docs/research-v2.md).

## Direction Research V3 (Phase D.9)

V3 remplace la direction terminale UP/DOWN par deux modèles triple-barrière séparés :
probabilité que le target LONG soit touché avant son stop et probabilité équivalente
pour SHORT. Le MetaGate choisit LONG, SHORT ou SKIP en fonction d'un seuil et d'un edge
BASE préenregistrés. Toutes les features restent causales et le HOLDOUT demeure scellé.
Voir [docs/research-v3.md](docs/research-v3.md).

## Methodology Audit (Phase D.9A)

Le replay V3 applique désormais une purge propre à chaque configuration avant chaque
fold : une ligne TRAIN n'est retenue que si `timestamp + timeout <= validation_start`.
Les 16 frontières configuration/fold sont auditées dans le rapport. Le replay purgé
conserve le verdict `V3_RESEARCH_REJECTED`. Le diagnostic montre séparément
`TARGET_FIRST`, `STOP_FIRST` et `NEITHER`; il ne modifie pas la formule MetaGate V3
préenregistrée. D.10 est uniquement spécifiée dans
[docs/d10-three-outcome-spec.md](docs/d10-three-outcome-spec.md), sans entraînement.

## Three-Outcome Expected Value Engine (Phase D.10)

D.10 modélise séparément `TARGET_FIRST`, `STOP_FIRST` et `NEITHER` pour LONG et SHORT,
puis combine les probabilités avec les payoffs BASE moyens appris dans TRAIN. Le buffer
d'incertitude est appris dans des folds nested purgés et la décision ne dépend plus du
seuil V3 `P(target)>=0,60`. Le contrôle interne produit seulement quatre candidats : le
verdict est `D10_INSUFFICIENT_EVIDENCE`. Voir [docs/research-d10.md](docs/research-d10.md).

## Predictive Ranking & Regime Stability Audit (Phase D.11)

D.11 reproduit 111 136 prédictions side-OOF D.10 sur le seul dataset RESEARCH. Le top
5 % agrégé reste négatif après coûts BASE (-7,29 points) et sa légère amélioration
(+0,09 point) change de signe dans le contrôle interne gelé. La corrélation Spearman
observation-level est négative (-0,0267). Verdict : `D11_UNSTABLE_RANKING_SIGNAL`.
Aucun seuil n'est créé. Voir [docs/research-d11.md](docs/research-d11.md).

## V4 Feature & Regime Discovery (Phase D.12)

D.12 sépare `REGIME_QUALITY` de `WITHIN_REGIME_DIRECTION` sur 17 941 326 ticks du
seul dataset RESEARCH. Elle compare cinq échantillonnages causaux, 88 features et six
horizons, avec folds purgés, block bootstrap et correction FDR sur 1 056 tests. Le
verdict préenregistré est `D12_FEATURES_WORTH_V4` : l'amplitude future est fortement
liée à la volatilité récente et un faible effet directionnel contrariant apparaît à
30–60 s. Ce verdict n'est pas une preuve de profitabilité et ne crée aucun V4. Voir
[docs/research-d12.md](docs/research-d12.md).

## Hierarchical V4 Research (Phase D.13)

D.13 réduit les features sur TRAIN, puis teste en nested purged walk-forward
`RegimeOpportunityModel → ContrarianDirectionModel → EconomicExecutionGate`. Le régime
est informatif, mais l'amplitude directionnelle prédite ne couvre jamais les coûts :
zéro candidat OOF et verdict `V4_NOT_READY_TO_FREEZE`. Le manifeste produit est un
snapshot verrouillé non autorisé pour le HOLDOUT. Voir
[docs/research-d13.md](docs/research-d13.md).

## Economic Feasibility Audit (Phase D.14)

D.14 vérifie les hashes puis décompose les 48 024 prédictions OOF D.13 sans aucun
réentraînement. Le signal courant atteint 50,30 % de sign accuracy ; son faible edge
brut (+0,235 point) devient déjà négatif après spread exécutable (-0,062 point), puis
atteint -6,672 points sous BASE. La magnitude prédite n'offre qu'un Spearman de 0,059
avec la magnitude réalisée. Verdict : `D14_DIRECTION_ECONOMICALLY_TOO_WEAK`. Voir
[docs/research-d14.md](docs/research-d14.md).
