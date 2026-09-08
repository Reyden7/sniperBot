# SNIPER

Phases A à D.9 : fondation read-only, diagnostic broker 10 EUR, pipeline historique,
backtester tick event-driven et moteurs de recherche V2/V3 EURUSD rejetés.
Aucun ordre live n'est implémenté.

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
