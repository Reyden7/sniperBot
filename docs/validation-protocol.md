# Validation des phases A a D.7

Commandes reproductibles apres `uv sync --locked` :

```text
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
```

Les tests unitaires du rapport utilisent un double MT5 strict avec prix et
parametres synthetiques explicitement definis dans tests/conftest.py. Ils verifient
les seuils nano-lots, les devises, les deux sens, les couts, les stops broker,
la marge, les donnees manquantes et les defauts de connexion/calcul.
Les tests de sizing et Hypothesis verifient le budget apres arrondi, la grille,
la maximalite du volume retenu et le refus quand le minimum est trop risque.
Les tests integration couvrent l'adaptateur et les deux commandes CLI sur ce double.
Une regression fixe conserve les montants attendus du scenario synthetique.

Les tests Phase B couvrent les conversions UTC/serveur/local, les transitions DST,
la distinction entre decalage de source non prouve et vraie rupture d'horloge locale,
la validation/de-duplication des ticks, les trous, les statistiques de spread,
l'idempotence Parquet et la consolidation de bougies partielles. Hypothesis verifie
les invariants OHLC et volume du constructeur incremental.

Les tests Phase C utilisent plusieurs petits Parquet synthetiques generes uniquement
dans les repertoires temporaires pytest. Les golden assertions figent les prix ASK/BID,
le spread, le slippage, la commission, le PnL et le journal complet. Les tests couvrent
BUY/SELL, les quatre SL/TP, les cinq latences, les rejets volume/marge/risque, la limite
d'une position, l'absence de tick futur et la reproductibilite du seed. Hypothesis
verifie l'identite financiere `net = brut - spread - slippage - commission`.

Un second test croise recalcule le PnL directement depuis les prix executes, le sens,
le contract size et le volume, puis soustrait la commission. Les rapports sont aussi
testes pour garantir l'affichage du profil simule, levier/grille de volume et
l'avertissement MetaQuotes-Demo.

Les tests Phase D couvrent toutes les familles de features V1, breakout/retest,
patterns, compression/expansion, invariants tick Hypothesis, filtrage des barres
partielles/futures, determinisme, seuils 80/90/95, BUY/SELL/WAIT et blockers fail-closed.
La CLI est testee de Parquet jusqu'au rapport et l'absence de tout champ volume dans le
signal est imposee.

Les durcissements couvrent `MARKET_DATA_STALE`, la mise a zero de la confirmation
tick, l'absence de bonus lorsque le tick rate est nul, les warm-ups explicites et
`FEATURE_WARMUP_INCOMPLETE`, ainsi que la separation biais/trigger et le refus d'un
trigger oppose a M5.

Les tests Phase D.5 verifient les rendements executables LONG/SHORT, MFE/MAE, couts et
ratio target/cout, les tranches fixes, l'absence d'optimisation, et surtout l'isolation
temporelle : modifier un prix futur peut changer l'outcome mais jamais le score a T.
La pagination journaliere et la reprise par manifeste de `collect-history` sont aussi
testees sans connexion MT5.

Les tests Phase D.6 figent les empreintes des sources du Signal Engine, comparent les
features tick du chemin streaming au Feature Engine, valident les sessions IANA avec
DST, les quatre etapes de couts, MFE/MAE monetaire, le profil broker simule et les sept
buckets. Le scan reel reste une operation separee sur les Parquet locaux.

Les tests Phase D.7 verrouillent l'échantillonnage à une seconde et les seuils
méthodologiques 90/85, vérifient qu'un retour 89→91 ne crée pas un nouvel épisode avant
un passage sous 85, distinguent `FIRST_CROSSING` tradable de `PEAK_SCORE` rétrospectif,
et contrôlent que la borne minute reste supérieure ou égale au score exact du moteur.

Un diagnostic reel se lance separement, sur Windows et un terminal MT5 connecte :

```text
uv run --locked --extra mt5 sniper broker-check --capital 10 --symbol EURUSD
```

Si aucun terminal n'est disponible, le code de sortie doit etre 2 avec une erreur
explicite, sans valeurs de remplacement. Le premier rapport sur un broker reel
reste une verification distincte des tests automatises sur doubles. Il n'est pas
necessaire d'activer Algo Trading pour lire les donnees ou faire ces estimations.

Le squelette EA n'est pas un EA de trading a valider dans le Strategy Tester ;
compilation et validation de l'execution appartiennent a la phase MQL5 ulterieure.

## Resultat local du 8 septembre 2026

- Python 3.14.6, pytest 9.1.1 : 200 tests reussis, dont les proprietes Hypothesis.
- Ruff (lint et format) : succes ; mypy strict : succes sur les 48 modules Python.
- Installation `uv sync --offline --locked --extra mt5` : succes depuis le cache local.
- Construction sdist et wheel 0.7.0 : succes ; archives verifiees sans cache, .venv,
  donnees de marche, manifestes ou `.env` secret (`.env.example` est conserve).
- SDK MetaTrader5 5.0.6180 reel : collecte EURUSD reussie sur un terminal connecte.
  Plage `[2026-09-08T07:46:00.000316Z, 2026-09-08T07:51:00.000316Z)` :
  354 ticks acceptes, 0 doublon, 0 invalide, 0 trou > 60 s ; spread en points
  min/mediane/p95/max = 0/1/1/1 ; bougies M1/M5/M15 = 5/2/1.

Sur cette machine, uv existe dans `C:\Users\quent\.local\bin` mais n'est pas dans
le PATH de la session. La CLI installee peut aussi etre lancee directement :

```powershell
.venv\Scripts\sniper.exe broker-check --capital 10 --symbol EURUSD
```
