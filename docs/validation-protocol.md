# Validation des phases A à D.9

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

Les tests Phase D.8 verrouillent les rôles `RESEARCH/VALIDATION/HOLDOUT/FORWARD`,
l'ordre strict des folds, le rôle `INTERNAL_FREEZE_CHECK` du fold 3, l'inclusion de la
quote à l'horizon exact, l'isolation des lignes du contrôle interne et le manifeste de
HOLDOUT scellé sans métrique de modèle. Les modèles et transformateurs sont ajustés
uniquement sur le TRAIN antérieur à chaque fenêtre.

Les tests Phase D.9 modifient les ticks futurs et prouvent l'invariance des nouvelles
features à T. Ils contrôlent aussi les prix exécutables ASK/BID, l'ordre target/stop,
le timeout, les quatre folds expanding TRAIN_ONLY et l'assemblage indépendant des
probabilités LONG/SHORT dans le MetaGate.

Les tests Phase D.9A vérifient les quatre purges propres aux timeouts 180/300/600/900 s.
Ils imposent sur chaque fold `max(label_end_time TRAIN) <= validation_start`, et non la
seule séparation des timestamps d'observation. Le replay conserve toutes les fenêtres
VALIDATION et tous les paramètres V3 inchangés.

Les tests Phase D.10 verrouillent le split développement/freeze, la purge B02 de 300 s,
le refus explicite d'une racine HOLDOUT, l'encodage des trois issues, la somme des
probabilités, le vrai payoff BASE de NEITHER, la reproductibilité du freeze manifest
et le bootstrap par blocs journaliers. Les challengers ne disposent d'aucun chemin de
promotion automatique.

Les tests Phase D.11 contrôlent les seuils de densité EV sans sélection, les quantiles
de ranking préenregistrés, les dix groupes de calibration, les métriques de changement
de régime, le marquage `EXPLORATORY_POST_HOC` et le refus du HOLDOUT avant tout accès
aux hashes ou aux modèles. La reproduction officielle doit retrouver 14/4 candidats
D.10 et l'empreinte OOF gelée.

Les tests Phase D.12 prouvent l'invariance des features secondes après ajout de quotes
futures, les cooldowns, l'usage Ask/Bid des labels, la définition 60 s de
compression-release, les quatre folds déclarés, les huit interactions, la correction
Benjamini-Hochberg et le refus du HOLDOUT avant lecture du protocole. Le replay officiel
contrôle en outre la purge propre à chacun des six horizons sur les 24 frontières
modèle/fold.

Les tests Phase D.13 verrouillent la réduction ordonnée à `|rho| >= 0,90`, le scaling
TRAIN-only borné, la suppression des colonnes quasi singulières, les coefficients et
prédictions Ridge finis, la décomposition exécutable Bid/Ask des coûts, les folds nested
purgés, la stabilité du hash canonique du manifeste et le refus du HOLDOUT avant lecture
du protocole. Le replay officiel retrouve 71 846 observations dans l'univers primaire,
48 024 sorties OOF et zéro candidat après coût.

Les tests Phase D.14 contrôlent les dix bins à effectifs égaux, le choix du meilleur
side par l'oracle, le maintien du spread observé dans la frontière de coûts et le refus
du HOLDOUT avant tout accès source. Le replay officiel vérifie les trois hashes d'entrée,
l'identité `source_index/timestamp/session` et retrouve une erreur nulle entre PnL BASE
persisté et recalculé.

Le premier lot Binance teste le refus double de LIVE, les filtres Spot dynamiques,
l'arrondi Decimal, le minimum notionnel, les frais/spread/slippage aller-retour, la
normalisation UTC des trades/book/klines, la composition des cinq flux WebSocket,
l'idempotence Parquet, les limites HTTP 418/429, la liste blanche REST et la sélection
dynamique des quotes éligibles. Les tests ne contactent pas Binance ; la
connexion publique réelle et la collecte bornée constituent un contrôle séparé.

Le deuxième lot Binance teste la séparation entre capacités du compte et permissions
de clé, l'univers EUR/USDC/USDT complet, la compatibilité du solde quote, la causalité
M15/M5/M1, les trois setups, le filtre économique, le classement sans sizing, le Risk
Engine, les états journaliers UTC, l'invariant d'une position et le recalcul du PnL depuis
les prix exécutés. Le replay officiel contient 518 400 M1 et 103 680 M5 sur quatre paires
EUR pendant 90 jours ; l'absence de Bid/Ask historique reste explicitement bloquante.

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

## Resultat local du 9 septembre 2026

- Python 3.14.6, pytest 9.1.1 : 262 tests réussis, dont les propriétés Hypothesis.
- Ruff (lint et format) : succès ; mypy strict : succès sur les 82 modules Python.
- Installation `uv sync --offline --locked --extra mt5` : succes depuis le cache local.
- Construction sdist et wheel 0.16.0 : succès ; archives vérifiées sans cache, .venv,
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
