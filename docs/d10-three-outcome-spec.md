# Phase D.10 — Three-Outcome Expected Value Engine

Statut : **LOCKED BEFORE FIRST D.10 TRAINING**. Le protocole sérialisé faisant foi est
`docs/research-protocol-d10.yaml`.

## Périmètre et split

D.10 utilise exclusivement EURUSD RESEARCH `[2026-06-10, 2026-09-08)`. Les trois
folds de `MODEL_DEVELOPMENT` utilisent respectivement les tranches temporelles
30–40 %, 40–50 % et 50–60 %, avec TRAIN expanding. `INTERNAL_FREEZE_CHECK` utilise
la tranche finale 60–100 %, soit environ six semaines calendaires. Toutes les lignes
TRAIN, y compris dans les splits nested, vérifient :

```text
observation_timestamp + 300 seconds <= validation_start
```

Les choix ci-dessous sont gelés avant le premier entraînement et ne peuvent pas être
adaptés après lecture du freeze check.

## Modèle économique primaire

Deux régressions logistiques multinomiales régularisées distinctes, LONG et SHORT,
estiment `P(TARGET_FIRST)`, `P(STOP_FIRST)` et `P(NEITHER)`. Les probabilités doivent
être finies, positives et sommer à un à la tolérance numérique près. Hyperparamètres :
`C=1.0`, `L2`, `max_iter=500`, seed `20260910`. Il n’y a aucune calibration post-hoc :
les probabilités softmax natives sont évaluées telles quelles. Le challenger fixe est
un `HistGradientBoostingClassifier`, `max_iter=50`, `max_depth=3`, learning rate
`0.08`, même seed. Aucun challenger ne peut remplacer le primaire après résultats.

Pour chaque fold et side, les trois payoffs sont les moyennes TRAIN de
`actual_BASE_net_pnl` conditionnelles à l’issue. L’espérance est :

```text
EV_BASE = Σ P(outcome) × mean_TRAIN(BASE pnl | outcome)
```

Tous les PnL sont les résultats exécutables constatés au premier hit ou au timeout.
`NEITHER` n’est remplacé ni par `-stop`, ni par zéro, ni par target. En l’absence
improbable d’une classe TRAIN, le fallback préenregistré est la moyenne BASE TRAIN du
side, et l’anomalie doit apparaître dans le rapport.

## Buffer d’incertitude

Dans le TRAIN de chaque fold externe, trois folds internes expanding 40–60 %, 60–80 %
et 80–100 % produisent des prédictions OOF purgées. Par side, on calcule le résidu
`actual_BASE_pnl - predicted_EV`, puis sa moyenne par jour UTC. Avec `D` jours :

```text
safety_buffer = max(0, -mean(daily_residual) + 1.645 × sd(daily_residual)/sqrt(D))
conservative_EV = predicted_EV - safety_buffer
```

Un side est éligible seulement si `conservative_EV > 0`. Si aucun side n’est positif,
la décision est `SKIP`; si les deux le sont, le plus élevé est retenu. Il n’existe plus
de seuil `P(target) >= 0.60`. Aucun sizing n’est produit.

## Challenger direct PnL et baselines

Les challengers directs, évalués mais jamais promus dans D.10, prédisent le BASE net
PnL par side : moyenne TRAIN, Ridge `alpha=1`, ElasticNet `alpha=0.001`,
`l1_ratio=0.5`, et HistGradientBoostingRegressor avec les paramètres HGB ci-dessus.
Les métriques sont MAE, RMSE et biais moyen.

Les baselines économiques sont `ALWAYS_SKIP`, random side (un side par observation,
seed fixe), unconditional TRAIN EV (side de moyenne TRAIN positive maximale sinon
SKIP) et le MetaGate V3 exact (`P(target)>=0.60` et ancienne formule conservatrice).

## Mesures et décision

Le rapport sépare `MODEL_DEVELOPMENT` et `INTERNAL_FREEZE_CHECK`. Il fournit métriques
multiclasses, calibration equal-width par classe, payoffs TRAIN, challengers directs,
count LONG/SHORT, gross/exécutable/BASE/STRESS expectancy, médiane BASE, PF, win rate,
issues, MFE/MAE, drawdown, semaines, sessions et baselines.

Les IC 95 % utilisent 2 000 réplications bootstrap de jours UTC entiers, seed
`20260910`; tous les candidats d’un jour sont rééchantillonnés ensemble. Les critères
du freeze check sont : au moins 100 candidats, six semaines actives, expectancy BASE
positive avec borne basse IC positive, STRESS non négative, PF BASE au moins 1,10,
au moins 60 % de semaines positives et au plus 60 % dans une seule session.

Verdicts exacts : `D10_RESEARCH_REJECTED`, `D10_INSUFFICIENT_EVIDENCE`,
`D10_NEEDS_MORE_RESEARCH`, `D10_FREEZE_CANDIDATE`.

## Frontières

B02_PRIMARY est l’unique configuration décisionnelle. B01/B03/B04 ne peuvent pas être
sélectionnées. Même en cas de `D10_FREEZE_CANDIDATE`, le HOLDOUT reste scellé jusqu’à
une validation explicite. D.10 ne commence ni Phase E ni trading live.
