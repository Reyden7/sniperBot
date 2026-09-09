# Phase D.13 — SNIPER V4 Hierarchical Research

Verdict final : **`V4_NOT_READY_TO_FREEZE`**.

D.13 a construit et testé l'architecture hiérarchique V4 sur le seul dataset
`EURUSD_RESEARCH_20260610_20260908`. Toutes les mesures sont donc
**RESEARCH PERFORMANCE ONLY**. Elles ne constituent pas une validation finale.

Le HOLDOUT historique reste `SEALED`, non ouvert et non évalué. Aucun ordre, sizing,
chemin live ou Phase E n'a été créé.

## Protocole préenregistré

Le protocole a été verrouillé avant le premier calcul de réduction ou de performance
D.13 :

- cible régime primaire : `FUTURE_TRADABLE_MOVEMENT > 2 × BASE_cost` à 60 s ;
- cible direction primaire : rendement signé exécutable à 30 s ;
- diagnostics secondaires 30 s et 300 s sans autorité de remplacement ;
- univers primaire : union dédupliquée par timestamp de `MINUTE_BASELINE` et
  `ACCELERATION_EVENT` ;
- `VOLATILITY_EVENT`, `BREAKOUT_EVENT` et `COMPRESSION_RELEASE_EVENT` restent
  diagnostiques ;
- 2 modèles régime × 2 modèles direction × 3 seuils × 3 buffers = 36 configurations ;
- seuils régime limités à 0,50 / 0,60 / 0,70 ;
- buffers limités à 0 / 1 / 2 points ;
- sélection nested purgée par score composite préenregistré ;
- minimum nested de 75 candidats ;
- purge de 60 s et preprocessing ajusté sur TRAIN uniquement ;
- bootstrap journalier de 2 000 réplications, seed 20260913.

## Architecture V4 étudiée

```text
RegimeOpportunityModel
        ↓ PASS seulement
ContrarianDirectionModel
        ↓ rendement signé attendu
EconomicExecutionGate
        ↓ coût BASE + uncertainty buffer
RESEARCH_LONG / RESEARCH_SHORT / RESEARCH_SKIP
```

Le DirectionModel ne reçoit aucune inversion artificielle du target. Il prédit le
rendement signé naturel ; le comportement contrariant doit être appris dans ses
coefficients ou ses splits. Sa sortie n'est utilisée économiquement que si le modèle
de régime passe son seuil.

La sortie est strictement de recherche. Elle ne contient aucun `BUY`/`SELL` actionnable
et n'appelle pas le Risk Engine.

## Univers et folds

| Mesure | Valeur |
|---|---:|
| Observations D.12 disponibles | 72 599 |
| Univers primaire dédupliqué | 71 846 |
| Observations outer OOF | 48 024 |
| Fold 1 validation | 9 233 |
| Fold 2 validation | 7 852 |
| Fold 3 validation | 5 991 |
| Internal research check | 24 948 |
| Folds nested audités | 15 |

Les quatre purges outer et les quinze purges nested vérifient :

```text
max(timestamp_train + 60 secondes) <= validation_start
```

Le faible nombre de lignes effectivement purgées aux frontières est dû à l'absence
d'observation événementielle dans la dernière minute avant ces frontières ; l'invariant
est néanmoins contrôlé explicitement.

## Réduction déterministe des features

La réduction est ajustée sur les 60 premiers pour cent temporels du DESIGN TRAIN. Une
feature placée plus tard dans l'ordre préenregistré est supprimée si son Spearman absolu
avec une représentante déjà conservée atteint 0,90.

### Features régime conservées

1. `realized_volatility_300s_points`
2. `volatility_percentile_300s`
3. `realized_move_to_base_cost_300s`
4. `micro_pullback_amplitude_60s_points`
5. `return_autocorrelation_lag1_900s`
6. `variance_ratio_5_900s`
7. `spread_vs_median_60s`

Aucune des sept paires candidates n'atteint le seuil de suppression. La corrélation la
plus élevée est `return_autocorrelation_lag1_900s` / `variance_ratio_5_900s` à 0,857.

### Features direction conservées

1. `current_signed_run_length`
2. `return_5s_points`
3. `return_15s_points`
4. `signed_tick_imbalance_1s`
5. `signed_tick_imbalance_5s`

### Features supprimées

| Feature supprimée | Représentante | Spearman TRAIN | Motif |
|---|---|---:|---|
| `displacement_per_tick_15s_points` | `return_15s_points` | +0,9484 | redondance ≥ 0,90 |
| `breakout_velocity_15s_points_per_second` | `return_15s_points` | +1,0000 | identité de ranking |

Les 42 comparaisons pairwise incluent également une mutual information discrétisée
10×10 diagnostique. Aucun choix n'utilise le HOLDOUT.

## Modèles et conditionnement Ridge

Les familles autorisées étaient :

- régime : LogisticRegression L2 primaire, HGBClassifier challenger ;
- direction : Ridge primaire, HGBRegressor challenger.

Le preprocessing TRAIN-only applique médiane, clipping aux quantiles 0,5 % / 99,5 %,
standardisation, suppression d'une colonne de scale < 1e-8 et clipping standardisé à
±10. Toutes les colonnes sont restées actives.

L'anomalie Ridge de D.12 est corrigée :

- tous les coefficients sont finis ;
- toutes les prédictions sont finies ;
- toutes les valeurs standardisées sont bornées à ±10 ;
- MAE par fit comprise approximativement entre 4,5 et 6,1 points ;
- MAE OOF globale : 5,069 points ;
- Spearman direction OOF : +0,0610.

Le snapshot final utilise le fallback préenregistré
`LOGISTIC_L2 + RIDGE + seuil 0,50 + buffer 0`, car aucune configuration nested
n'atteint le minimum de 75 candidats. Ce fallback n'est pas freeze-éligible.

## Modèle de régime et calibration

Sur les 48 024 observations OOF :

| Mesure | Valeur |
|---|---:|
| Taux de base `move > 2×cost` | 0,2276 |
| PR-AUC régime | 0,5393 |
| Brier | 0,1433 |
| Observations passant le seuil 0,50 | 4 645 |

Les dix bins de calibration sont rapportés dans le JSON complet. Le dernier décile a
une probabilité moyenne de 0,680 et un taux observé de 0,644. Aucune calibration
post-hoc n'est ajustée ; les probabilités natives sont seulement bornées numériquement
à `[1e-6, 0.999999]`.

Le critère « meilleur que le taux de base » passe. Le blocage V4 se situe après ce
niveau, dans la direction économique.

## EconomicExecutionGate

Pour chaque côté :

```text
predicted market move
- spread observé à T
- slippage BASE aller-retour
- commission BASE aller-retour
= predicted BASE net edge
```

La réalisation utilise directement les prix exécutables Bid/Ask D.12, puis retire le
slippage et la commission simulés. Le test croisé :

```text
market PnL - realized Bid/Ask spread - slippage - commission = BASE net PnL
```

présente une erreur absolue maximale de `7,11e-15` point sur les observations OOF.

Hypothèses simulées, qui ne représentent ni MetaQuotes-Demo ni un futur broker :

| Scénario | Slippage par côté | Commission EUR/lot/côté |
|---|---:|---:|
| BASE | 1 point | 2 EUR |
| STRESS | 2 points | 4 EUR |

Dans l'univers primaire, le spread moyen observé est 0,525 point, la commission BASE
aller-retour équivaut en moyenne à 4,598 points et le coût BASE total moyen vaut
7,123 points.

## Résultats nested et cause du rejet

Le meilleur nombre de candidats nested par fold est seulement 11 / 15 / 11 / 7. Sur
le DESIGN TRAIN final, la configuration la plus permissive produisant le plus de
candidats est :

```text
HGB_CLASSIFIER + HGB_REGRESSOR
threshold = 0.50
buffer = 0
candidates = 7
BASE expectancy = -4.720 points
```

Elle est très loin des 75 candidats nécessaires à la sélection nested et son espérance
est déjà négative. Elle ne peut donc pas remplacer le fallback.

Pour le fallback Ridge OOF :

- médiane de `|predicted signed return|` : 0,247 point ;
- percentile 99 : 1,039 point ;
- maximum : 1,731 point ;
- meilleur `predicted BASE net edge` : −4,817 points.

Le modèle détecte une petite relation directionnelle, mais son amplitude prédite ne
couvre jamais les coûts. Le RegimeGate laisse passer 4 645 observations ;
l'EconomicExecutionGate les rejette toutes.

## Performance économique RESEARCH

| Système | Candidats | BASE expectancy | STRESS expectancy | Observation |
|---|---:|---:|---:|---|
| ALWAYS_SKIP | 0 | 0 | 0 | aucune exposition |
| Random side | 0 | non estimable | non estimable | mêmes timestamps V4, inexistants |
| Contrarian return 5 s | 0 | non estimable | non estimable | amplitude TRAIN sous les coûts |
| Contrarian signed run | 0 | non estimable | non estimable | amplitude TRAIN sous les coûts |
| Unconditional TRAIN side | 4 645 | −7,221 | −13,830 | 0/10 semaine positive, PF 0,236 |
| D.10 gelé | 4 | +7,870 | +1,240 | horizon/univers différents, non comparable ; IC inclut zéro |
| V4 full hierarchy | 0 | non estimable | non estimable | EconomicGate rejette tout |

Avec zéro candidat, BASE/STRESS expectancy, profit factor, stabilité semaine/session et
IC bootstrap V4 ne sont pas estimables. Ils ne sont jamais assimilés à zéro ni présentés
comme favorables.

## Critères de readiness

Seuls quatre contrôles passent : régime meilleur que taux de base, conditionnement
Ridge, valeurs finies et invariants de purge. Échouent notamment :

- minimum 150 candidats ;
- minimum 8 semaines actives ;
- expectancy BASE > 0 et borne basse IC 95 % > 0 ;
- STRESS ≥ 0 ;
- PF BASE ≥ 1,15 ;
- au moins 60 % de semaines positives ;
- stabilité session mesurable ;
- supériorité aux deux baselines contrariennes ;
- sélection nested finale éligible.

Le verdict obligatoire est donc **`V4_NOT_READY_TO_FREEZE`**.

## V4_FREEZE_MANIFEST

Un snapshot de recherche verrouillé est produit, mais son statut est :

```text
LOCKED_RESEARCH_SNAPSHOT_NOT_AUTHORIZED_FOR_HOLDOUT
```

Il enregistre l'univers, les features et suppressions, les horizons, targets, modèles,
hyperparamètres, preprocessing, seuil, calibration, coûts, buffer, règle candidate et
critères d'acceptation.

- hash canonique interne : `f078629ca4d7ce9054c2cf72661e87a6029628e427b57bd1130916f655c93756` ;
- SHA-256 du fichier manifeste : `2051e53dad8b32cd92043c7b212278d8c43145fbaed7fff104ef12f45343e26c` ;
- moteur D.13 : `bcbc7f485b3759735f63937537b9da16d4aa9a4543d9c4e4380b01e38aad3484` ;
- protocole D.13 : `1f55213c8c503551b53f0086dbf6b4707c487209ad982bdbbd8e03598b81fd8f` ;
- observations OOF : `aadc3eaaaf046acbd9a566aecd11fb22abaf599acef71bca3909ad1a563e4ab0` ;
- rapport JSON : `f0117538f287e3c6b472f95cadcdd7e368495a94bf780a197b1fc9259e8a448a`.

Le hash du rapport sera recalculé après toute régénération ; les hashes moteur,
protocole, OOF et manifeste ci-dessus correspondent au replay officiel final.

## Validation future

Le futur dataset est réservé sous le nom :

```text
FORWARD_V4
start > 2026-09-08T00:00:00Z
```

Aucun résultat forward n'est requis ni produit en D.13. Un test forward postérieur au
développement serait temporellement plus fort que le HOLDOUT historique, qui précède
le TRAIN. Toutefois, V4 n'étant pas prête à geler, aucun de ces datasets ne doit être
ouvert pour tenter de la sauver.

D.13 s'arrête ici. Le HOLDOUT reste scellé, Phase E non commencée et LIVE désactivé.
