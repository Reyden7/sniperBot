# Phase D.10 — Three-Outcome Expected Value Engine

Verdict final : **D10_INSUFFICIENT_EVIDENCE**.

Le run utilise exclusivement les 92 143 observations RESEARCH issues de 17 941 326
ticks EURUSD. Les 55 568 observations des fenêtres externes sont labellisées et
évaluées sans lecture du HOLDOUT. Le calcul a duré 197,30 secondes, avec un pic mémoire
Python de 97,65 MiB.

## Protocole gelé

B02_PRIMARY est l’unique configuration décisionnelle : stop `1×ATR`, target
`max(2×ATR, 2×coût BASE)`, timeout 300 s. LONG et SHORT utilisent chacun une régression
logistique multinomiale L2 fixe. L’espérance combine les probabilités des trois issues
avec leurs payoffs BASE moyens calculés sur TRAIN uniquement. La règle est
`predicted_EV - safety_buffer > 0`; aucun seuil sur `P(target)` n’existe.

- protocole SHA-256 : `20d8967079e4503b581e8729aa140efe8bbdf7713e29e86c230ef4256b1cfcf5` ;
- freeze logique SHA-256 : `1a20efb21268db540c6c6f3547f1b76d0dff4c1950ef2f4570ecd468caca4699` ;
- fichier manifeste SHA-256 : `2e74cc2c4ce4d694c500294587cf255737b8c73b41bd49f374f19ef0bfe5ae75` ;
- moteur gelé SHA-256 : `93630689811933c21773ca50b4e0236b7407dcd207f6e358a0daafe3a16969dd` ;
- rapport JSON SHA-256 : `31836898ece3cf439682579055f0b3b9871ef73ee2b98e0b3445daccee05957d`.

## Classes du freeze check

| Side | TARGET_FIRST | STOP_FIRST | NEITHER |
|---|---:|---:|---:|
| LONG, 31 081 | 4 429 (14,250 %) | 15 479 (49,802 %) | 11 173 (35,948 %) |
| SHORT, 31 081 | 4 131 (13,291 %) | 15 905 (51,173 %) | 11 045 (35,536 %) |

Les probabilités somment à un avec une erreur maximale de `2,22e-16`. Les tables de
calibration equal-width complètes pour TARGET, STOP et NEITHER sont conservées dans le
rapport JSON. Elles montrent une calibration raisonnable dans les bins centraux très
peuplés et une forte incertitude dans les bins extrêmes à faible effectif.

## Métriques multiclasses

| Scope | Side | Modèle | Log-loss | Brier | Accuracy |
|---|---|---|---:|---:|---:|
| MODEL_DEVELOPMENT | LONG | Logistique primaire | 0,991559 | 0,602189 | 48,985 % |
| MODEL_DEVELOPMENT | LONG | HGB challenger | 0,993696 | 0,604633 | 48,777 % |
| MODEL_DEVELOPMENT | SHORT | Logistique primaire | 0,982212 | 0,597669 | 49,381 % |
| MODEL_DEVELOPMENT | SHORT | HGB challenger | 0,981473 | 0,597579 | 49,177 % |
| INTERNAL_FREEZE_CHECK | LONG | Logistique primaire | 0,981122 | 0,594575 | 50,549 % |
| INTERNAL_FREEZE_CHECK | LONG | HGB challenger | 0,980747 | 0,594845 | 50,439 % |
| INTERNAL_FREEZE_CHECK | SHORT | Logistique primaire | 0,966360 | 0,585695 | 51,884 % |
| INTERNAL_FREEZE_CHECK | SHORT | HGB challenger | 0,965811 | 0,586386 | 51,610 % |

Le challenger HGB n’est pas promu.

## Payoffs conditionnels TRAIN du freeze

| Side | TARGET_FIRST | STOP_FIRST | NEITHER |
|---|---:|---:|---:|
| LONG | +22,0263 | −21,3409 | −0,7065 |
| SHORT | +21,2761 | −21,3451 | −0,2289 |

Ces valeurs sont des moyennes BASE en points sur les 49 073 lignes TRAIN purgées. Le
payoff `NEITHER` est le vrai PnL exécutable au timeout ; aucun fallback de classe
manquante n’a été utilisé. Les payoffs de chaque fold figurent dans le JSON. Les
buffers nested du freeze valent 0 point pour les deux sides selon la formule gelée ;
ils n’ont pas été remplacés après observation du résultat.

## Résultats économiques

| Scope | Candidats L/S | Exécutable | BASE | STRESS | PF | Win rate | Drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| MODEL_DEVELOPMENT | 14 (11/3) | −35,0714 | −41,6483 | −48,2252 | 0,000 | 0 % | 583,0763 |
| INTERNAL_FREEZE_CHECK | 4 (2/2) | +14,5000 | +7,8700 | +1,24005 | 1,3271 | 50 % | 52,6261 |

En développement, aucun des 14 candidats n’est gagnant : 85,71 % sont STOP_FIRST et
14,29 % NEITHER. Le freeze contient deux TARGET_FIRST et deux STOP_FIRST. Sa médiane
BASE est négative à −5,6349 points malgré la moyenne positive.

IC 95 % par bootstrap de jours UTC entiers, 2 000 réplications :

| Mesure freeze | Estimation | IC 95 % |
|---|---:|---:|
| BASE expectancy | +7,8700 | [−43,6144 ; +32,3446] |
| STRESS expectancy | +1,24005 | [−50,2288 ; +25,6892] |
| PF BASE | 1,3271 | [0 ; 3,0416] |
| Win rate | 50 % | [0 % ; 100 %] |

Les quatre candidats ne couvrent que trois semaines actives : W32 `+21,37495`, W33
`−43,6144`, W35 `+32,3446` points BASE en moyenne. Trois candidats sur quatre viennent
de Londres/New York ; `largest_session_share=75 %`.

## Challenger DirectNetPnl — freeze

| Side | Modèle | MAE | RMSE | Biais actual−predicted |
|---|---|---:|---:|---:|
| LONG | TRAIN mean | 11,0468 | 14,1223 | +1,2553 |
| LONG | Ridge | 11,0334 | 14,0975 | +0,9479 |
| LONG | ElasticNet | 11,0324 | 14,0962 | +0,9480 |
| LONG | HGB regressor | 11,0435 | 14,1176 | +0,8501 |
| SHORT | TRAIN mean | 10,9207 | 13,9166 | +0,8139 |
| SHORT | Ridge | 10,9402 | 13,9001 | +0,4460 |
| SHORT | ElasticNet | 10,9379 | 13,8983 | +0,4469 |
| SHORT | HGB regressor | 10,9255 | 13,8955 | +0,4224 |

ElasticNet a émis des avertissements de convergence avec les paramètres préenregistrés.
Ils n’ont pas été modifiés après ouverture du freeze et ce challenger n’influence pas
la décision primaire.

## Baselines du freeze

| Baseline | Candidats | BASE expectancy | STRESS expectancy | PF | Win rate |
|---|---:|---:|---:|---:|---:|
| ALWAYS_SKIP | 0 | N/A | N/A | N/A | N/A |
| Random side | 31 081 | −7,28384 | −13,92075 | 0,30135 | 26,846 % |
| Unconditional TRAIN EV | 0 | N/A | N/A | N/A | N/A |
| V3 MetaGate exact | 1 | −43,6144 | −50,2288 | 0 | 0 % |

D.10 est ponctuellement supérieur à ces baselines sur quatre cas, mais cet effectif et
les IC empêchent d’interpréter cette différence comme un edge.

## Purge et critères

La purge externe retire `[0, 3, 1, 0]` lignes sur les folds 1–4. Les douze folds nested
retirent respectivement `[0,5,4, 5,4,5, 0,0,3, 4,0,5]` lignes. Les 16 invariants
`max(label_end_time TRAIN) <= validation_start` passent.

Le freeze réussit les tests ponctuels BASE, STRESS, PF et fraction de semaines positives.
Il échoue sur le minimum de 100 candidats, les six semaines actives, la borne basse IC
BASE positive et la concentration session maximale de 60 %. Le verdict préenregistré
est donc `D10_INSUFFICIENT_EVIDENCE`.

Le HOLDOUT reste SEALED, non lu et non évalué. Phase E n’est pas commencée, aucun lot
n’est calculé et aucun trading live n’est implémenté.
