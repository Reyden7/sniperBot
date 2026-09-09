# Phase D.11 — Predictive Ranking & Regime Stability Audit

Verdict final : **`D11_UNSTABLE_RANKING_SIGNAL`**.

D.11 est un diagnostic OOF sur le seul dataset `RESEARCH`. Elle ne crée ni seuil de
trading, ni sizing, ni modèle, ni feature. D.10 reste figée avec le verdict officiel
`D10_INSUFFICIENT_EVIDENCE`. Le HOLDOUT est resté scellé et n'a été ni ouvert ni évalué.

## Reproduction et périmètre

- période : `[2026-06-10T00:00:00Z, 2026-09-08T00:00:00Z)` ;
- observations source : 92 143 ;
- observations validation OOF : 55 568, déclinées en 111 136 observations hypothétiques
  side-level LONG/SHORT ;
- candidats D.10 reproduits : 14 en `MODEL_DEVELOPMENT`, 4 en
  `INTERNAL_FREEZE_CHECK` ;
- runtime : 270,45 s ; peak mémoire Python : 94,34 MiB ;
- toutes les empreintes D.10 ont été vérifiées avant le premier fit.

Les deux sides sont conservés pour mesurer le pouvoir de classement. Le scope `COMBINED`
concatène donc deux hypothèses side-level par timestamp ; il ne représente pas deux
trades exécutables simultanés.

## Résultat principal de ranking

| Mesure OOF, toutes observations | LONG | SHORT | COMBINED |
|---|---:|---:|---:|
| Spearman predicted EV / BASE | -0,02995 | -0,02191 | -0,02672 |
| Kendall tau predicted EV / BASE | -0,02037 | -0,01449 | -0,01793 |
| Spearman conservative EV / BASE | -0,02341 | -0,01574 | -0,01995 |

Les coefficients observation-level sont négatifs. Le diagnostic par moyennes de jours
UTC mesure un autre estimand : pour `COMBINED`, Spearman vaut 0,3673 avec bootstrap
95 % `[0,0216 ; 0,6432]` sur 45 moyennes journalières. Cet intervalle ne doit pas être
présenté comme un intervalle IID autour du coefficient observation-level négatif.
Il suggère au mieux un effet entre journées, non un classement robuste des observations
au sein des journées.

La relation entre numéro de décile EV et moyenne BASE du décile est positive
(Spearman 0,8303), mais tous les déciles restent négatifs. Le dernier décile réalise
-6,98 points contre une EV prédite moyenne de -5,83 points.

## Courbe de ranking agrégée

| Quantile COMBINED | N | BASE | Uplift vs inconditionnel | STRESS | Win rate | PF BASE |
|---|---:|---:|---:|---:|---:|---:|
| top 10 % | 11 114 | -6,98 | +0,39 | -13,60 | 33 % | 0,47 |
| top 5 % | 5 557 | -7,29 | +0,09 | -13,90 | 33 % | 0,50 |
| top 2 % | 2 223 | -6,97 | +0,41 | -13,58 | 36 % | 0,59 |
| top 1 % | 1 112 | -6,70 | +0,68 | -13,30 | 38 % | 0,65 |
| top 0,5 % | 556 | -6,91 | +0,47 | -13,50 | 40 % | 0,68 |
| top 0,25 % | 278 | -8,05 | -0,68 | -14,65 | 40 % | 0,67 |
| top 0,1 % | 112 | -6,33 | +1,05 | -12,92 | 41 % | 0,75 |

Le top 5 % agrégé contient 5 557 observations, reste fortement négatif après coûts
BASE et STRESS, et son uplift de +0,0876 point est économiquement minime. Sa plus grande
session représente 42,7 % des lignes (`LONDON_NEW_YORK` : 2 373), donc le critère de
concentration session est respecté. Les issues sont 13,60 % `TARGET_FIRST`, 45,85 %
`STOP_FIRST` et 40,54 % `NEITHER`.

### Stabilité du top 5 % COMBINED

| Fold | Rôle | N | BASE inconditionnel | BASE top 5 % | Uplift | Semaines actives |
|---|---|---:|---:|---:|---:|---:|
| 1 | MODEL_DEVELOPMENT | 922 | -8,16 | -7,70 | +0,46 | 2 |
| 2 | MODEL_DEVELOPMENT | 881 | -7,06 | -6,80 | +0,26 | 2 |
| 3 | MODEL_DEVELOPMENT | 646 | -6,86 | -4,33 | +2,53 | 1 |
| 4 | INTERNAL_FREEZE_CHECK | 3 109 | -7,35 | -7,52 | -0,17 | 6 |

L'uplift est positif dans trois folds sur quatre mais change de signe dans le fold 4,
qui est précisément le contrôle interne gelé et le fold le plus grand. Les corrélations
COMBINED sont négatives dans les quatre folds (-0,0090, -0,0380, -0,0381, -0,0385).
Le résultat agrégé ne satisfait donc pas le critère de corrélation positive.

## Densité des EV conservatrices

Ces seuils sont descriptifs et n'ont servi à sélectionner aucune règle.

| Fold | Side | N | >0 | >1 | >2 | >3 | >5 | >10 | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | LONG | 9 218 | 10 | 8 | 8 | 6 | 4 | 2 | 19,36 |
| 1 | SHORT | 9 218 | 2 | 0 | 0 | 0 | 0 | 0 | 0,42 |
| 2 | LONG | 8 810 | 0 | 0 | 0 | 0 | 0 | 0 | -2,54 |
| 2 | SHORT | 8 810 | 0 | 0 | 0 | 0 | 0 | 0 | -4,91 |
| 3 | LONG | 6 459 | 1 | 0 | 0 | 0 | 0 | 0 | 0,19 |
| 3 | SHORT | 6 459 | 1 | 1 | 1 | 0 | 0 | 0 | 2,19 |
| 4 | LONG | 31 081 | 2 | 1 | 0 | 0 | 0 | 0 | 1,52 |
| 4 | SHORT | 31 081 | 2 | 2 | 2 | 1 | 1 | 0 | 9,77 |

La rareté extrême des EV positives confirme que les 18 candidats D.10 ne constituent
pas une base suffisante pour inférer une règle de trading.

## Calibration économique

| Scope COMBINED | Mean(actual - predicted) | MAE EV | RMSE EV |
|---|---:|---:|---:|
| Fold 1 | +0,68 | 13,66 | 17,22 |
| Fold 2 | +1,48 | 11,27 | 13,99 |
| Fold 3 | +1,24 | 14,55 | 19,05 |
| Fold 4 | +0,98 | 10,94 | 14,03 |
| Tous folds | +1,04 | 11,86 | 15,25 |

La reliability curve complète à dix groupes de taille comparable se trouve dans le
JSON et le parquet dédiés. Les erreurs absolues restent grandes relativement aux petits
uplifts observés dans la courbe de ranking.

## Changement de régime

Entre `MODEL_DEVELOPMENT` et `INTERNAL_FREEZE_CHECK`, la volatilité et l'activité sont
plus faibles : ATR M1/M5/M15 ont des SMD de -0,38/-0,35/-0,43 et des PSI de
0,22/0,20/0,30 ; les tick rates ont des SMD de -0,22 à -0,37. Le spread change peu
(SMD +0,05), comme les probabilités et l'EV prédite (SMD absolus <= 0,13).

Les fréquences de session (PSI 0,0009) et d'issue (PSI 0,0018) sont très proches.
La rupture spectaculaire des 18 candidats (0 gagnant sur 14, puis 2 sur 4) n'est donc
pas une preuve de stabilité : elle coïncide avec un régime de volatilité différent et
repose surtout sur quatre observations seulement.

## Anatomie des candidats — exploratoire uniquement

`MODEL_DEVELOPMENT` : 14 candidats, EV prédite moyenne +5,04, EV conservatrice +4,51,
BASE moyenne -41,65 ; 0 target, 12 stops, 2 neither.

`INTERNAL_FREEZE_CHECK` : 4 candidats, EV prédite/conservatrice moyenne +3,57, BASE
moyenne +7,87 ; 2 targets, 2 stops. Trois des quatre appartiennent à
`LONDON_NEW_YORK`.

Toute différence entre ces deux groupes est **`EXPLORATORY_POST_HOC`**. Elle n'est ni
validée, ni utilisable pour modifier D.10 sur RESEARCH.

## Safety buffer inchangé

| Fold | Side | Jours UTC | Résidu moyen | Dispersion résiduelle | Buffer | Pénalité brute |
|---|---|---:|---:|---:|---:|---:|
| 1 | LONG | 11 | -0,141 | 0,835 | 0,555 | 0,555 |
| 1 | SHORT | 11 | -0,543 | 0,915 | 0,997 | 0,997 |
| 2 | LONG | 16 | +0,379 | 1,222 | 0,123 | 0,123 |
| 2 | SHORT | 16 | -0,283 | 1,031 | 0,707 | 0,707 |
| 3 | LONG | 20 | +0,694 | 1,219 | 0 | -0,245 |
| 3 | SHORT | 20 | +0,437 | 1,183 | 0 | -0,002 |
| 4 | LONG | 23 | +0,979 | 1,483 | 0 | -0,470 |
| 4 | SHORT | 23 | +0,673 | 0,962 | 0 | -0,343 |

Les buffers du freeze valent zéro parce que la pénalité brute préenregistrée
`-mean_daily_residual + 1,645 × standard_error` est négative. Le `max(0, ...)` gelé la
ramène donc mécaniquement à zéro ; aucune nouvelle formule n'a été testée.

## Application des critères préenregistrés

- top 5 % meilleur que l'inconditionnel dans au moins 3 folds : PASS (3/4) ;
- relation moyenne décile/EV positive : PASS ;
- corrélation observation-level positive : **FAIL** ;
- borne basse positive pour l'estimand agrégé par moyennes journalières : PASS, mais ce
  n'est pas le même estimand que le ranking observation-level ;
- aucune session au-dessus de 60 % dans le top 5 % : PASS.

Le faible uplift agrégé, le changement de signe au freeze, les corrélations négatives
dans chaque fold et l'espérance toujours négative excluent
`D11_RANKING_SIGNAL_WORTH_FURTHER_RESEARCH`. L'existence d'un minuscule uplift agrégé
préenregistré conduit exactement au verdict **`D11_UNSTABLE_RANKING_SIGNAL`**, pas à
`D11_NO_RANKING_SIGNAL`.

## Empreintes et livrables

- moteur D.11 : `e7bdce114bcc651c4d8cd6e789161d01dda09a4b94fa657d028603dcb5d070e1` ;
- protocole D.11 : `2730ccadbe929fc94fd67fbdcb253d904fd8e672c9b549033837e5f29dd0d89a` ;
- observations ranking : `017e864cf5910b935941f6524c949875cfa17a5b3528e021b80c9a4e0ae55f71` ;
- rapport JSON : `data/reports/research-d11.json` ;
- observations OOF : `data/evaluations/research-d11-20260610T000000Z-20260908T000000Z/ranking-observations.parquet` ;
- calibration : `data/evaluations/research-d11-20260610T000000Z-20260908T000000Z/ev-calibration.parquet` ;
- régime et anatomie : JSON + Markdown dans le même répertoire d'évaluation.

Fin de D.11 : D.12 n'est pas créée, Phase E n'est pas commencée, le sizing et le
trading live restent absents. Toute suite requiert une validation explicite.
