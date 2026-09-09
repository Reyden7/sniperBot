# Phase D.12 — V4 Feature & Regime Discovery

Verdict final : **`D12_FEATURES_WORTH_V4`**.

Ce verdict signifie uniquement que plusieurs familles de features causales méritent
d'être étudiées dans une éventuelle architecture V4. Il ne valide ni stratégie, ni
profitabilité, ni seuil d'exécution. D.12 ne crée aucun `TRADE/SKIP`, `BUY/SELL`,
SL/TP ou sizing.

## Périmètre et préenregistrement

- dataset : `EURUSD_RESEARCH_20260610_20260908`, intervalle UTC
  `[2026-06-10T00:00:00Z, 2026-09-08T00:00:00Z)` ;
- HOLDOUT `EURUSD_HOLDOUT_PRE_20260610` : `SEALED`, non ouvert, non évalué ;
- 88 features, 8 interactions préenregistrées, 6 horizons et 2 targets ;
- folds expanding/purgés : trois folds de développement puis un
  `INTERNAL_FREEZE_CHECK` ;
- coûts : spread observé à T, slippage simulé de 1 point par côté, commission
  simulée de 2 EUR/lot/côté ;
- profil : `SIMULATED_RESEARCH_PROFILE_NOT_METAQUOTES_DEMO` ;
- protocole verrouillé : `docs/research-protocol-d12.yaml`.

Un premier replay a été invalidé avant interprétation finale : le détecteur de
compression utilisait par erreur la plage courante 300 s alors que le protocole
préenregistrait la plage courante 60 s. Le code a été aligné sur le protocole, un test
causal ciblé a été ajouté, puis tout le replay a été recommencé. Seul ce second run est
rapporté ici.

## Volume traité et performance

| Mesure | Valeur |
|---|---:|
| Ticks réellement analysés | 17 941 326 |
| Secondes actives examinées | 3 728 883 |
| Événements uniques avant complétude | 116 482 |
| Événements complets, tous horizons | 72 599 |
| Observations VALIDATION | 48 519 |
| Features | 88 |
| Interactions préenregistrées | 8 |
| Tests feature × horizon × target | 1 056 |
| Temps total | 1 163,82 s |
| Débit | 15 415,86 ticks/s |
| Pic mémoire Python | 282,55 MiB |
| Plus grande frame ticks chargée | 30,82 MiB |

Les 72 599 observations se répartissent en 24 080 lignes de développement initial,
puis 9 332 / 7 919 / 6 060 / 25 208 lignes dans les folds 1 à 4. Chaque timestamp est
unique. La lecture des ticks est journalière ; les 17,9 millions de ticks et toutes les
features ne sont jamais conservés simultanément en mémoire.

## Échantillonnage événementiel

Les effectifs `VALIDATION` comptent les appartenances aux schémas. Une observation
peut porter plusieurs types d'événement ; leur somme n'est donc pas le nombre de
timestamps uniques.

| Schéma | VALIDATION | Déclenchés avant complétude, total | Complets, total | Jours VALIDATION | Semaines |
|---|---:|---:|---:|---:|---:|
| Minute baseline | 40 933 | 92 158 | 61 022 | 45 | 10 |
| Volatility event | 184 | 469 | 267 | 41 | 10 |
| Breakout event | 126 | 457 | 181 | 42 | 10 |
| Acceleration event | 7 321 | 22 993 | 11 145 | 45 | 10 |
| Compression release | 275 | 1 188 | 444 | 45 | 10 |

À 300 s, les ratios moyens `FUTURE_TRADABLE_MOVEMENT / BASE_cost` sont 3,43 pour la
baseline minute, 6,27 pour volatility, 7,16 pour breakout, 4,07 pour acceleration et
3,58 pour compression-release. Les deux schémas aux valeurs descriptives les plus
élevées ont seulement 184 et 126 observations VALIDATION : ils ne constituent pas une
preuve autonome et aucune sélection post-hoc de schéma n'est faite.

## Labels multi-horizons

Les rendements directionnels sont exécutables : entrée LONG à l'Ask et sortie au Bid,
entrée SHORT au Bid et sortie à l'Ask. Le target régime est indépendant de la direction
et rapporte le meilleur mouvement favorable brut au coût BASE observé à T.

| Horizon | N | Move/coût moyen | Médiane | P(>1,5×) | P(>2×) | P(>3×) | P(>4×) | Rendement signé moyen, points |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 s | 48 519 | 1,155 | 0,913 | 23,50 % | 10,88 % | 3,52 % | 1,41 % | 0,028 |
| 60 s | 48 519 | 1,591 | 1,280 | 41,93 % | 22,96 % | 8,75 % | 3,66 % | 0,095 |
| 180 s | 48 519 | 2,744 | 2,260 | 76,50 % | 56,91 % | 31,37 % | 16,64 % | 0,191 |
| 300 s | 48 519 | 3,541 | 2,889 | 88,07 % | 73,31 % | 47,81 % | 29,04 % | 0,250 |
| 600 s | 48 519 | 4,988 | 4,106 | 96,42 % | 89,23 % | 70,57 % | 51,45 % | 0,495 |
| 900 s | 48 519 | 6,091 | 4,991 | 98,54 % | 94,74 % | 81,69 % | 65,15 % | 0,651 |

Ces ratios décrivent la présence d'amplitude future, pas la capacité à choisir son
sens ni une espérance nette exploitable.

## Matrice univariée et correction multiple

La matrice contient exactement `88 × 6 × 2 = 1 056` tests. Une correction
Benjamini-Hochberg unique est appliquée avec `q = 0,10`. Un candidat exige
simultanément N ≥ 5 000, |Spearman| ≥ 0,03, un signe cohérent dans au moins 3 folds,
un freeze non opposé, une monotonie de bins ≥ 0,70, aucune session dominante, une
résistance au winsorizing, un IC journalier excluant zéro et FDR ≤ 0,10.

| Target | Aucun signal | Instable | Candidat recherche |
|---|---:|---:|---:|
| REGIME_QUALITY | 154 | 126 | 248 |
| WITHIN_REGIME_DIRECTION | 479 | 33 | 16 |
| Total | 633 | 159 | 264 |

### Régime

Le régime passe le critère préenregistré avec 43 noms de features, six familles et les
six horizons. Les effets dominants décrivent surtout la continuité de volatilité et
d'amplitude :

| Feature | Horizon | Spearman | IC 95 % jour | Folds 1 / 2 / 3 / freeze |
|---|---:|---:|---:|---:|
| realized_volatility_300s_points | 60 s | +0,538 | [+0,486 ; +0,529] | +0,506 / +0,498 / +0,576 / +0,533 |
| realized_volatility_300s_points | 300 s | +0,537 | [+0,484 ; +0,528] | +0,481 / +0,515 / +0,567 / +0,541 |
| range_900s_points | 300 s | +0,492 | [+0,415 ; +0,470] | +0,418 / +0,460 / +0,510 / +0,504 |
| realized_move_to_base_cost_300s | 300 s | +0,485 | [+0,416 ; +0,467] | +0,428 / +0,436 / +0,507 / +0,496 |

Ce résultat établit une prédictibilité de l'amplitude future conditionnelle à
l'amplitude récente. Il ne montre pas encore qu'une exécution directionnelle survit
aux coûts.

### Direction dans les régimes causaux

La direction passe de justesse le critère formel avec 11 noms de features, quatre
familles et uniquement les horizons 30 s et 60 s. Les effets sont faibles et
contrariants :

| Feature | Horizon | Spearman | IC 95 % jour | Folds 1 / 2 / 3 / freeze |
|---|---:|---:|---:|---:|
| current_signed_run_length | 30 s | −0,0536 | [−0,0635 ; −0,0376] | −0,0559 / −0,0990 / −0,0520 / −0,0388 |
| return_5s_points | 30 s | −0,0515 | [−0,0613 ; −0,0309] | −0,0720 / −0,0828 / −0,0294 / −0,0385 |
| breakout_velocity_15s_points_per_second | 30 s | −0,0462 | [−0,0579 ; −0,0249] | −0,0840 / −0,0831 / −0,0262 / −0,0243 |
| signed_tick_imbalance_1s | 30 s | −0,0420 | [−0,0498 ; −0,0245] | −0,0441 / −0,0781 / −0,0423 / −0,0295 |
| current_signed_run_length | 60 s | −0,0412 | [−0,0520 ; −0,0290] | −0,0308 / −0,0818 / −0,0391 / −0,0331 |

Plusieurs noms sont mathématiquement proches : `breakout_velocity_15s` dérive du
retour 15 s et son interaction low-spread est souvent équivalente. Le comptage formel
préenregistré les conserve séparément, mais une éventuelle spécification V4 devra
traiter explicitement cette redondance avant tout nouveau freeze.

## Diagnostics multivariés

Les modèles sont des sondes OOF, pas des candidats de trading. HGB direction obtient
des Spearman globaux de +0,0326 / +0,0182 / +0,0041 / −0,0004 / +0,0059 / +0,0111 aux
horizons 30/60/180/300/600/900 s. Il ne démontre donc pas une direction exploitable au
delà du très court terme.

Le modèle régime sépare bien les périodes d'amplitude, mais les PR-AUC doivent être
lues avec leurs taux de base croissants : par exemple 0,437 contre 0,109 à 30 s,
0,917 contre 0,733 à 300 s et 0,992 contre 0,947 à 900 s pour HGB. La volatilité est le
groupe dominant dans l'importance OOF du classifieur aux horizons courts.

Le Ridge directionnel présente des MAE numériquement aberrantes malgré des Spearman
proches de zéro. Ce diagnostic linéaire est considéré non interprétable en niveau ; il
n'est utilisé ni dans le verdict univarié, ni pour une décision, ni pour une
profitabilité. Cette anomalie de conditionnement devra être résolue avant toute
spécification V4.

Les 24 invariants de purge modèle/horizon/fold passent :
`max(timestamp_train + horizon) <= validation_start`.

## Hypothèse journalière D.11

Sur seulement 45 journées VALIDATION, la tradabilité moyenne à 300 s est corrélée à
l'Asian range (+0,461), à la volatilité des 30 premières minutes de Londres (+0,517)
et au range des 60 premières minutes de Londres (+0,376). Les features sont exposées
uniquement après que leur fenêtre est complète. Ces résultats restent marqués
`EXPLORATORY_DAY_LEVEL_HYPOTHESIS_NOT_VALIDATED` : peu de jours indépendants, p-values
non corrigées et aucune autorité de sélection.

## Verdict et limites

Les critères verrouillés passent pour les deux problèmes : régime (43 features,
6 groupes, 6 horizons) et direction (11 features, 4 groupes, 2 horizons), sans effet
opposé dans le fold 4 pour les lignes candidates. Le verdict mécanique est donc
`D12_FEATURES_WORTH_V4`.

Il faut toutefois conserver trois réserves :

1. l'information régime est principalement une persistance de volatilité/amplitude ;
2. l'information directionnelle est petite, contrariante, concentrée à 30–60 s et
   faible dans les modèles multivariés ;
3. les meilleurs schémas événementiels descriptifs ont de faibles effectifs.

D.12 s'arrête ici. Le HOLDOUT reste scellé. V4, Phase E, le Risk Engine d'exécution et
le trading live ne sont pas créés.

## Artefacts et empreintes

- rapport complet JSON : `data/reports/research-d12.json` ;
- matrice lisible : `data/evaluations/research-d12-20260610T000000Z-20260908T000000Z/feature-matrix.json` ;
- matrice Parquet : `data/evaluations/research-d12-20260610T000000Z-20260908T000000Z/feature-matrix.parquet` ;
- observations : `data/evaluations/research-d12-20260610T000000Z-20260908T000000Z/events.parquet` ;
- moteur D.12 SHA-256 : `aef820578dfa962c804b2764257386f81909f50df137736f4dcae7640f404593` ;
- protocole SHA-256 : `93091917b41e99c8a27c912d46d90c9a18970a6457c4922d85506531e8e74aeb` ;
- observations SHA-256 : `a23531949d3ea9857223caf911405d48aafcd3dbbd70f78f86a561b7a3bf9c0f` ;
- matrice SHA-256 : `d80c26de6123a0dd746440b8f3c15b93e99a8549d0aa07529f33c00682d7c8af`.
