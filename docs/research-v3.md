# Phase D.9 — SNIPER V3 Direction Research

V3 reformule la direction comme deux problèmes path-dependent indépendants :

```text
LongOutcomeModel  -> P(target LONG avant stop LONG)
ShortOutcomeModel -> P(target SHORT avant stop SHORT)
                              ↓
                    MetaGate TRADE / SKIP
```

Le modèle ne produit ni lot ni ordre. Le sizing reste réservé au Risk Engine et le
trading live est absent.

## Labels triple barrière

À T, une position LONG entre à l'ASK et toutes ses sorties sont évaluées au BID. Une
position SHORT entre au BID et sort à l'ASK. Pour chaque côté, le premier événement
parmi `TARGET_FIRST`, `STOP_FIRST` et `NEITHER` est conservé. Une collision au même
tick est classée `STOP_FIRST` de manière conservatrice.

```text
stop_points   = max(k_stop × ATR_M1(T), 1 point)
target_points = max(k_target × ATR_M1(T), minimum_ratio × coût_BASE(T))
```

Les configurations B01, B02_PRIMARY, B03 et B04 sont intégralement définies dans
`docs/research-protocol-v3.yaml`. Le primaire est B02_PRIMARY : stop 1×ATR, target
2×ATR avec plancher 2×coût BASE, timeout 300 secondes.

## Features causales

Les features V2 sont réutilisées. Les ajouts V3 sont calculés exclusivement avec les
quotes `timestamp <= T` : returns 5/15/30/60 secondes, distances au high/low 60 s,
position dans ce range, pente/momentum normalisés ATR, accélération du momentum,
imbalance signé, série de ticks directionnels, profondeur du micro-pullback, pression
corps/mèches et qualité du retest. Leurs formules exactes figurent dans le JSON.

La qualité du mouvement est `MFE / max(MAE, 1 point)`. Le plancher d'un point évite
qu'une MAE nulle transforme un ratio descriptif en valeur artificiellement infinie.

## Protocole

Le dataset `[2026-06-10, 2026-09-08)` reste exclusivement `RESEARCH`. Le split utilise
60 % de TRAIN initial, puis quatre fenêtres walk-forward de 10 %. À chaque fold,
scaler, encodeur et modèle sont ajustés sur le passé uniquement.

Les modèles LONG et SHORT comparent prévalence triviale, régression logistique
régularisée et HistGradientBoosting. La régression logistique est le modèle primaire
préenregistré. Une direction est éligible si `P(target-first) >= 0,60` et si son edge
BASE conservateur prévu est positif ; si les deux côtés passent, le meilleur edge est
retenu, sinon la décision est `SKIP`.

Le HOLDOUT `data/holdout-v2` reste scellé et n'apparaît dans aucun chemin de lecture du
moteur V3.

## Résultat du 8 septembre 2026

Le replay a traité 17 941 326 ticks, produit 92 143 observations et 248 398 labels
directionnels walk-forward. Sur B02_PRIMARY, LONG compte 4 429 `TARGET_FIRST`,
15 479 `STOP_FIRST` et 11 173 `NEITHER` ; SHORT compte respectivement 4 131, 15 905
et 11 045 événements.

La PR-AUC logistique progresse modestement par rapport à la prévalence : 0,1677 contre
0,1450 pour LONG et 0,1598 contre 0,1318 pour SHORT. L'accuracy équilibrée au seuil
0,5 reste cependant égale à environ 0,50. Au seuil MetaGate préenregistré de 0,60,
seuls deux candidats subsistent. Aucun n'atteint sa cible : expectancy exécutable
−39,000 points, nette BASE −45,625 et STRESS −52,250 points.

Les deux candidats proviennent des semaines W33 et W36 et uniquement de la session
Londres/New York. Les trois hypothèses D.7 ont toutes une expectancy BASE négative.
La conclusion est `V3_RESEARCH_REJECTED`. Aucune configuration secondaire ne remplace
B02_PRIMARY et le HOLDOUT reste scellé.
