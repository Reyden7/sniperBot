# Proposition Phase D.10 — moteur d’espérance à trois issues

Statut : **PROPOSED / NOT EXECUTED**. Cette spécification ne constitue ni une validation de
recherche, ni une autorisation d’ouvrir le HOLDOUT.

## Question primaire

Pour LONG et SHORT séparément, estimer les trois probabilités mutuellement exclusives :

- `P(TARGET_FIRST)` ;
- `P(STOP_FIRST)` ;
- `P(NEITHER)`.

La décision primaire sera fondée sur l’espérance nette BASE conditionnelle :

```text
EV_BASE =
    P(target)  * E[BASE pnl | target]
  + P(stop)    * E[BASE pnl | stop]
  + P(neither) * E[BASE pnl | neither]
```

Le PnL de `NEITHER` devra être le PnL exécutable constaté à l’expiration, avec exactement la
même décomposition des coûts que le reste du pipeline. Il ne devra pas être remplacé par la
perte du stop.

## Challenger séparé

Un challenger pourra effectuer une régression directe vers le `BASE net pnl` attendu. Il devra
rester séparé du modèle probabiliste à trois issues et être comparé selon un protocole déclaré
avant ses résultats.

## Protocole de sélection

Toutes les features, familles de modèles, hyperparamètres, règles de calibration, seuils et le
`safety_buffer` seront sélectionnés uniquement dans TRAIN au moyen d’un walk-forward purgé et
nested. La condition d’exécution future sera :

```text
expected_net_edge > safety_buffer
```

Le buffer devra refléter l’incertitude d’estimation et la résistance aux coûts ; il ne pourra pas
être choisi après lecture d’une fenêtre de contrôle. Les labels TRAIN devront respecter, pour
chaque configuration, `label_end_time <= validation_start`.

Avant tout contrôle final interne, la configuration complète devra être gelée : features,
modèle, hyperparamètres, calibration, barrières, coûts, horizon principal, métrique, buffer et
critères d’acceptation. Aucun élément ne pourra être modifié sur la base de ce contrôle.

## Critères minimaux à préenregistrer

- calibration multiclasses et score probabiliste approprié ;
- espérance BASE et STRESS, effectif exécutable et intervalles de confiance ;
- stabilité hebdomadaire et par session ;
- comparaison aux baselines et à V3 ;
- décision sans sélection opportuniste d’horizon, de coût ou de sous-groupe ;
- verdict explicite d’acceptation, de recherche supplémentaire ou de rejet.

## Frontières

Le HOLDOUT reste scellé jusqu’à ce qu’un candidat futur soit entièrement gelé et autorisé par
validation explicite. Cette spécification n’exécute aucun entraînement D.10, ne commence pas la
Phase E, ne dimensionne aucune position et n’implémente aucun trading live.
