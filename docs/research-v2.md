# Phase D.8 — SNIPER V2 Research Engine

V1 est conservé comme baseline historique `RESEARCH_REJECTED`. V2 ne réutilise pas le
score additif comme probabilité de trade et sépare trois responsabilités :

```text
Features connues à T
  ├─> OpportunityModel : amplitude/MFE/MAE et P(mouvement > coût)
  ├─> DirectionModel   : P(LONG), P(SHORT), rendement directionnel attendu
  └─> ExecutionGate    : gross - spread - slippage - commission = net
```

Le gate ne produit que des candidats. Il n'existe aucun sizing, envoi d'ordre ou
trading live ; un futur sizing restera exclusivement sous l'autorité du Risk Engine.

## Données et isolation temporelle

La plage `[2026-06-10T00:00:00Z, 2026-09-08T00:00:00Z)` est définitivement
`RESEARCH`. Le moteur échantillonne le premier tick de chaque minute UTC active. Les
features utilisent uniquement les ticks et bougies connus à T ; les labels 30, 60,
180, 300, 600 et 900 secondes sont construits séparément depuis les quotes futures.

Le walk-forward commence par 70 % de TRAIN. Les trois fenêtres suivantes couvrent
chacune 10 % de la durée :

- folds 1–2 : `MODEL_COMPARISON` ;
- fold 3 : `INTERNAL_FREEZE_CHECK` en lecture seule.

Chaque scaler, encodeur et modèle est réajusté uniquement sur le TRAIN antérieur à sa
fenêtre. Après calcul du fold 3, aucune feature, aucun hyperparamètre, seuil, horizon
principal ou scénario de coût ne peut être modifié à partir de ses résultats.

Le rapport principal a été préenregistré avant lecture des résultats : horizon 300 s,
scénario `BASE`, ratio mouvement/coût minimal 1,5. Les autres combinaisons sont des
diagnostics de sensibilité et ne peuvent pas remplacer le rapport principal.

## Modèles et coûts

OpportunityModel compare une moyenne constante, une régression logistique régularisée
et des arbres `HistGradientBoosting` peu profonds. DirectionModel compare aléatoire,
momentum, mean-reversion, régression logistique régularisée et
`HistGradientBoosting`. Les métriques incluent MAE/RMSE, accuracy équilibrée,
précision/rappel, PR-AUC, calibration et rendements brut/exécutable.

Les scénarios restent explicites :

- `OPTIMISTIC` : spread observé, sans slippage ni commission ;
- `BASE` : spread observé + 1 point de slippage par côté + 2 EUR/lot/côté ;
- `STRESS` : spread observé + 2 points par côté + 4 EUR/lot/côté.

Ce sont des hypothèses de recherche simulées, jamais une capacité déclarée de
MetaQuotes-Demo ou d'un futur broker.

## HOLDOUT

`collect-holdout` collecte dans une racine séparée, effectue uniquement des contrôles
de compte et de bornes temporelles, puis écrit `.holdout-sealed.json`. Le HOLDOUT ne
peut être lu par le Research Engine et aucune métrique, feature, prédiction ou
exploration de marché n'est autorisée avant un freeze V2 explicitement validé.

```powershell
.venv\Scripts\sniper.exe collect-holdout `
  --start 2026-02-01T00:00:00Z `
  --end 2026-06-10T00:00:00Z `
  --output data/holdout-v2
```

Le rapport machine complet est généré dans `data/reports/research-v2-*.json` et le
rapport humain correspondant dans `data/reports/research-v2-*.md`.
