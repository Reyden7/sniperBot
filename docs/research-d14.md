# SNIPER Phase D.14 — Economic Feasibility & Bottleneck Decomposition Audit

**Verdict principal : `D14_DIRECTION_ECONOMICALLY_TOO_WEAK`**

> Audit exclusivement diagnostique sur les prédictions OOF D.13 officielles. Aucun modèle n'a été réentraîné, aucun paramètre D.13 n'a été modifié et le HOLDOUT est resté scellé.

## Intégrité et périmètre

- Observations OOF : 48,024
- Hashes vérifiés avant analyse : True
- Erreur maximale du recalcul BASE : 0.000e+00 point
- Runtime : 1.428 s
- Peak mémoire Python : 19.76 MiB
- HOLDOUT : SEALED / NOT OPENED / NOT EVALUATED
- Phase E / sizing / live : NOT STARTED / DISABLED / DISABLED
- Validation logicielle : 244 tests, Ruff, format, mypy (66 modules) et build PASS

## Décomposition de l'edge

Population : CURRENT_REGIME (p >= 0.50) + CURRENT_DIRECTION figée.

| Étape | N | Expectancy (points) | PF |
|---|---:|---:|---:|
| GROSS_MARKET_EDGE | 4,645 | 0.234553 | 1.0490 |
| EXECUTABLE_AFTER_SPREAD | 4,645 | -0.062433 | 0.9874 |
| AFTER_SPREAD_AND_SLIPPAGE | 4,645 | -2.062433 | 0.6578 |
| BASE_NET | 4,645 | -6.671540 | 0.2638 |
| STRESS_NET | 4,645 | -13.280646 | 0.0810 |

Le spread exécutable est déjà incorporé dans les rendements Bid/Ask. BASE retire ensuite 2 points de slippage aller-retour et la commission simulée.

## Calibration directionnelle OOF

- Spearman signé global : 0.060962
- Spearman |prédiction| vs |mouvement réalisé| : 0.059473
- Sign accuracy : 50.3019%
- OLS : alpha=0.028224, beta=0.796380, R²=0.001176
- Diagnostic : `NO_DIRECTIONAL_MAGNITUDE_CALIBRATION`
- Alpha/beta n'ont été appliqués à aucune décision.

## Oracle bottleneck decomposition

| Système diagnostique | N | Gross | Executable | BASE | STRESS | PF BASE | Semaines +/actives |
|---|---:|---:|---:|---:|---:|---:|---:|
| CURRENT_REGIME_CURRENT_DIRECTION | 4,645 | 0.234553 | -0.062433 | -6.671540 | -13.280646 | 0.2638 | 0/10 |
| CURRENT_REGIME_ORACLE_DIRECTION | 4,645 | 9.816685 | 9.519699 | 2.910592 | -3.698515 | 2.6759 | 9/10 |
| ORACLE_REGIME_CURRENT_DIRECTION | 10,930 | 0.320860 | 0.001281 | -6.601521 | -13.204323 | 0.2801 | 0/10 |
| ORACLE_REGIME_ORACLE_DIRECTION | 10,930 | 10.830833 | 10.511253 | 3.908452 | -2.694350 | 4.6886 | 10/10 |

CURRENT_REGIME + CURRENT_DIRECTION donne -6.671540 point BASE, contre 2.910592 avec direction oracle. Les oracles emploient le futur et ne constituent pas des stratégies.

## Directional skill théorique requise

| Population | Break-even BASE | Break-even STRESS |
|---|---:|---:|
| ALL_OOF | not reached | not reached |
| REGIME_050 | 85% | not reached |
| REGIME_060 | 85% | not reached |
| REGIME_070 | 80% | not reached |

## Frontière de coûts diagnostique

- OBSERVED_SPREAD_ONLY : 3,685 candidats, expectancy 0.07924016282201223 point
- Break-even commission à 1 point de slippage/côté : None
- Break-even slippage à 2 EUR/lot/côté : None
- Ces scénarios ne remplacent pas BASE et ne représentent ni MetaQuotes-Demo ni un futur broker.

## HGB diagnostique

`INDIVIDUAL_ROWS_UNAVAILABLE_WITHOUT_FORBIDDEN_RETRAINING`. D.13 n'a pas persisté les lignes candidates HGB ; les reconstruire imposerait un réentraînement interdit. Les agrégats D.13 sont conservés dans le JSON sans promotion du modèle.

## Conclusion

Le protocole préenregistré conduit à `D14_DIRECTION_ECONOMICALLY_TOO_WEAK`. Ce verdict n'autorise ni D.15, ni l'ouverture du HOLDOUT, ni Phase E, ni sizing, ni trading live.

## Empreintes

- Protocole D.14 : `2c5604e10bbd67c4c9b258621bfd339efb6e63579c2899a6723534874e463457`
- research-d13.json : `f0117538f287e3c6b472f95cadcdd7e368495a94bf780a197b1fc9259e8a448a`
- outer-oof-predictions.parquet : `aadc3eaaaf046acbd9a566aecd11fb22abaf599acef71bca3909ad1a563e4ab0`
- d12-events.parquet : `a23531949d3ea9857223caf911405d48aafcd3dbbd70f78f86a561b7a3bf9c0f`
- research_d14.py : `2ce60de83ca34946cc13d60bdfd9fbfc0e16a0523dbfefb381cf4c9a5695923d`
- research-protocol-d14.yaml : `2c5604e10bbd67c4c9b258621bfd339efb6e63579c2899a6723534874e463457`
- edge_decomposition_parquet_sha256 : `4d42e2add9f99503976fbd4da14ca49767cd0e886c02065717fa1dabf3a9e0df`
- cost_frontier_json_sha256 : `81d35f9630ed8b8ac5f249b752a96d4e8d6a84dd7db8f101fd8c29ec03608aaf`
- oracle_decomposition_json_sha256 : `462f3f576548f919fa74e5193a1e0808d22437bdcdf921153a0b6e55ee896283`
- direction_calibration_json_sha256 : `b903a1d0ff2f8ee73155ee6a7acf5bce844e809d622e9906fcabc57f55241ea9`

## Limites

- All evidence is research OOF diagnostics, not independent final validation.
- Oracle systems use future outcomes and are descriptive upper bounds, never strategies.
- The abstract accuracy curve is a theoretical ceiling with hindsight-defined best side.
- D.13 did not persist row-level HGB candidate predictions; D.14 did not retrain them.
- Cost scenarios are simulated and do not claim MetaQuotes-Demo or broker capability.
