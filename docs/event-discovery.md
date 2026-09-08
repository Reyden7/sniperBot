# Phase D.7 — Event-Based Candidate Discovery

Le Feature Engine et le Signal Engine V1 restent gelés. Le scan vérifie leurs trois
empreintes SHA-256 avant de commencer. Il n'optimise aucun poids, seuil, feature ou
règle, ne retourne pas la stratégie et ne contient aucun chemin de trading live.

## Résolution temporelle et streaming

La politique retient le premier tick de chaque seconde UTC contenant au moins un tick.
Tous les ticks des 60 secondes visibles restent néanmoins utilisés par les features
tick inchangées. Une seconde sans tick ne crée pas d'observation. Les partitions sont
lues jour par jour avec 60 secondes de passé et 180 secondes de futur aux frontières.

Les features M15/M5/M1 sont constantes dans une minute et ne sont calculées qu'une fois
par minute. Deux évaluations synthétiques maximalement favorables, réalisées par le
Signal Engine lui-même, bornent le score atteignable dans cette minute pour BUY et SELL.
Si cette borne prouve que 90 est impossible, la machine d'état peut avancer sans appel
complet par seconde. Si 90 reste atteignable, ou si le réarmement sous 85 doit être
tranché, le premier tick de chaque seconde est évalué exactement. Cette accélération ne
change ni le score ni les règles du moteur.

## Épisodes 90/85

L'état de début de plage est censuré à gauche : il ne peut pas créer artificiellement
un épisode. Une fois armé, un épisode commence uniquement lorsque le score passe de
moins de 90 à au moins 90, sans blocker. Il reste actif tant que le score est au moins
90, sans blocker et dans la même direction. Il se termine sous 90, sur blocker ou sur
changement de direction.

Après sa fin, le détecteur reste désarmé jusqu'à un score strictement inférieur à 85.
Les seuils 90/85 sont imposés par la configuration méthodologique et ne peuvent pas
être explorés depuis la CLI.

`FIRST_CROSSING` est le seul instant tradable. `PEAK_SCORE` est le premier maximum de
l'épisode, identifié avec une information future ; il porte donc toujours
`tradable=false`.

## Diagnostics

Chaque instant conserve les rendements exécutables et mid à 5/10/30/60/180 secondes,
MFE/MAE, les coûts simulés, les prix courants et les features. L'orientation inverse est
calculée sur les mêmes timestamps et prix, mais reste `INVERSE_DIAGNOSTIC` : aucun BUY
ou SELL du moteur n'est modifié.

Les variables directionnelles numériques sont alignées sur la direction originale
avant Spearman ; les mesures de magnitude restent brutes. Les catégories sont décrites
par effectif, accuracy exécutable, rendement brut, MFE et MAE. Les cinq interactions
fixes sont : régime M15 × direction M5, direction M5 × momentum M1, momentum M1 ×
accélération tick, breakout/retest × session, volatilité × spread. Une cellule requiert
au moins 30 observations utilisables pour être déclarée suffisante.

Le Feature Engine n'expose qu'un champ combiné `breakout_retest`; D.7 le conserve tel
quel au lieu d'inventer deux nouveaux calculs. Les sessions sont dérivées de l'UTC avec
les règles IANA/DST. L'absence de calendrier historique de news reste une limitation
explicite.

## Coûts

Le coût total est la différence entre rendement mid avant coûts et rendement net
simulé. Il comprend donc spread observé, slippage simulé des deux côtés et commission
simulée. Le profil nano-lot n'est ni une capacité MetaQuotes-Demo ni une promesse sur un
futur broker réel.
