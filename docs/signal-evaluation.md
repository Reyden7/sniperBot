# Phase D.5 — Signal Evaluation

Cette phase mesure le score V1 tel quel. Elle ne modifie ni ses poids ni ses seuils,
ne choisit aucun lot et ne peut envoyer aucun ordre.

## Separation temporelle

A chaque pas regulier, le Feature Engine ne recoit que les bougies completes et les
ticks horodates au plus tard a T. Le Signal Engine produit ensuite une observation
immuable. Un composant d'evaluation distinct lit seulement apres cette decision les
ticks futurs necessaires aux horizons 5, 10, 30, 60 et 180 secondes. Les deux sens
sont toujours mesures, independamment du sens propose.

L'observation conserve separement son timestamp, le timestamp de la quote courante et
son age. Ainsi un Bid/Ask ancien ne peut pas etre confondu avec une quote recue a T.

Un tick d'horizon est le premier tick a ou apres l'horizon demande. S'il manque ou
arrive au-dela de la tolerance configuree, l'observation porte
`FUTURE_DATA_INCOMPLETE` et n'entre pas dans les agregats. Il n'y a ni interpolation,
ni forward fill, ni prix synthetique.

## Prix et excursions

- LONG : entree au ASK courant, sortie au BID futur.
- SHORT : entree au BID courant, sortie au ASK futur.
- MFE 30 s : meilleur mouvement executable dans le sens considere, borne a zero.
- MAE 30 s : pire mouvement executable contre le sens considere, borne a zero.

Le spread est donc deja incorpore une fois dans chaque rendement executable. Le
`potential_net_return` retranche en plus le slippage aller-retour et la commission
convertie en points ; il ne retranche pas une seconde fois le spread.

`expected_total_execution_cost_points` vaut spread observe + deux slippages par cote
et commission aller-retour. Le rapport expose `target_cost_ratio`, le respect du ratio
minimum de reference et la part du target consommee par les couts. Cette mesure ne
constitue pas encore une regle de Risk Engine.

## Interpretation

Les buckets sont fixes : 50-59, 60-69, 70-79, 80-89, 90-94 et 95-100. Le diagnostic
compare le win rate a 30 s dans `trigger_direction`, MFE, MAE et rendement net
potentiel. Moins de
deux buckets peuples produit `INSUFFICIENT_BUCKETS`. Sinon, le rapport indique
seulement si les quatre series sont monotones ; il ne calibre rien et ne conclut pas
a lui seul a une rentabilite statistiquement significative.

Les sessions sont de simples buckets UTC de recherche consignes dans le rapport. Les
statuts session/news utilises comme filtres restent explicites et fail-closed.
