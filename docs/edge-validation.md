# Phase D.6 — Edge Validation

Le Signal Engine V1 reste gele pendant cette phase. Avant chaque scan, les empreintes
SHA-256 de `features/engine.py`, `strategy/scorer.py` et `strategy/filters.py` sont
comparees a la baseline D.6. Une divergence interrompt l'evaluation.

## Echantillonnage et streaming

La politique choisit le premier tick de chaque minute UTC contenant des ticks. Elle
produit donc au maximum une observation par minute active, sans observations repetitives
pendant les week-ends. Le signal a T ne voit que les ticks `<= T` sur 60 secondes et
les barres completes `<= T`. Les ticks futurs ne sont lus qu'apres la decision.

Le lecteur ouvre une journee de quatre colonnes (`timestamp_utc`, Bid, Ask, spread) et
les bordures necessaires de 60/180 secondes. Les observations sont ecrites en Parquet
par jour avant l'agregation. Les 18 millions de ticks ne coexistent jamais en memoire.

## Mesures

Pour chaque observation non bloquee dont les horizons disposent d'un tick a moins de
deux secondes de la cible, le rapport calcule les rendements executables directionnels
a 5/10/30/60/180 secondes. Les observations WAIT utilisent `trigger_direction` afin de
mesurer la qualite predictive des scores inferieurs au seuil d'entree ; BUY/SELL/WAIT
reste la decision originale.

MFE et MAE utilisent les prix executables pendant les 30 secondes suivantes. Les
stops/targets sont exactement ceux proposes par le Signal Engine et leur premier hit
est observe sur 180 secondes.

La decomposition a 30 secondes est : rendement mid avant couts, rendement executable
apres spread, moins un slippage simule par cote, puis moins la commission simulee. Le
profil par defaut utilise 0,0001 lot, 1 point de slippage par cote et 2 EUR/lot/cote,
sans minimum. Ces valeurs ne representent ni MetaQuotes-Demo ni un broker reel.

## Statistiques et diagnostic

Les IC 95 % utilisent 5 000 bootstraps par blocs de jours UTC avec le seed `20260908`.
Spearman mesure score/accuracy, score/expectancy, score/MFE et score/MAE. Une relation
attendue compte comme presente seulement si `|rho| >= 0,05` dans le bon sens.

Un bucket requiert au moins 100 observations utilisables pour etre marque suffisant.
Le diagnostic global exige au minimum 5 000 observations utilisables, 200 candidats
BUY/SELL et huit semaines candidates. `EDGE_CANDIDATE` exige en plus une expectancy
nette positive, une borne basse d'IC positive, au moins 60 % de semaines positives et
une relation score/qualite coherente. Ces regles sont fixes avant lecture des resultats.

Les sessions partent toujours de l'instant UTC canonique, jamais de l'heure serveur.
Elles utilisent les regles IANA/DST de Londres, New York et Tokyo et les fenetres locales
08:00-17:00, avec une categorie explicite pour l'overlap Londres/New York.

Limite importante : aucun calendrier historique de news n'est disponible. Pour isoler
la valeur predictive des features, `news_clear=true` est une hypothese d'evaluation
declaree, pas une verification historique.
