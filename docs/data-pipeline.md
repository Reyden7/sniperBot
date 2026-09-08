# Pipeline de donnees EURUSD

## Temps

`timestamp_utc` est l'unique reference de stockage, de tri, de logique et de futur
backtest. Le SDK Python MT5 documente des bornes UTC ; aucune correction de plusieurs
heures n'est appliquee automatiquement.

- `timestamp_utc` : instant canonique UTC decode de `time_msc` ;
- `timestamp_server` : representation optionnelle du meme instant si un fuseau IANA
  serveur a ete fourni et verifie ; sinon `null` ;
- `timestamp_local` : representation d'affichage du meme instant, Europe/Paris par
  defaut ;
- `source_time_msc` : valeur epoch milliseconde originale de MT5.

Un decalage entre un tick et l'horloge hote donne `TIME_REFERENCE_UNVERIFIED`, jamais
`CLOCK_DESYNCHRONIZATION` a lui seul. Cette derniere valeur exige une divergence entre
le temps UTC ecoule et une horloge monotone independante. Les heures serveur ambigues
ou inexistantes pendant les transitions DST sont refusees.

## Validation et de-duplication

La plage de sortie est toujours semi-ouverte `[start, end)`, meme si l'API MT5 inclut
sa borne superieure. Bid et Ask doivent etre positifs, Ask >= Bid, les volumes et flags
non negatifs, et toutes les valeurs numeriques finies. Un doublon exact partage :

```text
source_time_msc, bid, ask, last, volume, volume_real, flags
```

Deux ticks differents portant la meme milliseconde sont donc conserves.

## Partitions et bougies

Les partitions journalieres utilisent Zstandard :

```text
data/ticks/symbol=EURUSD/date=YYYY-MM-DD/part-000.parquet
data/bars/timeframe=M1/symbol=EURUSD/date=YYYY-MM-DD/part-000.parquet
data/bars/timeframe=M5/symbol=EURUSD/date=YYYY-MM-DD/part-000.parquet
data/bars/timeframe=M15/symbol=EURUSD/date=YYYY-MM-DD/part-000.parquet
```

Chaque tick met a jour l'etat courant M1 en un seul passage. Les M1 fermees alimentent
ensuite les etats M5 et M15. Lorsqu'une nouvelle collecte touche une journee existante,
seule cette partition journaliere est consolidee afin de corriger proprement les
bougies partielles sans doubler les volumes.

Chaque barre porte `is_complete`. Le dernier bucket encore ouvert est conserve pour
l'audit mais marque `false` ; le Feature Engine l'exclut obligatoirement.

Le rapport contient la plage demandee et couverte, les volumes source/accepte,
doublons/invalides/hors plage, les trous au-dela du seuil, les spreads
min/mediane/p95/max et le
nombre de M1/M5/M15 produits. Une indisponibilite MT5 renvoie le code 2 et
`data_fabricated=false` en JSON.

## Collecte massive et reprise

`sniper collect-history --symbol EURUSD --days 90 --output data` prend par defaut les
90 derniers jours UTC complets et les requete un jour a la fois. Apres chaque tranche
terminee, un manifeste est ecrit sous
`data/manifests/history/symbol=EURUSD`. Une relance reprend les tranches ayant un
manifeste valide et rejoue les autres ; une journee de week-end vide peut donc etre
valablement terminee. Les donnees et manifestes locaux sont exclus des distributions.
