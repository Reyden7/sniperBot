# Rapport de compatibilite et sizing initial

Le capital de la CLI est en EUR ; la valeur par defaut est 10. Le compte reel
doit etre en EUR pour un verdict compatible dans ce premier lot. Pour un compte
USD ou en cents, les montants MT5 restent affiches dans leur devise, les
pourcentages rapportes au capital EUR sont inconnus et le verdict refuse la
compatibilite. Une conversion verifiee sera necessaire pour etendre ce support.

Le risque cible par defaut vaut 0,25 % ; le plafond absolu est 0,50 %.
Les valeurs de configuration sont des pourcentages (0.25), pas des fractions
(0.0025). Les calculs et la grille de volume utilisent Decimal ; seuls les
arguments de l'API native MT5 sont convertis en float.

Les stops 3/5/10 pips sont des scenarios configurables, pas des stops optimises.
Un pip est une convention d'affichage EUR/USD derivee de point et digits
(cotation a 4 ou 5 decimales). Aucune valeur monetaire de pip n'est codee en dur.
Les demandes au terminal utilisent ses prix, son tick size et son volume minimal.

- BUY : entree Ask, stop Bid moins la distance, arrondi vers le bas au tick size.
- SELL : entree Bid, stop Ask plus la distance, arrondi vers le haut au tick size.
- Le spread est donc deja inclus dans la perte au stop. Il n'est pas ajoute deux fois.
- Un stop qui ne respecte pas trade_stops_level est marque invalide.
- Le verdict utilise le plus petit stop configure valide dans les deux directions,
  puis la pire perte BUY/SELL a ce stop. Les autres scenarios restent visibles.
- Perte totale = perte au stop + commission aller-retour + slippage aller-retour estime.
- Commission aller-retour = 2 * max(volume * tarif par lot par cote, minimum par cote).
- Cout minimal aller-retour = cout du spread courant + commission. Le cout estime
  ajoute le slippage configure, exprime en somme de points adverses entree + sortie.
- La marge du volume minimum doit tenir dans min(capital, equity, marge libre),
  moins le buffer configure, dans les deux directions.

Le spread median reste inconnu : la collecte statistique n'appartient pas a ce lot.
Le rapport refuse les valeurs manquantes qui empechent le verdict, les donnees non
finies, les Bid/Ask invalides, les ticks perimes ou futurs et les autres paires.
Les drapeaux EA du compte ne prouvent pas l'autorisation contractuelle du scalping.

## Politique JSON optionnelle

`--config fichier.json` remplace la politique broker issue des valeurs par defaut
ou de l'environnement ; `--stop-pips` la surcharge ensuite. Exemple sans valeurs
broker inventees (il produit volontairement un rapport incomplet) :

```json
{
  "schema_version": 1,
  "stop_pips": ["3", "5", "10"],
  "risk_per_trade_pct": "0.25",
  "hard_max_risk_pct": "0.50",
  "minimum_margin_buffer": "0",
  "max_spread_points": null,
  "max_round_trip_cost_pct": null,
  "round_trip_slippage_points": null,
  "commission": null,
  "legally_accessible_from_france": null,
  "scalping_allowed": null,
  "historical_ticks_available": null,
  "terms_source": null
}
```

Quand le tarif est verifie, `commission` contient `currency`, `per_lot_per_side`,
`minimum_per_side` et `source` (reference du tarif). Un broker sans commission
doit avoir des zeros explicites et une source ; inconnu n'equivaut jamais a zero.
Un tarif non representable par ce modele reste inconnu. Les booleens de conditions
broker exigent une `terms_source`. Les seuils spread/cout/slippage doivent provenir
de mesures et de la calibration, sans valeur imposee par ce lot.

Les motifs sont cumulatifs ; le verdict principal suit cet ordre : TOO_COARSE,
INSUFFICIENT_MARGIN, SPREAD_TOO_HIGH, COST_TOO_HIGH, INCOMPATIBLE_FOR_10_EUR.
Toute presence de motif donne aussi `challenge_status=INCOMPATIBLE_FOR_10_EUR`.
COMPATIBLE n'etablit pas un edge rentable et ne peut jamais activer le live.

## Sizing de recherche

`size_position` cherche le plus grand volume `minimum + n * step <= maximum`
dont la perte tout compris est inferieure ou egale au budget et dont la marge
respecte le buffer. En dessous du volume minimum, il retourne zero et un motif.
Il evalue le volume exact avec des fonctions de perte/marge injectees et ne
multiplie pas une estimation arrondie a un lot. La recherche binaire exige des
fonctions non negatives, finies et non decroissantes en volume. Ces fonctions
doivent integrer un stop valide et tous les couts ; aucun ordre n'en resulte.
Le moteur de risque complet (daily loss, pertes consecutives, kill switch) est
reserve aux missions suivantes.
