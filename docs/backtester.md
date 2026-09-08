# Backtester EURUSD tick event-driven

## Perimetre de la Phase C

Le moteur est exclusivement `BACKTEST`, ne contient aucun chemin `order_send` et ne
peut pas activer le trading live. `DummyStrategy(TEST_ONLY)` sert seulement a produire
des ouvertures et fermetures connues. Aucun signal, indicateur ou optimisation de
strategie n'est implemente.

## Ordre des evenements

Pour chaque tick UTC, dans l'ordre stable des partitions Phase B :

1. executer les demandes dont l'instant theorique est atteint ;
2. evaluer le SL/TP de la position ;
3. exposer uniquement le tick courant a la strategie de test ;
4. executer les nouvelles demandes a latence zero ;
5. reevaluer les protections et marquer le compte au marche.

Une latence de 25 ms demandee a `t` execute au premier tick `>= t + 25 ms`. Si aucun
tick futur n'existe, l'ouverture est rejetee avec `REJECT_NO_EXECUTION_TICK`. Les SL/TP
sont consideres comme protections serveur et s'executent des le tick declencheur.

## Prix et decomposition comptable

```text
BUY  entree ASK, sortie BID
SELL entree BID, sortie ASK
```

Le mid n'est jamais un prix d'execution. Il sert uniquement au contrefactuel permettant
d'isoler les couts :

```text
gross_pnl  = mouvement mid entre les deux ticks
spread     = gross_pnl - PnL aux quotes ASK/BID
slippage   = PnL aux quotes - PnL aux prix executes
execution_pnl = gross_pnl - spread - slippage
net_pnl       = execution_pnl - commission
              = gross_pnl - spread - slippage - commission
```

Le test croise obligatoire recalcule independamment :

```text
direct_execution_pnl = side_sign * (exit_price - executed_price)
                       * contract_size * volume * conversion
net_pnl = direct_execution_pnl - commission
```

Les champs ne sont pas interchangeables : `requested_price` est la quote visible au
signal, `execution_quote_price` la quote ASK/BID du premier tick executable,
`entry_slippage_price` leur delta vers `executed_price`. Les quatre memes notions sont
repetees pour la sortie. Le rapport embarque aussi un dictionnaire `price_semantics`.

Pour EURUSD avec compte EUR, chaque montant USD est converti par le mid du tick de
sortie. Les tests unitaires peuvent neutraliser cette conversion afin de conserver des
golden calculs entiers faciles a auditer.

`FixedSlippage(1)` est le modele conservateur par defaut : un point defavorable a
l'entree et a la sortie. Une valeur fixe negative simule un scenario favorable.
`RandomSlippage(max_points, seed)` tire des points signes et reproduit exactement la
meme sequence avec le meme seed.

`NoCommission`, `PerLotCommission` et `MinimumCommission` facturent chaque cote. Le
risque pre-trade inclut le stop, le slippage defavorable borne et la commission
aller-retour. Une commission minimale trop elevee provoque `REJECT_RISK_LIMIT`.

## Courtier et compte simules

Le courtier applique les `VolumeConstraints` Phase A : minimum, maximum et pas de
volume. Il calcule aussi la marge, refuse une marge insuffisante et limite le risque par
rapport a l'equity courante. SNIPER V1 autorise une seule position ouverte ou en attente.

Chaque rapport embarque `broker_profile.kind=SIMULATED`, le levier effectif calcule,
volume minimum/maximum/step, contract size, point, marge par lot et budget de risque.
`is_real_broker_capability=false` et l'avertissement nomme explicitement
MetaQuotes-Demo : le nano-lot simule n'est jamais presente comme une capacite reelle.

Le compte maintient balance, equity, marge utilisee/libre, PnL realise et latent. Toute
position fermee produit un enregistrement contenant les sequences d'evenements, les
horodatages demande/theorique/reel, les quotes et prix executes, les couts, balances et
motif de sortie.

## Metriques

Le rapport contient starting/ending equity, rendement net, PnL brut/net, couts totaux,
nombre de trades/gagnants/perdants, win rate, gains/pertes moyens, expectancy, profit
factor, drawdown maximum en valeur/pourcentage et pertes consecutives maximales.
