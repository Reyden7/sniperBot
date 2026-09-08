# Feature Engine et Signal Engine V1

Le moteur analyse EURUSD uniquement. Il ne choisit jamais un volume et ne transmet
aucun ordre. Sa sortie porte `sizing_authority=RISK_ENGINE_ONLY`.

## Features limitees

- M15 : pente EMA, HH/HL ou LH/LL, ATR, regime TREND/RANGE ;
- M5 : EMA 3/8, momentum 3 barres, ATR, support/resistance, breakout/retest ;
- M1 : momentum 1/3/5, range/ATR, corps/meches, engulfing, hammer,
  shooting star, compression/expansion ;
- ticks : taux 1/5/15 s, ratio up/down, acceleration prix et tick rate, spread
  actuel et median sur la fenetre recente.

Seules les barres marquees `is_complete=true`, cloturees avant `as_of_utc`, et les ticks
`<= as_of_utc` sont visibles. Les fichiers peuvent contenir des donnees futures sans
modifier le resultat point-in-time.

## Score fixe non optimise

```text
M15 regime/context        0..15
M5 trend alignment        0..15
M1 momentum               0..15
Tick confirmation         0..15
Market structure          0..15
Volatility quality        0..10
Spread/execution quality  0..10
Session/news quality      0..5
```

```text
0..79   WAIT
80..89  WATCH (side reste WAIT)
90..94  CANDIDATE
95..100 PREMIUM_CANDIDATE
```

Un blocker force toujours `side=WAIT`, quel que soit le score. Le seuil de spread,
l'etat de session et l'etat news ne sont jamais devines : absents ou non verifies, ils
produisent des blockers explicites. Ces valeurs initiales ne font l'objet d'aucune
optimisation.
