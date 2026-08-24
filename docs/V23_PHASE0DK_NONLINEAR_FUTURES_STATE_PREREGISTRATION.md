# V2.3 Phase 0D-K Nonlinear Futures-State Preregistration

Frozen: 2026-08-24 before any Phase 0D-K scoring.

## Motivation

Phase 0D-J rejected the preregistered linear Ridge formulation for BTCUSDT and ETHUSDT. Both J1/J2 candidates were negative after the 12 bps primary cost and failed to improve on the J0 trade-flow reference. Phase 0D-K tests a distinct hypothesis: the same causal information may contain nonlinear conditional structure that Ridge cannot represent.

This phase does not rewrite or rescue Phase 0D-J. The historical holdout 2026-08-04 through 2026-08-23 remains sealed.

## Data, clock, target, execution and folds

All data/causality semantics are inherited unchanged from the frozen Phase 0D-J preregistration:

- BTCUSDT and ETHUSDT
- development: 2026-05-26 through 2026-08-03 UTC only
- 1-minute decision grid after the completed minute
- state inputs: markPriceKlines, indexPriceKlines, premiumIndexKlines
- trade-flow inputs sampled causally from the frozen one-second DEV grid
- target/execution price: last-trade price at minute :59 and exactly H minutes later
- no state price is used as an execution price
- 5 chronological outer folds identical to Phase 0D-J
- chronological inner validation = final 20% of eligible outer-train rows
- horizon purge at inner and outer boundaries
- one same-symbol position at a time; fixed exit at H; no pyramiding, TP/SL, sizing or leverage optimization

## Frozen feature blocks

No new feature engineering is allowed.

### K1 — nonlinear state-only

Exactly Phase 0D-J J1 futures-state features.

### K2 — nonlinear combined

Exactly Phase 0D-J J2: frozen trade-flow F0 + futures-state F1 + the five predeclared interactions.

K2 is the primary nonlinear hypothesis. K1 is retained as an ablation/reference. J0/J1/J2 Ridge results remain historical references and are not retuned.

## Candidate horizons

- 5 minutes
- 10 minutes
- 30 minutes

## Model family

XGBoost regressor only, histogram tree method. GPU execution is requested with `device="cuda"`; CPU fallback is forbidden for official Phase 0D-K scoring because the run environment must be explicit and reproducible.

Objective: `reg:squarederror`
Random seed: `20260824`
Subsample: `0.80`
Column subsample by tree: `0.80`
L2 regularization (`reg_lambda`): `1.0`
L1 regularization (`reg_alpha`): `0.0`
Minimum child weight: `5.0`
Tree method: `hist`
Device: `cuda`
No early stopping and no validation-driven number of trees.

Exactly four frozen model configurations are allowed:

- X1: max_depth=3, learning_rate=0.05, n_estimators=300
- X2: max_depth=3, learning_rate=0.05, n_estimators=600
- X3: max_depth=5, learning_rate=0.03, n_estimators=300
- X4: max_depth=5, learning_rate=0.03, n_estimators=600

No Optuna, Bayesian optimization, random search, additional depths, learning rates, estimators, regularization tuning or post-result rescue is allowed.

## Signal gates

Absolute prediction quantiles computed from training predictions only:

- 0.990
- 0.995
- 0.9975
- 0.999

Direction = sign(prediction).

## Cost grid

Round-trip costs unchanged:

- 5 bps diagnostic
- 8 bps diagnostic
- 10 bps diagnostic
- 12 bps PRIMARY
- 15 bps stress

Promotion requires positive performance at both 12 and 15 bps.

## Inner survival filters

At 12 bps:

- net expectancy > 0
- total net PnL > 0
- profit factor > 1.0
- median independent trades/day on active days >= 2

At 15 bps:

- net expectancy > 0
- total net PnL > 0

## Inner selection objective

Search only `(block K1/K2, horizon, frozen X1-X4 config, gate quantile)`.

Select lexicographically:

1. higher median net bps/day at 12 bps
2. higher worst 5-day rolling net PnL
3. higher median independent trades/day
4. lower max drawdown
5. K1 over K2 only when median net bps/day is within 1% relative difference
6. shorter horizon
7. higher gate quantile
8. lower model index X1 < X2 < X3 < X4 only as final deterministic tie-break

## Development promotion gate

A symbol becomes a candidate only if all hold:

- all 5 outer folds are scored
- >=4/5 outer folds have positive total net PnL at 12 bps
- pooled net expectancy >= +1.0 bps/trade at 12 bps
- pooled total net PnL > 0 at 12 bps
- pooled net expectancy > 0 and total net PnL > 0 at 15 bps
- pooled profit factor >= 1.15 at 12 bps
- >=55% positive active trading days at 12 bps
- pooled PnL/max-drawdown >= 2.0 at 12 bps
- no scored outer fold expectancy below -2.0 bps/trade at 12 bps
- median independent trades/day on active days >=2

Incremental nonlinear requirement:

- pooled K1/K2 12-bps expectancy must exceed the Phase 0D-J Ridge candidate pooled 12-bps expectancy for the same symbol; and
- pooled K1/K2 total 12-bps PnL must exceed the Phase 0D-J Ridge candidate total 12-bps PnL.

Because Phase 0D-J Ridge candidates were negative, this comparison prevents declaring success merely from using a more complex model without material economic improvement.

## Holdout rule

The 2026-08-04 through 2026-08-23 holdout remains sealed during development. If a symbol passes, freeze one final full-development configuration, code hash, data manifests and model environment before any one-shot holdout confirmation. No redesign after holdout metrics are observed.
