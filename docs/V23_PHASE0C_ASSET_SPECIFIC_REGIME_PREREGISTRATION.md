# V2.3 Phase 0C Preregistration — Asset-Specific, Regime-Gated Prediction

Date frozen: 2026-08-24

## Prior result

Phase 0A rejected the original core cross-market block with pooled six-bar incremental OOS R2 near -0.0047. Phase 0B then rejected the expanded dense cross-sectional representation more decisively. The Phase 0B result is treated as a representation failure, not as a license to select favorable sensors post hoc.

Phase 0C therefore changes the research question before any new predictive scoring: each target receives a small, economically declared linked-information set, its own causal price/volatility structure, and a causal regime state. The rejected dense 21-peer block is not reused.

## Objective

Determine whether a target-specific price/regime representation contains repeatable six-bar out-of-sample signal, and only then determine whether that signal survives an explicitly supplied round-trip cost model.

A statistical PASS is not a profitability claim. Economic promotion is impossible when no explicit cost assumption is supplied.

## Frozen targets

- EURUSD
- XAUUSD
- BTCUSD
- ETHUSD
- QQQ

Targets are evaluated independently. One target may pass while others fail. QQQ remains unavailable if the inherited `MIN_TRAIN_ROWS=5000` rule leaves fewer than four scored folds; the minimum is not lowered to make it available.

## Frozen linked sensor roles

The machine-readable manifest is `configs/v23_phase0c.json` and is the source of truth.

- EURUSD: UUP (USD), TLT (rates), HYG (risk/credit)
- XAUUSD: UUP (USD), TLT (rates), HYG (risk/credit)
- BTCUSD: ETHUSD (crypto), QQQ (equity risk), HYG (risk/credit)
- ETHUSD: BTCUSD (crypto), QQQ (equity risk), HYG (risk/credit)
- QQQ: TLT (rates), HYG (credit), XLP (defensive equity), UUP (USD)

The roles are declared from instrument meaning and the already-audited Phase 0B data universe, not from Phase 0B predictive contribution. GLD is deliberately not used as an XAUUSD peer in this first Phase 0C representation.

## Frozen target features

All target features are causal and computed at the decision bar using only data available at that timestamp:

- cumulative log-return bps: r1, r3, r6, r12, r24;
- realized volatility in bps: rv6, rv24, rv72;
- current bar log(high/low) range in bps;
- 24-close price z-score.

The current `MarketBar` schema contains OHLC only. Volume is therefore disabled in Phase 0C rather than synthesized or replaced with an unrelated proxy.

## Frozen linked-sensor features

Each linked sensor contributes only:

- r1;
- r6;
- r24.

A sensor packet must be available no more than one five-minute bar behind the target decision timestamp, pass the existing hard-eligibility rules, and have a contiguous causal window. There are no post-hoc availability fallbacks and no sensor ranking by observed target performance.

## Frozen regime features

- causal rv24 percentile against up to 120 prior valid rv24 observations;
- trend strength = abs(r24) / (rv24 + epsilon);
- largest absolute one-bar return over the most recent six bars divided by causal trailing sigma48;
- UTC time-of-day sine;
- UTC time-of-day cosine.

Regime variables are model inputs. Phase 0C does not split the data into separately trained low/mid/high-volatility models.

## Holdout isolation

G/H/I remain excluded exactly as reserved development holdouts:

- G: 2025-10-06 through 2025-10-24 UTC;
- H: 2026-02-02 through 2026-02-20 UTC;
- I: 2026-07-06 through 2026-07-24 UTC.

Phase 0C applies a stricter guard than simply dropping decision rows inside those windows: a development feature row is also rejected if its causal feature history would consume a G/H/I bar. A linked-sensor return packet is likewise rejected if its lookback touches a reserved window. Forward labels and executable return paths touching G/H/I are excluded.

This rule prevents indirect use of reserved prices after a holdout ends.

## Frozen horizon and chronological evaluation

- primary target horizon: six five-minute bars;
- statistical label: decision close to decision+6 close, log return in bps;
- executable diagnostic return: next-bar open to decision+6 close, log return in bps;
- inherited `MIN_TRAIN_ROWS=5000`;
- no random shuffle or random cross-validation;
- outer fold starts are loaded from each target's already-scored Phase 0B JSON `eval_start` timestamps rather than recomputed from the smaller Phase 0C sample;
- outer training rows must satisfy `label_end_timestamp < eval_start`.

This preserves the Phase 0B chronological boundaries despite Phase 0C having different feature availability.

## Frozen experiments

- C0: own features -> Ridge;
- C1: own + linked features -> Ridge;
- C2: own + linked + regime features -> Ridge;
- C3: same C2 features -> HistGradientBoostingRegressor.

C0 is the within-target baseline for all incremental R2 comparisons.

## Ridge selection

Ridge alpha candidates are frozen to:

- 0.1
- 1.0
- 10.0
- 100.0

For each outer fold and representation, alpha is selected only inside that outer training set using a chronological final-20% inner validation split with the six-bar completed-label purge. The outer evaluation fold is never used to choose alpha.

## Nonlinear model

C3 uses one frozen `HistGradientBoostingRegressor` configuration:

- learning_rate = 0.05
- max_iter = 200
- max_leaf_nodes = 15
- min_samples_leaf = 50
- l2_regularization = 1.0
- random_state = 0

No Optuna/grid search or post-result HGBR tuning is part of Phase 0C.

## Statistical metrics

For every scored target/fold/experiment:

- R2;
- MAE;
- RMSE;
- Spearman IC;
- Pearson correlation;
- sign accuracy;
- non-jump equivalents where available.

C1/C2/C3 also report delta R2 versus C0 on the exact same evaluation rows.

## Statistical candidate rule

Only C2 and C3 are promotion candidates. A candidate must satisfy all of:

1. pooled six-bar delta R2 versus C0 > 0;
2. pooled non-jump delta R2 versus C0 > 0;
3. at least four scored outer folds exist;
4. at least three of those folds have positive delta R2 versus C0.

If both C2 and C3 pass, C2 Ridge is preferred unless a separately preregistered later phase establishes a reason to prefer the more complex model.

## Frozen trade gate

Economic evaluation is optional at scoring time and only valid when the caller supplies a non-negative target-specific round-trip cost in bps.

Within each outer fold:

- confidence threshold = 75th percentile of absolute predictions on the outer training set;
- evaluation volatility percentile must be >= 0.60;
- absolute predicted return must be > 1.5 times the supplied round-trip cost;
- positive prediction -> LONG;
- negative prediction -> SHORT;
- otherwise -> NO_TRADE;
- only one position per target may be active at a time;
- entry = next bar open;
- exit = decision+6 close.

The confidence threshold and gate values are not tuned on the outer fold.

## Economic metrics and stress

When a cost is supplied, Phase 0C reports fold-level gross/net trade metrics, long/short splits, and cost stress at 1.0x, 1.5x, and 2.0x the supplied round-trip cost.

Economic promotion requires all of:

1. net expectancy > 0 bps/trade;
2. profit factor > 1.10;
3. unannualized per-trade Sharpe > 0.50;
4. 1.5x cost-stress aggregate net PnL > 0;
5. at least 30 actionable trades;
6. no single fold contributes more than 60% of positive aggregate net profit.

No compounding, leverage selection, Kelly sizing, or capital-growth simulation is part of Phase 0C.

## Target-level decisions

A target can have three distinct outcomes:

- `PROMOTED`: statistical and economic rules pass;
- `SIGNAL_CANDIDATE_REQUIRES_COST_MODEL`: statistical rules pass but no valid cost model was supplied;
- `FAIL`: no C2/C3 candidate passes the statistical rule, or economic rules fail when costs are evaluated.

Across targets, `PARTIAL_PASS` is valid because Phase 0C is explicitly asset-specific.

## Forbidden post-hoc actions

After the first Phase 0C target is scored, do not:

- inspect or tune on G/H/I;
- replace linked sensors using observed Phase 0B or Phase 0C performance;
- change the six-bar horizon;
- lower `MIN_TRAIN_ROWS` for QQQ;
- change the 75th-percentile confidence threshold to improve PnL;
- change the 60th-percentile volatility gate to improve PnL;
- search HGBR hyperparameters on outer-fold results;
- reinterpret missing costs as zero costs;
- reintroduce the rejected dense 21-peer representation and call it Phase 0C.

Any such change requires a newly named and preregistered phase.
