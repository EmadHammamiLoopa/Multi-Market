# V2.3 Phase 0C Preregistration — Asset-Specific Regime Audit

Date frozen: 2026-08-24

## Status of prior evidence

V2.3 Phase 0A and Phase 0B are closed and rejected representations.

Phase 0A tested the five-target core peer universe and failed the frozen promotion rule. Phase 0B then tested the complete 17-sensor expanded block and also failed. Phase 0B therefore rejects the dense generic cross-sectional representation; it does not establish that every individual economically motivated relationship is useless.

No Phase 0A/0B target, fold, lag, sensor contribution, coefficient, or predictive score is used to choose Phase 0C sensors. The Phase 0C sensor sets below are fixed by economic role before Phase 0C scoring.

D/E/F are consumed evidence and MUST NOT influence Phase 0C features, markets, models, hyperparameters, thresholds, or promotion. G/H/I remain reserved and untouched exactly as in the parent V2.3 preregistration:

- G: 2025-10-06T00:00:00Z through 2025-10-24T23:59:59Z
- H: 2026-02-02T00:00:00Z through 2026-02-20T23:59:59Z
- I: 2026-07-06T00:00:00Z through 2026-07-24T23:59:59Z

Phase 0C is an information-signal audit only. It MUST NOT produce or optimize PnL, LONG/SHORT/NO_TRADE decisions, leverage, take-profit, stop-loss, cost gates, probability thresholds, or position sizing.

## Research question

Does a small, predeclared asset-specific information set, combined with strictly causal target-market regime state, provide stable incremental six-bar out-of-sample predictive information beyond target-only price structure?

A secondary question is whether a fixed nonlinear tree model extracts stable interactions from exactly the same information used by the regime-aware linear representation.

## Frozen targets

- EURUSD
- XAUUSD
- BTCUSD
- ETHUSD
- QQQ

No target may be added or removed after scoring begins. QQQ may remain unscored if the existing `MIN_TRAIN_ROWS=5000` rule leaves insufficient chronological training history. The minimum training requirement MUST NOT be lowered to make QQQ available.

## Frozen primary horizon

The primary label remains the existing V2.3 six-bar / 30-minute forward close-to-close return:

`10000 * (close[t+6] / close[t] - 1)`

The one-bar horizon remains diagnostic only and cannot be used for promotion or to change the primary horizon.

## Frozen asset-specific sensor sets

Only these exact peers are admitted for each target:

| Target | Frozen peers | Economic role |
|---|---|---|
| EURUSD | UUP, TLT, HYG | USD, rates/duration, credit/risk |
| XAUUSD | UUP, TLT, HYG | USD, rates/duration, credit/risk |
| BTCUSD | ETHUSD, QQQ, HYG | crypto linkage, equity risk, credit/risk |
| ETHUSD | BTCUSD, QQQ, HYG | crypto linkage, equity risk, credit/risk |
| QQQ | TLT, HYG, XLP, UUP | rates/duration, credit/risk, defensive equity, USD |

The Phase 0B acquisition audit already established structurally usable five-minute data for TLT, HYG, XLP, and UUP. UUP has materially more zero-range/repeated bars than most expanded sensors; those bars remain hard-ineligible under the existing data-quality rules. UUP is retained because USD exposure was selected by economic role before Phase 0C scoring, not because of any Phase 0B predictive result.

No fallback sensor is permitted after scoring begins. If a required peer file is absent or structurally invalid, that target is `UNAVAILABLE` rather than silently substituting another sensor.

## Frozen target-only feature block

All target-only features are computed from completed hard-eligible target bars at or before the decision timestamp.

### Multi-horizon target returns

Log returns in basis points over completed horizons:

- `r1`
- `r3`
- `r6`
- `r12`
- `r24`

For horizon `k`:

`r{k} = 10000 * log(close[t] / close[t-k])`

The entire `t-k .. t` span must be hard-eligible and exactly five minutes apart.

### Realized-volatility structure

- `rv6`
- `rv24`
- `rv72`

For window `k`, `rv{k}` is the root-mean-square of the `k` completed one-bar log returns ending at `t`, in basis points. The full window must be hard-eligible and contiguous.

### Current completed-bar range

`range_1 = 10000 * log(high[t] / low[t])`

### Trailing price z-score

`zprice_24` is the z-score of `log(close[t])` against the 24 completed log closes ending at `t`. The 24-bar window must be hard-eligible and contiguous. Rows with zero trailing standard deviation are ineligible.

No RSI, MACD, moving-average zoo, or additional technical indicator may be added during Phase 0C.

## Frozen peer feature block

For each target's exact frozen peer set, only three peer returns are used:

- `peer_r1`
- `peer_r6`
- `peer_r24`

At decision timestamp `t`, Phase 0C selects the latest hard-eligible completed peer bar with timestamp `<= t`. For horizon `k`:

`peer_r{k} = 10000 * log(peer_close[p] / peer_close[p-k])`

where `p` is the as-of peer index. The peer span must be hard-eligible and exactly five minutes apart.

Every peer return has an explicit availability mask. Unavailable numeric peer values are zero-filled only together with availability `0`; availability is `1` for a valid packet. Missing peer packets never delete an otherwise valid target row.

No cross-peer breadth, mean, dispersion, max-absolute summary, dense lag ladder, or sensor-by-sensor selection is used in Phase 0C.

## Frozen regime block

The regime block is target-only and strictly causal.

### Cyclic time state

- UTC minute-of-day sine
- UTC minute-of-day cosine
- weekday sine
- weekday cosine

### Causal volatility percentile

`volatility_percentile` is the percentile rank of current `rv24` relative to at most the preceding 288 valid target `rv24` observations, excluding the current observation from the reference distribution. If no prior valid reference exists, the deterministic fallback is `0.5`.

### Trend strength

`trend_strength = abs(r24) / max(rv24 * sqrt(24), 1e-12)`

### Recent jump ratio

`recent_jump_ratio` is the maximum absolute one-bar log return across the six completed target returns ending at `t`, divided by the existing causal trailing 48-return sigma that excludes the current return.

### Robustness jump flag

For comparability with Phase 0A/0B, non-jump robustness continues to use the existing deterministic current-bar rule:

`abs(current_one_bar_log_return_bps) > 4.0 * trailing_sigma_48_bps`

where trailing sigma uses the preceding 48 one-bar returns and excludes the current return. This flag is robustness metadata, not an optimized execution gate.

## Frozen representations and models

Phase 0C evaluates four paired representations on identical eligible target rows.

### C0 — OWN-RIDGE

Target-only feature block -> Ridge.

### C1 — OWN+LINKED-RIDGE

Target-only block + exact frozen peer block -> Ridge.

### C2 — OWN+LINKED+REGIME-RIDGE

Target-only block + exact frozen peer block + regime block -> Ridge.

### C3 — OWN+LINKED+REGIME-HGBR

Exactly the same features as C2 -> `HistGradientBoostingRegressor`.

No feature is added only for C3.

### Ridge

Ridge remains fixed at:

- `alpha = 10.0`
- `StandardScaler` fit on training rows only
- no alpha search or inner tuning

Keeping the prior V2.3 Ridge regularization avoids introducing a new tuning degree of freedom after observing Phase 0A/0B.

### HistGradientBoostingRegressor

Frozen configuration:

- `learning_rate = 0.05`
- `max_iter = 200`
- `max_leaf_nodes = 15`
- `min_samples_leaf = 50`
- `l2_regularization = 1.0`
- `early_stopping = False`
- `random_state = 0`

`early_stopping=False` is mandatory so the model does not create an implicit random internal validation split.

No XGBoost, LightGBM, CatBoost, neural network, LSTM, Transformer, Optuna, grid search, or alternative tree configuration is permitted in Phase 0C.

## Chronological evaluation

Phase 0C inherits the V2.3 chronological protocol:

- G/H/I and any forward label touching them are excluded from development;
- development rows are sorted chronologically;
- five contiguous evaluation folds using the existing V2.3 fold construction;
- train only on rows strictly earlier than the evaluation fold;
- completed six-bar label embargo: training rows require `label_end_timestamp < eval_start`;
- `MIN_TRAIN_ROWS = 5000`;
- no random shuffle or random cross-validation.

The first fold may therefore be skipped. A target with no scored fold is `UNAVAILABLE`, not a failed target.

## Metrics

For each scored target, fold, horizon, and representation/model:

- OOS R-squared;
- MSE;
- RMSE;
- MAE;
- Spearman correlation;
- Pearson correlation;
- sign accuracy, descriptive only.

All C0-C3 predictions are pooled over the same scored fold rows for each target.

Primary incremental comparisons at six bars:

- `C1_R2 - C0_R2`: value of the small asset-specific linked set;
- `C2_R2 - C0_R2`: value of linked set plus regime state;
- `C3_R2 - C0_R2`: value of the fixed nonlinear regime-aware candidate;
- `C3_R2 - C2_R2`: descriptive nonlinear added value using identical C2/C3 inputs.

The same C1/C2/C3 versus C0 comparisons are reported on non-jump observations.

## Frozen target-level signal promotion

Promotion is target-specific. Phase 0C no longer requires a universal model to work for all five markets.

A candidate representation C1, C2, or C3 is a `SIGNAL_PASS` for a target only if ALL conditions hold at the six-bar horizon:

1. pooled incremental R-squared versus C0 is `> 0`;
2. at least three scored chronological folds have positive incremental R-squared versus C0;
3. pooled non-jump incremental R-squared versus C0 is `> 0`.

A target must have at least three scored folds to be eligible for `SIGNAL_PASS`.

If more than one representation passes for a target, Phase 0C does not optimize a trading rule. The simplest passing representation is preferred for later preregistration unless a more complex representation has both higher pooled and higher non-jump incremental R-squared and no fewer positive folds. This complexity rule is deterministic and is applied only among passing representations.

Possible overall outcomes:

- `PHASE0C_PROMOTION=PASS`: every structurally available target has at least one passing representation;
- `PHASE0C_PROMOTION=PARTIAL_PASS`: at least one but not every structurally available target has a passing representation;
- `PHASE0C_PROMOTION=FAIL`: no structurally available target has a passing representation.

A partial pass is scientifically valid because Phase 0C explicitly tests asset-specific rather than universal structure.

## What Phase 0C may conclude

A signal pass means only that the representation shows repeatable incremental development-set predictive information under the frozen chronology. It does not establish profitability or deployability.

Only targets/representations that pass Phase 0C may enter a separately preregistered economic stage. That later stage may introduce execution timing, transaction costs, confidence/expected-value gates, LONG/SHORT/NO_TRADE, and risk controls. Those choices MUST NOT be tuned using G/H/I.

## Forbidden post-hoc actions

After any Phase 0C scoring begins, do not:

- replace or remove a frozen peer because its result is weak;
- select peers from Phase 0A/0B coefficients or scores;
- add GLD to XAUUSD or another apparently favorable same-asset proxy after seeing results;
- change own-feature horizons;
- change peer horizons;
- add technical indicators;
- change Ridge alpha;
- tune HGBR parameters;
- lower `MIN_TRAIN_ROWS`;
- change the primary six-bar horizon;
- choose favorable folds;
- use D/E/F to select anything;
- inspect G/H/I outcomes;
- introduce PnL or trading thresholds into Phase 0C.

Any such change requires a newly named, separately preregistered experiment and may not overwrite Phase 0C evidence.
