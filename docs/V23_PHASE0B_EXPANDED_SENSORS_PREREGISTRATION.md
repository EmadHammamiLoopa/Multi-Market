# V2.3 Phase 0B Preregistration — Expanded Sensor Universe

Date frozen: 2026-08-23

## Prior result

V2.3 Phase 0A (core five-market universe, four peers per target) is closed and rejected. Full-capacity rerun reproduced the original results exactly:

- pooled 6-bar Ridge incremental R2 = -0.00476183
- pooled 6-bar ElasticNet incremental R2 = -0.00471197
- pooled non-jump Ridge incremental R2 = -0.00478279
- pooled non-jump ElasticNet incremental R2 = -0.00473447
- positive targets = 0/5
- unavailable targets = 1/5 (QQQ lacked sufficient rows for a scored fold under frozen MIN_TRAIN_ROWS=5000)
- positive target-fold cells = 1/16 (6.25%)
- all four frozen promotion conditions failed

No individual target, lag, model, or fold from Phase 0A may be selected into Phase 0B based on those results.

## Research question

Does a broader, fixed cross-sectional sensor universe provide incremental causal information for the existing five targets beyond the Phase 0A base representation, when the additional peers are admitted as one complete block rather than selected individually?

## Frozen targets

- EURUSD
- XAUUSD
- BTCUSD
- ETHUSD
- QQQ

## Frozen additional sensor block

All of the following must be acquired and audited as one block before Phase 0B predictive scoring:

### Broad equity
- SPY
- IWM
- DIA

### Rates / credit
- TLT
- HYG
- LQD

### Commodities / dollar
- GLD
- SLV
- USO
- UUP

### U.S. sectors
- XLK
- XLF
- XLE
- XLI
- XLY
- XLP
- XLV

This is a 17-sensor block. No member may be removed because its eventual predictive contribution is weak or negative. A sensor may be declared structurally unavailable only before predictive scoring and only for objective data-integrity reasons documented by the acquisition audit.

## Data requirements before predictive scoring

For every sensor:

1. five-minute OHLC data;
2. timestamp-normalized UTC rows;
3. deterministic source file hash;
4. structural session eligibility using U.S. regular trading hours, DST-aware;
5. no duplicate timestamps;
6. explicit bar count, first timestamp, last timestamp;
7. zero-range / repeated-OHLC diagnostics;
8. overlap audit against each frozen target development pool;
9. G/H/I remain reserved and excluded exactly as in Phase 0A.

Predictive scoring is forbidden until the complete acquisition audit is frozen.

## Session policy

The 17 ETF sensors are U.S.-listed instruments and use the same conservative regular-session policy as QQQ:

09:30 <= America/New_York local time < 16:00, weekdays only, DST-aware.

Pre-market and after-hours bars are structurally ineligible for the Phase 0B primary audit.

## Phase 0B feature semantics

Phase 0B inherits the frozen Phase 0A causal feature construction:

- target lag ladder: 1, 2, 3, 4, 6, 9, 12 five-minute bars;
- peer one-bar lag ladder: 1, 2, 3, 4, 6, 9, 12;
- peer return normalization by causal trailing 48-return sigma;
- cross-peer breadth, mean, dispersion, and max-absolute normalized return summaries;
- intraday sine/cosine time encoding;
- weekday encoding;
- causal volatility percentile;
- jump-state robustness flag;
- six-bar (30-minute) forward return remains the primary horizon;
- one-bar horizon remains diagnostic only.

The existing four core peers plus all 17 expanded sensors form the CROSS information set for each target. The target itself is excluded from its own peer set.

## Models

Unchanged from Phase 0A:

- Ridge alpha = 10.0
- ElasticNet alpha = 0.0005
- ElasticNet l1_ratio = 0.25
- ElasticNet max_iter = 50000 numerical ceiling
- StandardScaler fit on training data only
- no hyperparameter search

GPU use is not part of Phase 0B because the frozen Phase 0 linear models use the CPU implementation. Full-capacity CPU execution may be used because it changes runtime only, not model semantics.

## Chronological evaluation

Unchanged from Phase 0A:

- reserved G/H/I windows excluded from development;
- five contiguous chronological folds;
- train only on rows strictly earlier than each fold;
- six-bar completed-label embargo;
- MIN_TRAIN_ROWS = 5000;
- no random shuffle or random CV.

## Promotion rule

Phase 0B uses the same four-condition promotion philosophy and is evaluated on the full expanded block, never sensor-by-sensor selection:

1. pooled 6-bar incremental OOS R2 > 0 for both Ridge and ElasticNet;
2. at least 3 of 5 targets have positive mean-model incremental 6-bar OOS R2;
3. at least 60% of scored chronological target-fold cells have positive mean-model incremental 6-bar OOS R2;
4. pooled non-jump incremental 6-bar OOS R2 >= 0 for both models.

If any condition fails, the expanded block is rejected as a V2.3 Phase 1 candidate.

## Forbidden post-hoc actions

After Phase 0B scoring begins, do not:

- remove weak sensors;
- keep only favorable asset classes;
- change the lag ladder;
- lower MIN_TRAIN_ROWS;
- switch the primary horizon;
- tune Ridge/ElasticNet hyperparameters;
- choose favorable targets or folds;
- reuse D/E/F or inspect G/H/I outcomes;
- introduce PnL thresholds or trading decisions.

## Interpretation

A PASS means the complete expanded sensor block contains incremental predictive information worth promoting to a separately preregistered nonlinear/economic Phase 1 experiment. It does not establish profitability.

A FAIL rejects this specific expanded sensor representation. It does not justify cherry-picking individual sensors from the block.
