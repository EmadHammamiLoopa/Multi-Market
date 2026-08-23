# V2.3 Research Preregistration — Information Diffusion

Date frozen: 2026-08-23

## Status of prior versions

V2.2 is closed and rejected as a deployable model. Its D/E/F holdout is consumed evidence and MUST NOT be used for V2.3 feature, model, market, threshold, head, or hyperparameter selection.

The paired D/E/F comparison used identical 450 frozen timestamps. V2.1 produced 5 actionable trades, 1 winner, aggregate net -73.324225 bp, and expectancy -14.664845 bp/trade. V2.2 produced 3 actionable trades, 0 winners, aggregate net -119.880379 bp, and expectancy -39.960126 bp/trade. V2.2 is therefore classified as DEGRADED relative to frozen V2.1.

## Research question

Does a broader, strictly causal intraday information set based on cross-market information diffusion provide incremental predictive information beyond target-only history and the compact V2.1 peer packet, without relying on always-on monthly macro levels?

This is an information-set experiment first. Trading-policy changes are forbidden until the information audit is frozen.

## Evidence motivating the experiment

The design is motivated by published evidence, not by D/E/F mining.

1. Huddleston, Liu, and Stentoft, *Intraday Market Predictability: A Machine Learning Approach*, Journal of Financial Econometrics 21(2), 2023, DOI 10.1093/jjfinec/nbab007. The study uses five-minute returns and shows that lagged cross-sectional constituent returns contain predictive information beyond aggregate-market history and common trend/liquidity characteristics. Regularized linear and tree models both work, and forecast ensembles perform best across time. Publisher: https://academic.oup.com/jfec/article/21/2/485/6400345

2. Aleti, Bollerslev, and Siggaard, *Intraday Market Return Predictability Culled from the Factor Zoo*, Management Science 71(9), 2025, DOI 10.1287/mnsc.2023.01657. The study exploits lagged high-frequency cross-sectional factor returns, regularizes the large signal set, and explicitly separates continuous returns from discontinuous jumps. Publisher: https://pubsonline.informs.org/doi/10.1287/mnsc.2023.01657

3. Andersen, Bollerslev, Diebold, and Vega, *Micro Effects of Macro Announcements: Real-Time Price Discovery in Foreign Exchange*, American Economic Review 93(1), 2003. Announcement surprises, rather than stale monthly levels, produce high-frequency conditional-mean jumps; sign and timing matter. NBER: https://www.nber.org/papers/w8959

4. Andersen, Bollerslev, Diebold, and Vega, *Real-Time Price Discovery in Stock, Bond and Foreign Exchange Markets*, Journal of International Economics 73(2), 2007. High-frequency macro-news responses are linked across asset classes and are state dependent. NBER: https://www.nber.org/papers/w11312

The papers motivate hypotheses and architecture only. Their reported performance is not assumed to transfer to this dataset.

## Frozen conceptual changes from V2.2

### Remove from the candidate V2.3 information set

The 24 always-on V2.2 monthly macro-context features are not promoted to V2.3. They remain preserved as V2.2 evidence but are excluded from the new primary candidate block.

### Keep as the reference baseline

V2.1 remains the reference architecture:

- causal target price/history features;
- three regime features;
- compact causal peer packet;
- 15 minute maximum peer staleness;
- frozen V2 economic-event semantics for later paired testing.

### Candidate V2.3 information-diffusion block

For every available peer, V2.3 Phase 0 evaluates a dense lag ladder of one-bar peer returns at offsets:

`1, 2, 3, 4, 6, 9, 12` five-minute bars.

For each offset the value is the peer close-to-close return over exactly one five-minute bar ending at the lagged peer timestamp. No future peer bar may be used.

The candidate block also includes cross-peer summaries at each lag where at least two peers are available:

- breadth: fraction of available peer returns greater than zero;
- mean return;
- cross-peer standard deviation / dispersion;
- maximum absolute normalized peer return.

Peer returns are normalized by the peer's causal trailing 48-bar return standard deviation. The normalization window uses data at or before the lagged peer bar only.

Missing peer values are represented by an availability mask and zero-filled numeric value. A target row is never deleted solely because one peer is missing.

## Intraday state block

The candidate information audit includes deterministic time/state variables because published intraday predictability is strongly time dependent:

- UTC minute-of-day encoded as sine and cosine;
- weekday encoded as sine and cosine;
- target trailing 48-bar volatility percentile based only on prior target observations;
- target absolute one-bar return divided by trailing 48-bar volatility;
- jump-state indicator.

### Frozen jump-state rule

A bar is classified as a jump-like observation when:

`abs(one_bar_log_return_bps) > 4.0 * trailing_sigma_48_bps`

where `trailing_sigma_48_bps` is computed from the preceding 48 one-bar log returns and excludes the current return. The rule is a deterministic robustness flag; Phase 0 reports results for all rows and separately for non-jump rows. It does not delete holdout observations and is not tuned from D/E/F.

## Macro/event handling in V2.3

Always-on monthly macro levels are excluded from the primary candidate block.

Macro information may re-enter only as a separately preregistered event-gated block. Two permitted forms are:

1. standardized announcement surprise, but only if a historical point-in-time consensus source can be verified without guessed timestamps or reconstructed future information;
2. causal post-release market reaction, using price changes that have already occurred after a verified official release timestamp.

No event feature may be added during Phase 0 unless its exact construction and source audit are frozen in a separate amendment before evaluation.

## Phase 0 — Cross-sectional information audit

Phase 0 does NOT make LONG/SHORT/NO_TRADE decisions and does NOT optimize trading thresholds.

### Primary target

Six-bar forward close-to-close target return in basis points, preserving the existing V2 horizon of 30 minutes.

The label at target index `t` is:

`10000 * (close[t+6] / close[t] - 1)`

and is available only when all seven target bars `t..t+6` are hard-eligible and exactly five minutes apart.

### Secondary mechanism check

One-bar forward return is reported only as a mechanism diagnostic because the strongest published evidence is at very short horizons. It cannot be used to change the primary six-bar V2.3 horizon without a new preregistration.

### Models

Phase 0 uses only regularized linear models so that the incremental information content is measurable before introducing tree interactions:

- Ridge, alpha = 10.0;
- ElasticNet, alpha = 0.0005, l1_ratio = 0.25, max_iter = 10000.

Features are standardized using mean and standard deviation fit on the training segment only.

No hyperparameter search is permitted in Phase 0.

### Baseline versus candidate

For each target market and each chronological fold:

- BASE: target-only lag ladder + intraday state block;
- CROSS: BASE + dense causal peer lag block + cross-peer summaries.

The compact V2.1 peer packet is not included in BASE because Phase 0's question is whether a broad/dense cross-sectional block adds information to target history. A separate descriptive comparison against V2.1 can be reported later but is not the Phase 0 promotion criterion.

### Chronological evaluation

The development pool excludes every reserved G/H/I timestamp range and all rows whose forward label touches those ranges.

The remaining rows are sorted chronologically and divided into five contiguous evaluation folds. For fold `k`, the model trains only on rows strictly earlier than the fold and with the six-bar embargo satisfied. The first fold may be skipped if fewer than 5,000 training rows exist.

No random shuffle or random cross-validation is permitted.

### Metrics

For each model, target, fold, and horizon:

- out-of-sample R-squared versus zero-return prediction;
- mean squared error;
- Spearman correlation between prediction and realized return;
- Pearson correlation;
- sign accuracy, reported descriptively only.

Primary Phase 0 metric: pooled incremental six-bar OOS R-squared, `R2_CROSS - R2_BASE`, computed from pooled squared errors across all scored development rows for each target and model.

Secondary stability metrics:

- number of targets with positive incremental six-bar OOS R-squared;
- number of chronological folds with positive incremental OOS R-squared;
- non-jump incremental OOS R-squared.

No profit, PnL, take-profit, stop-loss, probability threshold, EV gate, or trade outcome is computed in Phase 0.

## Frozen Phase 0 promotion rule

The dense cross-sectional block is promoted into a full V2.3 model only if ALL conditions hold on the development pool:

1. pooled six-bar incremental OOS R-squared is positive for both Ridge and ElasticNet;
2. at least 3 of the 5 target markets have positive six-bar incremental OOS R-squared for the mean of Ridge and ElasticNet;
3. at least 60% of scored chronological target-fold cells have positive incremental six-bar OOS R-squared for the mean of Ridge and ElasticNet;
4. pooled non-jump incremental six-bar OOS R-squared is non-negative for both models.

If the rule fails, the block is rejected. Individual favorable assets, folds, lags, or models may not be cherry-picked into V2.3.

## Reserved untouched windows for the eventual V2.3 holdout

The following calendar ranges are reserved now and MUST NOT be used by Phase 0, model selection, feature selection, or threshold selection:

- G: 2025-10-06T00:00:00Z through 2025-10-24T23:59:59Z
- H: 2026-02-02T00:00:00Z through 2026-02-20T23:59:59Z
- I: 2026-07-06T00:00:00Z through 2026-07-24T23:59:59Z

Exact G/H/I timestamp manifests will be generated using structural eligibility only before any V2.3 holdout scoring. Their outcomes must not be inspected during Phase 0.

These windows are reserved as human-untouched evaluation evidence. Later chronological models may mechanically train on earlier historical observations as allowed by their point-in-time training rule; no human selection may use G/H/I outcomes.

## Expanded information universe

The five existing targets remain:

- EURUSD
- XAUUSD
- BTCUSD
- ETHUSD
- QQQ

Phase 0 code accepts any number of additional peer CSVs. The preferred expanded sensor universe, if five-minute point-in-time history can be acquired under the same integrity rules, is frozen as:

- SPY, IWM, DIA
- TLT, HYG, LQD
- GLD, SLV, USO, UUP
- XLK, XLF, XLE, XLI, XLY, XLP, XLV

Additional peers may be admitted only by a data-quality amendment frozen before running the expanded audit. No peer may be admitted or removed because of its predictive result.

The first Phase 0 run uses the currently available five-market universe and therefore four peers per target. This is a mechanism audit, not a claim that the broad-universe hypothesis has been fully tested. If the promotion rule passes with the core universe, the block qualifies for full V2.3 construction. If it fails, an expanded-universe Phase 0B may still be run only after the expanded universe is acquired and frozen as a complete block; core Phase 0 results cannot be used to select individual added peers.

## Full V2.3 model, only after Phase 0 promotion

If Phase 0 passes, the full model family is frozen conceptually as:

- Model A: regularized linear return model;
- Model B: XGBoost nonlinear model;
- Model C: fixed 50/50 standardized-score ensemble of A and B.

Training and policy details, including probability/EV translation and economic decision gates, require a separate Phase 1 preregistration before G/H/I scoring.

No deep neural network is included in V2.3 Phase 1 unless a later preregistered experiment justifies it independently.

## Leakage rules

1. Every feature timestamp must be at or before the decision timestamp.
2. Peer as-of selection must never choose a future peer bar.
3. All normalization statistics are fit using past data only.
4. Forward-label bars are never used in a feature.
5. Training rows whose label horizon intersects the decision-time embargo are excluded.
6. G/H/I outcomes are unavailable to Phase 0 by construction.
7. D/E/F are consumed evidence and may be reported as historical context only; they cannot determine V2.3 choices.

## Research interpretation rule

A positive Phase 0 result means only that the cross-sectional block contains incremental predictive information in consumed/development history. It does not imply profitability or deployability.

A negative Phase 0 result rejects this particular dense lead-lag representation for the available information universe. It does not prove that intraday information diffusion does not exist.
