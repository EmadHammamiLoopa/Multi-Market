# V2.3 Phase 0D-M — L2-Informed Passive Execution / Adverse-Selection Control

Date frozen: 2026-08-25
Status: **PREREGISTERED BEFORE PROSPECTIVE DATA COLLECTION AND BEFORE ANY PHASE M SCORE**

## 1. Motivation

Phase 0D-L failed its preregistered development gate for both BTCUSDT and ETHUSDT. No L0/L1/L2 configuration survived inner validation at the frozen taker/taker economic gates, and the 2026-08-01 confirmation day remained sealed.

The Phase L diagnostic post-mortem is used only for hypothesis generation. It showed that the best post-hoc gross executable expectancy before additional costs was generally small (roughly sub-3 bps for dynamic candidates), with ETH showing the strongest diagnostic values. These post-hoc values are not promotion evidence and must not be reused as validation.

Phase M tests a materially different economic hypothesis:

> L2 order-flow information may be more useful for **passive execution and adverse-selection control** than for paying the spread and taker costs twice in short-horizon directional trading.

Phase M is not a rescue or retuning of Phase L. It uses newly collected prospective data and a new execution objective.

## 2. Scientific status of old data

The following data are **not allowed to promote Phase M**:

- Phase L development days 2026-01-01 through 2026-07-01.
- Phase L sealed confirmation day 2026-08-01.
- The older Phase J/K sealed holdout 2026-08-04 through 2026-08-23.

Jan-Jul may be used only for deterministic parser/converter unit tests, schema validation, and reproducibility checks that do not select or score Phase M policy/model parameters.

2026-08-01 remains analytically sealed.

The older 2026-08-04..2026-08-23 holdout remains analytically sealed.

## 3. Prospective-only timeline

Define `P1` as the first **full UTC day** after all of the following are true:

1. Phase M collector code is frozen by commit SHA.
2. Phase M configuration is frozen.
3. A continuous dry-run has passed feed-integrity checks.
4. No Phase M prospective score has yet been viewed.

Prospective periods:

- Calibration: P1-P7 (7 full UTC days).
- Sealed evaluation: P8-P21 (14 full UTC days).

P8-P21 must not be scored or inspected for policy performance until calibration selection is frozen.

If a full day has a material feed gap, continuity failure, or collector outage, it is marked unavailable. It is not replaced selectively after seeing performance. The prospective clock continues to the next calendar day, and the missing day remains documented.

## 4. Symbols

- ETHUSDT: primary Phase M symbol.
- BTCUSDT: independently scored secondary comparator.

ETH priority is a hypothesis-generation choice made from Phase L diagnostics and is therefore allowed only because all Phase M promotion evidence is prospective.

No additional symbols may be added after P1 begins.

## 5. Prospective market data

Collect Binance USD-M perpetual data causally with local receipt timestamps:

- diff depth stream at the fastest stable documented interval, with 100 ms as the intended primary feed interval;
- aggTrade stream;
- bookTicker retained only as an integrity/BBO cross-check;
- REST depth snapshot for initialization and any required resynchronization.

For every event retain both exchange timestamp and local receipt timestamp where available.

Local-book reconstruction must follow documented Binance sequence continuity. Any gap, discontinuity, or failed bridge invalidates the local book until a new snapshot and valid bridge are established.

No interpolation or forward filling across invalid feed intervals is allowed.

## 6. L2 / queue limitation

The feed is Market-By-Price (L2), not Market-By-Order (L3). Exact queue position is unobserved.

Therefore Phase M must never use `price touched -> filled` as its primary fill assumption.

Primary passive-fill simulation uses a **conservative queue-position model** in which queue advancement is driven only by executed trade quantity at the order price (Risk-Averse queue semantics).

A probability queue model may be reported only as a diagnostic sensitivity analysis. It cannot rescue or promote a strategy that fails under the conservative primary queue model.

The replayed market cannot react to our simulated orders, so strategy order size must remain small relative to displayed liquidity and capacity must be audited separately.

## 7. Latency

Primary feed/decision semantics remain causal by local receipt time.

Primary fixed order latency assumption for backtest until measured order-latency data exist:

- order-entry latency: 250 ms;
- cancel/response latency: 250 ms.

This is deliberately conservative and preserves continuity with the Phase L reaction-latency assumption.

If actual prospective order-latency measurements are collected later, they may be used only in a separately frozen latency-validation analysis. The primary Phase M promotion result remains judged under the frozen 250/250 ms assumption unless the measured-latency substitution is frozen before P1.

No forward-looking feed-latency model may be used for promotion.

## 8. Fees and execution costs

Before P1, record an authenticated account-specific Binance USD-M maker and taker commission snapshot for BTCUSDT and ETHUSDT when available.

Those rates are frozen for the Phase M primary economic score. Fee rates may not be changed after seeing prospective PnL.

Observed bid/ask prices and simulated passive fill prices define execution prices. No midpoint fills or price improvement are assumed.

Primary promotion also applies an additional fixed **1.0 bps round-trip execution safety charge** on top of exact frozen account commissions.

Stress score applies an additional **2.0 bps round-trip safety charge** on top of exact commissions.

If account-specific commission rates are unavailable before P1, Phase M may collect data but cannot produce a promotion result until a fee schedule is frozen prospectively.

## 9. Feature information set

Phase M retains the already frozen causal Phase L L2 information family rather than expanding feature mining:

### Static state

- spread_bps
- microprice_minus_mid_bps
- OBI L1/L5/L10
- log bid/ask depth L1/L5/L10

### Event flow

- best-level OFI 250 ms / 1 s / 3 s
- MLOFI top-5 250 ms / 1 s / 3 s
- MLOFI top-10 250 ms / 1 s / 3 s
- aggressive trade quantity imbalance 250 ms / 1 s / 3 s
- aggressive trade count imbalance 250 ms / 1 s / 3 s

### Resiliency

- OBI L1/L5/L10 changes 250 ms / 1 s
- spread change 250 ms / 1 s
- microprice displacement change 250 ms / 1 s
- bid/ask replenishment and depletion top-5 over 1 s
- the three previously frozen interactions

No new predictive feature family may be added after P1 begins.

## 10. Decision clock

Feature state is sampled every 250 ms from events available locally at or before that timestamp.

Quotes may be reconsidered every 250 ms, but an existing passive order is not automatically modified on every decision tick because modification loses queue priority.

Minimum passive quote lifetime before discretionary reprice/cancel: 1 second, except for:

- book invalidation;
- post-only safety violation / would-cross condition;
- inventory hard limit;
- explicit adverse-selection kill condition from the frozen policy.

## 11. Policies

Phase M compares fixed execution policies rather than arbitrary strategy search.

### M0 — symmetric passive baseline

At valid state, place one post-only bid at current best bid and one post-only ask at current best ask, subject to inventory limit.

No L2 predictive filtering.

Purpose: measure whether simple passive quoting is profitable or merely harvested by informed flow.

### M1 — L2 adverse-selection filter

Same quoting mechanism as M0, but an L2 Ridge score may suppress the side expected to experience adverse short-horizon markout.

Examples conceptually:

- sufficiently positive predicted markout: suppress/cancel ask-side passive exposure;
- sufficiently negative predicted markout: suppress/cancel bid-side passive exposure;
- neutral score: allow both sides subject to inventory control.

The exact threshold is selected only on P1-P7 from the frozen grid below and then frozen before P8 is opened.

### M2 — L2 skewed passive quoting

Diagnostic/secondary candidate. Uses the same Ridge score to skew the quoted reservation price by at most one tick while remaining post-only. It cannot promote unless M1 also demonstrates non-negative economics under the primary queue model; this prevents a complex quoting policy from becoming a standalone rescue path.

## 12. Model

Low-capacity only:

- StandardScaler fitted on calibration-training rows only.
- Ridge regression only.
- alphas: 0.1, 1, 10, 100.

Primary prediction target is short-horizon future mid-price markout in bps, because the signal is used to avoid adverse passive fills rather than to authorize taker/taker directional trades.

Frozen horizons:

- 3 seconds
- 10 seconds
- 30 seconds

The 1-second Phase L horizon is excluded from Phase M because the new mechanism requires order-entry latency plus queue waiting and is not treated as an immediate-taking strategy.

Threshold grid, derived from calibration-training absolute predictions only:

- 0.990
- 0.995
- 0.9975
- 0.999

No nonlinear model, neural network, Transformer, DeepLOB, XGBoost, CatBoost, or hyperparameter optimizer is allowed in Phase M.

## 13. Calibration structure P1-P7

Chronological calibration only:

- P1-P5: training.
- P6-P7: validation.

For each symbol and M1 candidate `(horizon, alpha, quantile)`:

1. fit StandardScaler + Ridge on P1-P5;
2. derive threshold from P1-P5 predictions only;
3. replay P6-P7 with the conservative queue model, fixed latency, exact frozen fees, and fixed safety charge;
4. require at least 20 completed passive round trips across P6-P7;
5. require net expectancy > 0 and total net PnL > 0 under primary cost;
6. require net expectancy > 0 under stress cost.

Select one M1 configuration per symbol lexicographically by:

1. highest net expectancy/trade under primary cost;
2. if within 1% relative tie, higher total net PnL;
3. higher profit factor;
4. lower max drawdown;
5. higher completed-fill count;
6. shorter horizon;
7. higher quantile;
8. lower alpha.

If no M1 configuration survives calibration, that symbol cannot open P8-P21 for promotion scoring; collection may continue but evaluation remains analytically sealed for strategy PnL.

M0 has no predictive parameter selection and is scored as baseline.

## 14. Inventory and order size

Phase M is a mechanism screen, not a leverage study.

Use a fixed small notional equivalent to the intended initial personal deployment scale. Default research notional is **USD 100 per passive order** unless exchange minimum/step rules require a higher valid rounded amount.

Maximum absolute inventory: two filled order units per symbol.

No leverage optimization, Kelly sizing, compounding, averaging down, martingale, or dynamic capital scaling is allowed.

Capacity audit must report order quantity as a fraction of displayed same-side L1 quantity at submission and at fill.

## 15. Primary fill/exchange model

Promotion model:

- Market-By-Price replay.
- Conservative Risk-Averse queue model.
- Post-only (GTX-equivalent) passive orders.
- NoPartialFill-style conservative fill accounting for the fixed small order unit.
- 250 ms entry latency.
- 250 ms response/cancel latency.

Probability queue models and partial-fill models are diagnostics only and cannot promote a primary failure.

## 16. P8-P21 sealed prospective promotion gate

Once P1-P7 calibration selection is frozen by commit SHA, score P8-P21 exactly once.

M1 passes only if all of the following hold under the conservative primary queue model and primary costs:

- at least 140 completed round trips across available evaluation days;
- at least 10 available evaluation days;
- net expectancy >= +0.50 bps per completed round trip;
- total net PnL > 0;
- profit factor >= 1.25;
- pooled PnL / max drawdown >= 2.0;
- at least 65% of active evaluation days have positive net PnL;
- no single evaluation day loses more than 25% of total positive-day PnL accumulated over the full evaluation;
- stress-cost net expectancy > 0;
- stress-cost total net PnL > 0.

Additionally, M1 must beat M0 on both:

- net expectancy per completed round trip;
- total net PnL.

This is the incremental adverse-selection-control gate.

M2 may be reported, but cannot rescue M1 failure.

## 17. Fill-quality diagnostics

Report, but do not use to weaken gates after scoring:

- passive fill rate;
- median and tail queue wait time;
- 1 s / 3 s / 10 s markout after fill;
- maker-side adverse-selection rate;
- cancellation rate;
- quote lifetime;
- inventory occupancy;
- fill count by UTC hour;
- displayed L1 capacity ratio;
- conservative vs probability-queue fill sensitivity;
- primary vs stress cost sensitivity.

## 18. Promotion meaning

Phase M PASS does **not** authorize meaningful capital deployment.

A pass permits only a separately frozen shadow/live-small-size validation stage that compares predicted simulated fills with actual exchange behavior and measures real order latency, cancellation latency, fill probability, fees, and slippage.

Real-money scaling is outside Phase M.

## 19. No-rescue rules

After P1 begins, do not:

- add symbols;
- add feature families;
- add horizons;
- add model classes;
- change alpha or quantile grids;
- change queue model used for promotion;
- treat touch as fill;
- reduce fixed latency assumptions after seeing PnL;
- change fees/safety charges after seeing PnL;
- change inventory/order size to improve historical/prospective PnL;
- weaken trade-count, stability, or drawdown gates;
- reopen Phase L confirmation day 2026-08-01;
- reopen the older 2026-08-04..2026-08-23 holdout;
- use P8-P21 to retune and rescore.

Any material change requires a separately named future phase based on a new hypothesis.

## 20. External implementation basis checked before freeze

Before this preregistration was frozen, current documentation was checked to confirm that:

- Binance documents USD-M Futures APIs as an official supported product family.
- hftbacktest supports L2 Market-By-Price replay, local/exchange timestamps, order latency models, post-only GTX orders, and queue-position models.
- its Risk-Averse queue model is explicitly conservative: queue position advances only with trades at the same price level, while probability queue models allow estimated advancement from depth decreases.

These implementation capabilities motivate the Phase M simulation design; they are not performance evidence.
