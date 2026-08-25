# V2.3 Phase 0D-L — L2 Order-Book Mechanism Screen

Date frozen: 2026-08-24
Status: PREREGISTERED BEFORE SCORING
Historical sealed holdout from Phase J/K: **DO NOT OPEN**

## 1. Purpose

Phase 0D-L tests a materially new information set after the historical `aggTrades + mark/index/premium` lineage failed with both linear Ridge and nonlinear XGBoost.

The new hypothesis is that short-horizon directional edge, if any, is carried primarily by event-driven order-book state and order-flow dynamics rather than by trade flow or mark/index/premium state alone.

This phase is a **mechanism screen**, not a final trading system. It asks whether full L2 microstructure contains stable out-of-sample directional information large enough to survive conservative taker execution assumptions.

## 2. Research basis

The frozen feature families are motivated by published market-microstructure results rather than post-hoc search:

- Order Flow Imbalance (OFI): price changes are strongly related to net order-book event flow rather than trade volume alone (Cont, Kukanov & Stoikov).
- Multi-Level OFI: information is distributed across multiple depth levels, not only best bid/ask (Xu, Gould & Howison).
- Microprice / queue imbalance: best-level size imbalance can shift the conditional fair price away from the mid (Stoikov and related queue-imbalance work).
- Stationary order-flow representations: order-flow features can be more robust than raw LOB levels for forecasting across changing price scales (Kolm, Turiel & Westray).

DeepLOB / Transformers are explicitly **not** part of Phase 0D-L. Model complexity is not escalated until the new information set first demonstrates stable economic value with a low-capacity model.

## 3. Targets

Frozen symbols:

- BTCUSDT Binance USD-M perpetual
- ETHUSDT Binance USD-M perpetual

No additional symbols may be added after scoring begins.

## 4. Historical data source and screening design

### 4.1 Immediate historical mechanism screen

Use Tardis downloadable Binance Futures historical data only for deterministic **free sample days: the first UTC day of each calendar month**. No paid data will be assumed for the initial mechanism test.

Frozen development sample days:

- 2026-01-01
- 2026-02-01
- 2026-03-01
- 2026-04-01
- 2026-05-01
- 2026-06-01
- 2026-07-01

Frozen untouched confirmation sample days:

- 2026-08-01

If any listed day is unavailable for either target, the day is marked unavailable for both symbols; it is not replaced after seeing results.

The existing Phase J/K sealed holdout `2026-08-04..2026-08-23` remains completely untouched and is not repurposed for Phase L.

### 4.2 Historical raw event types

Required Tardis data types per symbol/day:

- `incremental_book_L2`
- `trades`

Quotes are reconstructed from the L2 book and are not taken from a separate quote feed in the core mechanism screen.

Optional exchange/context feeds are excluded from the primary Phase L screen.

### 4.3 Prospective confirmation

Historical mechanism success does not authorize real trading. If and only if the historical Phase 0D-L development gate passes, a prospective live collector must run using Binance USD-M:

- diff depth @100ms
- aggTrade
- bookTicker retained only as an integrity cross-check
- REST depth snapshot for local-book initialization/reinitialization

The prospective dataset must be newly collected after the final historical configuration is frozen. Historical results may not be used to alter the prospective feature list/model/horizons/gates.

## 5. Order-book reconstruction

### 5.1 Historical Tardis reconstruction

Reconstruct Market-By-Price L2 state causally from `incremental_book_L2` records using exchange timestamp order; `local_timestamp` is retained for latency diagnostics but does not reorder exchange events.

A snapshot record resets the reconstructed book. Subsequent price-level updates set quantity to the supplied amount; zero amount removes the level.

At all times require:

- non-empty bid and ask books
- best bid < best ask
- monotone top levels by price
- finite positive prices
- non-negative quantities

Any integrity failure invalidates the interval until the next valid snapshot.

### 5.2 Prospective Binance reconstruction

For Binance diff depth, follow the official local-order-book snapshot + buffered-diff procedure and enforce `pu == previous u` continuity after bridging. Any gap causes invalidation and resnapshot; no interpolation or silent fill is allowed.

## 6. Decision clock and latency

Historical normalized state is sampled every **250 ms** from the latest complete reconstructed book and trade events available at or before the decision timestamp.

To prevent zero-latency execution assumptions, every signal is evaluated under a frozen **250 ms reaction latency**:

- signal state timestamp = t
- entry touch is taken from the reconstructed book at the first valid state at or after `t + 250 ms`

Latency diagnostics:

- 100 ms
- 500 ms

Only 250 ms is primary. Diagnostics cannot promote a failed primary result.

If no valid book exists at the delayed entry timestamp, that signal is unavailable.

## 7. Execution semantics

Phase 0D-L primary execution is **taker/taker only** to avoid unobservable queue-position assumptions in Market-By-Price data.

For a long trade:

- entry = best ask at delayed entry timestamp
- exit = best bid at the fixed horizon exit timestamp

For a short trade:

- entry = best bid at delayed entry timestamp
- exit = best ask at the fixed horizon exit timestamp

No midpoint fills, no optimistic maker fills, no queue-priority assumptions, and no price improvement.

Execution size is assumed small relative to top-of-book liquidity. A separate capacity diagnostic will report entry quantity / displayed L1 quantity but cannot improve promotion.

## 8. Horizons and overlap

Frozen holding horizons:

- 1 second
- 3 seconds
- 10 seconds
- 30 seconds

Fixed exit at H.

At most one open position per symbol. Same-symbol signals that occur while a position is open are ignored. BTC and ETH positions may overlap because they are different instruments.

## 9. Frozen feature blocks

All features are causal and computed only from events/state available at or before the signal timestamp.

### L0 — static book baseline

- spread_bps
- microprice_minus_mid_bps
- OBI L1
- OBI L5
- OBI L10
- log bid depth L1/L5/L10
- log ask depth L1/L5/L10

### L1 — event-flow / OFI

L0 plus:

- best-level OFI over trailing 250 ms, 1 s, 3 s
- multi-level OFI over top 5 levels over trailing 250 ms, 1 s, 3 s
- multi-level OFI over top 10 levels over trailing 250 ms, 1 s, 3 s
- aggressive trade quantity imbalance over trailing 250 ms, 1 s, 3 s
- aggressive trade count imbalance over trailing 250 ms, 1 s, 3 s

OFI is computed from signed changes in displayed bid/ask quantity and price-level creation/removal, not from trades alone.

### L2 — resiliency / interaction

L1 plus only:

- change in OBI L1 over 250 ms and 1 s
- change in OBI L5 over 250 ms and 1 s
- change in OBI L10 over 250 ms and 1 s
- change in spread_bps over 250 ms and 1 s
- change in microprice_minus_mid_bps over 250 ms and 1 s
- bid replenishment quantity top-5 over 1 s
- ask replenishment quantity top-5 over 1 s
- bid depletion quantity top-5 over 1 s
- ask depletion quantity top-5 over 1 s
- trade_imbalance_1s × OBI_L5
- trade_imbalance_1s × microprice_minus_mid_bps
- OFI_L5_1s × spread_bps

No other features may be added after development scoring begins.

## 10. Labels

Primary target is future **touch-to-touch executable gross return**, not mid-price return.

For a signal direction predicted at time t and delayed entry timestamp e=t+250ms:

Long gross bps at horizon H:

`10000 * log(best_bid(e+H) / best_ask(e))`

Short gross bps at horizon H:

`10000 * log(best_bid(e) / best_ask(e+H))`

Model regression target for direction learning is future mid-price log return from delayed entry state to H. Economic scoring always uses the executable touch-to-touch gross return above.

Rows lacking valid entry or exit book state are unavailable.

## 11. Models

Phase 0D-L uses only StandardScaler + Ridge.

Frozen alphas:

- 0.1
- 1
- 10
- 100

Blocks L0, L1, L2 are evaluated separately.

No XGBoost, CatBoost, HGBR, neural network, DeepLOB, CNN, LSTM, Transformer, or hyperparameter optimizer is allowed in Phase 0D-L.

If L1/L2 demonstrates stable economic value but Ridge underfits diagnostics, a separately named future nonlinear phase may be preregistered before scoring it.

## 12. Signal threshold selection

For each training window, absolute predicted-return thresholds are derived only from training predictions at quantiles:

- 0.990
- 0.995
- 0.9975
- 0.999

No absolute threshold is chosen using evaluation outcomes.

## 13. Transaction-cost model

Because primary execution already crosses the observed bid/ask at entry and exit, spread is embedded directly in touch-to-touch gross PnL.

Additional fee + adverse-slippage round-trip cost scenarios:

- 5 bps diagnostic
- 8 bps primary
- 12 bps stress

These costs are applied on top of observed touch execution.

The primary 8 bps is deliberately conservative for a personal taker strategy and can later be replaced by verified account-specific commission in prospective validation, but Phase 0D-L promotion is judged at the frozen 8 bps primary and 12 bps stress values.

## 14. Historical walk-forward folds

Development days are chronological independent day-folds:

1. train Jan-Feb; evaluate Mar 1
2. train Jan-Mar; evaluate Apr 1
3. train Jan-Apr; evaluate May 1
4. train Jan-May; evaluate Jun 1
5. train Jan-Jun; evaluate Jul 1

The first two sample days are warm-up/training only and are never outer evaluation days.

Within each outer training set, inner validation is the final available training day; model/threshold selection uses only data strictly earlier than the outer evaluation day.

Labels whose horizon crosses a train/validation/evaluation boundary are purged.

## 15. Inner selection

For each symbol, horizon, block, alpha and quantile:

A configuration survives inner validation only if, at primary 8 bps additional cost:

- expectancy > 0
- total net PnL > 0
- profit factor > 1
- at least 20 non-overlapping trades on the inner validation day

and at 12 bps stress:

- expectancy > 0
- total net PnL > 0

Among survivors select lexicographically:

1. highest net expectancy/trade @8
2. if within 1% relative tie, higher total net PnL @8
3. higher profit factor @8
4. lower max drawdown @8
5. simpler block preference L0 > L1 > L2 only after economic ties
6. shorter horizon
7. higher quantile
8. lower alpha

## 16. Development promotion gate

A symbol passes Phase 0D-L development only if the selected outer configurations satisfy all of:

- scored configuration exists in all 5 outer folds
- at least 4/5 outer folds positive net expectancy @8
- pooled net expectancy >= +1.0 bps/trade @8
- pooled total net PnL > 0 @8
- pooled profit factor >= 1.20 @8
- pooled net expectancy > 0 @12
- pooled total net PnL > 0 @12
- no outer fold expectancy < -2 bps/trade @8
- at least 100 pooled non-overlapping trades
- at least 55% positive active evaluation hours
- pooled PnL/maxDD >= 2

## 17. Incremental information gate

A winning L1 or L2 candidate must beat the independently selected L0 static-book baseline on both:

- pooled net expectancy/trade @8
- pooled total net PnL @8

If L0 alone passes but L1/L2 does not improve it, the conclusion is `STATIC_BOOK_SIGNAL_ONLY`; this is not evidence for dynamic OFI/resiliency value.

If no L1/L2 candidate passes structural + incremental gates, Phase 0D-L is a failure and no August confirmation day is opened.

## 18. Confirmation day

If and only if development passes for a symbol:

- freeze exact block/horizon/alpha/quantile
- freeze code commit
- freeze SHA-256 of all development raw files
- then score 2026-08-01 exactly once

Confirmation requires:

- net expectancy > 0 @8
- total net PnL > 0 @8
- profit factor > 1
- net expectancy > 0 @12
- at least 20 trades

Failure ends the historical L2 candidate. No configuration is changed and rescored on the confirmation day.

## 19. Prospective gate

Historical confirmation is still not enough for real money.

A historically confirmed configuration must be frozen and run prospectively on newly collected Binance data with the same features, 250 ms reaction latency and taker/taker execution assumptions.

Prospective observation must include enough independent trades to assess slippage and stability; real-money deployment is not authorized by this preregistration.

## 20. No-rescue rules

After the first usable Phase 0D-L development metrics are produced, do not:

- change feature definitions
- add/remove horizons
- change cost assumptions
- alter reaction latency
- add symbols
- change threshold grid
- change Ridge alpha grid
- weaken trade-count or promotion gates
- substitute maker execution
- open Aug 1 unless development passes
- open the older Phase J/K sealed holdout

Any such change requires a separately named new phase based on a new hypothesis.
