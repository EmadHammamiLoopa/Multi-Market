# V2.3 Phase 0D-J Futures-State + Trade-Flow Preregistration

Frozen: 2026-08-24 before any Phase 0D-J scoring.

## Purpose

Phase 0D-H and Phase 0D-I rejected the `aggTrades`-only lineage for economic deployment. Phase 0D-J introduces a materially new information source: Binance USD-M futures-state price series (mark price, index price, premium index) combined with the already-frozen causal trade-flow features.

This phase does not rewrite any previous result. The historical holdout 2026-08-04 through 2026-08-23 remains sealed during development.

## Markets and development window

- BTCUSDT
- ETHUSDT
- development only: 2026-05-26 through 2026-08-03 UTC
- sealed confirmation window: 2026-08-04 through 2026-08-23 UTC

## New historical sources

Binance USD-M Futures public archives, 1-minute interval:

- `markPriceKlines`
- `indexPriceKlines`
- `premiumIndexKlines`

Acquisition must download the official ZIP and `.CHECKSUM` sidecar for every required symbol/day/stream. Missing archives, checksum mismatch, corrupt ZIPs, duplicate minute keys, or unresolved continuity defects are acquisition failures for official scoring.

Known public-data continuity concerns require explicit validation; no silent forward/back fill of missing state candles is allowed.

## Decision clock and causality

The model operates on a 1-minute decision grid.

For minute M, the decision timestamp is immediately after the complete 1-minute state candles for M are known. Only state candles whose close time is at or before the decision time may be used. Trade-flow features are sampled causally from the frozen one-second development grid at the final second available at or before that decision timestamp.

Rows with incomplete state-source alignment are unavailable; no future backfill.

### Frozen target and execution-price semantics

The predictive target and simulated entry/exit prices use the frozen `aggTrades`-derived one-second last-trade price grid, not mark/index/premium prices.

For minute M, the entry decision row uses the last-trade price at the final second of M (UTC second `:59`), after the completed state candles for M are available. For holding horizon H minutes, the gross signed future return is based on the last-trade price exactly `60*H` seconds later on the same contiguous frozen grid:

`r_H(t) = 10000 * log(P_trade(t + 60H) / P_trade(t))`

The state series are predictors only. They are never substituted as fill/execution prices. Round-trip cost deductions below are applied to the resulting trade-price return. Labels crossing an inner/outer boundary are purged by the corresponding `60*H` seconds.

## Feature blocks

### F0 — frozen trade-flow reference block

Use the prior T1 trade-flow family, sampled causally on the minute grid:

- ret1, ret3
- qfi/cfi 1s, 3s, 5s, 10s
- log total quantity 1s/5s
- log aggregate-trade count 1s/5s
- signed VWAP pressure
- buy/sell presence indicators

### F1 — futures-state block

Using completed 1-minute state candles only:

- mark/index basis close in bps: `(mark_close/index_close - 1) * 10000`
- premium close
- mark return 1m, 5m, 15m
- index return 1m, 5m, 15m
- basis change 1m, 5m, 15m
- premium change 1m, 5m, 15m
- basis rolling z-score over the 60 completed minutes immediately preceding the current minute
- premium rolling z-score over the 60 completed minutes immediately preceding the current minute
- mark-minus-index return differential over 1m, 5m, 15m

For `basis_z60` and `premium_z60`, the current minute is excluded from the rolling mean and standard deviation. At minute t, compute mean/std from minutes `t-60 ... t-1`, then score the current value t against that prior-only distribution. Population standard deviation (`ddof=0`) is frozen. Rows lacking all 60 prior completed minutes are unavailable. If the prior-window standard deviation is zero, the z-score is defined as 0.0.

### F2 — combined block

F0 + F1 plus only these pre-frozen interactions:

- qfi10 * basis_z60
- cfi10 * basis_z60
- qfi10 * premium_z60
- cfi10 * premium_z60
- vwap_pressure_bps * basis_z60

No other feature search is allowed after scoring begins.

## Candidate horizons

Primary candidate holding horizons:

- 5 minutes
- 10 minutes
- 30 minutes

Fixed exit exactly H minutes after entry. One open trade maximum per symbol; no overlapping same-symbol positions.

## Models

Compare incrementally:

- J0: F0 StandardScaler + Ridge
- J1: F1 StandardScaler + Ridge
- J2: F2 StandardScaler + Ridge

Frozen Ridge alpha grid: `{0.1, 1, 10, 100}`.

No HGBR/XGBoost/CatBoost/neural network in this phase. GPU models are reserved for a later phase only if the new information source first demonstrates robust incremental value.

## Signal gates

Absolute prediction quantiles from training predictions only:

- 0.990
- 0.995
- 0.9975
- 0.999

Direction = sign(prediction). Signals while a same-symbol position is open are ignored.

## Cost grid

Round-trip costs:

- 5 bps diagnostic
- 8 bps diagnostic
- 10 bps diagnostic
- 12 bps PRIMARY
- 15 bps stress

No promotion unless positive after 12 bps and still positive after 15 bps.

## Folds

Retain the five chronological outer folds:

1. 2026-06-15 through 2026-06-24
2. 2026-06-25 through 2026-07-04
3. 2026-07-05 through 2026-07-14
4. 2026-07-15 through 2026-07-24
5. 2026-07-25 through 2026-08-03

Training is strictly earlier than evaluation. Labels reaching/crossing evaluation start are purged by H. Inner chronological validation uses the final 20% of eligible outer-training rows, with the same horizon purge at the inner boundary.

## Selection objective

Within each outer fold, J0 is selected independently as the trade-flow reference using the same frozen `(H, alpha, gate_quantile)` grid. The Phase 0D-J candidate search considers only J1/J2 configurations, again over the frozen `(feature_block, H, alpha, gate_quantile)` grid, and selects lexicographically:

1. higher median net bps/day at 12 bps cost
2. higher worst 5-day rolling net PnL
3. higher median independent trades/day
4. lower maximum drawdown
5. simpler feature block (J1 before J2 only when performance is effectively tied)
6. shorter horizon
7. higher gate quantile

“Effectively tied” for the first objective is frozen as a <=1% relative difference in `median net bps/day` at 12 bps, using `max(abs(a), abs(b), 1e-12)` as the denominator. If the relative difference exceeds 1%, the larger median net bps/day wins before simplicity is considered.

J0 can never itself cause Phase 0D-J promotion; it is retained only for the frozen incremental-information comparison.

## Inner survival filters

At 12 bps:

- net expectancy > 0
- total net PnL > 0
- profit factor > 1.0
- median independent trades/day on active days >= 2

At 15 bps:

- net expectancy > 0
- total net PnL > 0

## Development promotion gate

A symbol becomes a candidate only if all hold:

- 4/5 or more outer folds positive at 12 bps
- pooled net expectancy >= +1.0 bps/trade at 12 bps
- pooled total net PnL > 0 at 12 bps
- pooled expectancy and total PnL > 0 at 15 bps
- pooled profit factor >= 1.15 at 12 bps
- >=55% positive active trading days
- pooled PnL/max-drawdown >= 2.0
- no scored outer fold expectancy below -2 bps/trade at 12 bps
- median independent trades/day on active days >= 2

Incremental-information requirement:

- the selected J1/J2 candidate must outperform J0 pooled 12-bps expectancy and pooled 12-bps total PnL; a trade-flow-only winner is not a Phase 0D-J success.

## Holdout rule

The 2026-08-04 through 2026-08-23 historical holdout remains sealed during development. If a symbol passes, freeze one final configuration plus code/data hashes before any one-shot confirmation step. No redesign after holdout metrics are observed.
