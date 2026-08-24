# V2.3 Phase 0D-L — Feature Semantics Freeze

Date frozen: 2026-08-24
Status: **FROZEN BEFORE ANY PREDICTIVE SCORING**

This document resolves implementation details left implicit in the original Phase 0D-L preregistration. It changes no symbols, sample days, horizons, model family, alpha grid, signal quantiles, latency, costs, execution semantics, folds, promotion gates, or confirmation rules.

## 1. Causal clock

All historical feature construction uses the previously frozen causal `local_timestamp` ordering. Equal-`local_timestamp` rows form one atomic message group. A group is fully applied before any resulting book state or event-flow contribution is exposed to the 250 ms sampler.

## 2. 250 ms bins

The feature grid is the exact UTC day grid at 250 ms spacing: 345,600 rows/day/symbol.

For a grid timestamp `t`, event-flow features for the 250 ms bin are the sum of complete atomic groups with local timestamps in `(t-250ms, t]`.

The first grid row of a day has zero prior-bin flow unless a complete group occurs exactly at the day-start timestamp.

## 3. Best-level OFI

For each complete L2 atomic group, capture top-of-book before and after the group. Best-level OFI uses the standard price/size event rule.

Bid contribution:

- if `bid_price_after > bid_price_before`: `+bid_qty_after`
- if equal: `bid_qty_after - bid_qty_before`
- if `bid_price_after < bid_price_before`: `-bid_qty_before`

Ask contribution:

- if `ask_price_after < ask_price_before`: `-ask_qty_after`
- if equal: `ask_qty_before - ask_qty_after`
- if `ask_price_after > ask_price_before`: `+ask_qty_before`

`OFI_L1_group = bid_contribution + ask_contribution`.

Groups for which either the pre-group or post-group book is not valid do not contribute OFI and instead reset the rolling flow continuity until a valid snapshot has re-established the book.

## 4. Multi-level OFI

For levels `m = 1..N`, use the same price/size rule independently on rank-m bid and ask before/after the complete atomic group. Missing rank-m levels contribute zero for that side/level only when both books are otherwise valid.

`MLOFI_L5_group = sum(level_OFI_m, m=1..5)`

`MLOFI_L10_group = sum(level_OFI_m, m=1..10)`

No depth normalization is applied inside the feature definition. StandardScaler in the frozen Ridge pipeline handles scale using training data only.

The 250 ms features are sums of group contributions in the bin. The 1 s and 3 s features are rolling sums over the latest 4 and 12 complete 250 ms bins respectively.

## 5. Top-5 replenishment and depletion

For each complete atomic group, collect the union of the exact price levels present in top-5 before and top-5 after the group, separately for bid and ask.

For each price in this union, compare displayed quantity before vs after at the same exact price:

- positive quantity delta contributes to replenishment;
- negative quantity delta contributes its absolute value to depletion.

This avoids treating pure rank migration as replenishment/depletion while still capturing creation, cancellation, refill, and removal at near-touch prices.

Frozen group quantities:

- `bid_replenish_l5_group`
- `ask_replenish_l5_group`
- `bid_deplete_l5_group`
- `ask_deplete_l5_group`

The Phase L L2 features use 1 s rolling sums of these quantities over four 250 ms bins.

## 6. Trade-flow bins

Tardis `trades` are assigned by causal `local_timestamp` to the same 250 ms grid.

For each bin accumulate:

- buy quantity
- sell quantity
- buy count
- sell count

`side=unknown` contributes to neither directional quantity nor directional count and is tracked separately for audit only.

For each window W in {250 ms, 1 s, 3 s}:

`trade_qty_imbalance_W = (buy_qty_W - sell_qty_W) / (buy_qty_W + sell_qty_W)` when denominator > 0, else 0.

`trade_count_imbalance_W = (buy_count_W - sell_count_W) / (buy_count_W + sell_count_W)` when denominator > 0, else 0.

For 1 s and 3 s, buy/sell quantities and counts are summed first, then the imbalance is computed. Per-bin imbalance values are never averaged.

## 7. Static block transforms

The frozen L0 log-depth features use natural `log1p` of displayed quantity:

- `log1p(bid_qty_l1)`
- `log1p(ask_qty_l1)`
- `log1p(bid_depth_l5)`
- `log1p(ask_depth_l5)`
- `log1p(bid_depth_l10)`
- `log1p(ask_depth_l10)`

## 8. L2 change features

For any state quantity `x`:

- 250 ms change = `x_t - x_{t-250ms}`
- 1 s change = `x_t - x_{t-1s}`

Applied only to:

- OBI L1/L5/L10
- spread_bps
- microprice_minus_mid_bps

Rows lacking a valid current state or the required valid lag state are feature-invalid for configurations requiring that feature.

## 9. Frozen interaction definitions

The three L2 interactions are exactly:

- `trade_qty_imbalance_1s * obi_l5`
- `trade_qty_imbalance_1s * microprice_minus_mid_bps`
- `mlofi_l5_1s * spread_bps`

The original prose label `OFI_L5_1s × spread_bps` is implemented as the preregistered multi-level L5 OFI quantity (`MLOFI_L5_1s`).

## 10. Feature validity

No forward fill is permitted across invalid-book intervals.

A final feature row is valid only if:

- the current BOOK250 row has `book_valid=1`;
- all lag states needed by its block are valid;
- all required rolling event-flow bins are causally continuous since the latest valid snapshot/book reinitialization;
- all numeric feature values are finite.

Invalid rows remain present in prepared files with a validity flag but may not enter model training, threshold fitting, selection, labels, or economic scoring.

## 11. No predictive use yet

This semantics freeze was created after BOOK250 integrity passed but before any Phase 0D-L labels, Ridge fits, predictions, fold metrics, trades, or PnL were produced.
