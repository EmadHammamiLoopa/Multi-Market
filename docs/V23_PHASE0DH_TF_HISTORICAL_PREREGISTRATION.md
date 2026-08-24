# V2.3 Phase 0D-H-TF Historical Trade-Flow Audit

Frozen: 2026-08-24 before any predictive scoring.

## Purpose

Phase 0D-H-TF tests whether historical Binance USD-M Futures aggressive trade flow contains short-horizon predictive information for BTCUSDT and ETHUSDT. It is a separate experiment from the unavailable historical L1 audit and from the live full-L2 Phase 0D collector.

## Frozen targets and venue

- BTCUSDT
- ETHUSDT
- Binance USD-M perpetual futures only

## Frozen historical window

- Acquisition: 2026-05-26 through 2026-08-23 UTC inclusive (90 complete calendar days)
- Development/scoring: 2026-05-26 through 2026-08-03 UTC inclusive (70 days)
- Sealed historical holdout: 2026-08-04 through 2026-08-23 UTC inclusive (20 days)

The holdout remains sealed unless a development-window candidate passes the frozen gate.

## Frozen source

Public Binance USD-M Futures daily `aggTrades` archives only.

Raw ZIP archives must be retained and hashed. Missing or corrupt days are acquisition failures and may not be silently imputed.

## Causal one-second normalization

Aggregate trades are assigned to UTC one-second buckets using exchange trade time. Buyer-maker flag determines aggressor side only:

- `m=true` => aggressive sell
- `m=false` => aggressive buy

No future-price inference is permitted.

For each one-second bucket derive:

- aggressive buy quantity
- aggressive sell quantity
- aggressive buy count
- aggressive sell count
- quantity flow imbalance `(buy_qty-sell_qty)/(buy_qty+sell_qty)`
- count flow imbalance `(buy_count-sell_count)/(buy_count+sell_count)`
- VWAP of aggressive buys when present
- VWAP of aggressive sells when present
- last trade price known by bucket end
- total traded quantity
- total aggregate-trade count

Causal trailing features may use 1 s, 3 s, 5 s, and 10 s windows only.

## Frozen feature sets

T0 baseline:

- trailing last-trade log return 1 s
- trailing last-trade log return 3 s

T1 trade-flow linear:

- T0
- quantity flow imbalance 1 s
- count flow imbalance 1 s
- trailing quantity flow imbalance 3 s, 5 s, 10 s
- trailing count flow imbalance 3 s, 5 s, 10 s
- log1p total quantity 1 s and 5 s
- log1p aggregate-trade count 1 s and 5 s
- signed VWAP pressure: aggressive-buy VWAP minus aggressive-sell VWAP, expressed in bps versus last known trade price, zero only when one side is absent as explicitly encoded by side-presence indicators
- buy-side-present and sell-side-present indicators

T2 uses the same T1 features with HistGradientBoostingRegressor.

No order-book, bookTicker, candle, funding, open-interest, liquidation, cross-market, neural-network, CatBoost, XGBoost, or unrestricted feature family is allowed in this phase.

## Labels

Primary: future last-trade log return in bps from decision second to decision + 10 seconds using the last trade known at or before each endpoint.

Diagnostics only: 3 s and 30 s.

Rows without a causal endpoint price are unavailable; no backward use of future prices.

## Models

- T0: StandardScaler + Ridge
- T1: StandardScaler + Ridge
- T2: HistGradientBoostingRegressor

Ridge alpha grid: `{0.1, 1.0, 10.0, 100.0}`, selected only inside each outer training window using chronological inner validation.

T2 fixed configuration:

- learning_rate = 0.05
- max_iter = 200
- max_leaf_nodes = 15
- min_samples_leaf = 100
- l2_regularization = 1.0
- random_state = 0

No tuning after observing official scores.

## Development walk-forward

Five chronological outer folds are created from date boundaries inside the first 70 days. Training is strictly earlier than evaluation. Labels crossing fold boundaries are purged.

No random cross-validation.

## Statistical promotion gate

T1 or T2 must satisfy all of:

- pooled delta R2 versus T0 > 0
- pooled Spearman IC > 0
- at least 4 scored folds
- at least 3 outer folds with positive delta R2 versus T0
- pooled directional accuracy > 0.50

Prefer T1 if both T1 and T2 pass.

If neither passes, Phase 0D-H-TF is FAIL and the 20-day historical holdout remains sealed.

If a candidate passes, freeze that exact candidate before opening the 20-day historical holdout once.

## Economic evaluation

Forbidden in this phase before a statistical candidate exists. Historical trade-only data cannot reconstruct full executable spread/order-book cost, so any later economic audit must be separately specified and conservative.

## Interpretation

PASS means aggressive trade flow deserves continued prospective confirmation with live L1/L2 data. FAIL rejects this trade-flow-only representation, not full order-book microstructure.
