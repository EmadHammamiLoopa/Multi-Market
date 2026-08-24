# V2.3 Phase 0D-I Longer-Horizon Trade-Flow Audit

Frozen: 2026-08-24 before any Phase 0D-I scoring.

## Purpose

Phase 0D-H established that short-horizon trade-flow-only features carry reproducible statistical information but insufficient gross edge to survive realistic execution costs over 10-300 second holding periods. Phase 0D-I tests a materially different hypothesis: whether persistent trade-flow state predicts sufficiently larger moves over longer intraday horizons to create positive cost-adjusted expectancy.

This phase does not rewrite any Phase 0D-H result. The 2026-08-04 through 2026-08-23 historical holdout remains sealed and is not used in Phase 0D-I development.

## Markets and source

- BTCUSDT Binance USD-M perpetual futures
- ETHUSDT Binance USD-M perpetual futures
- public historical `aggTrades` only
- development window: 2026-05-26 through 2026-08-03 UTC
- no order book, bookTicker, candles, funding, OI, liquidations, cross-market data or prospective live data

## Objective hierarchy

1. Profitability after conservative costs
2. Stability through time
3. Number of independent opportunities
4. Raw directional accuracy

## Frozen longer-horizon feature family

Retain all prior causal T1 features and add only longer causal trade-flow persistence summaries:

- existing T1 features: ret1, ret3, qfi/cfi 1/3/5/10 s, log quantity/count 1/5 s, VWAP pressure, buy/sell presence
- quantity-flow imbalance over 30 s, 60 s, 120 s, 300 s
- count-flow imbalance over 30 s, 60 s, 120 s, 300 s
- log1p total quantity over 30 s, 60 s, 120 s, 300 s
- log1p aggregate-trade count over 30 s, 60 s, 120 s, 300 s
- last-trade log return over 30 s, 60 s, 120 s, 300 s

All windows are causal and right-closed at the decision second. The first 300 seconds of the development dataset are excluded because a complete 300-second trailing history is unavailable at the acquisition boundary. No future observation may be used to fill that history.

The longer flow features are derived deterministically from the already-frozen one-second development dataset. Per-second total quantity/count are recovered from `log_qty1`/`log_count1`; buy/sell components are recovered algebraically from total and the corresponding one-second imbalance. This is a representation-preserving derivation and does not add a new information source.

## Candidate holding horizons

- 600 seconds (10 min)
- 1800 seconds (30 min)
- 3600 seconds (60 min)

No other horizon may be introduced after scoring begins.

## Model

Ridge only:

- StandardScaler + Ridge
- alpha grid `{0.1, 1.0, 10.0, 100.0}`
- no HGBR/XGBoost/CatBoost/neural network/Optuna

## Target

For horizon H, gross return is:

`r_H(t) = 10000 * log(P(t+H)/P(t))`

The model predicts signed future return in bps directly. Trade eligibility is determined by extreme absolute model predictions rather than by predicting every second.

## Signal gate candidates

Absolute prediction quantiles, estimated using past training predictions only:

- 0.990
- 0.995
- 0.9975
- 0.999
- 0.9995

Trade direction is the sign of the prediction.

## Independent opportunity execution

For each symbol:

- one open trade maximum;
- entry on first eligible flat-state signal;
- fixed exit exactly H seconds later using last trade price known at or before exit time;
- signals while open are ignored;
- next entry earliest one second after exit;
- no leverage, pyramiding, TP/SL search, position-size optimization or compounding.

## Cost grid

Round-trip costs:

- 5 bps diagnostic
- 8 bps lower-cost diagnostic
- 10 bps diagnostic
- 12 bps PRIMARY
- 15 bps stress

Promotion requires positive results at 12 bps and 15 bps. Lower-cost results are diagnostics only.

## Development folds

Retain the five frozen chronological evaluation folds:

1. 2026-06-15 through 2026-06-24
2. 2026-06-25 through 2026-07-04
3. 2026-07-05 through 2026-07-14
4. 2026-07-15 through 2026-07-24
5. 2026-07-25 through 2026-08-03

For each fold, training is strictly earlier than evaluation and labels crossing the boundary are purged by H.

## Nested selection

Inside each outer fold, use the final 20% of eligible outer-training history as chronological inner validation. Search only the frozen `(H, alpha, gate_quantile)` grid. Select lexicographically:

1. higher median net bps/day at 12 bps cost;
2. higher worst 5-day rolling net PnL;
3. higher median independent trades/day;
4. lower max drawdown;
5. shorter horizon;
6. higher gate quantile.

A configuration survives inner selection only if:

- net expectancy > 0 at 12 bps;
- total net PnL > 0 at 12 bps;
- profit factor > 1.0 at 12 bps;
- net expectancy > 0 at 15 bps;
- total net PnL > 0 at 15 bps;
- median independent trades/day on active days >= 2.

## Development promotion gate per symbol

All must hold:

- 4/5 or more outer folds positive total net PnL at 12 bps;
- pooled net expectancy >= +1.0 bps/trade at 12 bps;
- pooled total net PnL > 0 at 12 bps;
- pooled net expectancy > 0 and total net PnL > 0 at 15 bps;
- pooled profit factor >= 1.15 at 12 bps;
- >= 55% positive active trading days;
- pooled PnL/max-drawdown >= 2.0;
- no outer fold expectancy below -2 bps/trade at 12 bps;
- median independent trades/day on active days >= 2.

The lower opportunity-count requirement reflects longer holding periods. Profitability and stability remain higher priorities than trade count.

## Holdout rule

Phase 0D-I does not open the previous historical holdout during development. If a symbol passes, one final configuration must be frozen and hashed before any separate confirmation phase is proposed. The untouched 2026-08-04 through 2026-08-23 data may be opened only once under a separately recorded confirmation step.

## Interpretation

PASS means longer-horizon persistent trade flow produces an economically meaningful candidate deserving untouched confirmation and later live L2 validation. FAIL means trade-flow-only information remains economically insufficient even when allowed 10-60 minutes to realize a larger move; the next research direction should then rely on richer information rather than further retuning the same feature family.
