# V2.3 Phase 0D-H-OPPORTUNITY Preregistration

Frozen: 2026-08-24 before any Phase 0D-H-OPPORTUNITY scoring or holdout access.

## Objective hierarchy

The phase is explicitly optimized in this order:

1. Profitability after conservative trading costs
2. Stability across time and days
3. Number of independent tradable opportunities
4. Raw directional accuracy (reporting only; never a promotion gate)

The goal is not to emit a prediction every second. The goal is to identify the largest number of non-overlapping opportunities that retain positive net expectancy after costs and acceptable drawdown.

## Relationship to prior phase

Phase 0D-H-TF is closed as `FAIL_KEEP_HOLDOUT_SEALED` under its frozen directional-accuracy gate. Its development result showed reproducible incremental rank/magnitude information, especially T1 Ridge with positive delta R2 in 5/5 development folds for both BTCUSDT and ETHUSDT. This new phase is a separate post-result hypothesis and may not rewrite the prior decision.

The historical holdout 2026-08-04 through 2026-08-23 remains sealed until this phase passes its development gate and a final configuration is frozen.

## Frozen markets and data

- BTCUSDT Binance USD-M perpetual futures
- ETHUSDT Binance USD-M perpetual futures
- public historical `aggTrades` only
- development data: 2026-05-26 through 2026-08-03 UTC
- sealed historical holdout: 2026-08-04 through 2026-08-23 UTC
- no bookTicker, order-book, candles, funding, open interest, liquidation, cross-market, or live-smoke data in this phase

## Frozen feature family

Use exactly the prior T1 trade-flow feature family, with no additions or deletions:

- trailing last-trade log return 1 s
- trailing last-trade log return 3 s
- quantity-flow imbalance 1 s
- count-flow imbalance 1 s
- trailing quantity-flow imbalance 3 s, 5 s, 10 s
- trailing count-flow imbalance 3 s, 5 s, 10 s
- log1p total quantity 1 s and 5 s
- log1p aggregate-trade count 1 s and 5 s
- signed aggressive-buy VWAP minus aggressive-sell VWAP in bps versus last known trade price, zero when one side is absent and accompanied by presence indicators
- buy-side-present indicator
- sell-side-present indicator

All features are causal and known at or before the decision second.

## Cost-adjusted opportunity target

For a horizon H, define gross future return in bps:

`r_H(t) = 10000 * log(P(t+H) / P(t))`

where P is the last trade price known at or before each endpoint.

The training target is a signed cost-excess opportunity score using a frozen 12 bps round-trip screening cost:

`y_H(t) = sign(r_H(t)) * max(abs(r_H(t)) - 12, 0)`

This makes sub-cost moves explicitly non-opportunities while preserving direction and excess magnitude for moves large enough to cover the base screening cost.

The 12 bps value is not claimed to be the user's exact Binance fee. Binance Futures fees depend on maker/taker role, VIP tier and discounts. This phase therefore evaluates a conservative cost stress grid rather than relying on one optimistic account-specific fee.

## Frozen evaluation cost grid

Report all economic metrics at all of:

- 10 bps round trip: lower-cost diagnostic
- 12 bps round trip: PRIMARY screening cost
- 15 bps round trip: stress cost

No configuration can promote unless it is profitable at 12 bps and remains positive at 15 bps.

Funding is ignored because candidate holding horizons are at most five minutes; any future longer-horizon phase must model funding separately.

## Candidate horizons

The only candidate holding horizons are:

- 10 seconds
- 30 seconds
- 60 seconds
- 120 seconds
- 300 seconds

No other horizon may be introduced after seeing results in this phase.

## Model

Use T1 only:

- StandardScaler + Ridge
- alpha grid `{0.1, 1.0, 10.0, 100.0}`
- no HGBR, XGBoost, CatBoost, neural network, Transformer, LSTM or unrestricted search

The model predicts the signed cost-excess target for each candidate horizon.

## Signal gate candidates

For a fitted model, a trade is eligible only when the absolute prediction is in a frozen extreme-prediction quantile calculated from past training/calibration predictions.

Candidate absolute-prediction quantiles:

- 0.9900
- 0.9950
- 0.9975
- 0.9990
- 0.9995

Trade direction is the sign of the prediction. Raw zero-threshold sign accuracy over all seconds is not used for selection or promotion.

## Independent opportunity rule

For each symbol independently:

- at most one open trade at a time;
- entry occurs at the decision second when the gate first fires and the symbol is flat;
- exit occurs exactly H seconds later using the last trade price known at or before the exit second;
- all gate signals while a position is open are ignored;
- after exit, a new trade may enter no earlier than the next second;
- no pyramiding, averaging down, leverage optimization or concurrent multiple positions in the same symbol.

BTCUSDT and ETHUSDT may be open simultaneously. A combined equal-notional portfolio is reported separately.

## Development folds

Retain the five chronological outer development folds already frozen in Phase 0D-H-TF:

1. 2026-06-15 through 2026-06-24
2. 2026-06-25 through 2026-07-04
3. 2026-07-05 through 2026-07-14
4. 2026-07-15 through 2026-07-24
5. 2026-07-25 through 2026-08-03

Training is strictly earlier than evaluation. Labels crossing a training/evaluation boundary are purged using the candidate horizon H.

## Nested configuration selection inside each outer fold

Configuration selection must occur only inside the outer training history.

For each outer fold:

1. reserve the final 20% of eligible outer-training rows as chronological inner validation;
2. use the first 80% as inner training;
3. for every frozen `(H, alpha, gate_quantile)` configuration, fit only on inner training;
4. derive the absolute-prediction gate threshold only from predictions generated from data available before inner validation;
5. execute non-overlapping simulated trades on inner validation;
6. discard configurations that fail the minimum profitability/stability filters below;
7. choose among survivors using the frozen lexicographic objective below;
8. refit the selected `(H, alpha)` on the complete eligible outer-training history, recompute its gate threshold using only that training history, and evaluate once on the outer fold.

No outer-fold result may influence the configuration selected for that same fold.

## Inner minimum filters

A configuration is eligible for selection only if its inner-validation trades satisfy, at the 12 bps primary cost:

- total net PnL > 0 bps;
- net expectancy > 0 bps/trade;
- profit factor > 1.0;
- at least 5 independent trades per active validation day on median;

and at the 15 bps stress cost:

- total net PnL > 0 bps;
- net expectancy > 0 bps/trade.

If no configuration survives, that outer fold is `NO_CONFIGURATION` and counts as a failed fold.

## Frozen lexicographic selection objective

Among inner-surviving configurations, select in this exact order:

1. higher median net bps/day at 12 bps cost;
2. if tied within 1%, higher minimum 5-day rolling net PnL at 12 bps cost;
3. if still tied, higher median independent trades/day;
4. if still tied, lower maximum drawdown;
5. if still tied, lower horizon H;
6. if still tied, higher gate quantile.

Raw accuracy is never a selector.

## Metrics

For every outer fold, symbol and cost level, report:

- number of independent trades
- median and mean trades/day
- LONG and SHORT counts
- gross bps/trade
- net bps/trade (expectancy)
- total net bps
- median net bps/day
- fraction of positive trading days
- win rate (reporting only)
- average winner and average loser
- profit factor
- maximum peak-to-trough drawdown in cumulative net bps
- cumulative-net-bps / max-drawdown ratio
- worst 5-day rolling net PnL

## Development promotion gate per symbol

A symbol becomes a Phase 0D-H-OPPORTUNITY candidate only if all of the following hold across its outer development folds:

Profitability:
- at least 4 of 5 outer folds have positive total net PnL at 12 bps;
- pooled net expectancy at 12 bps >= +0.50 bps/trade;
- pooled total net PnL at 12 bps > 0;
- pooled net expectancy at 15 bps > 0;
- pooled total net PnL at 15 bps > 0.

Stability:
- pooled profit factor at 12 bps >= 1.10;
- at least 55% of active trading days are net positive at 12 bps;
- cumulative-net-bps / max-drawdown ratio at 12 bps >= 2.0;
- no outer fold has net expectancy below -1.0 bps/trade at 12 bps.

Opportunity count:
- median independent trades/day across active outer-fold days >= 10.

The opportunity-count requirement is deliberately below the aspirational 20-40/day level: the selection objective will prefer more opportunities only after profitability and stability are already satisfied.

## Final development configuration and holdout rule

If a symbol passes the development gate:

- run the same frozen nested-selection logic one final time using only the complete development window to freeze one `(H, alpha, gate_quantile)` configuration for that symbol;
- record code, data-manifest and result hashes;
- only then open the 2026-08-04 through 2026-08-23 holdout exactly once.

No configuration change is allowed after seeing any holdout metric.

## Holdout confirmation gate

The frozen configuration confirms on the holdout only if, at 12 bps:

- total net PnL > 0;
- net expectancy >= +0.50 bps/trade;
- profit factor >= 1.10;
- positive trading-day fraction >= 55%;
- cumulative-net-bps / max-drawdown ratio >= 1.5;
- median independent trades/day >= 10;

and at 15 bps:

- total net PnL > 0;
- net expectancy > 0.

If the holdout fails, the phase fails. Do not retune and reopen it.

## What this phase does not do

- no leverage optimization;
- no position-size optimization;
- no stop-loss or take-profit search;
- no maker-fill assumption;
- no account-specific fee optimization;
- no capital compounding;
- no reuse of the holdout for redesign.

If this phase confirms, exit-rule optimization and live full-L2 execution-cost validation become separate later phases.
