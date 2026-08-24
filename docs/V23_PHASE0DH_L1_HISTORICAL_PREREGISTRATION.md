# V2.3 Phase 0D-H-L1 Historical Preregistration

Frozen: 2026-08-24

## Purpose

Phase 0D-H-L1 is an historical screening audit designed to determine whether short-horizon Level-1 crypto microstructure contains incremental predictive information before waiting for the separate prospective Phase 0D full-L2 capture.

This is not the same experiment as the live full-L2 Phase 0D collector. Historical public Binance data provides `bookTicker` and `aggTrades`, but not the exact 100 ms reconstructed L5/L10 depth state used by Phase 0D. Therefore this phase is named separately and may not be used to claim full-L2 validation.

## Frozen targets and venue

- BTCUSDT
- ETHUSDT
- Binance USD-M perpetual futures only

## Frozen historical window

- Inclusive acquisition start: 2026-05-26 UTC
- Development/scoring window: 2026-05-26 through 2026-08-03 UTC (70 complete calendar days)
- Historical holdout: 2026-08-04 through 2026-08-23 UTC (20 complete calendar days)

The historical holdout must not be opened unless a development-window statistical candidate first passes the frozen gate. It may then be opened exactly once as historical confirmation.

Smoke-test data and prospective Phase 0D live data are excluded from this phase.

## Data

Public Binance historical daily archives:

- `bookTicker`
- `aggTrades`

Raw ZIP archives are retained. SHA-256 hashes are generated locally after acquisition. Missing days or malformed archives are acquisition failures, not silently imputed days.

## Causal normalization

Derived rows are sampled on UTC-aligned one-second boundaries.

For each second and symbol, the L1 state is the last `bookTicker` observation with event time less than or equal to the sample timestamp. Historical bookTicker records are sorted by `(event_time, update_id)` before normalization because archive row order must not be treated as a causality guarantee.

Aggregate trades are assigned by exchange trade/event timestamps to their corresponding one-second bucket. Buyer-maker flag is used only to determine aggressor side:

- buyer is maker => aggressive sell
- buyer is not maker => aggressive buy

No future-price inference, interpolation, or backward filling is allowed.

## Frozen L1 feature families

Baseline H0:

- trailing log mid-price return over 1 s
- trailing log mid-price return over 3 s

Microstructure H1/H2:

- H0 features
- spread bps
- L1 order-book imbalance `(bid_qty - ask_qty)/(bid_qty + ask_qty)`
- microprice-minus-mid bps
- aggregate aggressive-buy quantity over 1 s
- aggregate aggressive-sell quantity over 1 s
- trade-flow imbalance over 1 s
- trade-count imbalance over 1 s
- causal trailing 5 s trade-flow imbalance
- causal trailing 10 s trade-flow imbalance

No L5/L10 depth feature is present in this historical L1 phase.

## Labels

Primary label: future mid-price log return in bps from decision timestamp to decision + 10 seconds.

Diagnostics only: 3 s and 30 s.

Every feature observation must be known at or before the decision timestamp. No feature may use any record after that timestamp.

## Models

- H0: StandardScaler + Ridge on baseline features
- H1: StandardScaler + Ridge on L1 microstructure features
- H2: HistGradientBoostingRegressor on L1 microstructure features

Ridge alpha grid is frozen to `{0.1, 1.0, 10.0, 100.0}` and selected inside each outer training window using a chronological inner validation split. H2 uses a fixed configuration frozen before first official scoring; no tuning based on results.

No CatBoost, XGBoost, neural network, Transformer, LSTM, Optuna, or unrestricted search is allowed in this phase.

## Development walk-forward

Only the first 70 days may be used for development scoring. Five chronological outer folds are created from date boundaries without inspecting target outcomes. Labels crossing an evaluation boundary are purged from training.

No random cross-validation.

## Statistical promotion gate

A candidate H1 or H2 must satisfy all of:

- pooled delta R2 versus H0 > 0;
- pooled Spearman IC > 0;
- at least 4 scored folds;
- at least 3 outer folds with positive delta R2 versus H0.

H1 is preferred if both pass because it is the simpler model.

If neither candidate passes, Phase 0D-H-L1 is a historical FAIL and the 20-day historical holdout remains sealed.

If a candidate passes, freeze the candidate and then open the 20-day historical holdout once. No feature/model/threshold change is allowed after seeing that holdout.

## Economic evaluation

Forbidden before a statistical candidate exists. Historical economic evaluation, if later opened, must be separately specified because public historical L1 archives do not reproduce the full live execution path.

## Interpretation

PASS supports continuing the microstructure direction and prospective full-L2 confirmation. FAIL rejects this historical L1 representation only; it does not automatically reject full-L2 Phase 0D because L5/L10 depth information is absent.
