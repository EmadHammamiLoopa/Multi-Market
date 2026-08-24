# V2.3 Phase 0D Preregistration — Crypto Microstructure Information Audit

Frozen: 2026-08-24

## Motivation

Phase 0C-R restored evaluation feasibility but rejected the tested 5-minute price-only asset-specific representation for EURUSD, XAUUSD, BTCUSD, and ETHUSD. Phase 0D therefore changes the information source rather than tuning the rejected representation.

Phase 0D tests whether short-horizon crypto microstructure contains incremental predictive information beyond short trailing mid-price returns.

## Targets

- BTCUSDT
- ETHUSDT

Venue and market type are frozen to Binance USD-M perpetual futures for Phase 0D.

## Data source

Public Binance USD-M Futures market data only. No account credentials or private endpoints are required.

Collector inputs:

- diff book depth stream at 100 ms;
- aggregate trade stream;
- individual symbol book ticker stream;
- REST depth snapshot used only to initialize/reinitialize the local order book.

The collector must follow exchange sequence IDs and rebuild from a fresh snapshot after any detected depth-sequence discontinuity.

## Collection-only stage

The first Phase 0D implementation is data collection and integrity validation only. No official model scoring may begin until the minimum capture period below is satisfied.

Minimum capture before official scoring: 30 calendar days.
Preferred capture: 60–90 calendar days.

Partial data may be used only for collector/integrity tests and engineering validation, not for promotion decisions.

The engineering smoke-test capture performed before the official collection start is not part of the 30-day official dataset. Operational fixes discovered during smoke testing must be frozen before the official capture starts.

## Raw preservation

Raw websocket messages must be retained append-only as losslessly compressed JSONL (`.jsonl.gz`) with local receive timestamp. Gzip is storage-only and must preserve each JSON record exactly when decompressed. Normalized one-second snapshots are derived data and must never replace raw capture.

Each raw record must contain:

- local receive timestamp in UTC nanoseconds or an equivalent lossless UTC timestamp;
- stream name;
- symbol;
- exchange event timestamp when present;
- unmodified message payload.

## Local order book integrity

For each symbol:

1. obtain a REST depth snapshot;
2. buffer websocket depth messages while snapshot is acquired;
3. discard buffered messages ending before snapshot lastUpdateId;
4. apply only the first valid event that bridges the snapshot state;
5. thereafter require update-ID continuity using the exchange-provided update fields;
6. on any gap, mark the local book invalid, emit an integrity event, reacquire a snapshot, and resume only after a valid bridge.

No interpolation across gaps is allowed.

## Normalized sampling

Primary derived sampling interval: 1 second UTC-aligned snapshots.

A normalized row may be emitted only while the local depth book is valid. Trade-flow buckets from intervals with an invalid local book must be discarded rather than carried into a later valid second.

Frozen normalized fields include:

- timestamp_utc
- symbol
- best_bid
- best_ask
- bid_qty_l1
- ask_qty_l1
- mid
- spread_bps
- microprice
- microprice_minus_mid_bps
- bid_depth_l5
- ask_depth_l5
- bid_depth_l10
- ask_depth_l10
- obi_l1
- obi_l5
- obi_l10
- agg_buy_qty_1s
- agg_sell_qty_1s
- agg_buy_count_1s
- agg_sell_count_1s
- trade_flow_imbalance_1s
- trade_count_imbalance_1s
- depth_sequence_valid
- last_depth_update_id

Buyer/seller trade classification must use only exchange-provided aggressor/maker information from the trade message; no future-price inference is permitted. Websocket event routing must not depend on stream-name letter case; the exchange payload event type is authoritative when present.

## Later predictive features

The later scoring stage may derive only preregistered causal trailing features from the normalized stream:

- mid returns over 1 s and 3 s;
- spread bps;
- OBI at L1/L5/L10;
- microprice-minus-mid bps;
- trade-flow imbalance over 1 s, 5 s, and 10 s;
- trade-count imbalance;
- bid/ask depth at L5;
- causal depth-change imbalance.

Any additional feature family requires a newly named phase/specification before scoring.

## Frozen horizons for later scoring

Primary: 10 seconds.
Diagnostics: 3 seconds and 30 seconds.

The prediction timestamp must precede every label price observation. Labels are based on future executable or explicitly defined mid-price observations; the exact later execution label specification must be frozen before first model scoring.

## Frozen model families for later scoring

- D0: trailing mid returns only -> Ridge
- D1: microstructure block -> Ridge
- D2: microstructure block -> HistGradientBoostingRegressor
- D3: microstructure block -> CatBoost, only if dependency is explicitly frozen before scoring

No neural network, Transformer, LSTM, Optuna, or unrestricted hyperparameter search is part of Phase 0D.

## Later evaluation policy

Chronological only. No random CV.

Before official scoring, freeze exact date boundaries based solely on capture timestamps and without inspecting future target outcomes. Target plan:

- approximately first 70% development/training history;
- next approximately 20% chronological outer OOS evaluation;
- final approximately 10% untouched holdout.

The final holdout must not be used for feature choice, thresholds, hyperparameters, or debugging.

## Statistical promotion principle

Primary question: incremental predictive value versus D0, not raw standalone fit.

Required metrics will include:

- delta R2 versus D0;
- Spearman information coefficient;
- directional accuracy;
- fold consistency.

Exact numeric promotion thresholds must be frozen in a scoring preregistration before the first official model run and cannot be selected after observing Phase 0D scores.

## Economic evaluation

Economic evaluation is forbidden until a statistical candidate exists.

When opened, it must use actual bid/ask-side executable prices or an explicitly conservative execution approximation and must include:

- taker fee assumptions;
- spread;
- slippage;
- latency stress;
- 1.0x, 1.5x, and 2.0x cost stress.

No missing cost may be treated as zero.

## Forbidden post-hoc actions

After the official collection starts, do not silently change venue, market type, target symbols, sampling interval, raw-message preservation, order-book continuity requirements, or the minimum 30-day official-scoring gate.

After scoring begins, do not tune horizons/features/models/thresholds based on the first result. Any rescue requires a new preregistered phase.
