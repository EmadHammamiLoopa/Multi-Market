# V1 research protocol

V1 is the first learned intraday model. It must be evaluated as a point-in-time research system rather than tuned to one attractive backtest number.

## 1. Freeze V0 historical timestamps

Before extending the local data files, save the V0 time-machine JSON for each market. These timestamps are the paired V0/V1/V2 regression set.

```bash
multimarket-time-machine data/EURUSD_5m.csv \
  --symbol EURUSD --samples 30 --horizon-bars 6 --lookback-bars 24 \
  --confidence-threshold 0.60 \
  --output-json reports/EURUSD_V0_TIME_MACHINE_30.json
```

## 2. Backfill historical 5-minute data

Use Twelve Data in bounded chunks and merge by timestamp.

```bash
multimarket-backfill \
  --symbols EURUSD XAUUSD BTCUSD ETHUSD QQQ \
  --interval 5m --start 2025-08-01 --end 2026-08-22 --chunk-days 14
```

## 3. Run the multi-horizon V1 suite

```bash
multimarket-v1-suite data/EURUSD_5m.csv \
  --symbol EURUSD --horizons 3 6 12 \
  --confidence-threshold 0.60 --min-train-rows 5000 --retrain-every 500 \
  --output-json reports/EURUSD_V1_SUITE.json
```

On 5-minute bars, 3/6/12 bars correspond to 15/30/60 minutes.

## 4. Run threshold diagnostics

Threshold sweeps are exploratory. They reveal how accuracy, coverage, profit factor and non-overlapping execution change as V1 becomes more selective. Do not choose the best threshold on this history and then call the same history an untouched test.

```bash
multimarket-v1-research data/EURUSD_5m.csv \
  --symbol EURUSD \
  --horizons 3 6 12 \
  --thresholds 0.55 0.60 0.65 0.70 0.75 \
  --min-train-rows 5000 --retrain-every 500 \
  --v0-baseline-json reports/EURUSD_V0_TIME_MACHINE_30.json \
  --output-json reports/EURUSD_V1_RESEARCH.json
```

The research manifest records:

- exact dataset SHA-256, row count and date range
- model protocol and execution assumptions
- results for each horizon and threshold
- confidence-bucket accuracy
- non-overlapping execution metrics
- top XGBoost feature importances from the final historical fit
- frozen V0 benchmark metadata

## 5. Repeat the V0 time-machine timestamps with V1

```bash
multimarket-v1-time-machine data/EURUSD_5m.csv \
  --symbol EURUSD \
  --baseline-json reports/EURUSD_V0_TIME_MACHINE_30.json \
  --horizon-bars 6 --confidence-threshold 0.60 \
  --min-train-rows 5000 --retrain-every 500 \
  --output-json reports/EURUSD_V1_TIME_MACHINE_30.json
```

## 6. Produce a paired V0 -> V1 comparison

```bash
multimarket-version-compare \
  --symbol EURUSD \
  --v0 reports/EURUSD_V0_TIME_MACHINE_30.json \
  --v1 reports/EURUSD_V1_TIME_MACHINE_30.json \
  --output-json reports/EURUSD_V0_V1_COMPARE.json
```

The comparison only uses shared decision timestamps and reports paired verdict transitions such as WRONG->CORRECT and CORRECT->NO_TRADE.

## Acceptance philosophy

V1 is not accepted merely because one accuracy number is larger than V0. Evidence should include:

- improvement on frozen V0 time-machine timestamps
- sensible confidence/accuracy relationship
- adequate but selective coverage
- positive or materially improved expectancy after costs
- profit factor improvement
- realistic one-position-at-a-time execution metrics
- stability across more than one horizon and market

A threshold chosen after inspecting this evaluation history must be frozen and validated again on a later untouched period before it becomes a production candidate.
