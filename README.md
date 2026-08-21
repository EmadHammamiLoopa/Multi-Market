# Multi-Market

Point-in-time historical replay and evaluation framework for multi-market intraday trading research.

**V0 remains the immutable benchmark baseline. V1 is the first learned model.** Every future version is evaluated with leakage-resistant historical tests, explicit costs, selective LONG / SHORT / NO_TRADE decisions, and frozen time-machine timestamps.

## V0 baseline

V0 provides:

- strict point-in-time historical replay
- transparent momentum baseline predictor
- LONG / SHORT / NO_TRADE decisions with confidence filtering
- TP/SL barrier simulation and costs
- machine-readable metrics and dashboard reporting
- deterministic historical time-machine audits
- exact dataset SHA-256 recording

Run:

```bash
multimarket-replay data/EURUSD_5m.csv --symbol EURUSD --version-label V0-5M-30M --horizon-bars 6 --plot
multimarket-time-machine data/EURUSD_5m.csv --symbol EURUSD --samples 30 --horizon-bars 6
```

## V1 learned intraday model

V1 adds:

- 35 causal technical/time/session features
- expanding-window XGBoost classification
- completed-horizon-only training labels
- chronological historical validation and confidence calibration
- selective `NO_TRADE`
- 15 / 30 / 60 minute evaluation on 5-minute bars
- Twelve Data chunked historical backfill
- non-overlapping portfolio execution metrics
- V1 time-machine evaluation on the exact timestamps frozen by V0

Install V1 dependencies:

```bash
python -m pip install -e ".[all]"
```

Important: before backfilling the current V0 files, freeze their 30-point audit JSONs:

```bash
for SYMBOL in EURUSD XAUUSD BTCUSD ETHUSD QQQ; do
  multimarket-time-machine "data/${SYMBOL}_5m.csv" \
    --symbol "$SYMBOL" \
    --samples 30 \
    --horizon-bars 6 \
    --lookback-bars 24 \
    --confidence-threshold 0.60 \
    --output-json "reports/${SYMBOL}_V0_TIME_MACHINE_30.json"
done
```

Backfill approximately one year of 5-minute history:

```bash
multimarket-backfill \
  --symbols EURUSD XAUUSD BTCUSD ETHUSD QQQ \
  --interval 5m \
  --start 2025-08-01 \
  --end 2026-08-22 \
  --chunk-days 14
```

Run a full multi-horizon V1 suite:

```bash
multimarket-v1-suite data/EURUSD_5m.csv \
  --symbol EURUSD \
  --horizons 3 6 12 \
  --confidence-threshold 0.60 \
  --min-train-rows 5000 \
  --retrain-every 500 \
  --output-json reports/EURUSD_V1_SUITE.json
```

Re-test the exact frozen V0 historical timestamps with V1:

```bash
multimarket-v1-time-machine data/EURUSD_5m.csv \
  --symbol EURUSD \
  --baseline-json reports/EURUSD_V0_TIME_MACHINE_30.json \
  --horizon-bars 6 \
  --confidence-threshold 0.60 \
  --min-train-rows 5000 \
  --retrain-every 500 \
  --output-json reports/EURUSD_V1_TIME_MACHINE_30.json
```

See [`docs/V1.md`](docs/V1.md) for the full methodology and required run order.

## Accuracy is not enough

Primary evidence includes:

- selected directional accuracy
- coverage
- win rate
- profit factor
- expectancy after costs
- non-overlapping portfolio return and drawdown
- confidence calibration
- robustness across markets, horizons, and historical periods

A version is not considered better merely because one accuracy number increased.

## Research progression

- **V0:** replay engine + auditable momentum baseline + frozen historical audit
- **V1:** causal features + walk-forward XGBoost + calibration + realistic execution accounting
- **V2:** synchronized multi-market features and opportunity ranking
- **V3:** point-in-time macro/news/event store
- **V4:** ensemble + regime detector + news/event agent
- **V5:** paper-trading adapter behind a deterministic risk engine
- **V6:** live execution only after acceptance criteria are met

## Tests

```bash
python -m unittest discover -s tests -v
```

This software is for research and testing. Historical results do not guarantee future profit, and no live broker execution is included in V1.
