# V1 evaluation freeze — 2026-08-22

This note freezes the interpretation of the first complete V1 walk-forward research run before any V2 design or additional time-window validation.

## Frozen V1 configuration

- Model: XGBoost classifier
- Horizons: 3 / 6 / 12 five-minute bars (15 / 30 / 60 minutes)
- Confidence thresholds explored: 0.55 / 0.60 / 0.65 / 0.70 / 0.75
- Minimum training rows: 5,000
- Retrain cadence: every 500 decision bars
- Validation tail: 20%
- TP: 20 bp
- SL: 10 bp
- Round-trip cost: 2 bp
- Point-in-time expanding-window training only

Threshold rankings in this run are exploratory. They are not untouched-test selections.

## Full-history result

### EURUSD

All 15 horizon/threshold combinations were economically negative after costs. Accuracy increased strongly with confidence, but execution profit factor remained below 1 and expectancy remained negative.

At 30 minutes, threshold 0.75 reached 85.74% selected directional accuracy, but execution PF was 0.14, execution expectancy -1.59 bp/trade, and net return -18.87%.

### XAUUSD

All 15 combinations were economically negative after costs. The model often achieved extremely high directional accuracy while still losing under the fixed execution model.

At 15 minutes, threshold 0.75 reached 97.65% selected directional accuracy, but execution PF was 0.11 and execution expectancy -1.79 bp/trade. This is strong evidence that the current future-close direction target is not aligned with trade profitability.

### BTCUSD

BTCUSD was the only market with positive exploratory execution results.

At 15 minutes:

- threshold 0.70: 59.78% directional accuracy, 70 non-overlapping trades, execution PF 1.17, execution expectancy +0.81 bp/trade, net +0.56%, max DD 0.90%
- threshold 0.75: 67.31% directional accuracy, 40 non-overlapping trades, execution PF 1.50, execution expectancy +2.05 bp/trade, net +0.82%, max DD 0.52%

These are exploratory findings only because the thresholds were inspected on the same history. BTC 30-minute and 60-minute configurations remained negative.

### ETHUSD

All 15 combinations were economically negative. Higher confidence did not produce a stable improvement in directional accuracy or expectancy. No V1 ETH configuration qualifies as a positive trading candidate.

### QQQ

All 15 combinations were economically negative. Directional accuracy stayed near 50% and execution PF stayed below 1. No V1 QQQ configuration qualifies as a positive trading candidate.

## Frozen conclusion

V1 succeeds as a proof that walk-forward ML can produce selective directional classifications and can materially reduce coverage relative to V0. It does **not** qualify as a general profitable multi-market trading model.

The main design problem exposed by V1 is target mismatch: the classifier predicts whether the future close is above or below the decision close, even when the move is too small or the intrahorizon path is unfavorable after costs. High classification accuracy can therefore coexist with negative trading expectancy.

V1 must remain unchanged for the next validation stage. No threshold, feature, model parameter, TP, SL, or cost assumption should be tuned from the additional frozen time-window tests.

## Next validation stage

Before V2, run the unchanged V0 and unchanged V1 on two additional deterministic historical windows, 30 samples per market per window:

- Window B: 2026-03-02T00:00:00Z through 2026-03-20T23:59:59Z
- Window C: 2026-06-01T00:00:00Z through 2026-06-19T23:59:59Z

Samples must be selected deterministically and evenly from valid candles in each window. Dates are frozen before inspecting outcomes. Compare exact shared timestamps and report selected accuracy, coverage, and paired verdict transitions.

The existing August 2026 audit remains Window A and must not be regenerated or overwritten.
