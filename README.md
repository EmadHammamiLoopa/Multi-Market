# Multi-Market — V0

Point-in-time historical replay and evaluation framework for multi-market trading research.

**V0 is the benchmark baseline.** Its job is not to claim profitability; it gives every future model version (V1, V2, ...) the same leakage-resistant test harness and records comparable accuracy, profitability and risk metrics.

## What V0 does

- strict point-in-time historical replay — the predictor cannot see future bars
- transparent momentum baseline predictor
- LONG / SHORT / NO_TRADE decisions with confidence filtering
- TP/SL barrier trade simulation
- configurable spread/commission/slippage cost
- human-readable terminal results
- machine-readable metrics JSON
- optional PNG dashboard with running accuracy, equity curve, trade outcomes and version comparison
- persistent `results/benchmark_history.csv` so V0/V1/V2/... can be ranked on the **exact same dataset hash**
- unit tests for leakage, execution, metrics and reporting

## Install

Core backtest only:

```bash
python -m pip install -e .
```

With plots:

```bash
python -m pip install -e ".[plot]"
```

## Run V0

Input CSV must contain:

```text
timestamp,open,high,low,close
```

Example:

```bash
multimarket-replay data/eurusd_15m.csv \
  --symbol EURUSD \
  --version-label V0 \
  --horizon-bars 4 \
  --lookback-bars 20 \
  --confidence-threshold 0.60 \
  --take-profit-bps 20 \
  --stop-loss-bps 10 \
  --round-trip-cost-bps 2 \
  --plot \
  --output-jsonl reports/EURUSD_V0_predictions.jsonl
```

The terminal prints results such as:

```text
Multi-Market V0 | EURUSD
==========================================
Predictions          : ...
Trades               : ...
Coverage             : ...%
Directional accuracy : ...%
Trade win rate       : ...%
Profit factor        : ...
Expectancy           : ... bps/trade
Net return           : ...%
Max drawdown         : ...%
```

Generated files:

```text
reports/EURUSD_V0_metrics.json
reports/EURUSD_V0_dashboard.png
results/benchmark_history.csv
```

The dashboard contains:

1. running directional accuracy
2. simulated cumulative net return
3. win/loss counts
4. best recorded accuracy by model version on the same dataset

## Comparing V0, V1, V2, ...

Use the **same historical CSV** for every version and change only the model/version label. For example:

```bash
multimarket-replay data/eurusd_15m.csv --symbol EURUSD --version-label V0 --plot
multimarket-replay data/eurusd_15m.csv --symbol EURUSD --version-label V1 --plot
multimarket-replay data/eurusd_15m.csv --symbol EURUSD --version-label V2 --plot
```

Each run is appended to `results/benchmark_history.csv`. The CLI then prints a ranking like:

```text
Version ranking on this exact dataset
==========================================
 1. V2       accuracy=65.20%  win=61.80%  net=  7.200%  maxDD= 4.100%
 2. V1       accuracy=61.70%  win=58.50%  net=  4.900%  maxDD= 5.000%
 3. V0       accuracy=53.10%  win=51.20%  net=  0.800%  maxDD= 6.700%
```

The CSV SHA-256 is stored with every run, preventing us from accidentally calling a result "better" when it was tested on a different dataset.

## Accuracy is not enough

The main comparison metrics are:

- directional accuracy
- trade win rate
- coverage
- profit factor
- expectancy after costs
- net return after costs
- maximum drawdown

A future model is not considered better merely because accuracy increased. It should improve out-of-sample risk-adjusted results too.

## Research progression

- **V0:** replay engine + auditable momentum baseline + benchmark reporting
- **V1:** walk-forward folds + XGBoost + probability calibration
- **V2:** synchronized multi-market features and opportunity ranking
- **V3:** point-in-time macro/news/event store
- **V4:** ensemble + regime detector + news/event agent
- **V5:** paper-trading adapter behind a deterministic risk engine
- **V6:** live execution only after acceptance criteria are met

## Test

```bash
python -m unittest discover -s tests -v
```

This software is for research and testing. Historical accuracy does not guarantee future profit.
