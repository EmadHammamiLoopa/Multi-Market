# V2.3 Phase 0D-K Nonlinear Futures-State Result

Date: 2026-08-24
Status: `FAIL_KEEP_HOLDOUT_SEALED`
Historical holdout opened: **NO**
CUDA preflight: **PASS (`cuda:0`)**

## Environment correction before scoring

The initial Phase 0D-K CUDA preflight failed before any fold was scored because the default XGBoost 3.4.1 wheel was built against CUDA 13.3 while the installed NVIDIA driver exposed the RTX 5090 but could not provide a usable CUDA context to that build. No development fold had been run at that point.

The environment was corrected to the official `xgboost-cu12==3.4.0` package. Diagnostic output then confirmed:

- `USE_CUDA=True`
- XGBoost build CUDA version 12.9
- NVIDIA GeForce RTX 5090 Laptop GPU visible in WSL
- `libcuda.so.1` loadable
- XGBoost resolved device `cuda:0`
- `PHASE0DK_CUDA_DIAGNOSTIC=PASS`
- `PHASE0DK_CUDA_PREFLIGHT=PASS device=cuda:0`

No model family, hyperparameter, feature block, horizon, quantile, cost assumption, fold, selection rule, or promotion gate was changed as part of the environment correction.

## Official development score

### BTCUSDT

- development pass: `False`
- structural gate: `False`
- incremental vs Phase 0D-J Ridge: `False`
- boundary state minutes trimmed: `1`
- Phase 0D-J Ridge reference @12: `-11.725330 bps/trade`, total `-457.287884 bps`

Candidate @12 bps:

- trades: `73`
- net expectancy: `-14.382628 bps/trade`
- implied pooled gross expectancy before the 12 bps cost assumption: approximately `-2.382628 bps/trade`
- total net PnL: `-1049.931848 bps`
- profit factor: `0.359186`
- max drawdown: `1140.374117 bps`
- PnL/maxDD: `-0.920691`
- positive outer folds: `0`
- fold net expectancies: `[-10.105629, -19.419773, -15.069516, -inf, -inf]`
- median trades per active day: `3.0`
- positive active-day fraction: `0.40`

Candidate @15 bps:

- net expectancy: `-17.382628 bps/trade`
- total net PnL: `-1268.931848 bps`
- profit factor: `0.292682`
- max drawdown: `1335.374117 bps`
- positive outer folds: `0`

Outer-selection pattern:

1. Fold 1: `K1`, H=10m, X2 (depth=3, lr=0.05, 600 trees), q=0.99; inner survivors=5; outer net expectancy @12=`-10.105629`.
2. Fold 2: `K2`, H=10m, X2, q=0.995; inner survivors=18; outer net expectancy @12=`-19.419773`.
3. Fold 3: `K2`, H=5m, X2, q=0.99; inner survivors=10; outer net expectancy @12=`-15.069516`.
4. Fold 4: `NO_CONFIGURATION`.
5. Fold 5: `NO_CONFIGURATION`.

### ETHUSDT

- development pass: `False`
- structural gate: `False`
- incremental vs Phase 0D-J Ridge: `False`
- boundary state minutes trimmed: `1`
- Phase 0D-J Ridge reference @12: `-9.242934 bps/trade`, total `-739.434686 bps`

Candidate @12 bps:

- trades: `63`
- net expectancy: `-18.018040 bps/trade`
- implied pooled gross expectancy before the 12 bps cost assumption: approximately `-6.018040 bps/trade`
- total net PnL: `-1135.136548 bps`
- profit factor: `0.244072`
- max drawdown: `1169.395383 bps`
- PnL/maxDD: `-0.970704`
- positive outer folds: `0`
- fold net expectancies: `[-17.723139, -19.271373, -inf, -inf, -inf]`
- median trades per active day: `3.0`
- positive active-day fraction: `0.117647`

Candidate @15 bps:

- net expectancy: `-21.018040 bps/trade`
- total net PnL: `-1324.136548 bps`
- profit factor: `0.195743`
- max drawdown: `1355.395383 bps`
- positive outer folds: `0`

Outer-selection pattern:

1. Fold 1: `K2`, H=5m, X1 (depth=3, lr=0.05, 300 trees), q=0.99; inner survivors=1; outer net expectancy @12=`-17.723139`.
2. Fold 2: `K2`, H=5m, X2 (depth=3, lr=0.05, 600 trees), q=0.9975; inner survivors=15; outer net expectancy @12=`-19.271373`.
3. Fold 3: `NO_CONFIGURATION`.
4. Fold 4: `NO_CONFIGURATION`.
5. Fold 5: `NO_CONFIGURATION`.

Overall:

- `candidate_targets=NONE`
- `decision=FAIL_KEEP_HOLDOUT_SEALED`

## Interpretation

The failure is not explained by transaction-cost assumptions alone. At the primary 12 bps cost level, adding the cost back to the pooled net expectancy implies negative gross directional expectancy for both symbols: approximately `-2.38 bps/trade` for BTCUSDT and `-6.02 bps/trade` for ETHUSDT. The selected nonlinear signals therefore failed directionally out of sample, rather than merely producing a small positive gross edge that costs consumed.

Temporal stability also failed strongly. BTCUSDT had three scored outer folds followed by two folds with no viable inner configuration. ETHUSDT had only two scored outer folds followed by three folds with no viable inner configuration. Neither symbol had a single positive outer fold at the 12 bps primary cost assumption.

The selected feature block was not stable for BTCUSDT (`K1` in fold 1, then `K2`), while ETHUSDT selected `K2` when a configuration survived. This provides no evidence of a stable nonlinear state-conditioned signal.

The preregistered nonlinear XGBoost formulation therefore failed to rescue the mark/index/premium futures-state hypothesis and was worse than the already-negative frozen Phase 0D-J Ridge reference on both pooled expectancy and total PnL.

## Lineage closure

Phase 0D-K is closed. No post-result tuning of its four frozen XGBoost configurations, horizons, feature blocks, signal quantiles, or promotion gates is permitted under the Phase 0D-K name.

The broader `aggTrades + mark/index/premium` historical information-set lineage is also closed for model escalation. A larger tree ensemble, neural network, transformer, alternative depth, additional quantile, or another optimizer over the same information set would constitute a new post-hoc rescue attempt rather than a justified continuation.

A future phase must add materially new causal information, with full order-book microstructure (L2 depth, spread, microprice, depth imbalance/pressure/change, and trade-flow × book interactions) being the primary remaining hypothesis.

The sealed 2026-08-04 through 2026-08-23 historical holdout remains unopened.
