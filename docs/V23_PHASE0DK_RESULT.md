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
- pooled candidate trades at 12 bps: `73`
- pooled candidate net expectancy at 12 bps: `-14.382628 bps/trade`
- incremental vs Phase 0D-J Ridge: `False`

### ETHUSDT

- development pass: `False`
- pooled candidate trades at 12 bps: `63`
- pooled candidate net expectancy at 12 bps: `-18.018040 bps/trade`
- incremental vs Phase 0D-J Ridge: `False`

Overall:

- `candidate_targets=NONE`
- `decision=FAIL_KEEP_HOLDOUT_SEALED`

## Interpretation

The preregistered nonlinear XGBoost formulation failed strongly for both symbols. It did not rescue the mark/index/premium futures-state hypothesis and did not improve on the frozen Phase 0D-J Ridge reference. The primary 12 bps net expectancy was materially negative for both BTCUSDT and ETHUSDT.

Phase 0D-K is therefore closed. No post-result tuning of its four frozen XGBoost configurations, horizons, feature blocks, signal quantiles, or promotion gates is permitted under the Phase 0D-K name.

The sealed 2026-08-04 through 2026-08-23 historical holdout remains unopened.
