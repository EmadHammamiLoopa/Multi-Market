# V2.3 Phase 0D-J Futures-State Result

Date: 2026-08-24
Status: `FAIL_KEEP_HOLDOUT_SEALED`
Historical holdout opened: **NO**

## Development result

Phase 0D-J tested whether Binance USD-M futures-state information (`markPriceKlines`, `indexPriceKlines`, `premiumIndexKlines`) could add economically useful predictive value over the frozen trade-flow reference on a causal one-minute decision grid.

Acquisition and continuity prerequisites passed before scoring: all 540 required archives were valid and the development audit found no missing or duplicate minutes.

### BTCUSDT

- Phase 0D-J candidate development pass: `False`
- candidate trades at 12 bps: `39`
- candidate pooled net expectancy at 12 bps: `-11.725330 bps/trade`
- incremental vs J0: `False`

### ETHUSDT

- Phase 0D-J candidate development pass: `False`
- candidate trades at 12 bps: `80`
- candidate pooled net expectancy at 12 bps: `-9.242934 bps/trade`
- incremental vs J0: `False`

## Interpretation

The frozen linear Ridge formulation failed for both symbols. Neither selected J1/J2 candidate was profitable after the 12 bps primary cost assumption, and neither demonstrated incremental value over the J0 trade-flow reference.

This result does **not** prove that mark/index/premium state contains no useful information. It rejects the preregistered linear formulation and its frozen selection/execution rules for economic deployment.

No alpha, horizon, quantile, feature, cost, or promotion gate will be softened after this result. The 2026-08-04 through 2026-08-23 holdout remains sealed.

## Next hypothesis

A separate preregistered phase may test whether the same causal information has nonlinear conditional structure that Ridge cannot represent. Such a phase must use a fixed small nonlinear model family, retain the same chronology/cost/holdout discipline, and must not use the Phase 0D-J result to tune hyperparameters after scoring starts.
