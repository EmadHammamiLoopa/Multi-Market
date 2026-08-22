# V2 volatility-scaled label audit — 2026-08-22

This document freezes the first volatility-scaled V2 label diagnostic before any price-only V2 model is trained.

## Frozen label policy

- horizon: 6 x 5-minute bars (30 minutes)
- horizon must be exactly contiguous; gaps are rejected
- every bar in the decision-to-horizon span must pass the hard session/data-quality gate
- volatility estimator: causal EWMA of valid contiguous 5-minute log returns
- EWMA span: 48
- minimum volatility observations: 24
- TP distance: 1.0 x one-bar EWMA volatility x sqrt(6)
- SL distance: 1.0 x one-bar EWMA volatility x sqrt(6)
- round-trip cost: 2 bp
- same-bar TP/SL ambiguity: STOP first

The 1.0/1.0 symmetric volatility geometry is frozen here as the first V2 baseline. It was selected as a structural normalization policy, not by maximizing historical PnL.

## Results

| Market | Eligible | Mean TP/SL bp | LONG net+ | SHORT net+ | HORIZON | LONG mean net bp | SHORT mean net bp |
|---|---:|---:|---:|---:|---:|---:|---:|
| EURUSD | 76,949 | 4.824 | 32.83% | 34.34% | 47.96% | -2.068 | -1.947 |
| XAUUSD | 78,701 | 19.503 | 44.21% | 42.57% | 49.72% | -1.911 | -2.158 |
| BTCUSD | 110,967 | 28.521 | 46.11% | 46.80% | 49.60% | -2.233 | -1.818 |
| ETHUSD | 111,118 | 41.057 | 47.25% | 47.56% | 49.21% | -2.387 | -1.686 |
| QQQ | 19,049 | 22.767 | 46.36% | 44.40% | 52.03% | -2.068 | -1.966 |

Exit geometry is now materially more comparable across markets than the fixed TP20/SL10 diagnostic. HORIZON outcomes cluster near 48–52% rather than being dominated by EURUSD timeouts, while TP and SL each account for roughly one quarter of eligible events.

EURUSD retains a lower net-positive rate because a 2 bp cost is large relative to its mean 4.824 bp barrier. That is an economically meaningful difference and will not be normalized away merely to balance classes.

The unconditional side means remain near -2 bp, approximately the assumed round-trip cost, so this audit is not evidence of a trading edge. V2 must create value by selecting a conditional subset with positive out-of-sample expected net return.

## Next frozen comparison

The first price-only V2 model will use this label policy without changing the barrier multipliers. Model development must therefore be evaluated as a change in conditional selection quality, not as a retrospective relabeling exercise.
