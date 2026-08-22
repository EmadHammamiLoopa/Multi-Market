# V2 price-only A/B/C validation — 2026-08-22

This document freezes the first V2 price-only economic validation before any threshold or model tuning.

## Frozen policy

- horizon: 6 x 5-minute bars
- volatility labels: causal EWMA48, minimum 24 observations
- TP/SL: symmetric 1.0 x horizon volatility
- round-trip cost: 2 bp
- minimum positive probability: 0.55
- minimum predicted EV: > 0 bp
- minimum training rows: 5,000
- retrain every: 500 bars
- validation fraction: 20%
- embargo: 6 bars
- features: existing causal V1 price/session features, but V2 requires the entire 48-bar raw feature history to be contiguous and hard-eligible

No parameter above was changed after viewing these A/B/C results.

## Results

| Window | Frozen | Gate-blocked | Unavailable | Actionable | Net-positive | Net-nonpositive | NO_TRADE |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 150 | 40 | 1 | 1 | 0 | 1 | 148 |
| B | 150 | 36 | 10 | 0 | 0 | 0 | 140 |
| C | 150 | 38 | 1 | 4 | 0 | 4 | 145 |
| **Total** | **450** | **114** | **12** | **5** | **0** | **5** | **433** |

Aggregate actionable coverage over the 450 frozen timestamps was 1.11% (5/450). Selected win rate was 0% (0/5).

The five executed trades were:

- QQQ Window A: SHORT, realized -6.57 bp
- ETHUSD Window C: SHORT, realized -56.66 bp
- QQQ Window C: LONG, realized -28.68 bp
- QQQ Window C: LONG, realized -25.80 bp
- QQQ Window C: LONG, realized -37.18 bp

Approximate aggregate realized net over these five selections: -154.89 bp, or -30.98 bp per selected trade.

## Interpretation

The first V2 price-only baseline is rejected as a deployable model. It is both too sparse and economically unsuccessful on the frozen A/B/C evidence.

This result must not be repaired by lowering the 0.55 probability threshold or changing the EV threshold on the same frozen evidence. Doing that would tune directly to the validation sample.

The next diagnostic is ranking-only: evaluate whether side-specific probability and EV scores rank realized economic outcomes on eligible frozen timestamps, without changing the live decision policy. If ranking signal is absent, the next research step should change the feature/model information set rather than simply relaxing gates.

QQQ also shows a structural evaluation limitation: a strict 48-bar contiguous feature window blocks many regular-session timestamps because the history cannot cross overnight gaps. Some QQQ Window B samples are unavailable because 5,000 clean historical training rows were not yet available. These are separate from model abstention and should remain visible in diagnostics.
