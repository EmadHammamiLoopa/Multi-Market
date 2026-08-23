# V2.2 D/E/F paired validation — 2026-08-23

## Verdict

**V2.2 is rejected as a deployable model and classified as DEGRADED relative to frozen V2.1.**

The comparison uses the identical preregistered 450 D/E/F timestamps.

| Metric | V2.1 | V2.2 | Delta |
|---|---:|---:|---:|
| Samples | 450 | 450 | 0 |
| Scored | 344 | 344 | +0 |
| Actionable | 5 | 3 | -2 |
| Winners | 1 | 0 | -1 |
| Coverage | 1.111111% | 0.666667% | -0.444444% |
| Aggregate net | -73.324225 bp | -119.880379 bp | -46.556153 bp |
| Expectancy | -14.664845 bp | -39.960126 bp | -25.295281 bp |
| Win rate | 20.000000% | 0.000000% | -20.000000% |
| Profit factor | 0.140090 | 0.000000 | -0.140090 |

## Interpretation

V2.2 reduced actionable coverage, removed the only winning selected trade,
increased aggregate loss, and materially worsened expectancy.

Ranking changes were mixed and do not demonstrate a stable cross-market,
cross-window advantage.

No threshold, feature, market, window, or model-head selection was performed
after opening D/E/F.

D/E/F are now consumed holdout evidence and must not be reused for V2.3 tuning.

The result rejects this V2.2 macro representation, not the general hypothesis
that macroeconomic information can be useful.
