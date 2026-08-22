# V2.1 cross-market + regime A/B/C validation — 2026-08-22

This document freezes the first full V2.1 A/B/C evaluation before any V2.1 threshold, feature, peer, or model tuning.

## Frozen specification

V2.1 preserves the frozen V2 economic/model policy and adds only the preregistered causal regime + cross-market information set:

- H = 6 bars
- causal EWMA span = 48, minimum observations = 24
- TP / SL = 1.0x / 1.0x horizon volatility
- round-trip cost = 2 bp
- minimum training rows = 5,000
- retrain every = 500 bars
- validation fraction = 20%
- probability gate = 0.55
- predicted EV gate = > 0 bp
- embargo = 6 bars
- XGBoost tree method = hist / CPU
- random state = 42
- 35 frozen V2 target features
- 3 causal target-regime features
- 6 fields for each of the other 4 markets (availability, staleness fraction, ret1, ret6, ret12, vol12)
- maximum peer staleness = 15 minutes
- hard peer data/session quality gating and exact contiguous raw 5-minute feature history
- missing peers are masked; missing peer data never deletes a target row

Total V2.1 feature count: 62.

A/B/C are development diagnostics and are fully consumed by this validation. They must not be treated as untouched holdout evidence in any later version.

## Economic result

Across the 450 frozen timestamps:

- actionable: 6 / 450 = 1.33%
- net-positive: 3
- net-nonpositive: 3
- selected win rate: 50.0%
- approximate aggregate realized net from printed per-window summaries: -33.35 bp
- approximate mean net expectancy: -5.56 bp / selected trade

The selected trades came only from:

- QQQ A: 2 trades, 1 positive / 1 nonpositive, mean -1.388 bp
- EURUSD C: 1 trade, positive, +7.583 bp
- BTCUSD C: 1 trade, nonpositive, -15.638 bp
- QQQ C: 2 trades, 1 positive / 1 nonpositive, mean -11.258 bp

All other market/window combinations produced zero actionable trades.

### Frozen V2 comparison

Frozen V2 price-only on the same 450 timestamps had:

- actionable: 5 / 450 = 1.11%
- net-positive: 0
- net-nonpositive: 5
- selected win rate: 0%
- approximate aggregate realized net: -154.89 bp
- mean net expectancy: approximately -30.98 bp / trade

V2.1 is materially less bad economically than V2 on these diagnostics, but it remains negative-expectancy and extremely sparse. Six selected trades are also far too few to support a deployment claim.

## Ranking result by market/window

| Market | Window | Scored | pLong rho | pShort rho | EVlong rho | EVshort rho | Top20 pLong bp | Top20 pShort bp | Top20 EVlong bp | Top20 EVshort bp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EURUSD | A | 22 | -0.0175 | +0.1948 | +0.2535 | +0.2468 | +0.5097 | -0.5907 | -0.7117 | -0.4944 |
| XAUUSD | A | 22 | -0.4873 | -0.4794 | -0.0152 | -0.2185 | -4.5855 | -21.1078 | +2.4584 | -14.1402 |
| BTCUSD | A | 28 | +0.2950 | +0.1746 | +0.3673 | +0.3290 | +17.3127 | -7.8883 | +9.9496 | +1.8552 |
| ETHUSD | A | 28 | +0.2042 | +0.0909 | +0.3087 | +0.1757 | +2.3490 | -8.9924 | +20.0045 | -8.5653 |
| QQQ | A | 9 | -0.3167 | +0.0500 | +0.0333 | -0.1167 | -14.6887 | -1.3879 | -15.1114 | +6.6361 |
| EURUSD | B | 22 | -0.1406 | -0.4229 | -0.2185 | +0.0096 | -3.4091 | -5.4897 | -0.9302 | -7.9863 |
| XAUUSD | B | 22 | +0.0683 | -0.2208 | +0.1022 | +0.2660 | -7.6259 | -13.7253 | -7.9891 | +3.8670 |
| BTCUSD | B | 30 | -0.2792 | -0.3255 | -0.0923 | -0.0919 | -13.7217 | -21.6850 | -11.8240 | +3.5969 |
| ETHUSD | B | 30 | +0.2191 | +0.2423 | +0.0905 | -0.0719 | +15.8709 | +7.9521 | +12.7481 | -5.0971 |
| QQQ | B | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| EURUSD | C | 22 | +0.0819 | +0.2151 | +0.0841 | +0.1892 | -0.7947 | +0.8878 | -3.3111 | +1.9193 |
| XAUUSD | C | 22 | -0.3857 | -0.2151 | -0.2863 | -0.2953 | -15.4001 | -16.5841 | -24.0296 | +2.5792 |
| BTCUSD | C | 30 | +0.2992 | +0.1057 | +0.1502 | +0.2805 | +4.3456 | +5.5282 | +13.1467 | +5.9339 |
| ETHUSD | C | 30 | -0.3775 | -0.1822 | -0.6018 | -0.5867 | -44.8707 | +8.4670 | -30.3771 | -22.9845 |
| QQQ | C | 7 | -0.6429 | -0.3571 | -0.6429 | -0.3929 | -28.6808 | +6.1647 | -25.8012 | +6.1647 |

Total scored ranking rows = 324. The audit also recorded 102 feature-history blocks and 12 future-event blocks; the remaining 12 frozen timestamps were unavailable for scoring/training/data reasons.

## Cross-window interpretation

No market exhibits a stable all-window ranking edge across the four score heads.

- EURUSD: some positive EV ranking in A and C, but B reverses and top-20 realized results are mostly negative.
- XAUUSD: broadly poor, especially probability ranking and top-20 realized returns; no stable edge.
- BTCUSD: the most promising exploratory structure, especially EV ranking in A/C and positive EV top-20 means in A/C, but B reverses strongly. This is not stable enough to claim an edge and must not be tuned using A/B/C.
- ETHUSD: A/B contain positive pockets, but C collapses sharply negative across all four ranking heads. No stable edge.
- QQQ: very small scored samples and generally negative long-side ranking; B has no scored rows. Not suitable for an edge claim.

Unweighted means across the 14 market/windows with ranking scores are also not positive overall:

- mean Spearman pLong: -0.1057
- mean Spearman pShort: -0.0807
- mean Spearman EVlong: -0.0334
- mean Spearman EVshort: -0.0198
- mean top-20 pLong realized: -6.6707 bp
- mean top-20 pShort realized: -4.8894 bp
- mean top-20 EVlong realized: -4.4127 bp
- mean top-20 EVshort realized: -1.9083 bp

These unweighted means are descriptive only; they are not pooled correlations and are not used as a statistical significance test.

## Verdict

**V2.1 is rejected as a deployable model.**

The cross-market + regime information set improves the economic failure magnitude relative to frozen V2, and produces several locally encouraging ranking pockets, but the effect is not stable across A/B/C and aggregate ranking remains negative. The frozen decision gate remains too sparse and still has negative selected-trade expectancy.

The correct response is not to lower 0.55, relax EV > 0, select only favorable A/B/C windows, or post-hoc choose BTC/ETH heads. Any such change would tune directly to consumed development evidence.

## Next research step

A materially new information set should be preregistered as V2.2 before seeing any new holdout outcomes. The existing research plan places point-in-time macro/news context after price/regime/cross-market features. V2.2 should therefore focus on causal point-in-time external context rather than another threshold adjustment to V2.1.

Before V2.2 evaluation:

1. freeze exact external features, release-time semantics, missing-data behavior, model policy, and evaluation metrics;
2. reserve new untouched evaluation periods distinct from A/B/C;
3. verify release/vintage causality independently before model training;
4. keep frozen V2/V2.1 results as baselines;
5. do not inspect the new holdout outcomes until V2.2 is frozen.
