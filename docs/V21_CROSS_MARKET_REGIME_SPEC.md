# V2.1 Cross-Market + Regime Specification

Date frozen: 2026-08-22

## Purpose

V2.1 tests whether causal regime context and causal cross-market context improve the frozen V2 price-only economic model. It does **not** change the V2 label geometry, costs, model hyperparameters, probability gate, EV gate, embargo, or training schedule.

The historical A/B/C windows have already been inspected during V1/V2 research. They are development diagnostics, not untouched final holdouts. V2.1 parameters below are frozen before evaluating V2.1 on A/B/C.

## Frozen V2 policy retained

- Horizon: 6 bars
- EWMA volatility span: 48
- Minimum volatility observations: 24
- Take-profit barrier: 1.0x horizon volatility
- Stop-loss barrier: 1.0x horizon volatility
- Round-trip cost: 2.0 bp
- Minimum training rows: 5000
- Retrain interval: 500 decision bars
- Chronological validation fraction: 20%
- Positive-probability gate: >= 0.55
- Predicted EV gate: > 0 bp
- Embargo: 6 bars
- XGBoost `tree_method="hist"`, CPU, same four frozen V2 heads
- Random state and all XGBoost hyperparameters remain identical to V2 price-only

No threshold or hyperparameter may be changed in response to A/B/C V2.1 results.

## Target-market features

V2.1 retains the 35 V1/V2 causal target-price features. The V2 rule requiring a full 48-bar contiguous, hard-eligible target feature history remains unchanged.

## Regime features

Exactly three additional target regime features are added:

1. `regime_vol48_percentile_288`
   - Percentile rank of the current clean `vol_48_bps` against the previous up-to-288 clean target feature observations.
   - Uses only information available at or before the decision timestamp.
   - If no prior clean history exists, neutral value 0.5 is used.

2. `regime_abs_ret12_percentile_288`
   - Percentile rank of the current absolute 12-bar return against the previous up-to-288 clean target feature observations.
   - Uses only information available at or before the decision timestamp.
   - If no prior clean history exists, neutral value 0.5 is used.

3. `regime_trend_consistency_12`
   - Fraction of the last 12 one-bar target returns whose sign agrees with the sign of the cumulative 12-bar return.
   - If the cumulative 12-bar return is exactly zero, neutral value 0.5 is used.
   - The target clean-history gate guarantees those raw bars are contiguous and hard eligible.

These features describe the *state relative to recent valid history* rather than adding another copy of an existing raw return/volatility field.

## Cross-market peer features

For every configured peer, V2.1 adds exactly six fields in deterministic alphabetical peer-symbol order:

- `<PEER>_available`
- `<PEER>_staleness_fraction`
- `<PEER>_ret_1_bps`
- `<PEER>_ret_6_bps`
- `<PEER>_ret_12_bps`
- `<PEER>_vol_12_bps`

A peer is `available=1` only when:

- the selected peer bar timestamp is <= the target decision timestamp;
- peer staleness is <= 15 minutes;
- all raw peer bars required for ret_1/ret_6/ret_12/vol_12 are exact contiguous 5-minute bars; and
- all required peer bars satisfy the same V2 hard-quality rule: session eligible, non-zero-range, non-repeated-full-OHLC.

When a peer is unavailable or feature-incomplete:

- `<PEER>_available = 0`
- `<PEER>_staleness_fraction = 1.25` (explicit sentinel outside the valid [0, 1] range)
- all four peer return/volatility values are zero

When usable, staleness fraction is peer staleness divided by the frozen 15-minute maximum and therefore lies in [0, 1]. The availability mask distinguishes true zero market movement from missing peer context.

No row is discarded merely because one or more peers are unavailable. This is necessary because the final causal alignment audit found QQQ complete coverage of only about 16-19% when the target is a non-US market, while BTC/ETH are nearly continuous and EUR/XAU have session-dependent availability.

## Causal alignment evidence frozen before V2.1

The final 15-minute cross-market audit reported `future=0` for every target-peer relationship. Hard-quality complete coverage showed the intended heterogeneity:

- BTC/ETH peers: approximately 99.8-100% complete in most target datasets.
- EUR/XAU peers: high but session-dependent; approximately 69-80% for several non-QQQ target relationships.
- QQQ peer: approximately 16-19% complete for EURUSD/XAUUSD/BTCUSD/ETHUSD targets.
- For QQQ as target, EURUSD/XAUUSD/BTCUSD/ETHUSD peers are approximately 99-100% complete.

Therefore V2.1 uses masks rather than requiring simultaneous availability of every market.

## Evaluation rule

V2.1 is evaluated against frozen V2 price-only using the same A/B/C decision timestamps and the same economic event construction.

Primary economic diagnostics:

- actionable trades / coverage
- net-positive and net-nonpositive selected trades
- selected win rate
- mean net expectancy after costs
- profit factor

Ranking diagnostics are also retained because a fixed `.55` / `EV>0` gate can hide score-ordering information:

- Spearman probability-long vs realized-long
- Spearman probability-short vs realized-short
- Spearman predicted-long-EV vs realized-long
- Spearman predicted-short-EV vs realized-short
- top-20% realized outcome for each of the four score heads

A/B/C results may reject V2.1 or motivate a *new preregistered version*, but they may not be used to tune V2.1 thresholds or choose favorable peer subsets after inspection.

External macro/news data is excluded from V2.1. Any such experiment belongs to a separately frozen V2.2 ablation.
