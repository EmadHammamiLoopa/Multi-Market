# V2 price-only ranking diagnostic — 2026-08-22

This document freezes the ranking-only diagnostic run after the V2 price-only baseline was rejected. No trading threshold or model hyperparameter was changed before or during this diagnostic.

## Aggregate availability

Across 450 frozen A/B/C decision timestamps:

- scored eligible rows: 324
- blocked feature history: 102
- blocked future event: 12
- unavailable training: 12
- unavailable data/other: 0

QQQ Window B produced no scoreable rows: 18 feature-history blocks, 2 future-event blocks, and 10 rows without the minimum 5,000 clean training events.

## Main conclusion

The existing generic single-market price-only feature set does not show a stable cross-market ranking edge. Spearman correlations are generally small, unstable in sign, or materially negative in some windows. Top-20% realized outcomes are likewise inconsistent for most market/side/head combinations.

### EURUSD

Probability top-20% realized net returns were negative for both LONG and SHORT in all A/B/C windows. Mean Spearman across windows was approximately -0.064 for LONG probability and -0.161 for SHORT probability.

### XAUUSD

Probability top-20% realized net returns were negative for both LONG and SHORT in all A/B/C windows. EV-SHORT had positive top-20% means in B/C but Spearman changed sign and was negative in B/C, so this is not accepted as stable ranking evidence.

### ETHUSD

Ranking was weak in A/B and deteriorated sharply in C. In Window C, probability Spearman was about -0.40 on both sides and top-20% probability realized means were strongly negative. Generic ETH price-only ranking is rejected.

### QQQ

Only 9 rows in A, 0 in B, and 7 in C were scored. A looked positive while C reversed sharply. Sample availability is insufficient and behavior is unstable; no QQQ edge is claimed.

### BTCUSD exploratory observation

BTCUSD LONG probability is the only combination whose top-20% realized mean was positive in every A/B/C window:

- A: +6.9718 bp
- B: +4.2047 bp
- C: +9.3357 bp

However, LONG-probability Spearman was only +0.2031, -0.0073, +0.1551. Because this pattern was identified after inspecting all markets, sides, heads, and windows, it is explicitly **exploratory** and must not be treated as a confirmed edge or used to tune the existing A/B/C evidence.

## Frozen interpretation

1. V2 price-only baseline remains rejected for deployment.
2. Lowering the 0.55 probability gate or changing the EV gate based on A/B/C is prohibited.
3. BTCUSD LONG probability is registered only as a hypothesis for a future pre-declared holdout.
4. Next research should enrich the information set rather than tune the rejected gate: causal regime/cross-market features first, then point-in-time macro/news as a separate ablation.
5. Any external news/macro layer must preserve publication/revision availability time and provenance; no current-value backfill is allowed.
