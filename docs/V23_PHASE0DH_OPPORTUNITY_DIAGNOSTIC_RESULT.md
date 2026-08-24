# V2.3 Phase 0D-H-OPPORTUNITY Diagnostic Result

Date: 2026-08-24
Status: exploratory diagnostic complete
Promotion decision changed: **NO**
Historical holdout opened: **NO**

## Official phase status remains

`FAIL_KEEP_HOLDOUT_SEALED`

The diagnostic does not alter the frozen Phase 0D-H-OPPORTUNITY result.

## Diagnostic finding

Across the 500 tested inner configurations per symbol (5 folds x 100 frozen configurations), no configuration survived the original profitability filters.

### BTCUSDT

- Best gross expectancy observed among the fold-level top configurations: **4.845045822689925 bps/trade**.
- All 500 tested configurations failed positive expectancy at 12 bps round-trip cost.
- All 500 failed positive expectancy at 15 bps round-trip cost.
- All 500 failed positive total PnL at 12 bps.
- All 500 failed positive total PnL at 15 bps.
- All 500 had profit factor <= 1 at 12 bps.
- 8 configurations additionally failed the median >=5 independent trades/day inner filter.

### ETHUSDT

- Best gross expectancy observed among the fold-level top configurations: **4.118044793379122 bps/trade**.
- All 500 tested configurations failed positive expectancy at 12 bps round-trip cost.
- All 500 failed positive expectancy at 15 bps round-trip cost.
- All 500 failed positive total PnL at 12 bps.
- All 500 failed positive total PnL at 15 bps.
- All 500 had profit factor <= 1 at 12 bps.

## Interpretation

This is not primarily an opportunity-count failure. It is an economic-edge failure for the tested information set and horizons.

The best observed gross expectancy is below 5 bps/trade for both symbols. Therefore even a hypothetical 5 bps round-trip execution cost would eliminate the best observed average gross edge. Reducing the frozen cost assumption from 12/15 bps merely to manufacture a PASS is forbidden.

The correct conclusion is:

- aggressive trade-flow-only features contain reproducible statistical information;
- under the tested 10/30/60/120/300-second horizons and frozen Ridge/gating family, that information is too small to support robust net profitability after realistic execution friction;
- the 2026-08-04 through 2026-08-23 historical holdout remains sealed;
- do not retune quantiles, alphas, or costs on the same phase;
- any next experiment must change the hypothesis materially, such as longer holding horizons and/or richer order-book information.
