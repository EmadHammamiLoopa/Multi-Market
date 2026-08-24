# V2.3 Phase 0D-I Longer-Horizon Flow Result

Date: 2026-08-24
Status: `FAIL_KEEP_HOLDOUT_SEALED`
Historical holdout opened: **NO**

## Official development result

Phase 0D-I tested BTCUSDT and ETHUSDT using the frozen longer-horizon trade-flow hypothesis with 10, 30 and 60 minute holding horizons. The frozen development gate was not met for either symbol.

### BTCUSDT

- Fold 1: no inner configuration survived.
- Fold 2: 8/60 inner configurations survived. Selected `(H=600s, alpha=0.1, gate=0.999)` had inner net expectancy `+4.968611 bps/trade` at 12 bps cost and `+1.968611 bps/trade` at 15 bps cost, but outer net expectancy was `-9.229765 bps/trade` at 12 bps.
- Fold 3: 8/60 inner configurations survived. Selected `(H=3600s, alpha=0.1, gate=0.999)` had inner net expectancy `+28.402994 bps/trade` at 12 bps and `+25.402994 bps/trade` at 15 bps, but outer net expectancy was `-23.770932 bps/trade` at 12 bps.
- Folds 4 and 5: no inner configuration survived.
- Pooled 12 bps: 38 trades, `-16.500348 bps/trade`, total `-627.013237 bps`, profit factor `0.504652`, 0 positive outer folds.
- Pooled 15 bps: `-19.500348 bps/trade`, total `-741.013237 bps`.

### ETHUSDT

No configuration survived inner selection in any of the five outer folds. No outer trades were scored.

## Interpretation

The longer-horizon extension does not rescue the `aggTrades`-only lineage. In BTCUSDT, configurations that appeared strongly profitable in inner validation failed sharply on immediately subsequent outer periods. This is evidence of temporal instability / selection overfit rather than a cost-only problem. ETHUSDT showed no viable inner configuration.

The correct conclusion is:

- short-horizon and longer-horizon `aggTrades`-only hypotheses are both rejected for economic deployment;
- do not retune alphas, gate quantiles, horizons, costs or feature windows on this same information lineage;
- the 2026-08-04 through 2026-08-23 historical holdout remains sealed;
- the next experiment must introduce a materially new information source rather than further optimize trade flow alone.
