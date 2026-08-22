# V2.2 macro-context preregistration

Frozen: 2026-08-23, before any V2.2 holdout outcomes are inspected.

## Research question

Does adding strictly point-in-time U.S. macroeconomic context to the frozen V2.1 price/regime/cross-market information set improve economic selection and score ranking without changing the frozen V2 decision policy?

V2.2 is an information-set experiment, not a threshold-tuning experiment.

## Frozen base policy

V2.2 inherits V2.1 unchanged:

- horizon: 6 five-minute bars
- causal EWMA span: 48; minimum observations: 24
- TP/SL: 1.0x / 1.0x horizon volatility
- round-trip cost: 2 bp
- minimum training rows: 5,000
- retrain every: 500 decision bars
- chronological validation fraction: 20%
- positive-probability gate: >= 0.55
- predicted-EV gate: > 0 bp
- embargo: 6 bars
- frozen XGBoost hyperparameters and random states
- frozen 62 V2.1 price/regime/cross-market features
- peer maximum staleness: 15 minutes
- hard target and peer quality gating

No V2/V2.1 parameter may be changed after V2.2 outcomes are inspected.

## External source policy

Only official/first-party macroeconomic data with auditable historical availability may enter V2.2.

The initial V2.2 macro layer uses six FRED/ALFRED series corresponding to BLS/BEA releases:

| Family | Series | Meaning | Unit used by feature |
|---|---|---|---|
| CPI | CPIAUCSL | CPI, all urban consumers | percent change |
| CPI | CPILFESL | Core CPI | percent change |
| Employment | PAYEMS | Total nonfarm payrolls | level change, thousands |
| Employment | UNRATE | Unemployment rate | percentage-point change |
| PCE | PCEPI | PCE price index | percent change |
| PCE | PCEPILFE | Core PCE price index | percent change |

No consensus forecast, vendor surprise field, revised current-history download, news sentiment, GDP, policy-rate feature, yield curve, or manually entered macro value is part of V2.2.

Those require a separately preregistered later version.

## Point-in-time contract

The model consumes a normalized macro ledger. Every ledger row means:

> this exact value for this observation was knowable no earlier than `known_at`.

Required columns:

- `series_id`
- `known_at` — exact timezone-aware timestamp normalized to UTC
- `observation_date`
- `value`

Rules:

1. A row is invisible when `known_at > decision_timestamp`.
2. For duplicate `(series_id, observation_date)` rows, the model may use only the latest row whose `known_at <= decision_timestamp`.
3. A later revision must never rewrite or replace earlier ledger history.
4. The normalized ledger must be append-only for research evidence.
5. `known_at` must come from an audited official release timestamp mapping; date-only observations are insufficient.
6. If an exact release timestamp cannot be established, the value is excluded rather than guessed.
7. Future mutation/revision of the ledger must not change features at an earlier decision timestamp.

## Frozen macro features

For each of the six series, V2.2 adds exactly four fields in deterministic series order:

1. `<SERIES>_available`
2. `<SERIES>_age_days`
3. `<SERIES>_latest_change`
4. `<SERIES>_previous_change`

Total macro fields: 6 x 4 = 24.

Total V2.2 feature count with the normal four-peer configuration: 62 + 24 = **86**.

### Availability

`available = 1` only when at least three distinct observation dates for the series are knowable at the decision timestamp. Three observations are required because both latest and previous one-period changes must be computable.

Otherwise the packet is masked atomically:

- available = 0
- age_days = 999.0 sentinel
- latest_change = 0.0
- previous_change = 0.0

Missing macro context never deletes a target decision row.

### Age

For an available packet:

`age_days = (decision_timestamp - latest_known_at).total_seconds() / 86400`

It must be >= 0. No clipping or normalization is applied.

### Change transforms

For CPIAUCSL, CPILFESL, PCEPI, PCEPILFE:

`change = (new_value / prior_value - 1) * 100`

The feature is percentage change, not basis points.

For PAYEMS:

`change = new_value - prior_value`

The feature remains in thousands of jobs.

For UNRATE:

`change = new_value - prior_value`

The feature remains in percentage points.

At each decision timestamp, the three observation values used to form latest/previous changes are the latest point-in-time values knowable for those observation dates at that timestamp. This deliberately allows an officially released revision to prior observations to affect features only from the revision's `known_at` onward.

No macro feature directly contains a future release date or time-to-next-release.

## Release-boundary semantics

A release with `known_at = T` is:

- unavailable for decisions strictly before T;
- available at T and afterward, subject to the three-observation requirement.

Five-minute target bars are timestamped at their existing repository timestamp semantics. V2.2 does not shift market bars to make macro releases fit.

## Development evidence already consumed

The following windows are fully consumed and may not be called untouched again:

- A: original August 2026 30-sample window
- B: 2026-03-02 through 2026-03-20
- C: 2026-06-01 through 2026-06-19

V2/V2.1 results from these windows remain baselines only.

## Frozen V2.2 evaluation windows

Before V2.2 outcome inspection, freeze three new UTC windows:

- **D:** 2025-11-03T00:00:00Z through 2025-11-21T23:59:59Z
- **E:** 2026-01-05T00:00:00Z through 2026-01-23T23:59:59Z
- **F:** 2026-04-06T00:00:00Z through 2026-04-24T23:59:59Z

For each market/window, select exactly 30 decision timestamps using the repository's deterministic evenly-spaced structural sampling rule. Timestamp selection may inspect timestamps and structural eligibility only; it may not inspect returns, labels, model predictions, macro values, or outcomes.

Total planned frozen V2.2 evaluation points: 5 markets x 3 windows x 30 = 450 before structural/unavailable accounting.

D/E/F are not to be opened for V2.2 outcome scoring until:

1. the macro ledger has passed causal audits;
2. V2.2 feature tests pass;
3. the V2.2 model implementation is frozen in a reviewed/green PR;
4. the exact D/E/F timestamp baseline JSON files are generated without inspecting outcomes for model tuning.

## Evaluation metrics

Primary economic metrics, unchanged from V2.1:

- frozen samples
- gate blocked
- unavailable
- actionable / coverage
- net-positive / net-nonpositive
- selected win rate
- mean realized net expectancy after frozen 2 bp cost
- profit factor

Ranking diagnostics, unchanged from V2.1:

- Spearman pLong vs realized long return
- Spearman pShort vs realized short return
- Spearman EVlong vs realized long return
- Spearman EVshort vs realized short return
- top-20% realized mean for each of the four score heads

Ranking diagnostics may expose scores by using the same diagnostic-only threshold relaxation as V2/V2.1; this never changes the economic decision policy.

## Decision rule for V2.2 research

V2.2 is not considered deployable merely because it beats V2.1 on one market/window.

Evidence must show materially improved economics and ranking stability across independent D/E/F periods. Sparse selected trades must be reported as sparse rather than converted into an accuracy claim.

There is no post-hoc lowering of 0.55, relaxing EV > 0, choosing only favorable markets, choosing only favorable macro series, or selecting only favorable D/E/F windows.

If V2.2 fails, reject it or preregister a materially new V2.3 before further outcome inspection.

## Explicit exclusions

V2.2 does not contain:

- news text or LLM sentiment
- FinBERT
- consensus-surprise data
- economic-calendar vendor scores
- FOMC statement embeddings
- order-book data
- options data
- alternative data
- threshold/model hyperparameter tuning from A/B/C

These exclusions prevent the first macro experiment from mixing multiple new information sources and make causal failure analysis possible.
