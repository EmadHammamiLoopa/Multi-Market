# V2.3 Phase 0C-R Preregistration — Asynchronous Sensor Feasibility Repair

Date frozen: 2026-08-24

## Status of Phase 0C

Phase 0C is closed as **INCONCLUSIVE** because the frozen synchronous five-minute sensor-alignment rule produced too few eligible rows to satisfy the preregistered four-scored-fold statistical gate.

Observed first-run feasibility counts:

- EURUSD: 1,753 eligible rows
- XAUUSD: 1,757 eligible rows
- BTCUSD: 8,684 eligible rows
- ETHUSD: 8,684 eligible rows
- QQQ: unavailable because inherited Phase 0B evidence contained no scored folds

Official repaired summary classification:

- `PHASE0C_PROMOTION=INCONCLUSIVE`
- `decision=REQUIRES_FEASIBILITY_REPAIR`

This result is not interpreted as evidence against the asset-specific/regime hypothesis.

## Repair question

Does Phase 0C become evaluable, and does any target satisfy the unchanged Phase 0C statistical promotion rule, when asynchronous linked markets are represented causally rather than requiring every linked market to have a valid packet within the same five-minute bar?

## What is allowed to change

Only linked-sensor alignment semantics change.

For each linked sensor, Phase 0C-R uses the most recent **fully valid causal sensor packet** available at or before the target decision timestamp. The model receives both the sensor-return packet and explicit staleness metadata.

No target, sensor membership, horizon, model family, promotion threshold, cost rule, holdout rule, or MIN_TRAIN_ROWS value may be changed in Phase 0C-R.

## Frozen targets

- EURUSD
- XAUUSD
- BTCUSD
- ETHUSD
- QQQ

QQQ remains structurally unavailable if its inherited Phase 0B evidence has no scored fold. Phase 0C-R does not invent new QQQ fold boundaries.

## Frozen linked sensor sets

Identical to Phase 0C:

- EURUSD: UUP, TLT, HYG
- XAUUSD: UUP, TLT, HYG
- BTCUSD: ETHUSD, QQQ, HYG
- ETHUSD: BTCUSD, QQQ, HYG
- QQQ: TLT, HYG, XLP, UUP

No sensor may be added, removed, swapped, or selected after scoring.

## Frozen target features

Identical to Phase 0C:

- returns: 1, 3, 6, 12, 24 bars
- realized volatility: 6, 24, 72 bars
- one-bar high/low range
- 24-bar price z-score

All target feature windows must be causal, contiguous under the target's structural session policy, and must not touch G/H/I.

## Frozen regime features

Identical to Phase 0C:

- causal 24-bar volatility percentile using the previous 120 valid values
- trend strength
- recent-jump strength
- UTC intraday sine/cosine encoding

## Asynchronous linked-sensor packet

For each linked sensor, a packet is considered valid only when all three frozen sensor returns are computable at its source bar:

- 1-bar return
- 6-bar return
- 24-bar return

The source packet itself must obey the sensor's structural eligibility rules, contiguous-bar rules, and strict G/H/I exclusion.

At target decision time `t`, select the latest valid sensor packet whose source timestamp is `<= t`.

The linked representation for each sensor is frozen as:

1. sensor return 1
2. sensor return 6
3. sensor return 24
4. `log1p(age_hours)` where age is target decision time minus packet source time
5. `fresh_5m`, equal to 1 only when age is at most one five-minute bar, otherwise 0

There is no maximum staleness cutoff in Phase 0C-R. Staleness is exposed to the model rather than causing row deletion.

A target row is unavailable only when a linked sensor has never yet produced any valid historical packet at or before the decision time.

## Critical causality rules

Phase 0C-R forbids:

- backward fill from a future sensor bar;
- interpolation using bars on both sides of the target timestamp;
- selecting the nearest bar when the nearest bar is in the future;
- constructing a sensor packet whose own return window touches G/H/I;
- constructing a target feature or label whose window touches G/H/I;
- computing age from a future packet;
- using G/H/I to choose staleness transformations or thresholds.

A packet from before a reserved holdout may remain the last known packet after the holdout. This is allowed because its source information predates the reserved interval and its age is explicitly represented.

## Frozen experiments

Unchanged:

- C0: own features -> Ridge
- C1: own + linked features -> Ridge
- C2: own + linked + regime -> Ridge
- C3: own + linked + regime -> HistGradientBoostingRegressor

## Frozen Ridge selection

Unchanged:

- alpha grid: 0.1, 1, 10, 100
- chronological inner split on training data only
- StandardScaler fit on training data only

## Frozen HGBR

Unchanged:

- learning_rate = 0.05
- max_iter = 200
- max_leaf_nodes = 15
- min_samples_leaf = 50
- l2_regularization = 1.0
- random_state = 0

## Frozen chronological evaluation

Unchanged:

- Phase 0B scored-fold evaluation starts are inherited exactly
- train rows must be strictly earlier than evaluation start
- completed-label purge: `label_end_timestamp < eval_start`
- MIN_TRAIN_ROWS = 5000
- no random shuffle
- no random CV

## Frozen primary horizon

- six bars / 30 minutes

No alternate horizon is tested in Phase 0C-R.

## Frozen statistical promotion rule

For C2 or C3 to be a statistical signal candidate for a target, all conditions remain unchanged:

1. pooled delta R2 versus C0 > 0;
2. pooled non-jump delta R2 versus C0 > 0;
3. at least four scored folds exist;
4. at least three of those folds have positive delta R2 versus C0.

C2 is preferred over C3 if both satisfy the full promotion gate after economic evaluation.

## Economic evaluation

Unchanged from Phase 0C and not run unless an explicit target cost model is supplied:

- confidence threshold = 75th percentile of absolute training predictions;
- volatility percentile >= 60%;
- predicted absolute edge > 1.5 x round-trip cost;
- non-overlapping positions;
- cost stress at 1.0x, 1.5x, and 2.0x;
- no missing cost is ever treated as zero cost.

The first Phase 0C-R run should omit cost inputs and remain statistical-only.

## Feasibility interpretation

If a target still has fewer than four scored folds, classify it as **INCONCLUSIVE_INSUFFICIENT_FOLDS**, not as predictive failure.

If the asynchronous repair restores sufficient folds but C2 and C3 fail the frozen statistical gate, that target is a legitimate Phase 0C-R statistical FAIL.

If at least one target becomes a statistical signal candidate, only those targets proceed to a separately frozen cost-model run.

## Forbidden post-hoc actions

After the first Phase 0C-R scoring begins, do not:

- change the sensor sets;
- add or remove age/freshness fields;
- cap age based on observed outcomes;
- change the staleness transformation;
- lower MIN_TRAIN_ROWS;
- reduce the four-fold requirement;
- alter the primary horizon;
- tune Ridge/HGBR settings outside the frozen procedure;
- choose favorable sessions, folds, or regimes;
- inspect or use G/H/I;
- add transaction costs merely to rescue a statistical result.

Any such change requires a new preregistered phase.
