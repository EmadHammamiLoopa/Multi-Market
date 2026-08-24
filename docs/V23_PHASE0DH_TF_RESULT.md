# V2.3 Phase 0D-H-TF Development Result

Date: 2026-08-24
Status: `FAIL_KEEP_HOLDOUT_SEALED`
Historical holdout opened: **NO**

## Official development decision

The frozen Phase 0D-H-TF promotion gate was not met for BTCUSDT or ETHUSDT because pooled directional accuracy did not exceed 0.50. The result is therefore an official development FAIL under the preregistered rules. The 2026-08-04 through 2026-08-23 historical holdout remains sealed.

## BTCUSDT

T0 pooled:
- R2 = 0.0019213800625034194
- Spearman = 0.07024833650701952
- directional accuracy = 0.4475572633942522

T1 pooled:
- R2 = 0.003977640666763205
- delta R2 vs T0 = +0.0020562606042597853
- Spearman = 0.07604349843567808
- directional accuracy = 0.44255813145985484
- positive delta-R2 folds = 5/5
- frozen gate = FAIL

T2 pooled:
- R2 = 0.004413780437785397
- delta R2 vs T0 = +0.002492400375281978
- Spearman = 0.09179707099846467
- directional accuracy = 0.4469917475896712
- positive delta-R2 folds = 4/5
- frozen gate = FAIL

## ETHUSDT

T0 pooled:
- R2 = 0.0006513156254621677
- Spearman = 0.03634088221729988
- directional accuracy = 0.4559043507448003

T1 pooled:
- R2 = 0.0014337828447261725
- delta R2 vs T0 = +0.0007824672192640048
- Spearman = 0.04550640347011357
- directional accuracy = 0.46304193335571014
- positive delta-R2 folds = 5/5
- frozen gate = FAIL

T2 pooled:
- R2 = 0.0007588713585860996
- delta R2 vs T0 = +0.00010755573312393185
- Spearman = 0.055130736705729724
- directional accuracy = 0.4670716096251114
- positive delta-R2 folds = 3/5
- frozen gate = FAIL

## Interpretation

This result does **not** support overriding the frozen gate. It also does not imply that trade flow contains no predictive information. T1 improved R2 over T0 in all 5 development folds for both symbols and retained positive rank correlation, while raw zero-threshold directional accuracy remained below the frozen 0.50 requirement.

The correct conclusion is therefore:

- Phase 0D-H-TF as preregistered: FAIL.
- Historical holdout remains sealed.
- The development evidence motivates, but does not validate, a separately preregistered rank/strength-gated hypothesis.
- Any new hypothesis must be frozen before opening the untouched holdout and the holdout must not be repeatedly reused for redesign.
