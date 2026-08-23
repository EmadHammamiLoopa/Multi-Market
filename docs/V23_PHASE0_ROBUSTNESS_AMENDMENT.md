# V2.3 Phase 0 Robustness Amendment

Frozen: 2026-08-23, after the first core-universe execution exposed numerical/reporting issues and before any Phase 0 promotion decision was finalized.

## Observed implementation issues

1. QQQ produced 5,048 eligible development rows. Under the preregistered five chronological folds and `MIN_TRAIN_ROWS=5000`, no QQQ fold can be scored. The original CLI attempted to format missing pooled metrics as floats and crashed.
2. ElasticNet emitted convergence warnings for XAUUSD, BTCUSD, and ETHUSD with `max_iter=10000`.

## Permitted corrections

These are implementation/numerical corrections only. They do not change the research question, feature set, labels, reserved G/H/I windows, alpha, l1 ratio, promotion thresholds, or market selection.

- QQQ with zero scored folds is recorded as structurally unavailable under the preregistered minimum-training rule. We do **not** lower `MIN_TRAIN_ROWS`, change the folds, or manufacture a QQQ result after observing the core-market results.
- Missing pooled metrics are printed and serialized as `null`/`n/a` rather than causing a CLI crash.
- ElasticNet keeps `alpha=0.0005` and `l1_ratio=0.25`; only the numerical iteration ceiling is increased from 10,000 to 50,000 so the solver has more opportunity to converge to the same frozen objective. No tolerance or regularization parameter is changed.
- The promotion summary pools only scored targets. An unavailable target is never counted as positive. The preregistered condition `at least 3 of the 5 target markets positive` still uses denominator/requirement 5 and therefore cannot be relaxed because QQQ is unavailable.

## Interpretation constraint

The first run already showed negative six-bar incremental R-squared for both Ridge and ElasticNet on EURUSD, XAUUSD, BTCUSD, and ETHUSD. Those observations are consumed development evidence. This amendment cannot be used to reinterpret them favorably or to select a subset of markets.

A full clean rerun after this amendment is required before freezing the Phase 0 summary. If the frozen promotion rule fails, the core dense cross-sectional block is rejected and no favorable market/lag/model subset may be promoted.
