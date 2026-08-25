# V2.3 Phase 0D-L — Pre-Score Method Amendment

Date frozen: 2026-08-24
Status: **FROZEN BEFORE ANY PREDICTIVE SCORING**
Reason: correctness clarification discovered during raw-data audit / implementation review.

## Scope

This amendment changes no symbols, days, feature families, horizons, Ridge grid, threshold grid, latency scenarios, cost assumptions, execution semantics, inner-selection rules, promotion gates, or confirmation rules.

It clarifies only the causal ordering semantics used to reconstruct historical Tardis L2 state.

## Correct causal ordering

The original preregistration said historical Tardis L2 would be reconstructed in exchange-timestamp order. Implementation review against Tardis/HftBacktest semantics showed that this is not the safest causal rule: exchange timestamps can be non-monotone relative to message receipt, while `local_timestamp` records the capture/receipt order. Reordering a later-received message backward solely because of its exchange timestamp could make information appear available before it was actually observed by the feed capture.

Therefore, before any Phase 0D-L predictive metric exists, the historical reconstruction rule is frozen as follows:

1. Process Tardis rows in nondecreasing `local_timestamp` (capture/arrival) order.
2. All rows sharing one `local_timestamp` are one atomic feed message/group. Apply the full group before exposing the resulting book state to the decision sampler.
3. Preserve the original row order within an equal-`local_timestamp` group.
4. `timestamp` (exchange timestamp) is retained for latency/event-time diagnostics but does not reorder the causal feed stream.
5. A snapshot group resets the corresponding side/book state according to the Tardis snapshot semantics before subsequent incremental groups are applied.
6. Rows before the first usable snapshot are unavailable for state sampling.
7. No state sampled at decision time `t` may contain a row with `local_timestamp > t`.

## Decision / latency clock

For the historical mechanism screen, the 250 ms state grid and the frozen 250 ms primary reaction latency are measured on the causal `local_timestamp` clock.

- signal state at local time `t` uses only complete atomic groups with `local_timestamp <= t`;
- delayed entry uses the first valid sampled state at or after `t + 250 ms`;
- 100 ms and 500 ms remain diagnostics only;
- exchange timestamps remain available to report feed-latency characteristics but cannot alter promotion.

## Why this is not post-hoc tuning

No predictive score, target statistic, fold result, trade result, or PnL from Phase 0D-L had been produced when this clarification was frozen. The change is solely to prevent causal reordering / look-ahead risk discovered while reviewing the raw event semantics and external open-source converter behavior.
