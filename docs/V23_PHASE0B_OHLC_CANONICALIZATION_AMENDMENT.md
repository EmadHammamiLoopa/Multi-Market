# V2.3 Phase 0B OHLC Canonicalization Amendment

Date: 2026-08-23
Status: pre-scoring structural amendment

## Trigger

The frozen 17-sensor acquisition audit was run before any Phase 0B predictive scoring. Twelve sensors passed immediately. Five source CSVs (IWM, TLT, GLD, UUP, XLI) contained raw OHLC bars where the reported high/low did not contain the reported open/close.

A read-only full scan found 43 such rows in total across the affected files. No Phase 0B predictive output, PnL, target outcome, or reserved G/H/I outcome was inspected before this amendment.

## Frozen canonicalization rule

The rule is applied identically to all 17 sensor CSVs, not only the five files already observed to contain violations:

- canonical_high = max(raw_high, open, close)
- canonical_low = min(raw_low, open, close)
- open is unchanged
- close is unchanged
- timestamps and row ordering are unchanged

Raw source files are never modified. Canonical copies are written to a separate directory.

## Fail-closed guardrail

For each required high/low expansion, the absolute adjustment is measured in basis points relative to the relevant open/close boundary. Any required adjustment greater than 5.0 bps aborts canonicalization for the complete block.

This 5 bps guardrail is frozen before predictive scoring. It is not a model parameter and may not be changed in response to Phase 0B predictive results.

## Evidence

The canonicalization report records, per sensor:

- raw SHA-256;
- canonical SHA-256;
- row count;
- mutation count;
- every mutated CSV line and timestamp;
- open and close;
- high/low before and after;
- adjustment size in basis points.

The complete canonicalized 17-sensor block must pass the acquisition audit before Phase 0B predictive scoring is allowed.

## Research-integrity constraints

This amendment changes only structural OHLC validity. It does not alter the frozen sensor universe, lag ladder, models, folds, MIN_TRAIN_ROWS, promotion rules, reserved windows, or target outcomes. No sensor may be selected or removed based on predictive performance.
