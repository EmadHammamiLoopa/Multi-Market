# V1.1 clean-session validation freeze — 2026-08-22

This document freezes the session-clean V1.1 paired replay results before V2 target/model development.

## Protocol

V1.1 keeps the V1 model and frozen research parameters unchanged:

- 5-minute bars
- horizon: 6 bars / 30 minutes
- confidence threshold: 0.60
- minimum training rows: 5,000
- retrain cadence: 500 decision bars
- validation fraction: 0.20
- same V1 feature set and XGBoost hyperparameters

Only the V1.1 hard data gate changes eligibility:

- session must be structurally eligible;
- decision/future endpoint must not be zero-range;
- decision/future endpoint must not be repeated-full-OHLC;
- tiny-range alone remains eligible.

Original V0/V1 outputs remain unchanged.

## Frozen three-window results

| Window | Market | Gate-blocked | Actionable | Correct | Wrong | NO_TRADE | Selected accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| A | EURUSD | 7 | 0 | 0 | 0 | 30 | n/a |
| A | XAUUSD | 7 | 1 | 1 | 0 | 29 | 100.00% |
| A | BTCUSD | 1 | 0 | 0 | 0 | 30 | n/a |
| A | ETHUSD | 0 | 0 | 0 | 0 | 30 | n/a |
| A | QQQ | 0 | 1 | 1 | 0 | 29 | 100.00% |
| B | EURUSD | 7 | 0 | 0 | 0 | 30 | n/a |
| B | XAUUSD | 7 | 0 | 0 | 0 | 30 | n/a |
| B | BTCUSD | 0 | 0 | 0 | 0 | 30 | n/a |
| B | ETHUSD | 0 | 0 | 0 | 0 | 30 | n/a |
| B | QQQ | 0 | 1 | 0 | 1 | 29 | 0.00% |
| C | EURUSD | 7 | 0 | 0 | 0 | 30 | n/a |
| C | XAUUSD | 7 | 0 | 0 | 0 | 30 | n/a |
| C | BTCUSD | 0 | 0 | 0 | 0 | 30 | n/a |
| C | ETHUSD | 0 | 1 | 0 | 1 | 29 | 0.00% |
| C | QQQ | 0 | 1 | 0 | 1 | 29 | 0.00% |

Aggregate across all 450 frozen decisions:

- gate-blocked: 43
- actionable: 5
- correct: 2
- wrong: 3
- NO_TRADE: 445
- coverage: 1.11%
- selected accuracy: 40.00%

For comparison, frozen V1 across the same 450 decisions was:

- actionable: 39
- correct: 32
- wrong: 7
- coverage: 8.67%
- selected accuracy: 82.05%

## Interpretation

The apparent V1 high selected accuracy does not survive session/data-quality correction.

March and June EURUSD/XAUUSD fall to zero actionable trades under the clean-session protocol. This directly supports the prior diagnosis that much of the spectacular V1 EUR/XAU accuracy was learned from closure/stale-like observations rather than a normal tradable intraday edge.

BTCUSD still produces no actionable 30-minute / 0.60 calls, even though its session structure is 24/7. ETHUSD produces one June action and it is wrong. QQQ produces one action in each frozen window: August correct, March wrong, June wrong.

Therefore V1.1 is not a deployable trading model and should not be tuned further.

## Frozen verdict

V1/V1.1 have now served their intended research purpose:

1. prove the point-in-time walk-forward infrastructure;
2. expose the difference between directional accuracy and economic value;
3. expose a market-session/data-quality artifact;
4. show that correcting the artifact removes most apparent edge;
5. justify replacing the future-close sign target rather than tuning the old classifier.

V2 must begin with a new economically aligned target and stricter horizon integrity.

## V2 entry requirements

Before any V2 signal is considered valid:

- the full prediction horizon must be contiguous at the expected bar interval for that instrument/session;
- all bars used for the decision and outcome must be structurally eligible;
- labels must represent trading outcomes or net economic value after costs, not only future-close direction;
- same-bar TP/SL ambiguity remains conservative;
- all tuning occurs on development history only, with untouched temporal validation kept separate;
- later macro/news features must be point-in-time with publication-time provenance and must be evaluated by ablation against price-only V2.
