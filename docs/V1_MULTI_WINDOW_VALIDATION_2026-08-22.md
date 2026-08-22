# V1 three-window validation freeze — 2026-08-22

This document freezes the paired V0/V1 historical validation evidence before any V1.1/V2 data-cleaning, target redesign, model tuning, news integration, or live/demo validation.

## Protocol

Unchanged V1 paired protocol in all windows:

- 5-minute bars
- horizon: 6 bars / 30 minutes
- confidence threshold: 0.60
- minimum training rows: 5,000
- retrain cadence: 500 decision bars
- validation fraction: 0.20
- TP: 20 bp
- SL: 10 bp
- round-trip cost: 2 bp
- expanding point-in-time walk-forward training
- scoring target: frozen V0 actual direction on exact shared timestamps

No threshold/model/feature tuning was performed between windows.

## Frozen windows

- Window A: original August 2026 frozen audit, 30 samples per market
- Window B: 2026-03-02T00:00:00Z through 2026-03-20T23:59:59Z, 30 deterministic valid dataset candles per market
- Window C: 2026-06-01T00:00:00Z through 2026-06-19T23:59:59Z, 30 deterministic valid dataset candles per market

Total: 450 paired historical decisions.

## Aggregate results by window

| Window | V0 actionable | V0 correct | V0 wrong | V0 coverage | V0 selected accuracy | V1 actionable | V1 correct | V1 wrong | V1 coverage | V1 selected accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A — August | 95 | 44 | 51 | 63.33% | 46.32% | 12 | 10 | 2 | 8.00% | 83.33% |
| B — March | 120 | 62 | 58 | 80.00% | 51.67% | 14 | 11 | 3 | 9.33% | 78.57% |
| C — June | 106 | 64 | 42 | 70.67% | 60.38% | 13 | 11 | 2 | 8.67% | 84.62% |
| **All 450** | **321** | **170** | **151** | **71.33%** | **52.96%** | **39** | **32** | **7** | **8.67%** | **82.05%** |

The raw aggregate makes V1 look strongly selective. It is not sufficient evidence of an investable edge.

## Aggregate paired transitions across all 450 decisions

| Transition | Count |
|---|---:|
| CORRECT -> CORRECT | 1 |
| CORRECT -> NO_TRADE | 164 |
| CORRECT -> WRONG | 5 |
| NO_TRADE -> CORRECT | 31 |
| NO_TRADE -> WRONG | 2 |
| NO_TRADE -> NO_TRADE | 96 |
| WRONG -> NO_TRADE | 151 |

Important structural result: every one of the 151 V0 wrong decisions became V1 NO_TRADE. However, V1 also abstained on 164 of 170 V0 correct decisions and converted five V0-correct decisions into V1 wrong decisions. Only one V0-correct action was retained as V1-correct. V1's 32 correct calls consist of one retained V0 correct plus 31 new actions that V0 had classified as NO_TRADE.

This is an aggressive replacement/filtering behavior, not simple error correction.

## Market-level consistency

### EURUSD

Across the three windows V1 is highly selective, but the March/June selected calls reveal a market-session/data artifact. In March, five of six V1 actions occur on Saturday/Sunday dataset timestamps. In June, all five V1 actions occur on Saturday/Sunday timestamps. These decisions often have tiny terminal moves and negative post-cost horizon returns.

The apparent directional accuracy therefore cannot be treated as a normal tradable EURUSD edge.

### XAUUSD

XAUUSD is the strongest artifact case. March reports 7/7 V1 selected calls correct; June reports 6/6 correct. Yet the selected timestamps cluster on weekends or late-Friday closure-like periods, with terminal moves commonly around only a few hundredths to a few tenths of a basis point. The simulated horizon return after the fixed 2 bp round-trip cost is negative on these calls.

The 100% directional accuracy is economically meaningless and strongly suggests that V1 learned a stale/closed-market pattern in the supplied XAUUSD bars.

### BTCUSD

At the frozen 30-minute / 0.60 protocol V1 makes zero actionable calls in all three windows. This is consistent selectivity but provides no evidence of a 30-minute BTC trading edge.

The separate full-history exploratory result at the 15-minute horizon / 0.75 threshold remains a candidate only; it is not validated by these 30-minute frozen audits.

### ETHUSD

V1 makes no actions in March or August and one action in June; the June action is wrong. No validated V1 ETH edge exists.

### QQQ

V1 makes one action per frozen window: August correct, March wrong, June wrong. Three actions are far too few for inference, and the observed 1/3 correctness does not support a stable edge.

## Data-quality finding

The supplied EURUSD/XAUUSD datasets contain bars at timestamps that should not be assumed to be normal liquid/tradable market observations. The present benchmark sampler correctly sampled valid rows from the dataset, but 'row exists in CSV' is not the same as 'instrument was in a normal tradable session.'

This distinction must become an explicit first-class rule before V2.

The original V0/V1 results must remain unchanged as historical evidence of this failure mode. Do not silently regenerate them after filtering.

## Frozen V1 verdict

V1 is rejected as a deployable multi-market trading model.

What V1 demonstrated:

1. point-in-time walk-forward ML infrastructure works;
2. the classifier can become highly selective;
3. classification confidence can increase while economic expectancy remains negative;
4. dataset/session artifacts can create extremely convincing but non-investable accuracy;
5. future-close direction is not a sufficient economic target;
6. BTC 15-minute / 0.75 is an exploratory candidate, not validated evidence.

No V1 tuning should be performed from this point. Any correction to session eligibility, target construction, calibration, execution assumptions, or model architecture belongs to a new version (V1.1 diagnostic or V2).
