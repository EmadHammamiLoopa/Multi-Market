# V2 research plan

V2 must solve the failure modes demonstrated by V0/V1 rather than merely increase model complexity.

## 1. First gate: market-data/session integrity

Before any new predictive model is trained:

- define explicit tradable-session eligibility per instrument;
- distinguish 24/7 crypto from session-bound FX/metals/equities;
- reject or flag stale/closure-like bars;
- detect repeated/near-static OHLC sequences;
- ensure the prediction horizon itself remains in a tradable interval;
- preserve the original V0/V1 benchmarks unchanged as evidence of the failure mode;
- create a new diagnostic benchmark version rather than rewriting historical results.

A row existing in the source CSV is not sufficient evidence that it represents a normal executable market state.

## 2. Replace the V1 target

V1 predicts future-close sign. That target produced very high directional accuracy on economically tiny moves while still losing after costs.

V2 should evaluate targets that are directly aligned with trading value. Candidate design:

### Primary event target

For each decision timestamp and side, estimate:

- P(long TP before long SL)
- P(short TP before short SL)
- probability of neither barrier before horizon

Use point-in-time OHLC path rules and conservative same-bar ambiguity treatment.

### Expected-value layer

Compute an estimated net value after transaction costs/slippage assumptions:

`EV_long = p_win_long * net_TP - p_loss_long * net_SL + p_timeout_long * expected_timeout_return`

and analogously for short.

Emit LONG/SHORT only when:

- estimated EV is positive by a pre-frozen safety margin;
- probability/calibration criteria pass;
- tradability/session criteria pass;
- deterministic risk rules permit the trade.

Otherwise emit NO_TRADE.

### Alternative/diagnostic labels

Also test a three-class economically meaningful label:

- LONG if future net move exceeds +economic threshold
- SHORT if future net move exceeds -economic threshold
- NO_TRADE otherwise

The threshold must be derived from costs/volatility rules and frozen before untouched validation; it must not be selected by maximizing the final benchmark.

## 3. Validation controls

V2 validation must be stricter than V1.

Required:

- expanding/purged point-in-time walk-forward evaluation;
- embargo or equivalent horizon separation where folds could overlap;
- no future-dependent feature normalization;
- exact tracking of every parameter/model/threshold trial;
- untouched temporal windows after exploratory development;
- multi-market and multi-regime reporting;
- economic metrics after costs, not directional accuracy alone;
- coverage, PF, expectancy, net return, drawdown, calibration, and sample counts;
- uncertainty intervals where sample sizes are small;
- Probability of Backtest Overfitting / CSCV-style diagnostics when a strategy family has many tested variants;
- Deflated Sharpe Ratio or an equivalent multiple-testing-aware performance statistic before treating an optimized Sharpe result as evidence.

Relevant literature:

- Bailey, Borwein, López de Prado & Zhu, "The Probability of Backtest Overfitting", Journal of Computational Finance. DOI: 10.21314/JCF.2016.322.
- Bailey & López de Prado, "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality", Journal of Portfolio Management 40(5), 94-107. DOI: 10.3905/jpm.2014.40.5.094.

These references motivate explicit control of multiple testing and backtest selection bias; they do not by themselves validate any trading strategy.

## 4. Feature architecture

Use modular causal feature families so ablation is possible.

### A. Market micro/price features

Retain causal technical features but re-evaluate them under session-clean data. Add only features with an economic hypothesis, e.g.:

- multi-horizon returns
- realized volatility
- ATR/range
- trend/momentum
- intraday/session position
- time since session open
- gap/overnight state where relevant
- cross-market returns/volatility with strict timestamp alignment

### B. Regime features

Create a separate regime layer for:

- volatility level
- trend/range state
- risk-on/risk-off proxy state
- session/liquidity state

Regime detection must be causal and should gate or contextualize prediction rather than provide hindsight labels.

### C. News and macro features — later V2 intelligence layer

News is not part of V0/V1 and should be added only after the clean price-only V2 core exists.

Requirements:

- preserve original publication timestamp and revision timestamp where available;
- only expose content released at/before the decision timestamp;
- distinguish scheduled macro events from unscheduled news;
- model time-since-release and surprise, not merely headline existence;
- avoid retrospective article timestamps or edited summaries that introduce leakage;
- store source provenance.

Candidate features:

- economic calendar event type
- actual/forecast/previous and standardized surprise
- central-bank event flags
- earnings/event flags for equity instruments
- headline/document sentiment
- topic/event embeddings
- source/reliability metadata
- decay since release

For financial text sentiment, FinBERT is a literature-supported baseline for domain-specific sentiment representation:

- Araci, "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models", arXiv:1908.10063.

FinBERT is a feature candidate, not a trading oracle. Any sentiment feature must pass point-in-time ablation tests after costs.

### D. Order-book features — only if genuine LOB data becomes available

Do not imitate LOB features from OHLC candles. If true order-book data is later acquired, architectures such as DeepLOB are relevant background:

- Zhang, Zohren & Roberts, "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books", arXiv:1808.03668.

Until real bid/ask depth is available, this remains out of scope.

## 5. Model architecture

Do not jump immediately to a large deep model.

Recommended sequence:

1. calibrated gradient-boosted baseline on economically aligned labels;
2. separate long and short probability heads or multiclass classifier;
3. regime-conditioned ensemble only if baseline evidence supports it;
4. probability calibration evaluated out of sample;
5. optional neural/time-series models only when they add stable untouched-test value.

Model complexity must earn its place through ablation.

## 6. News/research distinction

Research papers and journals should primarily inform methodology, architecture, validation, and feature hypotheses. They are not intraday predictive inputs.

News, macro releases, and event data can be actual point-in-time predictive inputs because they arrive through time. They therefore require strict timestamp provenance and leakage controls.

## 7. Deterministic risk engine

The predictive model must not have unrestricted control of position size or capital.

Risk layer remains deterministic and separate:

- maximum position/risk per trade
- maximum daily loss
- maximum concurrent exposure
- market-specific exposure limits
- cooldown/circuit breaker
- stale-data rejection
- spread/slippage guard
- NO_TRADE on invalid/incomplete inputs

## 8. Evaluation progression

V2 should progress through:

1. data/session integrity audit;
2. clean historical replay diagnostics;
3. exploratory development period;
4. frozen model/policy selection;
5. untouched temporal windows;
6. live/demo forward test;
7. only then very small real capital if evidence remains positive;
8. gradual scale only after additional forward evidence.

No profit guarantee is implied at any stage.

## Immediate implementation order

1. Build `market_calendar.py` / tradability policy and a dataset-quality audit CLI.
2. Produce a **new** session-clean diagnostic benchmark without overwriting V0/V1 evidence.
3. Re-measure how much of the V1 apparent accuracy disappears after excluding invalid/closure-like intervals.
4. Implement economically aligned V2 labels/barrier outcomes.
5. Build V2 price-only baseline.
6. Freeze development policy and validate on untouched periods.
7. Only after price-only V2 is credible, add macro/news ingestion with timestamp provenance and ablation.
