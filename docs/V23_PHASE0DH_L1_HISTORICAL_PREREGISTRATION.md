# V2.3 Phase 0D-H-L1 Historical Audit — CLOSED INFEASIBLE

Frozen: 2026-08-24
Closed: 2026-08-24 before any predictive scoring.

## Intended question

Test whether historical BTCUSDT/ETHUSDT Binance USD-M Futures Level-1 microstructure (`bookTicker` plus `aggTrades`) contains short-horizon predictive information.

## Intended acquisition window

- 2026-05-26 through 2026-08-23 UTC
- BTCUSDT and ETHUSDT
- Binance USD-M perpetual futures
- `bookTicker` + `aggTrades`

## Acquisition probe outcome

A one-day probe for 2026-08-23 was executed before any model scoring.

- BTCUSDT `aggTrades`: available
- ETHUSDT `aggTrades`: available
- BTCUSDT `bookTicker`: HTTP 404
- ETHUSDT `bookTicker`: HTTP 404

The required 2026 public `bookTicker` history is not available from the frozen Binance archive source. Therefore the intended 2026 L1 representation cannot be reconstructed from the preregistered source.

## Formal decision

`PHASE0DH_L1_STATUS=INFEASIBLE_DATA_UNAVAILABLE`

No predictive model was fit and no predictive metric was observed. This is not a statistical rejection of L1 microstructure; it is an acquisition infeasibility result.

## Forbidden reinterpretation

Do not silently replace missing historical `bookTicker` with candles, forward-looking reconstruction, another venue, or later-collected live L1 data and continue under the same phase name.

A trade-flow-only historical audit must be separately named and preregistered. Full L1/L2 prospective confirmation remains a separate live-data experiment.
