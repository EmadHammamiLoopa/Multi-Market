# Real historical data in V0

V0 can fetch real OHLC candles from Yahoo Finance's public chart endpoint and normalize them into the exact CSV schema consumed by `multimarket-replay`.

## Fetch

```bash
multimarket-fetch --symbols EURUSD XAUUSD BTCUSD ETHUSD NDX --interval 15m --range 60d
```

Files are written to `data/` by default:

```text
data/EURUSD_15m.csv
data/XAUUSD_15m.csv
data/BTCUSD_15m.csv
data/ETHUSD_15m.csv
data/NDX_15m.csv
```

## Symbol mapping

- `EURUSD` -> `EURUSD=X` (EUR/USD FX)
- `XAUUSD` -> `GC=F` (COMEX Gold futures proxy; this is **not** spot XAU/USD)
- `BTCUSD` -> `BTC-USD`
- `ETHUSD` -> `ETH-USD`
- `NDX` / `NASDAQ` -> `^NDX`
- `SPX` / `SP500` -> `^GSPC`

Unknown inputs are passed through as raw Yahoo Finance symbols.

## Important provider limits

Yahoo Finance restricts how much intraday history is available for short intervals. If an interval/range combination is unsupported, the fetcher reports the provider error instead of silently changing the request.

For the first V0 intraday benchmark, use:

```bash
--interval 15m --range 60d
```

For longer historical comparisons, use a larger interval such as `1h` or `1d` and an appropriate range.

## Replay a fetched file

```bash
multimarket-replay data/EURUSD_15m.csv --symbol EURUSD --version-label V0 --plot
```

Repeat for every market. The generated benchmark history stores a SHA-256 hash of each exact CSV, so later V1/V2 comparisons can be restricted to identical datasets.

## Data-quality note

This zero-key connector is for establishing a reproducible V0 baseline. For research-grade later versions, especially news/macro point-in-time work and production trading, we should add explicit licensed/provider APIs and preserve source timestamps/revisions.
