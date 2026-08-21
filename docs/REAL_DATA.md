# Real historical data in V0

V0 supports two real historical OHLC providers and normalizes both into the exact CSV schema consumed by `multimarket-replay`.

## Preferred provider: Twelve Data

If `TWELVE_DATA_API_KEY` is exported in your shell, `multimarket-fetch` automatically prefers Twelve Data. The key is read from the environment and is never written to the repository.

```bash
export TWELVE_DATA_API_KEY="..."
```

Then fetch 5-minute data:

```bash
multimarket-fetch \
  --provider twelve-data \
  --symbols EURUSD XAUUSD BTCUSD ETHUSD NDX \
  --interval 5m \
  --outputsize 5000
```

Twelve Data mappings used by Multi-Market:

- `EURUSD` -> `EUR/USD`
- `GBPUSD` -> `GBP/USD`
- `USDJPY` -> `USD/JPY`
- `XAUUSD` / `GOLD` -> `XAU/USD` (Gold Spot / U.S. dollar)
- `BTCUSD` -> `BTC/USD`
- `ETHUSD` -> `ETH/USD`
- `NDX` / `NASDAQ` -> `NDX`
- `SPX` / `SP500` -> `SPX`

Provider availability can vary by instrument and plan. If a symbol is not available on the free plan, use `--provider auto` and the fetcher will try Twelve Data first and fall back to Yahoo Finance.

## Automatic mode

This is the recommended default once the key is configured:

```bash
multimarket-fetch \
  --provider auto \
  --symbols EURUSD XAUUSD BTCUSD ETHUSD NDX \
  --interval 5m \
  --outputsize 5000
```

`auto` behavior:

1. If `TWELVE_DATA_API_KEY` exists, try Twelve Data.
2. If Twelve Data fails for that symbol/request, print a warning.
3. Fall back to Yahoo Finance.
4. Save the normalized CSV in the same format regardless of provider.

## Yahoo Finance fallback

Yahoo requires no API key:

```bash
multimarket-fetch \
  --provider yahoo \
  --symbols EURUSD XAUUSD BTCUSD ETHUSD NDX \
  --interval 15m \
  --range 60d
```

Yahoo mappings:

- `EURUSD` -> `EURUSD=X`
- `XAUUSD` -> `GC=F` (COMEX Gold futures proxy; **not** spot XAU/USD)
- `BTCUSD` -> `BTC-USD`
- `ETHUSD` -> `ETH-USD`
- `NDX` / `NASDAQ` -> `^NDX`
- `SPX` / `SP500` -> `^GSPC`

## Output files

For a 5-minute fetch:

```text
data/EURUSD_5m.csv
data/XAUUSD_5m.csv
data/BTCUSD_5m.csv
data/ETHUSD_5m.csv
data/NDX_5m.csv
```

Each file contains:

```text
timestamp,open,high,low,close
```

Timestamps are normalized to UTC.

## Replay a fetched file

Example 30-minute forecast horizon from 5-minute candles (`6` future bars):

```bash
multimarket-replay data/EURUSD_5m.csv \
  --symbol EURUSD \
  --version-label V0 \
  --horizon-bars 6 \
  --lookback-bars 24 \
  --confidence-threshold 0.60 \
  --plot
```

The benchmark history stores a SHA-256 hash of the exact CSV, so later model versions can be compared on identical data.

## Security

Never commit the API key. `.env` is ignored by Git, but the simplest current setup is an exported shell environment variable:

```bash
export TWELVE_DATA_API_KEY="your-key"
```

To confirm that it is set without revealing it:

```bash
python -c 'import os; print("configured" if os.getenv("TWELVE_DATA_API_KEY") else "missing")'
```

## Research note

V0 remains a simple momentum control model. Adding better data does not turn V0 into a strong predictor; it gives us cleaner and faster intraday observations for the V1/XGBoost experiments. Historical accuracy is not evidence of guaranteed future profit.
