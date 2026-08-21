from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


YAHOO_SYMBOL_MAP = {
    "EURUSD": ("EURUSD=X", "EUR/USD spot FX"),
    "GBPUSD": ("GBPUSD=X", "GBP/USD spot FX"),
    "USDJPY": ("JPY=X", "USD/JPY spot FX"),
    "XAUUSD": ("GC=F", "COMEX Gold futures proxy (not spot XAU/USD)"),
    "GOLD": ("GC=F", "COMEX Gold futures"),
    "BTCUSD": ("BTC-USD", "Bitcoin / U.S. dollar"),
    "ETHUSD": ("ETH-USD", "Ether / U.S. dollar"),
    "NDX": ("^NDX", "NASDAQ-100 index"),
    "NASDAQ": ("^NDX", "NASDAQ-100 index"),
    "SPX": ("^GSPC", "S&P 500 index"),
    "SP500": ("^GSPC", "S&P 500 index"),
}

TWELVE_SYMBOL_MAP = {
    "EURUSD": ("EUR/USD", "EUR/USD spot FX"),
    "GBPUSD": ("GBP/USD", "GBP/USD spot FX"),
    "USDJPY": ("USD/JPY", "USD/JPY spot FX"),
    "XAUUSD": ("XAU/USD", "Gold Spot / U.S. dollar"),
    "GOLD": ("XAU/USD", "Gold Spot / U.S. dollar"),
    "BTCUSD": ("BTC/USD", "Bitcoin / U.S. dollar"),
    "ETHUSD": ("ETH/USD", "Ether / U.S. dollar"),
    "NDX": ("NDX", "NASDAQ-100 index"),
    "NASDAQ": ("NDX", "NASDAQ-100 index"),
    "SPX": ("SPX", "S&P 500 index"),
    "SP500": ("SPX", "S&P 500 index"),
}

TWELVE_INTERVAL_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "45m": "45min",
    "60m": "1h",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1day",
    "1wk": "1week",
    "1mo": "1month",
}


@dataclass(frozen=True)
class FetchResult:
    requested_symbol: str
    provider: str
    provider_symbol: str
    description: str
    rows: int
    output_path: Path
    start: str
    end: str


def resolve_symbol(symbol: str, provider: str = "yahoo") -> tuple[str, str]:
    key = symbol.strip().upper()
    if provider == "twelve-data":
        return TWELVE_SYMBOL_MAP.get(key, (symbol, "Twelve Data symbol"))
    return YAHOO_SYMBOL_MAP.get(key, (symbol, "Yahoo Finance symbol"))


def build_yahoo_url(provider_symbol: str, interval: str, range_: str) -> str:
    params = urlencode(
        {
            "interval": interval,
            "range": range_,
            "includePrePost": "false",
            "events": "div,splits",
        }
    )
    encoded_symbol = quote(provider_symbol, safe="")
    return f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}?{params}"


def build_twelve_url(
    provider_symbol: str,
    interval: str,
    outputsize: int,
    api_key: str,
) -> str:
    twelve_interval = TWELVE_INTERVAL_MAP.get(interval)
    if not twelve_interval:
        raise ValueError(
            f"Interval {interval!r} is not supported by the Twelve Data adapter"
        )
    params = urlencode(
        {
            "symbol": provider_symbol,
            "interval": twelve_interval,
            "outputsize": outputsize,
            "timezone": "UTC",
            "order": "ASC",
            "apikey": api_key,
        }
    )
    return f"https://api.twelvedata.com/time_series?{params}"


def _default_get(url: str) -> dict:
    request = Request(
        url,
        headers={
            "User-Agent": "Multi-Market-V0/1.0",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def parse_yahoo_chart(payload: dict) -> list[dict[str, object]]:
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise RuntimeError(f"Yahoo Finance error: {chart['error']}")

    results = chart.get("result") or []
    if not results:
        raise RuntimeError("Yahoo Finance returned no chart result")

    result = results[0]
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quote_rows = indicators.get("quote") or []
    if not quote_rows:
        raise RuntimeError("Yahoo Finance response has no OHLC quote data")

    quote_data = quote_rows[0]
    rows: list[dict[str, object]] = []
    for index, timestamp in enumerate(timestamps):
        try:
            open_ = quote_data["open"][index]
            high = quote_data["high"][index]
            low = quote_data["low"][index]
            close = quote_data["close"][index]
        except (KeyError, IndexError):
            continue

        if any(value is None for value in (open_, high, low, close)):
            continue

        rows.append(
            {
                "timestamp": datetime.fromtimestamp(
                    int(timestamp), tz=timezone.utc
                ).isoformat().replace("+00:00", "Z"),
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
            }
        )

    if not rows:
        raise RuntimeError("Yahoo Finance returned no complete OHLC rows")
    return rows


def parse_twelve_time_series(payload: dict) -> list[dict[str, object]]:
    if payload.get("status") == "error" or payload.get("code"):
        message = payload.get("message") or payload.get("status") or "unknown error"
        raise RuntimeError(f"Twelve Data error: {message}")

    values = payload.get("values") or []
    if not values:
        raise RuntimeError("Twelve Data returned no time-series values")

    rows: list[dict[str, object]] = []
    for item in values:
        try:
            timestamp = datetime.fromisoformat(str(item["datetime"]).replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            timestamp = timestamp.astimezone(timezone.utc)
            rows.append(
                {
                    "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                    "open": float(item["open"]),
                    "high": float(item["high"]),
                    "low": float(item["low"]),
                    "close": float(item["close"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

    rows.sort(key=lambda row: str(row["timestamp"]))
    if not rows:
        raise RuntimeError("Twelve Data returned no complete OHLC rows")
    return rows


# Backward-compatible name used by the original V0 tests.
parse_chart = parse_yahoo_chart


def _write_rows(
    symbol: str,
    interval: str,
    output_dir: str | Path,
    rows: list[dict[str, object]],
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.upper().replace("/", "")
    output_path = output_dir / f"{safe_symbol}_{interval}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "open", "high", "low", "close"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def fetch_symbol(
    symbol: str,
    *,
    interval: str = "15m",
    range_: str = "60d",
    output_dir: str | Path = "data",
    provider: str = "yahoo",
    outputsize: int = 5000,
    api_key: str | None = None,
    getter: Callable[[str], dict] = _default_get,
) -> FetchResult:
    if provider not in {"yahoo", "twelve-data"}:
        raise ValueError(f"Unknown provider: {provider}")

    provider_symbol, description = resolve_symbol(symbol, provider)
    if provider == "twelve-data":
        key = api_key or os.getenv("TWELVE_DATA_API_KEY")
        if not key:
            raise RuntimeError(
                "TWELVE_DATA_API_KEY is not set. Export it in your shell before using Twelve Data."
            )
        rows = parse_twelve_time_series(
            getter(build_twelve_url(provider_symbol, interval, outputsize, key))
        )
    else:
        rows = parse_yahoo_chart(
            getter(build_yahoo_url(provider_symbol, interval, range_))
        )

    output_path = _write_rows(symbol, interval, output_dir, rows)
    return FetchResult(
        requested_symbol=symbol.upper(),
        provider=provider,
        provider_symbol=provider_symbol,
        description=description,
        rows=len(rows),
        output_path=output_path,
        start=str(rows[0]["timestamp"]),
        end=str(rows[-1]["timestamp"]),
    )


def fetch_symbol_auto(
    symbol: str,
    *,
    interval: str,
    range_: str,
    output_dir: str | Path,
    outputsize: int,
    getter: Callable[[str], dict] = _default_get,
) -> FetchResult:
    if os.getenv("TWELVE_DATA_API_KEY"):
        try:
            return fetch_symbol(
                symbol,
                interval=interval,
                output_dir=output_dir,
                provider="twelve-data",
                outputsize=outputsize,
                getter=getter,
            )
        except Exception as exc:
            print(f"WARN {symbol}: Twelve Data failed ({exc}); falling back to Yahoo")

    return fetch_symbol(
        symbol,
        interval=interval,
        range_=range_,
        output_dir=output_dir,
        provider="yahoo",
        getter=getter,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch real historical OHLC data for Multi-Market"
    )
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument(
        "--provider",
        choices=["auto", "twelve-data", "yahoo"],
        default="auto",
        help="auto prefers Twelve Data when TWELVE_DATA_API_KEY is set and falls back to Yahoo",
    )
    parser.add_argument(
        "--interval",
        default="15m",
        choices=["1m", "2m", "5m", "15m", "30m", "45m", "60m", "90m", "1h", "2h", "4h", "1d", "5d", "1wk", "1mo", "3mo"],
    )
    parser.add_argument(
        "--range",
        dest="range_",
        default="60d",
        help="Yahoo range such as 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max",
    )
    parser.add_argument(
        "--outputsize",
        type=int,
        default=5000,
        help="Twelve Data bars to request (1..5000)",
    )
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args(argv)

    if not 1 <= args.outputsize <= 5000:
        parser.error("--outputsize must be between 1 and 5000")

    failures = 0
    for symbol in args.symbols:
        try:
            if args.provider == "auto":
                result = fetch_symbol_auto(
                    symbol,
                    interval=args.interval,
                    range_=args.range_,
                    output_dir=args.output_dir,
                    outputsize=args.outputsize,
                )
            else:
                result = fetch_symbol(
                    symbol,
                    interval=args.interval,
                    range_=args.range_,
                    output_dir=args.output_dir,
                    provider=args.provider,
                    outputsize=args.outputsize,
                )
            print(
                f"{result.requested_symbol:8s} [{result.provider:11s}] -> "
                f"{result.provider_symbol:12s} rows={result.rows:6d}  "
                f"{result.start} .. {result.end}"
            )
            print(f"           {result.description}; saved {result.output_path}")
        except Exception as exc:
            failures += 1
            print(f"ERROR {symbol}: {exc}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
