from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


SYMBOL_MAP = {
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


@dataclass(frozen=True)
class FetchResult:
    requested_symbol: str
    provider_symbol: str
    description: str
    rows: int
    output_path: Path
    start: str
    end: str


def resolve_symbol(symbol: str) -> tuple[str, str]:
    key = symbol.strip().upper()
    return SYMBOL_MAP.get(key, (symbol, "Yahoo Finance symbol"))


def build_url(provider_symbol: str, interval: str, range_: str) -> str:
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


def _default_get(url: str) -> dict:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 Multi-Market-V0/1.0",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def parse_chart(payload: dict) -> list[dict[str, object]]:
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


def fetch_symbol(
    symbol: str,
    *,
    interval: str = "15m",
    range_: str = "60d",
    output_dir: str | Path = "data",
    getter: Callable[[str], dict] = _default_get,
) -> FetchResult:
    provider_symbol, description = resolve_symbol(symbol)
    rows = parse_chart(getter(build_url(provider_symbol, interval, range_)))

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

    return FetchResult(
        requested_symbol=symbol.upper(),
        provider_symbol=provider_symbol,
        description=description,
        rows=len(rows),
        output_path=output_path,
        start=str(rows[0]["timestamp"]),
        end=str(rows[-1]["timestamp"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch real historical OHLC data for Multi-Market V0"
    )
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument(
        "--interval",
        default="15m",
        choices=[
            "1m",
            "2m",
            "5m",
            "15m",
            "30m",
            "60m",
            "90m",
            "1h",
            "1d",
            "5d",
            "1wk",
            "1mo",
            "3mo",
        ],
    )
    parser.add_argument(
        "--range",
        dest="range_",
        default="60d",
        help="Yahoo range such as 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max",
    )
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args(argv)

    failures = 0
    for symbol in args.symbols:
        try:
            result = fetch_symbol(
                symbol,
                interval=args.interval,
                range_=args.range_,
                output_dir=args.output_dir,
            )
            print(
                f"{result.requested_symbol:8s} -> {result.provider_symbol:12s} "
                f"rows={result.rows:6d}  {result.start} .. {result.end}"
            )
            print(f"           {result.description}; saved {result.output_path}")
        except Exception as exc:
            failures += 1
            print(f"ERROR {symbol}: {exc}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
