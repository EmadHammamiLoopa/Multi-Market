from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .fetcher import TWELVE_INTERVAL_MAP, parse_twelve_time_series, resolve_symbol


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_api(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _chunk_id(start: datetime, end: datetime) -> str:
    return f"{_format_api(start)}__{_format_api(end)}"


def build_backfill_url(
    provider_symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    api_key: str,
) -> str:
    twelve_interval = TWELVE_INTERVAL_MAP.get(interval)
    if not twelve_interval:
        raise ValueError(f"unsupported Twelve Data interval: {interval}")
    params = urlencode(
        {
            "symbol": provider_symbol,
            "interval": twelve_interval,
            "start_date": _format_api(start),
            "end_date": _format_api(end),
            "timezone": "UTC",
            "order": "ASC",
            "apikey": api_key,
        }
    )
    return f"https://api.twelvedata.com/time_series?{params}"


def _get(url: str) -> dict:
    request = Request(
        url,
        headers={"User-Agent": "Multi-Market-V1/1.0", "Accept": "application/json"},
    )
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def _load_existing(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "timestamp": _parse_utc(row["timestamp"]).isoformat().replace("+00:00", "Z"),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
            )
    return rows


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["timestamp", "open", "high", "low", "close"]
        )
        writer.writeheader()
        writer.writerows(rows)


def _load_state(path: Path, *, data_exists: bool) -> set[str]:
    if not data_exists or not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    return {str(value) for value in payload.get("completed_chunks", [])}


def _write_state(path: Path, *, symbol: str, interval: str, completed: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "symbol": symbol,
                "interval": interval,
                "completed_chunks": sorted(completed),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def backfill_symbol(
    symbol: str,
    *,
    interval: str,
    start: datetime,
    end: datetime,
    chunk_days: int = 14,
    output_dir: str | Path = "data",
    state_dir: str | Path | None = None,
    api_key: str | None = None,
    getter=_get,
) -> Path:
    if start >= end:
        raise ValueError("start must be before end")
    if chunk_days <= 0:
        raise ValueError("chunk_days must be positive")
    key = (api_key or os.getenv("TWELVE_DATA_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("TWELVE_DATA_API_KEY is not set")

    provider_symbol, _ = resolve_symbol(symbol, "twelve-data")
    safe_symbol = symbol.upper().replace("/", "")
    output_dir = Path(output_dir)
    path = output_dir / f"{safe_symbol}_{interval}.csv"
    state_root = Path(state_dir) if state_dir is not None else output_dir / ".backfill_state"
    state_path = state_root / f"{safe_symbol}_{interval}.json"

    collected = _load_existing(path)
    by_timestamp = {str(row["timestamp"]): row for row in collected}
    completed = _load_state(state_path, data_exists=path.exists())

    cursor = start
    request_number = 0
    skipped = 0
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=chunk_days), end)
        chunk_key = _chunk_id(cursor, chunk_end)
        if chunk_key in completed:
            skipped += 1
            print(
                f"{symbol:8s} skip completed: "
                f"{_format_api(cursor)} .. {_format_api(chunk_end)}"
            )
            cursor = chunk_end
            continue

        request_number += 1
        payload = getter(build_backfill_url(provider_symbol, interval, cursor, chunk_end, key))
        rows = parse_twelve_time_series(payload)
        for row in rows:
            by_timestamp[str(row["timestamp"])] = row

        merged = sorted(by_timestamp.values(), key=lambda row: str(row["timestamp"]))
        _write(path, merged)
        completed.add(chunk_key)
        _write_state(state_path, symbol=symbol, interval=interval, completed=completed)

        print(
            f"{symbol:8s} chunk {request_number:3d}: "
            f"{_format_api(cursor)} .. {_format_api(chunk_end)} rows={len(rows):4d} "
            f"checkpoint={len(merged)}"
        )
        cursor = chunk_end

    merged = sorted(by_timestamp.values(), key=lambda row: str(row["timestamp"]))
    if not merged:
        raise RuntimeError(f"no rows available after backfill for {symbol}")
    _write(path, merged)
    _write_state(state_path, symbol=symbol, interval=interval, completed=completed)
    print(
        f"{symbol:8s} merged rows={len(merged)} "
        f"{merged[0]['timestamp']} .. {merged[-1]['timestamp']} -> {path} "
        f"(requested={request_number}, skipped={skipped})"
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill Twelve Data history in checkpointed date chunks and merge into replay CSVs"
    )
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--interval", default="5m", choices=sorted(TWELVE_INTERVAL_MAP))
    parser.add_argument("--start", required=True, help="UTC ISO date/datetime, e.g. 2025-08-01")
    parser.add_argument("--end", required=True, help="UTC ISO date/datetime, e.g. 2026-08-21")
    parser.add_argument("--chunk-days", type=int, default=14)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument(
        "--state-dir",
        default=None,
        help="checkpoint directory (default: <output-dir>/.backfill_state)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    start = _parse_utc(args.start)
    end = _parse_utc(args.end)
    failures = 0
    for symbol in args.symbols:
        try:
            backfill_symbol(
                symbol,
                interval=args.interval,
                start=start,
                end=end,
                chunk_days=args.chunk_days,
                output_dir=args.output_dir,
                state_dir=args.state_dir,
            )
        except Exception as exc:
            failures += 1
            print(f"ERROR {symbol}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
