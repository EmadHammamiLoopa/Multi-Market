from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import httpx


BASE_URL = "https://data.binance.vision/data/futures/um/daily"
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT")
DEFAULT_STREAMS = ("bookTicker", "aggTrades")
DEFAULT_START = date(2026, 5, 26)
DEFAULT_END = date(2026, 8, 23)


def _date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _days(start: date, end: date) -> Iterable[date]:
    if end < start:
        raise ValueError("end date precedes start date")
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def archive_url(stream: str, symbol: str, day: date) -> str:
    stamp = day.isoformat()
    name = f"{symbol}-{stream}-{stamp}.zip"
    return f"{BASE_URL}/{stream}/{symbol}/{name}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_zip(path: Path) -> tuple[int, list[str]]:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"corrupt zip member: {bad}")
        members = [info.filename for info in archive.infolist() if not info.is_dir()]
        if not members:
            raise ValueError("zip archive contains no files")
        return len(members), members


def download_one(client: httpx.Client, url: str, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        members_count, members = validate_zip(destination)
        return {
            "status": "EXISTS_VALID",
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "members_count": members_count,
            "members": members,
        }

    tmp = destination.with_suffix(destination.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    try:
        with client.stream("GET", url) as response:
            if response.status_code == 404:
                return {"status": "MISSING_404"}
            response.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    handle.write(chunk)
        tmp.replace(destination)
        members_count, members = validate_zip(destination)
        return {
            "status": "DOWNLOADED",
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "members_count": members_count,
            "members": members,
        }
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download frozen Phase 0D-H-L1 Binance USD-M historical archives"
    )
    parser.add_argument("--output-dir", default="data/v23_phase0dh_l1_raw")
    parser.add_argument("--start", type=_date, default=DEFAULT_START)
    parser.add_argument("--end", type=_date, default=DEFAULT_END)
    parser.add_argument("--symbol", action="append", dest="symbols", default=None)
    parser.add_argument("--stream", action="append", dest="streams", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Record missing archives but do not fail acquisition; not allowed for official scoring.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    symbols = tuple(dict.fromkeys(s.upper() for s in (args.symbols or DEFAULT_SYMBOLS)))
    streams = tuple(dict.fromkeys(args.streams or DEFAULT_STREAMS))
    unknown_symbols = sorted(set(symbols) - set(DEFAULT_SYMBOLS))
    unknown_streams = sorted(set(streams) - set(DEFAULT_STREAMS))
    if unknown_symbols:
        raise SystemExit(f"frozen symbols only: {DEFAULT_SYMBOLS}; got {unknown_symbols}")
    if unknown_streams:
        raise SystemExit(f"frozen streams only: {DEFAULT_STREAMS}; got {unknown_streams}")

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    missing: list[str] = []

    timeout = httpx.Timeout(args.timeout_seconds, connect=min(20.0, args.timeout_seconds))
    headers = {"User-Agent": "Multi-Market-V23-Phase0DH-L1/0.2.27"}
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        for day in _days(args.start, args.end):
            for symbol in symbols:
                for stream in streams:
                    url = archive_url(stream, symbol, day)
                    filename = url.rsplit("/", 1)[-1]
                    destination = root / stream / symbol / filename
                    print(f"[{day}] {symbol} {stream}", flush=True)
                    result = download_one(client, url, destination)
                    record = {
                        "date": day.isoformat(),
                        "symbol": symbol,
                        "stream": stream,
                        "url": url,
                        "path": str(destination),
                        **result,
                    }
                    records.append(record)
                    if result["status"] == "MISSING_404":
                        missing.append(f"{day}:{symbol}:{stream}")
                        print("  MISSING_404", flush=True)
                    else:
                        print(
                            f"  {result['status']} bytes={result['bytes']} sha256={result['sha256']}",
                            flush=True,
                        )

    manifest = {
        "phase": "V2.3-PHASE0DH-L1-HISTORICAL",
        "start_date": args.start.isoformat(),
        "end_date": args.end.isoformat(),
        "symbols": list(symbols),
        "streams": list(streams),
        "base_url": BASE_URL,
        "archives_expected": ((args.end - args.start).days + 1) * len(symbols) * len(streams),
        "archives_seen": len(records),
        "missing": missing,
        "official_acquisition_complete": not missing,
        "records": records,
    }
    manifest_path = root / "ACQUISITION_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest={manifest_path}")
    print(f"missing_archives={len(missing)}")
    if missing and not args.allow_missing:
        print("PHASE0DH_L1_ACQUISITION=FAIL_MISSING_ARCHIVES")
        return 2
    print("PHASE0DH_L1_ACQUISITION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
