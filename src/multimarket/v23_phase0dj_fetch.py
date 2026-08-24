from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx

BASE_URL = "https://data.binance.vision/data/futures/um/daily"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
STREAMS = ("markPriceKlines", "indexPriceKlines", "premiumIndexKlines")
INTERVAL = "1m"
DEFAULT_START = date(2026, 5, 26)
DEFAULT_END = date(2026, 8, 23)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _days(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _name(symbol: str, day: date) -> str:
    return f"{symbol}-{INTERVAL}-{day.isoformat()}.zip"


def archive_url(stream: str, symbol: str, day: date) -> str:
    return f"{BASE_URL}/{stream}/{symbol}/{INTERVAL}/{_name(symbol, day)}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_zip(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f"corrupt zip member: {bad}")
        members = [x.filename for x in zf.infolist() if not x.is_dir()]
        if not members:
            raise ValueError("empty archive")
        return members


def _download_bytes(client: httpx.Client, url: str) -> bytes:
    r = client.get(url)
    if r.status_code == 404:
        raise FileNotFoundError(url)
    r.raise_for_status()
    return r.content


def _official_checksum(text: str, expected_filename: str) -> str:
    line = text.strip().splitlines()[0].strip()
    parts = line.replace("*", " ").split()
    if not parts or len(parts[0]) != 64:
        raise ValueError(f"invalid CHECKSUM format: {line!r}")
    checksum = parts[0].lower()
    if len(parts) >= 2:
        stated = Path(parts[-1]).name
        if stated != expected_filename:
            raise ValueError(f"CHECKSUM filename mismatch: expected {expected_filename}, got {stated}")
    return checksum


def acquire_one(root: Path, stream: str, symbol: str, day: date, timeout: float) -> dict[str, object]:
    url = archive_url(stream, symbol, day)
    checksum_url = url + ".CHECKSUM"
    dest = root / stream / symbol / INTERVAL / _name(symbol, day)
    checksum_dest = Path(str(dest) + ".CHECKSUM")
    dest.parent.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "Multi-Market-V23-Phase0DJ/0.2"}
    with httpx.Client(timeout=httpx.Timeout(timeout, connect=min(20.0, timeout)), follow_redirects=True, headers=headers) as client:
        checksum_bytes = _download_bytes(client, checksum_url)
        checksum_dest.write_bytes(checksum_bytes)
        official = _official_checksum(checksum_bytes.decode("utf-8", errors="strict"), dest.name)

        status = "EXISTS_VALID"
        if not dest.exists() or dest.stat().st_size == 0 or sha256_file(dest) != official:
            tmp = Path(str(dest) + ".part")
            if tmp.exists():
                tmp.unlink()
            data = _download_bytes(client, url)
            tmp.write_bytes(data)
            tmp.replace(dest)
            status = "DOWNLOADED"

    actual = sha256_file(dest)
    if actual != official:
        raise ValueError(f"SHA256 mismatch for {dest}: official={official} actual={actual}")
    members = validate_zip(dest)
    return {
        "date": day.isoformat(),
        "stream": stream,
        "symbol": symbol,
        "interval": INTERVAL,
        "url": url,
        "checksum_url": checksum_url,
        "path": str(dest),
        "checksum_path": str(checksum_dest),
        "status": status,
        "bytes": dest.stat().st_size,
        "sha256": actual,
        "official_sha256": official,
        "members": members,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Acquire checksum-verified Binance futures-state archives for Phase 0D-J")
    p.add_argument("--output-dir", default="data/v23_phase0dj_state_raw")
    p.add_argument("--start", type=_parse_date, default=DEFAULT_START)
    p.add_argument("--end", type=_parse_date, default=DEFAULT_END)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--timeout-seconds", type=float, default=60.0)
    args = p.parse_args(argv)
    if args.end < args.start:
        raise SystemExit("end precedes start")

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    jobs = [(stream, symbol, day) for day in _days(args.start, args.end) for symbol in SYMBOLS for stream in STREAMS]
    records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    workers = max(1, min(args.workers, 24))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="phase0dj-fetch") as ex:
        futs = {ex.submit(acquire_one, root, s, sym, d, args.timeout_seconds): (s, sym, d) for s, sym, d in jobs}
        for fut in as_completed(futs):
            stream, symbol, day = futs[fut]
            try:
                rec = fut.result()
                records.append(rec)
                print(f"[{day}] {symbol} {stream} {rec['status']} bytes={rec['bytes']}", flush=True)
            except Exception as exc:
                failures.append({"date": day.isoformat(), "symbol": symbol, "stream": stream, "error": repr(exc)})
                print(f"[{day}] {symbol} {stream} FAIL {exc}", flush=True)

    records.sort(key=lambda r: (r["date"], r["symbol"], r["stream"]))
    manifest = {
        "phase": "V2.3-PHASE0DJ-FUTURES-STATE",
        "start_date": args.start.isoformat(),
        "end_date": args.end.isoformat(),
        "symbols": list(SYMBOLS),
        "streams": list(STREAMS),
        "interval": INTERVAL,
        "archives_expected": len(jobs),
        "archives_valid": len(records),
        "failures": failures,
        "official_acquisition_complete": len(records) == len(jobs) and not failures,
        "records": records,
    }
    path = root / "ACQUISITION_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest={path}")
    print(f"archives_expected={len(jobs)} archives_valid={len(records)} failures={len(failures)}")
    if failures or len(records) != len(jobs):
        print("PHASE0DJ_ACQUISITION=FAIL")
        return 2
    print("PHASE0DJ_ACQUISITION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
