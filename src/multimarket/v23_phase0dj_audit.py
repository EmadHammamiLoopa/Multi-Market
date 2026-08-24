from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

STREAMS = ("markPriceKlines", "indexPriceKlines", "premiumIndexKlines")
SYMBOLS = ("BTCUSDT", "ETHUSDT")
INTERVAL = "1m"
DEV_END = date(2026, 8, 3)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _read_rows(zip_path: Path) -> list[list[str]]:
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
        if len(members) != 1:
            raise ValueError(f"expected one member in {zip_path}, got {members}")
        with zf.open(members[0]) as fh:
            text = (line.decode("utf-8").strip() for line in fh)
            rows = [row for row in csv.reader(text) if row]
    if rows and not rows[0][0].lstrip("-").isdigit():
        rows = rows[1:]
    return rows


def _audit_archive(path: Path, day: date) -> dict[str, object]:
    rows = _read_rows(path)
    open_times = [int(r[0]) for r in rows]
    expected_start_ms = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)
    expected = [expected_start_ms + i * 60_000 for i in range(1440)]
    duplicates = len(open_times) - len(set(open_times))
    missing = sorted(set(expected) - set(open_times))
    extras = sorted(set(open_times) - set(expected))
    monotonic = all(b > a for a, b in zip(open_times, open_times[1:]))
    return {
        "rows": len(rows),
        "duplicates": duplicates,
        "missing_minutes": len(missing),
        "extra_minutes": len(extras),
        "strictly_increasing": monotonic,
        "first_open_ms": open_times[0] if open_times else None,
        "last_open_ms": open_times[-1] if open_times else None,
        "missing_open_ms": missing[:100],
        "extra_open_ms": extras[:100],
        "pass": len(rows) == 1440 and duplicates == 0 and not missing and not extras and monotonic,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Audit Phase 0D-J 1m futures-state archives before scoring")
    p.add_argument("--raw-dir", default="data/v23_phase0dj_state_raw")
    p.add_argument("--start", type=_parse_date, default=date(2026, 5, 26))
    p.add_argument("--end", type=_parse_date, default=DEV_END)
    p.add_argument("--output", default="evidence/v23/phase0dj_state_audit.json")
    args = p.parse_args(argv)
    if args.end > DEV_END:
        raise SystemExit("development audit may not cross sealed holdout start 2026-08-04")

    root = Path(args.raw_dir)
    results: list[dict[str, object]] = []
    failures: list[str] = []
    d = args.start
    while d <= args.end:
        for symbol in SYMBOLS:
            for stream in STREAMS:
                path = root / stream / symbol / INTERVAL / f"{symbol}-{INTERVAL}-{d.isoformat()}.zip"
                if not path.exists():
                    failures.append(f"MISSING_ARCHIVE:{d}:{symbol}:{stream}")
                    results.append({"date": d.isoformat(), "symbol": symbol, "stream": stream, "path": str(path), "pass": False, "reason": "MISSING_ARCHIVE"})
                    continue
                try:
                    audit = _audit_archive(path, d)
                    results.append({"date": d.isoformat(), "symbol": symbol, "stream": stream, "path": str(path), **audit})
                    if not audit["pass"]:
                        failures.append(f"CONTINUITY:{d}:{symbol}:{stream}")
                except Exception as exc:
                    failures.append(f"ERROR:{d}:{symbol}:{stream}:{exc}")
                    results.append({"date": d.isoformat(), "symbol": symbol, "stream": stream, "path": str(path), "pass": False, "reason": repr(exc)})
        d += timedelta(days=1)

    by_stream = defaultdict(lambda: {"archives": 0, "pass": 0, "missing_minutes": 0, "duplicates": 0})
    for r in results:
        key = f"{r['symbol']}:{r['stream']}"
        by_stream[key]["archives"] += 1
        by_stream[key]["pass"] += int(bool(r.get("pass")))
        by_stream[key]["missing_minutes"] += int(r.get("missing_minutes", 0) or 0)
        by_stream[key]["duplicates"] += int(r.get("duplicates", 0) or 0)

    payload = {
        "phase": "V2.3-PHASE0DJ-FUTURES-STATE-AUDIT",
        "development_only": True,
        "historical_holdout_opened": False,
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "archives_expected": len(results),
        "failures": failures,
        "summary": dict(by_stream),
        "pass": not failures,
        "records": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"audit={out}")
    print(f"failures={len(failures)}")
    print("PHASE0DJ_STATE_AUDIT=PASS" if not failures else "PHASE0DJ_STATE_AUDIT=FAIL")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
