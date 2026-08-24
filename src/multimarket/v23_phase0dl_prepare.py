from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

DEV_DAYS = tuple(date(2026, m, 1) for m in range(1, 8))
SEALED_CONFIRMATION_DAY = date(2026, 8, 1)
SYMBOLS = ("BTCUSDT", "ETHUSDT")
EXPECTED_ROWS = 345_600


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build(root: Path, build_dir: Path) -> Path:
    src = root / "tools" / "v23_phase0dl_depth250.cpp"
    if not src.exists():
        raise FileNotFoundError(src)
    cxx = shutil.which("g++")
    if cxx is None:
        raise RuntimeError("g++ not found; install build-essential")
    build_dir.mkdir(parents=True, exist_ok=True)
    exe = build_dir / "v23_phase0dl_depth250"
    stamp = build_dir / "v23_phase0dl_depth250.source.sha256"
    source_sha = _sha256(src)
    if exe.exists() and stamp.exists() and stamp.read_text().strip() == source_sha:
        return exe
    cmd = [cxx, "-std=c++17", "-O3", "-DNDEBUG", str(src), "-lz", "-o", str(exe)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"C++ build failed\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    stamp.write_text(source_sha + "\n")
    return exe


def _bounds_us(d: date) -> tuple[int, int]:
    start = int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1_000_000)
    return start, start + 86_400_000_000


def _count_rows_and_validate(path: Path) -> int:
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or header[0] != "local_timestamp_us" or header[-1] != "book_valid":
            raise RuntimeError(f"unexpected prepared header: {path}")
        prev = None
        for row in r:
            rows += 1
            if len(row) != len(header):
                raise RuntimeError(f"row width mismatch in {path} at row {rows}")
            ts = int(row[0])
            if prev is not None and ts - prev != 250_000:
                raise RuntimeError(f"non-250ms grid in {path} at row {rows}: {prev}->{ts}")
            prev = ts
    if rows != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} rows in {path}, got {rows}")
    return rows


def _one(exe: Path, raw: Path, out: Path, d: date, symbol: str) -> dict[str, object]:
    src = raw / "incremental_book_L2" / symbol / f"{d.isoformat()}.csv.gz"
    dst = out / symbol / f"{d.isoformat()}_BOOK250.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return {"day": d.isoformat(), "symbol": symbol, "pass": False, "reason": "MISSING_INPUT", "input": str(src)}
    start, end = _bounds_us(d)
    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    p = subprocess.run([str(exe), str(src), str(tmp), str(start), str(end)], capture_output=True, text=True)
    if p.returncode != 0:
        if tmp.exists(): tmp.unlink()
        return {
            "day": d.isoformat(), "symbol": symbol, "pass": False, "reason": "RECONSTRUCTOR_FAIL",
            "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr, "input": str(src),
        }
    tmp.replace(dst)
    try:
        rows = _count_rows_and_validate(dst)
    except Exception as exc:
        return {"day": d.isoformat(), "symbol": symbol, "pass": False, "reason": "OUTPUT_AUDIT_FAIL", "error": repr(exc), "output": str(dst)}
    return {
        "day": d.isoformat(), "symbol": symbol, "pass": True, "input": str(src), "output": str(dst),
        "input_sha256": _sha256(src), "output_sha256": _sha256(dst), "rows": rows,
        "stderr": p.stderr.strip(),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Prepare frozen Phase 0D-L development L2 book states")
    p.add_argument("--raw-dir", default="data/v23_phase0dl_l2_raw")
    p.add_argument("--output-dir", default="evidence/v23/phase0dl_book250")
    p.add_argument("--build-dir", default=".build/phase0dl")
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args(argv)

    root = _repo_root()
    raw = Path(a.raw_dir)
    out = Path(a.output_dir)
    exe = _build(root, Path(a.build_dir))
    jobs = [(d, s) for d in DEV_DAYS for s in SYMBOLS]
    results: list[dict[str, object]] = []
    workers = max(1, min(int(a.workers), 4))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, exe, raw, out, d, s): (d, s) for d, s in jobs}
        for fut in as_completed(futs):
            rec = fut.result(); results.append(rec)
            print(f"{rec['day']} {rec['symbol']} pass={rec['pass']} reason={rec.get('reason')} {rec.get('stderr','')}", flush=True)
    results.sort(key=lambda x: (str(x["day"]), str(x["symbol"])))
    failures = [x for x in results if not x["pass"]]
    manifest = {
        "phase": "V2.3-PHASE0DL-L2-MECHANISM",
        "stage": "BOOK250_PREPARATION",
        "development_days": [d.isoformat() for d in DEV_DAYS],
        "sealed_confirmation_day": SEALED_CONFIRMATION_DAY.isoformat(),
        "confirmation_analytically_opened": False,
        "grid_ms": 250,
        "expected_jobs": len(jobs),
        "completed_jobs": len(results) - len(failures),
        "failures": failures,
        "files": results,
        "pass": not failures and len(results) == len(jobs),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "BOOK250_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"expected_jobs={len(jobs)} completed={manifest['completed_jobs']} failures={len(failures)}")
    print("PHASE0DL_BOOK250=" + ("PASS" if manifest["pass"] else "FAIL"))
    return 0 if manifest["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
