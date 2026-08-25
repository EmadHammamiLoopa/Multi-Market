from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

DEV_DAYS = tuple(date(2026, m, 1) for m in range(1, 8))
SYMBOLS = ("BTCUSDT", "ETHUSDT")
EXPECTED_ROWS = 345_600


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bounds(d: date) -> tuple[int, int]:
    s = int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1_000_000)
    return s, s + 86_400_000_000


def build(build_dir: Path) -> Path:
    src = root() / "tools" / "v23_phase0dl_features250.cpp"
    cxx = shutil.which("g++")
    if not cxx:
        raise RuntimeError("g++ not found")
    build_dir.mkdir(parents=True, exist_ok=True)
    exe = build_dir / "v23_phase0dl_features250"
    p = subprocess.run([cxx, "-std=c++17", "-O3", "-DNDEBUG", str(src), "-o", str(exe)], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr)
    return exe


def one(exe: Path, book_dir: Path, flow_dir: Path, trade_dir: Path, snapshot_dir: Path, out_dir: Path, d: date, symbol: str) -> dict[str, object]:
    book = book_dir / symbol / f"{d.isoformat()}_BOOK250.csv"
    flow = flow_dir / symbol / f"{d.isoformat()}_FLOW250.csv"
    trade = trade_dir / symbol / f"{d.isoformat()}_TRADE250.csv"
    snap = snapshot_dir / symbol / f"{d.isoformat()}_SNAPSHOTS.csv"
    dst = out_dir / symbol / f"{d.isoformat()}_FEATURES250.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    for p in (book, flow, trade, snap):
        if not p.exists():
            return {"day": d.isoformat(), "symbol": symbol, "pass": False, "reason": "MISSING_INPUT", "input": str(p)}
    start, end = bounds(d)
    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    p = subprocess.run([str(exe), str(book), str(flow), str(trade), str(snap), str(tmp), str(start), str(end), symbol], capture_output=True, text=True)
    if p.returncode:
        if tmp.exists():
            tmp.unlink()
        return {"day": d.isoformat(), "symbol": symbol, "pass": False, "reason": "FEATURE_ASSEMBLY_FAIL", "returncode": p.returncode, "stderr": p.stderr.strip()}
    tmp.replace(dst)
    with dst.open("r", encoding="utf-8") as f:
        rows = sum(1 for _ in f) - 1
    if rows != EXPECTED_ROWS:
        return {"day": d.isoformat(), "symbol": symbol, "pass": False, "reason": "ROW_COUNT_FAIL", "rows": rows}
    return {"day": d.isoformat(), "symbol": symbol, "pass": True, "rows": rows, "stderr": p.stderr.strip(), "output": str(dst), "sha256": sha256(dst)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book-dir", default="evidence/v23/phase0dl_book250")
    ap.add_argument("--flow-dir", default="evidence/v23/phase0dl_flow250")
    ap.add_argument("--trade-dir", default="evidence/v23/phase0dl_trade250")
    ap.add_argument("--snapshot-dir", default="evidence/v23/phase0dl_snapshots")
    ap.add_argument("--output-dir", default="evidence/v23/phase0dl_features250")
    ap.add_argument("--build-dir", default=".build/phase0dl")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)

    exe = build(Path(a.build_dir))
    jobs = [(d, s) for d in DEV_DAYS for s in SYMBOLS]
    rec: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(a.workers, 8))) as pool:
        futs = [pool.submit(one, exe, Path(a.book_dir), Path(a.flow_dir), Path(a.trade_dir), Path(a.snapshot_dir), Path(a.output_dir), d, s) for d, s in jobs]
        for fut in as_completed(futs):
            r = fut.result(); rec.append(r)
            print(r["day"], r["symbol"], "pass=" + str(r["pass"]), r.get("stderr", r.get("reason", "")), flush=True)
    rec.sort(key=lambda x: (x["day"], x["symbol"]))
    failures = [x for x in rec if not x["pass"]]
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    manifest = {"phase": "V2.3-PHASE0DL-L2-MECHANISM", "stage": "FEATURE250_ASSEMBLY_AND_INTEGRITY", "development_only": True, "confirmation_analytically_opened": False, "expected_jobs": len(jobs), "files": rec, "failures": failures, "pass": not failures and len(rec) == len(jobs)}
    (out / "FEATURE250_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"expected_jobs={len(jobs)} completed={len(rec)-len(failures)} failures={len(failures)}")
    print("PHASE0DL_FEATURE250=" + ("PASS" if manifest["pass"] else "FAIL"))
    return 0 if manifest["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
