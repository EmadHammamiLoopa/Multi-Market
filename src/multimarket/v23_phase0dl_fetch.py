from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx

EXCHANGE = "binance-futures"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
DATA_TYPES = ("incremental_book_L2", "trades")
SAMPLE_DAYS = tuple(date(2026, m, 1) for m in range(1, 9))
BASE = "https://datasets.tardis.dev/v1"


@dataclass(frozen=True)
class Item:
    day: date
    symbol: str
    data_type: str

    @property
    def url(self) -> str:
        d = self.day
        return f"{BASE}/{EXCHANGE}/{self.data_type}/{d:%Y/%m/%d}/{self.symbol}.csv.gz"

    def path(self, root: Path) -> Path:
        return root / self.data_type / self.symbol / f"{self.day.isoformat()}.csv.gz"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _gzip_header(path: Path) -> str:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        return fh.readline().strip()


def _download(item: Item, root: Path, timeout: float) -> dict[str, object]:
    path = item.path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    if path.exists() and path.stat().st_size > 0:
        try:
            header = _gzip_header(path)
            if header:
                return {
                    "day": item.day.isoformat(), "symbol": item.symbol,
                    "data_type": item.data_type, "url": item.url,
                    "path": str(path), "bytes": path.stat().st_size,
                    "sha256": _sha256(path), "header": header,
                    "status": "VALID_EXISTING",
                }
        except Exception:
            pass
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout, connect=20.0), follow_redirects=True) as client:
            with client.stream("GET", item.url) as r:
                r.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in r.iter_bytes(1024 * 1024):
                        if chunk:
                            fh.write(chunk)
        if not tmp.exists() or tmp.stat().st_size == 0:
            raise RuntimeError("empty download")
        header = _gzip_header(tmp)
        if not header:
            raise RuntimeError("empty gzip dataset")
        tmp.replace(path)
        return {
            "day": item.day.isoformat(), "symbol": item.symbol,
            "data_type": item.data_type, "url": item.url,
            "path": str(path), "bytes": path.stat().st_size,
            "sha256": _sha256(path), "header": header,
            "status": "DOWNLOADED",
        }
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        return {
            "day": item.day.isoformat(), "symbol": item.symbol,
            "data_type": item.data_type, "url": item.url,
            "path": str(path), "status": "FAIL", "error": repr(exc),
        }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download frozen free Tardis Phase 0D-L L2 samples")
    p.add_argument("--output-dir", default="data/v23_phase0dl_l2_raw")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=float, default=600.0)
    a = p.parse_args(argv)
    root = Path(a.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    items = [Item(d, s, t) for d in SAMPLE_DAYS for s in SYMBOLS for t in DATA_TYPES]
    results: list[dict[str, object]] = []
    workers = max(1, min(int(a.workers), 8))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_download, item, root, a.timeout): item for item in items}
        for fut in as_completed(futs):
            r = fut.result(); results.append(r)
            print(f"{r['day']} {r['symbol']} {r['data_type']} {r['status']}", flush=True)
    results.sort(key=lambda x: (str(x['day']), str(x['symbol']), str(x['data_type'])))
    failures = [r for r in results if r["status"] == "FAIL"]
    manifest = {
        "phase": "V2.3-PHASE0DL-L2-MECHANISM",
        "source": "Tardis downloadable CSV free first-of-month samples",
        "exchange": EXCHANGE,
        "symbols": list(SYMBOLS),
        "data_types": list(DATA_TYPES),
        "sample_days": [d.isoformat() for d in SAMPLE_DAYS],
        "development_days": [d.isoformat() for d in SAMPLE_DAYS[:-1]],
        "sealed_confirmation_day": SAMPLE_DAYS[-1].isoformat(),
        "confirmation_analytically_opened": False,
        "expected_files": len(items),
        "valid_files": len(results) - len(failures),
        "failures": failures,
        "files": results,
        "complete": not failures and len(results) == len(items),
    }
    (root / "ACQUISITION_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"expected={len(items)} valid={manifest['valid_files']} failures={len(failures)}")
    print("PHASE0DL_ACQUISITION=" + ("PASS" if manifest["complete"] else "FAIL"))
    return 0 if manifest["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
