from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Sequence

from .v23_phase0b_acquisition_audit import SENSORS

MAX_ADJUSTMENT_BPS = 5.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bps(delta: float, reference: float) -> float:
    return 0.0 if reference == 0 else abs(delta) / abs(reference) * 10_000.0


def canonicalize_file(src: Path, dst: Path, symbol: str) -> dict[str, object]:
    mutations: list[dict[str, object]] = []
    with src.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"{symbol}: CSV has no header")
        rows = list(reader)

    for csv_line, row in enumerate(rows, start=2):
        try:
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{symbol}: parse error at line {csv_line}: {exc}") from exc
        if min(o, h, l, c) <= 0:
            raise ValueError(f"{symbol}: non-positive OHLC at line {csv_line}")

        new_h = max(h, o, c)
        new_l = min(l, o, c)
        high_delta = new_h - h
        low_delta = l - new_l
        high_bps = _bps(high_delta, max(o, c))
        low_bps = _bps(low_delta, min(o, c))
        max_bps = max(high_bps, low_bps)

        if max_bps > MAX_ADJUSTMENT_BPS:
            raise ValueError(
                f"{symbol}: required OHLC adjustment {max_bps:.6f} bps exceeds "
                f"{MAX_ADJUSTMENT_BPS:.6f} bps at line {csv_line}"
            )

        if high_delta > 0 or low_delta > 0:
            mutations.append({
                "csv_line": csv_line,
                "timestamp": row.get("timestamp", ""),
                "open": o,
                "close": c,
                "high_before": h,
                "high_after": new_h,
                "low_before": l,
                "low_after": new_l,
                "high_adjustment_bps": high_bps,
                "low_adjustment_bps": low_bps,
            })
            if high_delta > 0:
                row["high"] = format(new_h, ".15g")
            if low_delta > 0:
                row["low"] = format(new_l, ".15g")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    return {
        "symbol": symbol,
        "source_path": str(src),
        "canonical_path": str(dst),
        "source_sha256": _sha256(src),
        "canonical_sha256": _sha256(dst),
        "row_count": len(rows),
        "mutation_count": len(mutations),
        "mutations": mutations,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create canonical Phase 0B sensor copies by minimally expanding high/low "
            "to contain open/close. Raw files are never modified."
        )
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="data/v23_phase0b_canonical")
    parser.add_argument(
        "--report-json",
        default="evidence/v23/phase0b/V23_PHASE0B_OHLC_CANONICALIZATION.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    report = Path(args.report_json)

    results: list[dict[str, object]] = []
    try:
        for symbol in SENSORS:
            src = data_dir / f"{symbol}_5m.csv"
            dst = output_dir / f"{symbol}_5m.csv"
            if not src.is_file():
                raise FileNotFoundError(f"Missing sensor file: {src}")
            results.append(canonicalize_file(src, dst, symbol))
    except (OSError, ValueError) as exc:
        payload = {
            "version": "V2.3-PHASE0B-OHLC-CANONICALIZATION",
            "rule": "high=max(raw_high,open,close); low=min(raw_low,open,close)",
            "max_adjustment_bps": MAX_ADJUSTMENT_BPS,
            "raw_files_modified": False,
            "status": "FAIL",
            "error": str(exc),
            "results": results,
        }
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("===== V2.3 PHASE 0B OHLC CANONICALIZATION =====")
        print(f"status=FAIL error={exc}")
        print("raw_files_modified=NO")
        print(f"Output: {report}")
        return 1

    total_mutations = sum(int(result["mutation_count"]) for result in results)
    payload = {
        "version": "V2.3-PHASE0B-OHLC-CANONICALIZATION",
        "rule": "high=max(raw_high,open,close); low=min(raw_low,open,close)",
        "max_adjustment_bps": MAX_ADJUSTMENT_BPS,
        "raw_files_modified": False,
        "status": "PASS",
        "total_mutations": total_mutations,
        "results": results,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("===== V2.3 PHASE 0B OHLC CANONICALIZATION =====")
    for result in results:
        print(
            f"{result['symbol']:4s} rows={result['row_count']:6d} "
            f"mutations={result['mutation_count']:3d} "
            f"raw_sha256={result['source_sha256']} canonical_sha256={result['canonical_sha256']}"
        )
    print(f"total_mutations={total_mutations}")
    print(f"max_adjustment_bps={MAX_ADJUSTMENT_BPS}")
    print("raw_files_modified=NO")
    print("status=PASS")
    print(f"Output: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
