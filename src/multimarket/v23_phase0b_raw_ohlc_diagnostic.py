from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

DEFAULT_SYMBOLS = ("IWM", "TLT", "GLD", "UUP", "XLI")


def diagnose_file(path: Path, symbol: str) -> dict[str, object]:
    anomalies: list[dict[str, object]] = []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for csv_line, row in enumerate(reader, start=2):
            rows.append(row)
            try:
                o = float(row["open"])
                h = float(row["high"])
                l = float(row["low"])
                c = float(row["close"])
            except (KeyError, TypeError, ValueError) as exc:
                anomalies.append({
                    "csv_line": csv_line,
                    "timestamp": row.get("timestamp", ""),
                    "kind": "PARSE_ERROR",
                    "error": str(exc),
                    "raw": row,
                })
                continue

            violations: list[str] = []
            high_required = max(o, c)
            low_required = min(o, c)
            if min(o, h, l, c) <= 0:
                violations.append("NON_POSITIVE_PRICE")
            if h < high_required:
                violations.append("HIGH_BELOW_OPEN_OR_CLOSE")
            if l > low_required:
                violations.append("LOW_ABOVE_OPEN_OR_CLOSE")
            if violations:
                anomalies.append({
                    "csv_line": csv_line,
                    "timestamp": row.get("timestamp", ""),
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "violations": violations,
                    "high_shortfall": max(0.0, high_required - h),
                    "low_excess": max(0.0, l - low_required),
                })

    for anomaly in anomalies:
        line = int(anomaly["csv_line"])
        index = line - 2
        start = max(0, index - 2)
        end = min(len(rows), index + 3)
        anomaly["neighbors"] = [
            {"csv_line": i + 2, **rows[i]} for i in range(start, end)
        ]

    return {
        "symbol": symbol,
        "path": str(path),
        "row_count": len(rows),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read raw Phase 0B sensor CSVs and report OHLC invariant violations without modifying data."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--symbol", action="append", dest="symbols")
    parser.add_argument(
        "--output-json",
        default="evidence/v23/phase0b/V23_PHASE0B_RAW_OHLC_DIAGNOSTIC.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    symbols = tuple(dict.fromkeys((args.symbols or DEFAULT_SYMBOLS)))
    data_dir = Path(args.data_dir)
    results = [diagnose_file(data_dir / f"{symbol}_5m.csv", symbol) for symbol in symbols]
    payload = {
        "version": "V2.3-PHASE0B-RAW-OHLC-DIAGNOSTIC",
        "data_modified": False,
        "predictive_scoring_performed": False,
        "results": results,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("===== V2.3 PHASE 0B RAW OHLC DIAGNOSTIC =====")
    total = 0
    for result in results:
        count = int(result["anomaly_count"])
        total += count
        print(f"{result['symbol']:4s} anomalies={count}")
        for anomaly in result["anomalies"]:
            print(
                f"  line={anomaly['csv_line']} ts={anomaly.get('timestamp','')} "
                f"O={anomaly.get('open')} H={anomaly.get('high')} "
                f"L={anomaly.get('low')} C={anomaly.get('close')} "
                f"violations={','.join(anomaly.get('violations', []))} "
                f"high_shortfall={anomaly.get('high_shortfall', 0.0):.12g} "
                f"low_excess={anomaly.get('low_excess', 0.0):.12g}"
            )
    print(f"total_anomalies={total}")
    print("data_modified=NO")
    print("predictive_scoring_performed=NO")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
