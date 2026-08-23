from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .data import load_ohlc_csv
from .v21_common import hard_eligible_indices


SAMPLE_COUNT = 30
WINDOWS = {
    "D": (
        datetime(2025, 11, 3, 0, 0, 0, tzinfo=timezone.utc),
        datetime(2025, 11, 21, 23, 59, 59, tzinfo=timezone.utc),
    ),
    "E": (
        datetime(2026, 1, 5, 0, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 23, 23, 59, 59, tzinfo=timezone.utc),
    ),
    "F": (
        datetime(2026, 4, 6, 0, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 24, 23, 59, 59, tzinfo=timezone.utc),
    ),
}


def _iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def evenly_spaced_positions(candidate_count: int, sample_count: int = SAMPLE_COUNT) -> tuple[int, ...]:
    """Return deterministic endpoint-inclusive integer positions.

    This is the frozen V2.2 structural sampling rule. For k samples from n
    chronologically ordered structurally eligible candidates, position j is
    floor(j * (n - 1) / (k - 1)), j=0..k-1. With n >= k this is unique,
    deterministic, endpoint-inclusive, and does not inspect prices, returns,
    labels, predictions, macro values, or outcomes.
    """
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    if candidate_count < sample_count:
        raise ValueError(
            f"need at least {sample_count} structurally eligible candidates; got {candidate_count}"
        )
    positions = tuple(
        (j * (candidate_count - 1)) // (sample_count - 1)
        for j in range(sample_count)
    )
    if len(set(positions)) != sample_count:
        raise AssertionError("evenly spaced sampling produced duplicate positions")
    return positions


def select_timestamps(
    timestamps: Iterable[datetime],
    *,
    sample_count: int = SAMPLE_COUNT,
) -> tuple[datetime, ...]:
    ordered = tuple(sorted(ts.astimezone(timezone.utc) for ts in timestamps))
    positions = evenly_spaced_positions(len(ordered), sample_count)
    return tuple(ordered[pos] for pos in positions)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    source_csv: Path,
    symbol: str,
    window_label: str,
    candidate_timestamps: Iterable[datetime],
) -> dict[str, object]:
    if window_label not in WINDOWS:
        raise ValueError(f"unsupported frozen V2.2 window: {window_label}")
    start, end = WINDOWS[window_label]
    candidates = tuple(sorted(ts.astimezone(timezone.utc) for ts in candidate_timestamps))
    selected = select_timestamps(candidates)
    return {
        "version": "V2.2-MACRO-CONTEXT",
        "purpose": "selection-only frozen decision timestamp manifest",
        "symbol": symbol.upper(),
        "window": window_label,
        "window_start": _iso(start),
        "window_end": _iso(end),
        "source_csv": str(source_csv),
        "source_csv_sha256": _sha256(source_csv),
        "structural_eligibility": "session_eligible AND NOT zero_range AND NOT repeated_ohlc",
        "sampling_rule": "position_j=floor(j*(n-1)/(30-1)), j=0..29, chronological structurally eligible candidates only",
        "candidate_count": len(candidates),
        "sample_count": SAMPLE_COUNT,
        "rows": [{"decision_timestamp": _iso(ts)} for ts in selected],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze V2.2 D/E/F decision timestamp manifests using structural eligibility only. "
            "This command does not compute labels, returns, predictions, or macro outcomes."
        )
    )
    parser.add_argument("csv")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--output-dir", default="evidence/v22/manifests")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.csv)
    symbol = args.symbol.strip().upper()
    output_dir = Path(args.output_dir)

    bars = load_ohlc_csv(source)
    eligible = hard_eligible_indices(bars, symbol)
    source_hash = _sha256(source)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"V2.2 selection-only manifest freeze | {symbol}")
    print(f"source_sha256={source_hash}")
    print("outcomes_inspected=NO")

    for label, (start, end) in WINDOWS.items():
        candidates = [
            bar.timestamp.astimezone(timezone.utc)
            for index, bar in enumerate(bars)
            if index in eligible and start <= bar.timestamp.astimezone(timezone.utc) <= end
        ]
        manifest = build_manifest(
            source_csv=source,
            symbol=symbol,
            window_label=label,
            candidate_timestamps=candidates,
        )
        out = output_dir / f"{symbol}_{label}_V22_TIMESTAMP_MANIFEST.json"
        out.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        rows = manifest["rows"]
        assert isinstance(rows, list)
        first = rows[0]["decision_timestamp"]
        last = rows[-1]["decision_timestamp"]
        print(
            f"{label}: candidates={len(candidates)} selected={len(rows)} "
            f"first={first} last={last} output={out}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
