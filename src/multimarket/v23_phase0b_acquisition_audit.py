from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .data import load_ohlc_csv
from .data_quality import audit_bars


SENSORS = (
    "SPY", "IWM", "DIA",
    "TLT", "HYG", "LQD",
    "GLD", "SLV", "USO", "UUP",
    "XLK", "XLF", "XLE", "XLI", "XLY", "XLP", "XLV",
)
TARGETS = ("EURUSD", "XAUUSD", "BTCUSD", "ETHUSD", "QQQ")

REQUIRED_START = datetime(2025, 8, 1, tzinfo=timezone.utc)
REQUIRED_END = datetime(2026, 8, 21, 23, 59, 59, tzinfo=timezone.utc)
MIN_RTH_BARS = 10_000


@dataclass(frozen=True, slots=True)
class SensorAudit:
    symbol: str
    path: str
    sha256: str
    bars: int
    first_timestamp: str
    last_timestamp: str
    hard_eligible_bars: int
    session_ineligible: int
    zero_range: int
    repeated_ohlc: int
    covers_required_start: bool
    covers_required_end: bool
    enough_hard_eligible_bars: bool
    status: str
    load_error: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iso(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def audit_sensor(path: Path, symbol: str) -> SensorAudit:
    sha = _sha256(path)
    try:
        bars = load_ohlc_csv(path)
    except (OSError, ValueError) as exc:
        return SensorAudit(
            symbol=symbol,
            path=str(path),
            sha256=sha,
            bars=0,
            first_timestamp="",
            last_timestamp="",
            hard_eligible_bars=0,
            session_ineligible=0,
            zero_range=0,
            repeated_ohlc=0,
            covers_required_start=False,
            covers_required_end=False,
            enough_hard_eligible_bars=False,
            status="FAIL_INVALID_OHLC",
            load_error=str(exc),
        )

    quality = audit_bars(bars, symbol=symbol)
    hard_eligible = [
        row for row in quality
        if row.session_eligible and not row.zero_range and not row.repeated_ohlc
    ]
    first = bars[0].timestamp.astimezone(timezone.utc)
    last = bars[-1].timestamp.astimezone(timezone.utc)
    covers_start = first <= REQUIRED_START
    covers_end = last >= REQUIRED_END
    enough = len(hard_eligible) >= MIN_RTH_BARS
    passed = covers_start and covers_end and enough
    return SensorAudit(
        symbol=symbol,
        path=str(path),
        sha256=sha,
        bars=len(bars),
        first_timestamp=_iso(first),
        last_timestamp=_iso(last),
        hard_eligible_bars=len(hard_eligible),
        session_ineligible=sum(not row.session_eligible for row in quality),
        zero_range=sum(row.zero_range for row in quality),
        repeated_ohlc=sum(row.repeated_ohlc for row in quality),
        covers_required_start=covers_start,
        covers_required_end=covers_end,
        enough_hard_eligible_bars=enough,
        status="PASS" if passed else "FAIL",
        load_error=None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the complete frozen V2.3 Phase 0B 17-sensor acquisition block. "
            "No predictive scoring is performed."
        )
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--output-json",
        default="evidence/v23/phase0b/V23_PHASE0B_ACQUISITION_AUDIT.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)

    missing = [
        symbol for symbol in SENSORS
        if not (data_dir / f"{symbol}_5m.csv").is_file()
    ]
    if missing:
        payload = {
            "version": "V2.3-PHASE0B-ACQUISITION-AUDIT",
            "frozen_sensors": list(SENSORS),
            "required_start": _iso(REQUIRED_START),
            "required_end": _iso(REQUIRED_END),
            "minimum_hard_eligible_bars": MIN_RTH_BARS,
            "missing_sensors": missing,
            "sensor_results": [],
            "gate_pass": False,
            "decision": "BLOCK_PHASE0B_SCORING",
        }
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("===== V2.3 PHASE 0B ACQUISITION AUDIT =====")
        print(f"missing_sensors={','.join(missing)}")
        print("PHASE0B_ACQUISITION_GATE=FAIL")
        print("decision=BLOCK_PHASE0B_SCORING")
        print(f"Output: {output}")
        return 1

    results = [audit_sensor(data_dir / f"{symbol}_5m.csv", symbol) for symbol in SENSORS]
    gate_pass = all(result.status == "PASS" for result in results)
    invalid = [result for result in results if result.load_error]
    payload = {
        "version": "V2.3-PHASE0B-ACQUISITION-AUDIT",
        "frozen_sensors": list(SENSORS),
        "required_start": _iso(REQUIRED_START),
        "required_end": _iso(REQUIRED_END),
        "minimum_hard_eligible_bars": MIN_RTH_BARS,
        "missing_sensors": [],
        "invalid_sensor_files": [result.symbol for result in invalid],
        "sensor_results": [asdict(result) for result in results],
        "gate_pass": gate_pass,
        "decision": "ALLOW_PHASE0B_SCORING" if gate_pass else "BLOCK_PHASE0B_SCORING",
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("===== V2.3 PHASE 0B ACQUISITION AUDIT =====")
    for result in results:
        if result.load_error:
            print(
                f"{result.symbol:4s} status={result.status} "
                f"sha256={result.sha256} error={result.load_error}"
            )
            continue
        print(
            f"{result.symbol:4s} status={result.status} bars={result.bars:6d} "
            f"eligible={result.hard_eligible_bars:6d} "
            f"start={result.first_timestamp} end={result.last_timestamp} "
            f"zero={result.zero_range} repeated={result.repeated_ohlc}"
        )
    print(f"invalid_sensor_files={','.join(result.symbol for result in invalid) or 'NONE'}")
    print(f"PHASE0B_ACQUISITION_GATE={'PASS' if gate_pass else 'FAIL'}")
    print(f"decision={payload['decision']}")
    print(f"Output: {output}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
