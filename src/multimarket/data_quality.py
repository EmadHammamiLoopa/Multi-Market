from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .data import load_ohlc_csv
from .market_calendar import tradability_at
from .models import MarketBar
from .reporting import sha256_file


@dataclass(frozen=True, slots=True)
class BarQuality:
    index: int
    timestamp: str
    session_eligible: bool
    session_reason: str
    zero_range: bool
    tiny_range: bool
    repeated_close: bool
    repeated_ohlc: bool


def _range_bps(bar: MarketBar) -> float:
    return (bar.high / bar.low - 1.0) * 10_000.0


def audit_bars(
    bars: list[MarketBar],
    *,
    symbol: str,
    tiny_range_bps: float = 0.10,
) -> list[BarQuality]:
    if tiny_range_bps < 0:
        raise ValueError("tiny_range_bps must be non-negative")

    rows: list[BarQuality] = []
    previous: MarketBar | None = None
    for index, bar in enumerate(bars):
        session = tradability_at(symbol, bar.timestamp)
        zero_range = bar.high == bar.low
        tiny_range = _range_bps(bar) <= tiny_range_bps
        repeated_close = previous is not None and bar.close == previous.close
        repeated_ohlc = (
            previous is not None
            and bar.open == previous.open
            and bar.high == previous.high
            and bar.low == previous.low
            and bar.close == previous.close
        )
        rows.append(
            BarQuality(
                index=index,
                timestamp=bar.timestamp.isoformat().replace("+00:00", "Z"),
                session_eligible=session.eligible,
                session_reason=session.reason,
                zero_range=zero_range,
                tiny_range=tiny_range,
                repeated_close=repeated_close,
                repeated_ohlc=repeated_ohlc,
            )
        )
        previous = bar
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit OHLC data for structural market-session eligibility and stale/near-flat bars. "
            "This diagnostic never rewrites the source dataset."
        )
    )
    parser.add_argument("csv")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--tiny-range-bps", type=float, default=0.10)
    parser.add_argument("--output-json")
    parser.add_argument(
        "--show-flagged",
        type=int,
        default=20,
        help="Print up to N flagged rows (0 disables row listing)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.show_flagged < 0:
        raise SystemExit("--show-flagged must be non-negative")

    bars = load_ohlc_csv(args.csv)
    rows = audit_bars(bars, symbol=args.symbol, tiny_range_bps=args.tiny_range_bps)
    reasons = Counter(row.session_reason for row in rows)
    session_invalid = [row for row in rows if not row.session_eligible]
    zero_range = [row for row in rows if row.zero_range]
    tiny_range = [row for row in rows if row.tiny_range]
    repeated_close = [row for row in rows if row.repeated_close]
    repeated_ohlc = [row for row in rows if row.repeated_ohlc]
    flagged = [
        row
        for row in rows
        if (
            not row.session_eligible
            or row.zero_range
            or row.tiny_range
            or row.repeated_ohlc
        )
    ]

    print(f"Multi-Market data-quality audit | {args.symbol}")
    print("=" * 78)
    print(f"Dataset SHA-256       : {sha256_file(args.csv)}")
    print(f"Bars                  : {len(rows)}")
    print(f"Session-ineligible    : {len(session_invalid)} ({len(session_invalid) / len(rows):.2%})")
    print(f"Zero-range bars       : {len(zero_range)} ({len(zero_range) / len(rows):.2%})")
    print(f"Tiny-range <= {args.tiny_range_bps:.3f}bp : {len(tiny_range)} ({len(tiny_range) / len(rows):.2%})")
    print(f"Repeated close        : {len(repeated_close)} ({len(repeated_close) / len(rows):.2%})")
    print(f"Repeated full OHLC    : {len(repeated_ohlc)} ({len(repeated_ohlc) / len(rows):.2%})")
    print("Session reasons       : " + ", ".join(f"{key}={value}" for key, value in sorted(reasons.items())))
    print("Mutation rule         : diagnostic only; source CSV is never rewritten")

    if args.show_flagged:
        print("\nFlagged sample")
        print("-" * 78)
        for row in flagged[: args.show_flagged]:
            flags = []
            if not row.session_eligible:
                flags.append(row.session_reason)
            if row.zero_range:
                flags.append("zero_range")
            elif row.tiny_range:
                flags.append("tiny_range")
            if row.repeated_ohlc:
                flags.append("repeated_ohlc")
            print(f"{row.timestamp:25s} " + ",".join(flags))

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "symbol": args.symbol,
            "dataset_sha256": sha256_file(args.csv),
            "bars": len(rows),
            "tiny_range_bps": args.tiny_range_bps,
            "session_ineligible": len(session_invalid),
            "zero_range": len(zero_range),
            "tiny_range": len(tiny_range),
            "repeated_close": len(repeated_close),
            "repeated_ohlc": len(repeated_ohlc),
            "session_reasons": dict(sorted(reasons.items())),
            "flagged_rows": [asdict(row) for row in flagged],
            "note": "diagnostic only; original V0/V1 datasets and benchmarks remain unchanged",
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Audit JSON            : {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
