from __future__ import annotations

import argparse
from datetime import datetime, timezone

from .v22_macro import MACRO_SERIES, MacroLedgerIndex, ledger_summary, load_macro_ledger_csv


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("audit timestamp must include an explicit timezone")
    return parsed.astimezone(timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a normalized V2.2 point-in-time macro ledger at one or more timestamps"
    )
    parser.add_argument("ledger_csv")
    parser.add_argument(
        "--timestamp",
        action="append",
        required=True,
        help="ISO-8601 decision timestamp; may be supplied multiple times",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_macro_ledger_csv(args.ledger_csv)
    ledger = MacroLedgerIndex(rows)
    counts = ledger_summary(rows)

    print("Multi-Market V2.2 macro causality audit")
    print("=" * 72)
    print(f"Ledger rows: {len(rows)}")
    for series in MACRO_SERIES:
        print(f"{series:10s}: {counts[series]}")

    for raw_timestamp in args.timestamp:
        ts = _parse_timestamp(raw_timestamp)
        print()
        print(ts.isoformat().replace("+00:00", "Z"))
        print("-" * 72)
        for series in MACRO_SERIES:
            point_values = ledger.point_in_time_values(series, ts)
            packet = ledger.packet(series, ts)
            latest_known = point_values[-1].known_at if point_values else None
            latest_observation = point_values[-1].observation_date if point_values else None
            latest_known_text = (
                latest_known.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
                if latest_known is not None
                else "-"
            )
            latest_observation_text = str(latest_observation) if latest_observation is not None else "-"
            print(
                f"{series:10s} available={int(packet.available)} "
                f"known_obs={len(point_values):3d} latest_obs={latest_observation_text} "
                f"known_at={latest_known_text} age_days={packet.age_days:9.4f} "
                f"latest_change={packet.latest_change:+12.6f} "
                f"previous_change={packet.previous_change:+12.6f}"
            )
            if latest_known is not None and latest_known.astimezone(timezone.utc) > ts:
                raise AssertionError("future macro release leaked into causal audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
