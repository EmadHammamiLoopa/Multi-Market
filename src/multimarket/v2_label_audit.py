from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .data import load_ohlc_csv
from .data_quality import audit_bars
from .models import Direction
from .reporting import sha256_file
from .v2_labels import build_economic_event, horizon_is_contiguous


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Audit V2 realized barrier/economic labels before model training"
    )
    p.add_argument("csv")
    p.add_argument("--symbol", required=True)
    p.add_argument("--horizon-bars", type=int, default=6)
    p.add_argument("--take-profit-bps", type=float, default=20.0)
    p.add_argument("--stop-loss-bps", type=float, default=10.0)
    p.add_argument("--round-trip-cost-bps", type=float, default=2.0)
    p.add_argument("--output-json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bars = load_ohlc_csv(args.csv)
    quality = audit_bars(bars, symbol=args.symbol)
    hard_eligible = {
        row.index
        for row in quality
        if row.session_eligible and not row.zero_range and not row.repeated_ohlc
    }

    counts: Counter[str] = Counter()
    exit_long: Counter[str] = Counter()
    exit_short: Counter[str] = Counter()
    long_nets: list[float] = []
    short_nets: list[float] = []

    max_decision = len(bars) - args.horizon_bars
    for decision_index in range(max_decision):
        if not horizon_is_contiguous(bars, decision_index, args.horizon_bars):
            counts["rejected_gap"] += 1
            continue
        event = build_economic_event(
            bars,
            decision_index,
            horizon_bars=args.horizon_bars,
            take_profit_bps=args.take_profit_bps,
            stop_loss_bps=args.stop_loss_bps,
            round_trip_cost_bps=args.round_trip_cost_bps,
            eligible_indices=hard_eligible,
        )
        if event is None:
            counts["rejected_quality_or_session"] += 1
            continue

        counts["eligible_events"] += 1
        exit_long[event.long.exit_reason] += 1
        exit_short[event.short.exit_reason] += 1
        long_nets.append(event.long.net_return_bps)
        short_nets.append(event.short.net_return_bps)
        if event.long.net_return_bps > 0:
            counts["long_net_positive"] += 1
        if event.short.net_return_bps > 0:
            counts["short_net_positive"] += 1
        counts[f"best_{event.best_realized_direction.value.lower()}"] += 1

    eligible = counts["eligible_events"]

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    payload = {
        "symbol": args.symbol,
        "version": "V2-LABEL-AUDIT",
        "dataset_sha256": sha256_file(args.csv),
        "horizon_bars": args.horizon_bars,
        "take_profit_bps": args.take_profit_bps,
        "stop_loss_bps": args.stop_loss_bps,
        "round_trip_cost_bps": args.round_trip_cost_bps,
        "candidate_decisions": max_decision,
        "eligible_events": eligible,
        "rejected_gap": counts["rejected_gap"],
        "rejected_quality_or_session": counts["rejected_quality_or_session"],
        "long_net_positive": counts["long_net_positive"],
        "short_net_positive": counts["short_net_positive"],
        "best_long": counts["best_long"],
        "best_short": counts["best_short"],
        "best_no_trade": counts["best_no_trade"],
        "long_exit_reasons": dict(sorted(exit_long.items())),
        "short_exit_reasons": dict(sorted(exit_short.items())),
        "mean_realized_long_net_bps": mean(long_nets),
        "mean_realized_short_net_bps": mean(short_nets),
        "note": (
            "Label diagnostic only. best_* is ex-post descriptive and must not be used "
            "as a live decision rule. V2 predictors will estimate side-specific probabilities/EV."
        ),
    }

    print(f"Multi-Market V2 economic-label audit | {args.symbol}")
    print("=" * 76)
    print(f"Dataset SHA-256             : {payload['dataset_sha256']}")
    print(f"Candidate decisions         : {max_decision}")
    print(f"Eligible contiguous events  : {eligible}")
    print(f"Rejected time gaps          : {counts['rejected_gap']}")
    print(f"Rejected session/quality    : {counts['rejected_quality_or_session']}")
    if eligible:
        print(
            f"LONG net-positive            : {counts['long_net_positive']} "
            f"({counts['long_net_positive'] / eligible:.2%})"
        )
        print(
            f"SHORT net-positive           : {counts['short_net_positive']} "
            f"({counts['short_net_positive'] / eligible:.2%})"
        )
        print(
            "Best realized diagnostic   : "
            f"LONG={counts['best_long']} SHORT={counts['best_short']} "
            f"NO_TRADE={counts['best_no_trade']}"
        )
        print(f"Mean LONG net bps            : {mean(long_nets):+.3f}")
        print(f"Mean SHORT net bps           : {mean(short_nets):+.3f}")
    print("LONG exits                   : " + json.dumps(dict(sorted(exit_long.items()))))
    print("SHORT exits                  : " + json.dumps(dict(sorted(exit_short.items()))))

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Audit JSON                   : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
