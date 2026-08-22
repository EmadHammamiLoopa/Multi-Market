from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import timedelta
from pathlib import Path

from .cross_market import causal_peer_snapshot
from .data import load_ohlc_csv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit causal as-of cross-market alignment without training a model")
    p.add_argument("target_csv")
    p.add_argument("--target-symbol", required=True)
    p.add_argument("--peer", action="append", default=[], metavar="SYMBOL=CSV", help="Peer market mapping; repeatable")
    p.add_argument("--max-staleness-minutes", type=int, default=15)
    p.add_argument("--output-json")
    return p


def _parse_peer(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError("--peer must use SYMBOL=CSV")
    symbol, path = value.split("=", 1)
    symbol = symbol.strip().upper()
    path = path.strip()
    if not symbol or not path:
        raise ValueError("--peer must use non-empty SYMBOL=CSV")
    return symbol, path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_staleness_minutes < 0:
        raise SystemExit("--max-staleness-minutes must be non-negative")
    target_bars = load_ohlc_csv(args.target_csv)
    peers = dict(_parse_peer(value) for value in args.peer)
    if not peers:
        raise SystemExit("at least one --peer SYMBOL=CSV is required")
    peer_bars = {symbol: load_ohlc_csv(path) for symbol, path in peers.items()}
    max_staleness = timedelta(minutes=args.max_staleness_minutes)

    payload_peers: dict[str, object] = {}
    print(f"Multi-Market causal cross-market alignment audit | {args.target_symbol.upper()}")
    print("=" * 86)
    print(f"Target bars              : {len(target_bars)}")
    print(f"Max peer staleness       : {args.max_staleness_minutes} minutes")

    for symbol, bars in peer_bars.items():
        available = 0
        future_violations = 0
        stale_or_missing = 0
        staleness_counts: Counter[int] = Counter()
        feature_complete = 0
        for target in target_bars:
            snapshot = causal_peer_snapshot(bars, target.timestamp, max_staleness=max_staleness)
            if snapshot is None:
                stale_or_missing += 1
                continue
            available += 1
            if snapshot.peer_timestamp > target.timestamp:
                future_violations += 1
            minutes = int(snapshot.staleness.total_seconds() // 60)
            staleness_counts[minutes] += 1
            if None not in (snapshot.ret_1_bps, snapshot.ret_6_bps, snapshot.ret_12_bps, snapshot.vol_12_bps):
                feature_complete += 1

        coverage = available / len(target_bars)
        complete_coverage = feature_complete / len(target_bars)
        print(
            f"{symbol:8s} available={available:7d} ({coverage:6.2%}) "
            f"complete={feature_complete:7d} ({complete_coverage:6.2%}) "
            f"stale/missing={stale_or_missing:7d} future={future_violations}"
        )
        payload_peers[symbol] = {
            "peer_bars": len(bars),
            "available": available,
            "coverage": coverage,
            "feature_complete": feature_complete,
            "feature_complete_coverage": complete_coverage,
            "stale_or_missing": stale_or_missing,
            "future_violations": future_violations,
            "staleness_minutes_histogram": dict(sorted(staleness_counts.items())),
        }

    payload = {
        "target_symbol": args.target_symbol.upper(),
        "target_bars": len(target_bars),
        "max_staleness_minutes": args.max_staleness_minutes,
        "peers": payload_peers,
        "note": "diagnostic only; as-of alignment uses latest peer timestamp <= target timestamp and never interpolates future data",
    }
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Audit JSON               : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
