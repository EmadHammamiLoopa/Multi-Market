from __future__ import annotations

import argparse
import json
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from math import log1p
from pathlib import Path
from typing import Sequence

from .data import load_ohlc_csv
from .v21_common import load_peer_markets
from .v21_features import PeerMarket
from .v23_phase0c import (
    PRIMARY_HORIZON,
    SENSOR_RETURN_WINDOWS,
    Phase0CRow,
    _forward_close_return_bps,
    _forward_executable_return_bps,
    _own_and_regime_features,
    _rv24_series,
    _window_return_bps,
    build_phase0c_rows as _unused_build_phase0c_rows,
    evaluate_rows,
    hard_eligible_indices,
    load_phase0b_fold_windows,
    load_phase0c_manifest,
    validate_linked_peers,
    _is_reserved,
)


@dataclass(frozen=True, slots=True)
class AsyncSensorSeries:
    timestamps: tuple[datetime, ...]
    packets: tuple[tuple[float, float, float], ...]


def _sha256_file(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _prepare_async_sensor(peer: PeerMarket) -> AsyncSensorSeries:
    timestamps: list[datetime] = []
    packets: list[tuple[float, float, float]] = []
    bars = peer.bars
    eligible = peer.eligible_indices

    for index, bar in enumerate(bars):
        if index not in eligible or _is_reserved(bar.timestamp):
            continue
        values: list[float] = []
        valid = True
        for window in SENSOR_RETURN_WINDOWS:
            value = _window_return_bps(bars, index, window, eligible)
            if value is None:
                valid = False
                break
            values.append(float(value))
        if valid:
            timestamps.append(bar.timestamp)
            packets.append((values[0], values[1], values[2]))

    return AsyncSensorSeries(tuple(timestamps), tuple(packets))


def _async_sensor_packet(
    sensor: AsyncSensorSeries,
    decision_timestamp: datetime,
) -> tuple[float, ...] | None:
    pos = bisect_right(sensor.timestamps, decision_timestamp) - 1
    if pos < 0:
        return None
    source_timestamp = sensor.timestamps[pos]
    age_seconds = (decision_timestamp - source_timestamp).total_seconds()
    if age_seconds < 0:
        raise AssertionError("future sensor packet selected")
    age_hours = age_seconds / 3600.0
    fresh_5m = 1.0 if age_seconds <= 300.0 else 0.0
    return sensor.packets[pos] + (log1p(age_hours), fresh_5m)


def build_phase0cr_rows(
    bars,
    *,
    symbol: str,
    linked_peers: dict[str, PeerMarket],
) -> list[Phase0CRow]:
    eligible = hard_eligible_indices(bars, symbol)
    prepared = {name: _prepare_async_sensor(peer) for name, peer in linked_peers.items()}
    rv24 = _rv24_series(bars, eligible)
    result: list[Phase0CRow] = []

    for index, bar in enumerate(bars):
        if index not in eligible or _is_reserved(bar.timestamp):
            continue

        linked: list[float] = []
        available = True
        for name in prepared:  # frozen manifest insertion order
            packet = _async_sensor_packet(prepared[name], bar.timestamp)
            if packet is None:
                available = False
                break
            linked.extend(packet)
        if not available:
            continue

        own_state = _own_and_regime_features(bars, index, eligible, rv24=rv24)
        if own_state is None:
            continue
        own, regime, vol_pct, jump_state = own_state

        forward = _forward_close_return_bps(bars, index, eligible)
        executable = _forward_executable_return_bps(bars, index, eligible)
        if forward is None or executable is None:
            continue
        executable_return, exit_timestamp = executable
        result.append(
            Phase0CRow(
                timestamp=bar.timestamp,
                label_end_timestamp=bars[index + PRIMARY_HORIZON].timestamp,
                execution_exit_timestamp=exit_timestamp,
                own_features=own,
                linked_features=own + tuple(linked),
                regime_features=own + tuple(linked) + regime,
                volatility_percentile=vol_pct,
                forward_6_bps=forward,
                executable_forward_6_bps=executable_return,
                jump_state=jump_state,
            )
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V2.3 Phase 0C-R asynchronous linked-sensor feasibility repair"
    )
    parser.add_argument("csv")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--peer", action="append", default=[], metavar="SYMBOL=CSV")
    parser.add_argument("--phase0b-json", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--round-trip-cost-bps", type=float, default=None)
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bars = load_ohlc_csv(args.csv)
    peers = load_peer_markets(args.peer, target_symbol=args.symbol)
    manifest, linked_order = load_phase0c_manifest(args.manifest, symbol=args.symbol)
    validate_linked_peers(peers, linked_order)
    peers = {name: peers[name] for name in linked_order}
    fold_windows = load_phase0b_fold_windows(args.phase0b_json)

    print(
        f"Building V2.3 Phase 0C-R rows | {args.symbol.upper()} | "
        f"linked={','.join(linked_order)} | alignment=ASYNC_LAST_VALID",
        flush=True,
    )
    rows = build_phase0cr_rows(bars, symbol=args.symbol, linked_peers=peers)
    print(f"eligible_phase0cr_rows={len(rows)}", flush=True)

    payload = evaluate_rows(
        rows,
        symbol=args.symbol,
        fold_windows=fold_windows,
        round_trip_cost_bps=args.round_trip_cost_bps,
    )
    payload["version"] = "V2.3-PHASE0CR-ASYNC-SENSOR-FEASIBILITY-REPAIR"
    payload["alignment"] = {
        "mode": "ASYNC_LAST_VALID_CAUSAL_PACKET",
        "sensor_return_windows": list(SENSOR_RETURN_WINDOWS),
        "staleness_features": ["log1p_age_hours", "fresh_5m"],
        "maximum_staleness_cutoff": None,
        "future_fill": False,
    }
    payload["frozen_manifest"] = {
        "path": str(Path(args.manifest)),
        "sha256": _sha256_file(args.manifest),
        "linked_sensor_order": list(linked_order),
        "manifest_version": manifest.get("version"),
    }
    payload["phase0b_boundary_source"] = {
        "path": str(Path(args.phase0b_json)),
        "sha256": _sha256_file(args.phase0b_json),
    }

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"signal_candidate={payload['signal_candidate']}", flush=True)
    print(f"promoted_candidate={payload['promoted_candidate']}", flush=True)
    print(f"Output: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
