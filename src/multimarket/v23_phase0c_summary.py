from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .v23_phase0c import EXPECTED_SENSORS


TARGETS = tuple(EXPECTED_SENSORS)


def summarize(payloads: Sequence[dict[str, object]]) -> dict[str, object]:
    by_symbol: dict[str, dict[str, object]] = {}
    for payload in payloads:
        symbol = str(payload.get("symbol", "")).upper()
        if symbol not in EXPECTED_SENSORS:
            raise ValueError(f"unexpected Phase 0C target: {symbol!r}")
        if symbol in by_symbol:
            raise ValueError(f"duplicate Phase 0C target payload: {symbol}")
        if payload.get("version") != "V2.3-PHASE0C-ASSET-SPECIFIC-REGIME":
            raise ValueError(f"unexpected Phase 0C payload version for {symbol}")
        by_symbol[symbol] = payload

    missing = [symbol for symbol in TARGETS if symbol not in by_symbol]
    target_status: dict[str, object] = {}
    available: list[str] = []
    passed: list[str] = []

    for symbol in TARGETS:
        payload = by_symbol.get(symbol)
        if payload is None:
            target_status[symbol] = {
                "status": "MISSING_PAYLOAD",
                "signal_pass": False,
                "selected_representation": None,
            }
            continue

        folds = payload.get("folds", [])
        scored = [fold for fold in folds if isinstance(fold, dict) and fold.get("status") == "SCORED"]
        if not scored:
            target_status[symbol] = {
                "status": "UNAVAILABLE",
                "signal_pass": False,
                "selected_representation": None,
                "scored_folds": 0,
            }
            continue

        available.append(symbol)
        signal_pass = bool(payload.get("target_signal_pass"))
        if signal_pass:
            passed.append(symbol)
        target_status[symbol] = {
            "status": "SIGNAL_PASS" if signal_pass else "SIGNAL_FAIL",
            "signal_pass": signal_pass,
            "selected_representation": payload.get("selected_representation"),
            "scored_folds": len(scored),
            "candidate_status": payload.get("candidate_status"),
        }

    if available and len(passed) == len(available):
        promotion = "PASS"
        decision = "PROMOTE_ALL_AVAILABLE_TARGETS_TO_PHASE0C_ECONOMIC_PREREGISTRATION"
    elif passed:
        promotion = "PARTIAL_PASS"
        decision = "PROMOTE_ONLY_SIGNAL_PASS_TARGETS_TO_PHASE0C_ECONOMIC_PREREGISTRATION"
    else:
        promotion = "FAIL"
        decision = "REJECT_PHASE0C_PRICE_LINKED_REGIME_CANDIDATES"

    return {
        "version": "V2.3-PHASE0C-SUMMARY",
        "phase0c_promotion": promotion,
        "decision": decision,
        "expected_targets": list(TARGETS),
        "missing_payloads": missing,
        "available_targets": available,
        "signal_pass_targets": passed,
        "signal_fail_targets": [symbol for symbol in available if symbol not in passed],
        "unavailable_targets": [
            symbol
            for symbol in TARGETS
            if target_status[symbol]["status"] in {"UNAVAILABLE", "MISSING_PAYLOAD"}
        ],
        "target_status": target_status,
        "holdout_status": "G/H/I_UNTOUCHED_REQUIRED",
        "economic_interpretation": (
            "Signal promotion only. Profitability, trading thresholds, transaction-cost gates, "
            "position sizing, and G/H/I scoring remain forbidden until separately preregistered."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize frozen V2.3 Phase 0C target signal results")
    parser.add_argument("json", nargs="+")
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.json]
    result = summarize(payloads)

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print(f"PHASE0C_PROMOTION={result['phase0c_promotion']}")
    print(f"decision={result['decision']}")
    print(f"available_targets={','.join(result['available_targets']) or 'NONE'}")
    print(f"signal_pass_targets={','.join(result['signal_pass_targets']) or 'NONE'}")
    print(f"unavailable_targets={','.join(result['unavailable_targets']) or 'NONE'}")
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
