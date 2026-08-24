from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


REQUIRED_STATISTICAL_FOLDS = 4


def _candidate_scored_folds(payload: dict[str, object]) -> int:
    candidates = payload.get("candidates", {})
    if not isinstance(candidates, dict):
        return 0
    counts: list[int] = []
    for name in ("C2", "C3"):
        block = candidates.get(name, {})
        if isinstance(block, dict):
            try:
                counts.append(int(block.get("scored_folds", 0)))
            except (TypeError, ValueError):
                counts.append(0)
    return max(counts, default=0)


def summarize(payloads: Sequence[dict[str, object]]) -> dict[str, object]:
    targets: list[dict[str, object]] = []
    promoted: list[str] = []
    signal_candidates: list[str] = []
    unavailable: list[str] = []
    inconclusive: list[str] = []

    for payload in payloads:
        symbol = str(payload.get("symbol", "UNKNOWN")).upper()
        status = str(payload.get("evaluation_status", "SCORED"))
        if status != "SCORED":
            unavailable.append(symbol)
            targets.append(
                {
                    "symbol": symbol,
                    "evaluation_status": status,
                    "reason": payload.get("reason"),
                }
            )
            continue

        scored_folds = _candidate_scored_folds(payload)
        if scored_folds < REQUIRED_STATISTICAL_FOLDS:
            inconclusive.append(symbol)
            targets.append(
                {
                    "symbol": symbol,
                    "evaluation_status": "INCONCLUSIVE_INSUFFICIENT_FOLDS",
                    "reason": (
                        f"Phase 0C statistical promotion requires at least "
                        f"{REQUIRED_STATISTICAL_FOLDS} scored folds; observed {scored_folds}"
                    ),
                    "scored_folds": scored_folds,
                    "required_scored_folds": REQUIRED_STATISTICAL_FOLDS,
                    "row_count": payload.get("row_count"),
                    "candidates": payload.get("candidates", {}),
                }
            )
            continue

        signal_candidate = payload.get("signal_candidate")
        promoted_candidate = payload.get("promoted_candidate")
        cost_status = str(payload.get("cost_model_status", "MISSING"))
        if signal_candidate:
            signal_candidates.append(symbol)
        if promoted_candidate:
            promoted.append(symbol)

        targets.append(
            {
                "symbol": symbol,
                "evaluation_status": "SCORED",
                "signal_candidate": signal_candidate,
                "promoted_candidate": promoted_candidate,
                "promotion_pass": bool(payload.get("promotion_pass")),
                "cost_model_status": cost_status,
                "scored_folds": scored_folds,
                "candidates": payload.get("candidates", {}),
            }
        )

    scored_count = sum(item.get("evaluation_status") == "SCORED" for item in targets)
    pending_cost_targets = [
        str(item["symbol"])
        for item in targets
        if item.get("evaluation_status") == "SCORED"
        and item.get("signal_candidate")
        and item.get("cost_model_status") == "MISSING"
    ]

    if promoted:
        promotion = "PASS" if len(promoted) == scored_count and scored_count > 0 else "PARTIAL_PASS"
        decision = "PROMOTE_ASSET_SPECIFIC_TARGETS"
    elif pending_cost_targets:
        promotion = "PENDING_COST_MODEL"
        decision = "SIGNAL_CANDIDATES_REQUIRE_COST_MODEL"
    elif scored_count > 0 and inconclusive:
        promotion = "PARTIAL_INCONCLUSIVE"
        decision = "NO_BLOCK_DECISION_MIXED_FEASIBILITY"
    elif scored_count > 0:
        promotion = "FAIL"
        decision = "REJECT_PRICE_ONLY_ASSET_SPECIFIC_BLOCK"
    else:
        promotion = "INCONCLUSIVE"
        decision = "REQUIRES_FEASIBILITY_REPAIR"

    return {
        "version": "V2.3-PHASE0C-SUMMARY",
        "required_statistical_folds": REQUIRED_STATISTICAL_FOLDS,
        "targets": targets,
        "scored_targets": scored_count,
        "inconclusive_targets": inconclusive,
        "unavailable_targets": unavailable,
        "signal_candidate_targets": signal_candidates,
        "pending_cost_targets": pending_cost_targets,
        "promoted_targets": promoted,
        "phase0c_promotion": promotion,
        "decision": decision,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize V2.3 Phase 0C target-level results")
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
    print(f"signal_candidate_targets={','.join(result['signal_candidate_targets']) or 'NONE'}")
    print(f"promoted_targets={','.join(result['promoted_targets']) or 'NONE'}")
    print(f"inconclusive_targets={','.join(result['inconclusive_targets']) or 'NONE'}")
    print(f"unavailable_targets={','.join(result['unavailable_targets']) or 'NONE'}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
