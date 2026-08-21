from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _row_map(payload: dict) -> dict[str, dict]:
    return {
        str(row["decision_timestamp"]): row
        for row in payload.get("rows", [])
        if row.get("decision_timestamp")
    }


def compare_time_machine(v0: dict, v1: dict) -> dict[str, object]:
    v0_rows = _row_map(v0)
    v1_rows = _row_map(v1)
    shared = sorted(set(v0_rows) & set(v1_rows))
    if not shared:
        raise ValueError("no shared decision timestamps between time-machine JSON files")

    transitions: dict[str, int] = {}
    paired: list[dict[str, object]] = []
    for timestamp in shared:
        left = v0_rows[timestamp]
        right = v1_rows[timestamp]
        left_verdict = str(left.get("verdict"))
        right_verdict = str(right.get("verdict"))
        key = f"{left_verdict}->{right_verdict}"
        transitions[key] = transitions.get(key, 0) + 1
        paired.append(
            {
                "decision_timestamp": timestamp,
                "v0_verdict": left_verdict,
                "v1_verdict": right_verdict,
                "v0_direction": left.get("prediction_direction"),
                "v1_direction": right.get("prediction_direction"),
                "actual_direction": right.get("actual_direction", left.get("actual_direction")),
                "v0_confidence": left.get("confidence"),
                "v1_confidence": right.get("confidence"),
            }
        )

    def summary(rows: dict[str, dict]) -> dict[str, object]:
        selected = [rows[timestamp] for timestamp in shared]
        correct = sum(str(row.get("verdict")) == "CORRECT" for row in selected)
        wrong = sum(str(row.get("verdict")) == "WRONG" for row in selected)
        no_trade = sum(str(row.get("verdict")) == "NO_TRADE" for row in selected)
        actionable = correct + wrong
        return {
            "samples": len(selected),
            "actionable": actionable,
            "correct": correct,
            "wrong": wrong,
            "no_trade": no_trade,
            "coverage": actionable / len(selected) if selected else 0.0,
            "selected_accuracy": correct / actionable if actionable else None,
        }

    left_summary = summary(v0_rows)
    right_summary = summary(v1_rows)
    left_accuracy = left_summary["selected_accuracy"]
    right_accuracy = right_summary["selected_accuracy"]
    delta = None
    if left_accuracy is not None and right_accuracy is not None:
        delta = float(right_accuracy) - float(left_accuracy)

    return {
        "shared_timestamps": len(shared),
        "v0": left_summary,
        "v1": right_summary,
        "selected_accuracy_delta": delta,
        "transitions": transitions,
        "paired_rows": paired,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired comparison of V0 and V1 on identical time-machine timestamps"
    )
    parser.add_argument("--v0", required=True)
    parser.add_argument("--v1", required=True)
    parser.add_argument("--symbol", default="UNKNOWN")
    parser.add_argument("--output-json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compare_time_machine(_load(args.v0), _load(args.v1))
    result.update({"symbol": args.symbol, "v0_path": args.v0, "v1_path": args.v1})

    print(f"Multi-Market paired time-machine comparison | {args.symbol}")
    print("=" * 68)
    print(f"Shared timestamps : {result['shared_timestamps']}")
    for name in ("v0", "v1"):
        row = result[name]
        accuracy = row["selected_accuracy"]
        accuracy_text = f"{accuracy:.2%}" if accuracy is not None else "n/a"
        print(
            f"{name.upper():3s} accuracy={accuracy_text:>7s}  coverage={row['coverage']:.2%}  "
            f"actionable={row['actionable']:3d} correct={row['correct']:3d} wrong={row['wrong']:3d}"
        )
    delta = result["selected_accuracy_delta"]
    print(f"Accuracy delta    : {delta:+.2%}" if delta is not None else "Accuracy delta    : n/a")
    print("\nPaired verdict transitions")
    for key, value in sorted(result["transitions"].items()):
        print(f"  {key:24s} {value:3d}")

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Comparison JSON   : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
