from __future__ import annotations

import argparse
import json
from pathlib import Path

from .v23_phase0_summary import MODELS, PRIMARY_HORIZON, _sse_and_sst


def _target_increment(payload: dict[str, object], model: str) -> float | None:
    block = payload.get("pooled", {}).get(model, {}).get(PRIMARY_HORIZON, {})
    value = block.get("incremental_r2")
    return None if value is None else float(value)


def _pooled_incremental(
    payloads: list[dict[str, object]],
    *,
    model_name: str,
    non_jump: bool,
) -> float | None:
    base_sse = cross_sse = total_sst = 0.0
    key = "non_jump" if non_jump else "all"
    used_targets = 0

    for payload in payloads:
        block = payload.get("pooled", {}).get(model_name, {}).get(PRIMARY_HORIZON, {})
        base_block = block.get("base")
        cross_block = block.get("cross")
        if not base_block or not cross_block:
            continue
        base = base_block.get(key)
        cross = cross_block.get(key)
        if base is None or cross is None:
            continue

        base_piece, base_sst = _sse_and_sst(base)
        cross_piece, cross_sst = _sse_and_sst(cross)
        if abs(base_sst - cross_sst) > max(1e-8, 1e-8 * abs(base_sst)):
            raise AssertionError("BASE and CROSS must share realized target variance")
        base_sse += base_piece
        cross_sse += cross_piece
        total_sst += base_sst
        used_targets += 1

    if used_targets == 0 or total_sst <= 0.0:
        return None
    base_r2 = 1.0 - base_sse / total_sst
    cross_r2 = 1.0 - cross_sse / total_sst
    return cross_r2 - base_r2


def summarize(payloads: list[dict[str, object]]) -> dict[str, object]:
    if len(payloads) != 5:
        raise ValueError(f"expected exactly five target payloads; got {len(payloads)}")
    symbols = [str(payload["symbol"]).upper() for payload in payloads]
    if len(set(symbols)) != 5:
        raise ValueError("target symbols must be unique")

    pooled = {
        model: _pooled_incremental(payloads, model_name=model, non_jump=False)
        for model in MODELS
    }
    pooled_non_jump = {
        model: _pooled_incremental(payloads, model_name=model, non_jump=True)
        for model in MODELS
    }

    target_rows: list[dict[str, object]] = []
    positive_targets = 0
    unavailable_targets = 0
    for payload in payloads:
        increments = [_target_increment(payload, model) for model in MODELS]
        if any(value is None for value in increments):
            unavailable_targets += 1
            target_rows.append(
                {
                    "symbol": str(payload["symbol"]).upper(),
                    "status": "UNAVAILABLE_NO_SCORED_FOLDS",
                    "ridge_incremental_r2": increments[0],
                    "elasticnet_incremental_r2": increments[1],
                    "mean_model_incremental_r2": None,
                    "positive": False,
                }
            )
            continue
        values = [float(value) for value in increments if value is not None]
        mean_increment = sum(values) / len(values)
        positive_targets += int(mean_increment > 0.0)
        target_rows.append(
            {
                "symbol": str(payload["symbol"]).upper(),
                "status": "SCORED",
                "ridge_incremental_r2": values[0],
                "elasticnet_incremental_r2": values[1],
                "mean_model_incremental_r2": mean_increment,
                "positive": mean_increment > 0.0,
            }
        )

    scored_cells = positive_cells = 0
    fold_rows: list[dict[str, object]] = []
    for payload in payloads:
        symbol = str(payload["symbol"]).upper()
        for fold in payload["folds"]:
            if fold["status"] != "SCORED":
                continue
            increments = [
                float(fold["models"][model][PRIMARY_HORIZON]["incremental_r2"])
                for model in MODELS
            ]
            mean_increment = sum(increments) / len(increments)
            scored_cells += 1
            positive_cells += int(mean_increment > 0.0)
            fold_rows.append(
                {
                    "symbol": symbol,
                    "fold": int(fold["fold"]),
                    "mean_model_incremental_r2": mean_increment,
                    "positive": mean_increment > 0.0,
                }
            )

    positive_cell_fraction = positive_cells / scored_cells if scored_cells else 0.0
    conditions = {
        "pooled_positive_both_models": all(
            pooled[model] is not None and pooled[model] > 0.0 for model in MODELS
        ),
        "at_least_three_positive_targets": positive_targets >= 3,
        "at_least_60pct_positive_target_fold_cells": positive_cell_fraction >= 0.60,
        "non_jump_nonnegative_both_models": all(
            pooled_non_jump[model] is not None and pooled_non_jump[model] >= 0.0
            for model in MODELS
        ),
    }
    promoted = all(conditions.values())

    return {
        "version": "V2.3-PHASE0-INFORMATION-DIFFUSION-SUMMARY",
        "targets": sorted(symbols),
        "primary_horizon_bars": 6,
        "pooled_incremental_r2": pooled,
        "pooled_non_jump_incremental_r2": pooled_non_jump,
        "target_results": sorted(target_rows, key=lambda row: row["symbol"]),
        "positive_targets": positive_targets,
        "unavailable_targets": unavailable_targets,
        "positive_target_requirement_denominator": 5,
        "scored_target_fold_cells": scored_cells,
        "positive_target_fold_cells": positive_cells,
        "positive_target_fold_fraction": positive_cell_fraction,
        "fold_results": sorted(fold_rows, key=lambda row: (row["symbol"], row["fold"])),
        "promotion_conditions": conditions,
        "promotion_pass": promoted,
        "decision": "PROMOTE_TO_V23_PHASE1" if promoted else "REJECT_DENSE_CROSS_SECTIONAL_BLOCK",
        "post_hoc_selection_allowed": False,
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.8f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply frozen V2.3 Phase 0 promotion rule with unavailable-target handling")
    parser.add_argument("result_json", nargs=5)
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.result_json]
    summary = summarize(payloads)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print("===== V2.3 PHASE 0 PROMOTION =====")
    for model, value in summary["pooled_incremental_r2"].items():
        print(f"pooled_6bar_{model}_incremental_r2={_fmt(value)}")
    for model, value in summary["pooled_non_jump_incremental_r2"].items():
        print(f"pooled_non_jump_6bar_{model}_incremental_r2={_fmt(value)}")
    print(f"positive_targets={summary['positive_targets']}/5")
    print(f"unavailable_targets={summary['unavailable_targets']}/5")
    print(
        f"positive_target_fold_cells={summary['positive_target_fold_cells']}/"
        f"{summary['scored_target_fold_cells']} "
        f"({summary['positive_target_fold_fraction']:.2%})"
    )
    for name, passed in summary["promotion_conditions"].items():
        print(f"{name}={'PASS' if passed else 'FAIL'}")
    print(f"PHASE0_PROMOTION={'PASS' if summary['promotion_pass'] else 'FAIL'}")
    print(f"decision={summary['decision']}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
