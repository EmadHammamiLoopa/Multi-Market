from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .data import load_ohlc_csv
from .data_quality import audit_bars
from .models import Direction
from .v1_model import V1Config, WalkForwardXGBoostPredictor


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_baseline(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    if not rows:
        raise ValueError("baseline audit JSON contains no rows")
    return payload, rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run V1.1 on frozen V0 timestamps using a hard session/data-quality eligibility gate")
    p.add_argument("csv")
    p.add_argument("--symbol", required=True)
    p.add_argument("--baseline-json", required=True)
    p.add_argument("--horizon-bars", type=int, default=6)
    p.add_argument("--confidence-threshold", type=float, default=0.60)
    p.add_argument("--min-train-rows", type=int, default=5000)
    p.add_argument("--retrain-every", type=int, default=500)
    p.add_argument("--validation-fraction", type=float, default=0.20)
    p.add_argument("--output-json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bars = load_ohlc_csv(args.csv)
    quality = audit_bars(bars, symbol=args.symbol)
    hard_eligible = {
        row.index for row in quality
        if row.session_eligible and not row.zero_range and not row.repeated_ohlc
    }
    baseline, frozen_rows = _load_baseline(Path(args.baseline_json))
    index_by_timestamp = {bar.timestamp.astimezone(timezone.utc): i for i, bar in enumerate(bars)}
    config = V1Config(
        horizon_bars=args.horizon_bars,
        confidence_threshold=args.confidence_threshold,
        min_train_rows=args.min_train_rows,
        retrain_every=args.retrain_every,
        validation_fraction=args.validation_fraction,
    )
    predictor = WalkForwardXGBoostPredictor(bars, config, eligible_indices=hard_eligible)

    print(f"Multi-Market V1.1 clean-session paired audit | {args.symbol}")
    print("=" * 92)
    print("Hard gate: session eligible AND non-zero-range AND non-repeated-full-OHLC")
    print("Tiny-range alone remains eligible")

    output_rows = []
    correct = wrong = no_trade = gated = unavailable = 0
    for frozen in frozen_rows:
        decision_ts = _parse_timestamp(str(frozen["decision_timestamp"]))
        future_ts = _parse_timestamp(str(frozen["future_end_timestamp"]))
        idx = index_by_timestamp.get(decision_ts)
        future_idx = index_by_timestamp.get(future_ts)
        actual = Direction(str(frozen["actual_direction"]))
        if idx is None or future_idx is None or future_idx != idx + args.horizon_bars:
            unavailable += 1
            continue
        gate_ok = idx in hard_eligible and future_idx in hard_eligible
        if not gate_ok:
            direction = Direction.NO_TRADE
            confidence = None
            verdict = "NO_TRADE"
            gated += 1
            no_trade += 1
        else:
            try:
                pred = predictor.predict_at_index(idx)
            except ValueError:
                unavailable += 1
                continue
            direction = pred.direction
            confidence = pred.confidence
            if direction is Direction.NO_TRADE:
                verdict = "NO_TRADE"
                no_trade += 1
            elif direction is actual:
                verdict = "CORRECT"
                correct += 1
            else:
                verdict = "WRONG"
                wrong += 1
        conf_text = "-" if confidence is None else f"{confidence:.1%}"
        print(f"{decision_ts.strftime('%Y-%m-%d %H:%MZ'):19s} {direction.value:9s} {conf_text:>6s}  {actual.value:6s}  {verdict:9s} gate={'PASS' if gate_ok else 'BLOCK'}")
        output_rows.append({
            "decision_timestamp": decision_ts.isoformat().replace("+00:00", "Z"),
            "future_end_timestamp": future_ts.isoformat().replace("+00:00", "Z"),
            "prediction_direction": direction.value,
            "confidence": confidence,
            "actual_direction": actual.value,
            "verdict": verdict,
            "hard_gate_pass": gate_ok,
        })

    actionable = correct + wrong
    accuracy = correct / actionable if actionable else None
    print("\nAudit summary")
    print("=" * 58)
    print(f"Frozen samples        : {len(frozen_rows)}")
    print(f"Evaluated             : {len(output_rows)}")
    print(f"Gate-blocked          : {gated}")
    print(f"Unavailable           : {unavailable}")
    print(f"Actionable            : {actionable}")
    print(f"Correct               : {correct}")
    print(f"Wrong                 : {wrong}")
    print(f"NO_TRADE              : {no_trade}")
    print(f"Selected accuracy     : {accuracy:.2%}" if accuracy is not None else "Selected accuracy     : n/a")

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "symbol": args.symbol,
            "version": "V1.1-CLEAN-SESSION",
            "baseline_json": args.baseline_json,
            "baseline_dataset_sha256": baseline.get("dataset_sha256"),
            "hard_gate": "session eligible AND non-zero-range AND non-repeated-full-OHLC; tiny-range alone allowed",
            "frozen_samples": len(frozen_rows),
            "evaluated": len(output_rows),
            "gate_blocked": gated,
            "unavailable": unavailable,
            "actionable": actionable,
            "correct": correct,
            "wrong": wrong,
            "no_trade": no_trade,
            "selected_accuracy": accuracy,
            "rows": output_rows,
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Audit JSON            : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
