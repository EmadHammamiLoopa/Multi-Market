from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .data import load_ohlc_csv
from .execution import simulate_barrier_trade
from .models import Direction
from .v1_model import V1Config, WalkForwardXGBoostPredictor


@dataclass(frozen=True, slots=True)
class FrozenBaselineRow:
    decision_timestamp: datetime
    future_end_timestamp: datetime
    reference_price: float
    future_end_price: float
    actual_direction: Direction
    actual_return_bps: float


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_baseline(path: Path) -> tuple[dict[str, object], list[FrozenBaselineRow]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    frozen: list[FrozenBaselineRow] = []
    for row in rows:
        frozen.append(
            FrozenBaselineRow(
                decision_timestamp=_parse_timestamp(str(row["decision_timestamp"])),
                future_end_timestamp=_parse_timestamp(str(row["future_end_timestamp"])),
                reference_price=float(row["reference_price"]),
                future_end_price=float(row["future_end_price"]),
                actual_direction=Direction(str(row["actual_direction"])),
                actual_return_bps=float(row["actual_return_bps"]),
            )
        )
    if not frozen:
        raise ValueError("baseline audit JSON contains no rows")
    return payload, frozen


def _drift_bps(current: float, frozen: float) -> float:
    if frozen == 0:
        return 0.0
    return (current / frozen - 1.0) * 10_000.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run V1 on the exact timestamps and frozen targets from a V0 time-machine audit JSON"
        )
    )
    parser.add_argument("csv")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--horizon-bars", type=int, default=6)
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    parser.add_argument("--min-train-rows", type=int, default=1000)
    parser.add_argument("--retrain-every", type=int, default=250)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--take-profit-bps", type=float, default=20.0)
    parser.add_argument("--stop-loss-bps", type=float, default=10.0)
    parser.add_argument("--round-trip-cost-bps", type=float, default=2.0)
    parser.add_argument("--output-json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bars = load_ohlc_csv(args.csv)
    index_by_timestamp = {
        bar.timestamp.astimezone(timezone.utc): i for i, bar in enumerate(bars)
    }
    baseline_payload, frozen_rows = _load_baseline(Path(args.baseline_json))
    baseline_horizon = baseline_payload.get("horizon_bars")
    if baseline_horizon is not None and int(baseline_horizon) != args.horizon_bars:
        raise SystemExit(
            f"baseline horizon is {baseline_horizon} bars but --horizon-bars is {args.horizon_bars}"
        )

    config = V1Config(
        horizon_bars=args.horizon_bars,
        confidence_threshold=args.confidence_threshold,
        min_train_rows=args.min_train_rows,
        retrain_every=args.retrain_every,
        validation_fraction=args.validation_fraction,
    )
    predictor = WalkForwardXGBoostPredictor(bars, config)

    print(f"Multi-Market V1 strict paired time-machine audit | {args.symbol}")
    print("=" * 108)
    print(f"Frozen V0 samples    : {len(frozen_rows)}")
    print(
        "Comparison rule     : exact V0 decision timestamp + exact V0 future timestamp + frozen V0 label"
    )
    print("Training rule       : V1 sees only labels whose complete horizon ended before decision time")
    print()
    print(
        "decision time       prediction  conf    frozen   move(bps)  verdict    price drift(dec/end)  trade*"
    )
    print("-" * 108)

    output_rows: list[dict[str, object]] = []
    correct = wrong = no_trade = unavailable = horizon_mismatch = 0
    drifted_prices = 0
    max_abs_decision_drift_bps = 0.0
    max_abs_future_drift_bps = 0.0

    for frozen in frozen_rows:
        index = index_by_timestamp.get(frozen.decision_timestamp)
        future_index = index_by_timestamp.get(frozen.future_end_timestamp)
        if index is None or future_index is None:
            unavailable += 1
            continue

        expected_future_index = index + args.horizon_bars
        if expected_future_index >= len(bars) or expected_future_index != future_index:
            horizon_mismatch += 1
            unavailable += 1
            print(
                f"{frozen.decision_timestamp.strftime('%Y-%m-%d %H:%MZ'):19s} "
                f"{'UNAVAILABLE':9s} {'-':6s}  {frozen.actual_direction.value:6s} "
                f"{frozen.actual_return_bps:+9.2f}  HORIZON_MISMATCH"
            )
            continue

        try:
            prediction = predictor.predict_at_index(index)
        except ValueError:
            unavailable += 1
            continue

        current_decision_close = bars[index].close
        current_future_close = bars[future_index].close
        decision_drift = _drift_bps(current_decision_close, frozen.reference_price)
        future_drift = _drift_bps(current_future_close, frozen.future_end_price)
        max_abs_decision_drift_bps = max(max_abs_decision_drift_bps, abs(decision_drift))
        max_abs_future_drift_bps = max(max_abs_future_drift_bps, abs(future_drift))
        if abs(decision_drift) > 1e-9 or abs(future_drift) > 1e-9:
            drifted_prices += 1

        # Directional scoring is against the original frozen V0 outcome, not a re-fetched label.
        actual_direction = frozen.actual_direction
        if prediction.direction is Direction.NO_TRADE:
            verdict = "NO_TRADE"
            trade_outcome = "-"
            no_trade += 1
            trade = None
        else:
            verdict = "CORRECT" if prediction.direction is actual_direction else "WRONG"
            if verdict == "CORRECT":
                correct += 1
            else:
                wrong += 1

            # Trade path is diagnostic only: intermediate OHLC bars may have been revised on re-fetch.
            future = bars[index + 1 : future_index + 1]
            trade = simulate_barrier_trade(
                prediction,
                future,
                take_profit_bps=args.take_profit_bps,
                stop_loss_bps=args.stop_loss_bps,
                round_trip_cost_bps=args.round_trip_cost_bps,
            )
            trade_outcome = f"{trade.exit_reason} {trade.net_return_bps:+.2f}bp"

        print(
            f"{frozen.decision_timestamp.strftime('%Y-%m-%d %H:%MZ'):19s} "
            f"{prediction.direction.value:9s} {prediction.confidence:6.1%}  "
            f"{actual_direction.value:6s} {frozen.actual_return_bps:+9.2f}  "
            f"{verdict:9s}  {decision_drift:+.3f}/{future_drift:+.3f}bp  {trade_outcome}"
        )
        output_rows.append(
            {
                "decision_timestamp": frozen.decision_timestamp.isoformat().replace("+00:00", "Z"),
                "future_end_timestamp": frozen.future_end_timestamp.isoformat().replace("+00:00", "Z"),
                "prediction_direction": prediction.direction.value,
                "confidence": prediction.confidence,
                "actual_direction": actual_direction.value,
                "actual_return_bps": frozen.actual_return_bps,
                "verdict": verdict,
                "frozen_reference_price": frozen.reference_price,
                "current_reference_price": current_decision_close,
                "decision_price_drift_bps": decision_drift,
                "frozen_future_end_price": frozen.future_end_price,
                "current_future_end_price": current_future_close,
                "future_price_drift_bps": future_drift,
                "trade_exit_reason": trade.exit_reason if trade else None,
                "trade_net_return_bps": trade.net_return_bps if trade else None,
                "trade_note": "diagnostic on re-fetched OHLC path; directional label is frozen V0",
            }
        )

    actionable = correct + wrong
    accuracy = correct / actionable if actionable else None
    print("\nAudit summary")
    print("=" * 58)
    print(f"Frozen V0 samples     : {len(frozen_rows)}")
    print(f"Evaluated             : {len(output_rows)}")
    print(f"Unavailable           : {unavailable}")
    print(f"Horizon mismatches    : {horizon_mismatch}")
    print(f"Actionable            : {actionable}")
    print(f"Correct               : {correct}")
    print(f"Wrong                 : {wrong}")
    print(f"NO_TRADE              : {no_trade}")
    print(f"Selected accuracy     : {accuracy:.2%}" if accuracy is not None else "Selected accuracy     : n/a")
    print(f"Rows with price drift : {drifted_prices}")
    print(f"Max decision drift    : {max_abs_decision_drift_bps:.6f} bp")
    print(f"Max future drift      : {max_abs_future_drift_bps:.6f} bp")
    print("Scoring target        : frozen V0 actual direction")
    print("Trade outcome note    : diagnostic only when re-fetched OHLC differs")

    if args.output_json:
        payload = {
            "symbol": args.symbol,
            "version": "V1-XGB-STRICT-PAIRED-TIME-MACHINE",
            "baseline_json": args.baseline_json,
            "baseline_dataset_sha256": baseline_payload.get("dataset_sha256"),
            "frozen_samples": len(frozen_rows),
            "evaluated": len(output_rows),
            "unavailable": unavailable,
            "horizon_mismatch": horizon_mismatch,
            "actionable": actionable,
            "correct": correct,
            "wrong": wrong,
            "no_trade": no_trade,
            "selected_accuracy": accuracy,
            "drifted_prices": drifted_prices,
            "max_abs_decision_drift_bps": max_abs_decision_drift_bps,
            "max_abs_future_drift_bps": max_abs_future_drift_bps,
            "scoring_target": "frozen V0 actual_direction",
            "rows": output_rows,
        }
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Audit JSON            : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
