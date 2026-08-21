from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .data import load_ohlc_csv
from .execution import simulate_barrier_trade
from .models import Direction
from .v1_model import V1Config, WalkForwardXGBoostPredictor


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _baseline_timestamps(path: Path) -> list[datetime]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    timestamps = [_parse_timestamp(str(row["decision_timestamp"])) for row in rows]
    if not timestamps:
        raise ValueError("baseline audit JSON contains no rows")
    return timestamps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run V1 on the exact timestamps frozen by a V0 time-machine audit JSON"
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
    index_by_timestamp = {bar.timestamp.astimezone(timezone.utc): i for i, bar in enumerate(bars)}
    timestamps = _baseline_timestamps(Path(args.baseline_json))
    config = V1Config(
        horizon_bars=args.horizon_bars,
        confidence_threshold=args.confidence_threshold,
        min_train_rows=args.min_train_rows,
        retrain_every=args.retrain_every,
        validation_fraction=args.validation_fraction,
    )
    predictor = WalkForwardXGBoostPredictor(bars, config)

    print(f"Multi-Market V1 time-machine audit | {args.symbol}")
    print("=" * 98)
    print(f"Baseline timestamps : {len(timestamps)}")
    print("Comparison rule     : exact V0 decision timestamps; V1 sees only prior known labels")
    print()
    print("decision time       prediction  conf    actual   move(bps)  verdict    trade outcome")
    print("-" * 98)

    output_rows: list[dict[str, object]] = []
    correct = wrong = no_trade = unavailable = 0
    for timestamp in timestamps:
        index = index_by_timestamp.get(timestamp)
        if index is None or index + args.horizon_bars >= len(bars):
            unavailable += 1
            continue
        try:
            prediction = predictor.predict_at_index(index)
        except ValueError:
            unavailable += 1
            continue

        future = bars[index + 1 : index + 1 + args.horizon_bars]
        actual_return = future[-1].close / bars[index].close - 1.0
        actual_direction = Direction.LONG if actual_return > 0 else Direction.SHORT
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
            trade = simulate_barrier_trade(
                prediction,
                future,
                take_profit_bps=args.take_profit_bps,
                stop_loss_bps=args.stop_loss_bps,
                round_trip_cost_bps=args.round_trip_cost_bps,
            )
            trade_outcome = f"{trade.exit_reason} {trade.net_return_bps:+.2f}bp"

        print(
            f"{timestamp.strftime('%Y-%m-%d %H:%MZ'):19s} "
            f"{prediction.direction.value:9s} {prediction.confidence:6.1%}  "
            f"{actual_direction.value:6s} {actual_return * 10000:+9.2f}  "
            f"{verdict:9s}  {trade_outcome}"
        )
        output_rows.append(
            {
                "decision_timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                "prediction_direction": prediction.direction.value,
                "confidence": prediction.confidence,
                "actual_direction": actual_direction.value,
                "actual_return_bps": actual_return * 10_000.0,
                "verdict": verdict,
                "trade_exit_reason": trade.exit_reason if trade else None,
                "trade_net_return_bps": trade.net_return_bps if trade else None,
            }
        )

    actionable = correct + wrong
    accuracy = correct / actionable if actionable else None
    print("\nAudit summary")
    print("=" * 46)
    print(f"Frozen V0 samples    : {len(timestamps)}")
    print(f"Evaluated            : {len(output_rows)}")
    print(f"Unavailable          : {unavailable}")
    print(f"Actionable           : {actionable}")
    print(f"Correct              : {correct}")
    print(f"Wrong                : {wrong}")
    print(f"NO_TRADE             : {no_trade}")
    print(f"Selected accuracy    : {accuracy:.2%}" if accuracy is not None else "Selected accuracy    : n/a")

    if args.output_json:
        payload = {
            "symbol": args.symbol,
            "version": "V1-XGB-TIME-MACHINE",
            "baseline_json": args.baseline_json,
            "frozen_samples": len(timestamps),
            "evaluated": len(output_rows),
            "unavailable": unavailable,
            "actionable": actionable,
            "correct": correct,
            "wrong": wrong,
            "no_trade": no_trade,
            "selected_accuracy": accuracy,
            "rows": output_rows,
        }
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Audit JSON           : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
