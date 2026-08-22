from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .data import load_ohlc_csv
from .execution import simulate_barrier_trade
from .models import Direction, MarketBar, Prediction
from .reporting import sha256_file
from .replay import ReplayConfig
from .strategy import MomentumBaselinePredictor, Predictor


@dataclass(frozen=True, slots=True)
class TimeMachineRow:
    decision_timestamp: datetime
    reference_price: float
    prediction_direction: Direction
    confidence: float
    future_end_timestamp: datetime
    future_end_price: float
    actual_direction: Direction
    actual_return_bps: float
    verdict: str
    trade_exit_timestamp: datetime | None = None
    trade_exit_reason: str | None = None
    trade_net_return_bps: float | None = None


def _apply_confidence_threshold(
    prediction: Prediction, confidence_threshold: float
) -> Prediction:
    if prediction.confidence >= confidence_threshold:
        return prediction
    return Prediction(
        timestamp=prediction.timestamp,
        direction=Direction.NO_TRADE,
        confidence=prediction.confidence,
        reference_price=prediction.reference_price,
        horizon_bars=prediction.horizon_bars,
    )


def audit_at_index(
    bars: Sequence[MarketBar],
    index: int,
    *,
    predictor: Predictor,
    config: ReplayConfig,
    min_history_bars: int,
) -> TimeMachineRow:
    """Predict at one historical timestamp, then reveal only the later outcome.

    The predictor receives ``bars[:index + 1]`` only. Future candles are sliced
    after prediction and are used exclusively for scoring/execution simulation.
    """
    if min_history_bars <= 0:
        raise ValueError("min_history_bars must be positive")
    if index < min_history_bars - 1:
        raise ValueError("decision index does not have enough historical bars")
    if index + config.horizon_bars >= len(bars):
        raise ValueError("decision index does not have a complete future horizon")

    history = bars[: index + 1]
    raw_prediction = predictor.predict(history, config.horizon_bars)
    current = bars[index]

    if raw_prediction.timestamp != current.timestamp:
        raise ValueError("predictor timestamp must equal history endpoint")
    if raw_prediction.reference_price != current.close:
        raise ValueError("predictor reference_price must equal current close")

    prediction = _apply_confidence_threshold(
        raw_prediction, config.confidence_threshold
    )

    # Future is intentionally materialized only after the prediction is frozen.
    future = bars[index + 1 : index + 1 + config.horizon_bars]
    future_end = future[-1]
    actual_return = future_end.close / current.close - 1.0
    actual_direction = Direction.LONG if actual_return > 0 else Direction.SHORT

    if prediction.direction is Direction.NO_TRADE:
        return TimeMachineRow(
            decision_timestamp=current.timestamp,
            reference_price=current.close,
            prediction_direction=prediction.direction,
            confidence=prediction.confidence,
            future_end_timestamp=future_end.timestamp,
            future_end_price=future_end.close,
            actual_direction=actual_direction,
            actual_return_bps=actual_return * 10_000.0,
            verdict="NO_TRADE",
        )

    correct = prediction.direction is actual_direction
    trade = simulate_barrier_trade(
        prediction,
        future,
        take_profit_bps=config.take_profit_bps,
        stop_loss_bps=config.stop_loss_bps,
        round_trip_cost_bps=config.round_trip_cost_bps,
    )
    return TimeMachineRow(
        decision_timestamp=current.timestamp,
        reference_price=current.close,
        prediction_direction=prediction.direction,
        confidence=prediction.confidence,
        future_end_timestamp=future_end.timestamp,
        future_end_price=future_end.close,
        actual_direction=actual_direction,
        actual_return_bps=actual_return * 10_000.0,
        verdict="CORRECT" if correct else "WRONG",
        trade_exit_timestamp=trade.exit_timestamp,
        trade_exit_reason=trade.exit_reason,
        trade_net_return_bps=trade.net_return_bps,
    )


def evenly_spaced_indices(
    bar_count: int,
    *,
    min_history_bars: int,
    horizon_bars: int,
    samples: int,
) -> list[int]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    start = min_history_bars - 1
    end = bar_count - horizon_bars - 1
    if end < start:
        raise ValueError("not enough bars for requested history and horizon")

    total = end - start + 1
    if samples >= total:
        return list(range(start, end + 1))
    if samples == 1:
        return [(start + end) // 2]

    return [
        start + (i * (end - start)) // (samples - 1)
        for i in range(samples)
    ]


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def evenly_spaced_window_indices(
    bars: Sequence[MarketBar],
    *,
    start_timestamp: str,
    end_timestamp: str,
    min_history_bars: int,
    horizon_bars: int,
    samples: int,
) -> list[int]:
    """Select deterministic evenly spaced valid decisions inside a UTC window.

    Selection is based only on timestamps and structural eligibility. Outcomes,
    predictions, returns, and prices are never inspected, preventing cherry-picking.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    start = _parse_timestamp(start_timestamp)
    end = _parse_timestamp(end_timestamp)
    if end < start:
        raise ValueError("end timestamp must be at or after start timestamp")

    first_valid = min_history_bars - 1
    last_valid = len(bars) - horizon_bars - 1
    candidates = [
        index
        for index in range(max(0, first_valid), max(-1, last_valid) + 1)
        if start <= bars[index].timestamp.astimezone(timezone.utc) <= end
    ]
    if not candidates:
        raise ValueError("no structurally valid decision candles exist inside requested window")
    if samples >= len(candidates):
        return candidates
    if samples == 1:
        return [candidates[(len(candidates) - 1) // 2]]

    last = len(candidates) - 1
    return [candidates[(i * last) // (samples - 1)] for i in range(samples)]


def indices_for_timestamps(
    bars: Sequence[MarketBar],
    timestamps: Sequence[str],
    *,
    min_history_bars: int,
    horizon_bars: int,
) -> list[int]:
    by_timestamp = {bar.timestamp.astimezone(timezone.utc): i for i, bar in enumerate(bars)}
    indices: list[int] = []
    for raw in timestamps:
        timestamp = _parse_timestamp(raw)
        if timestamp not in by_timestamp:
            raise ValueError(f"timestamp not found exactly in CSV: {raw}")
        index = by_timestamp[timestamp]
        if index < min_history_bars - 1:
            raise ValueError(f"timestamp lacks {min_history_bars} history bars: {raw}")
        if index + horizon_bars >= len(bars):
            raise ValueError(f"timestamp lacks {horizon_bars} future bars: {raw}")
        indices.append(index)
    return indices


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%MZ")


def _serialize_row(row: TimeMachineRow) -> dict[str, object]:
    payload = asdict(row)
    for key in ("decision_timestamp", "future_end_timestamp", "trade_exit_timestamp"):
        value = payload[key]
        if isinstance(value, datetime):
            payload[key] = value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    payload["prediction_direction"] = row.prediction_direction.value
    payload["actual_direction"] = row.actual_direction.value
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Point-in-time historical audit: predict at old timestamps using only "
            "then-known candles, then reveal the future and mark CORRECT/WRONG."
        )
    )
    parser.add_argument("csv", help="OHLC CSV path")
    parser.add_argument("--symbol", default="UNKNOWN")
    parser.add_argument("--version-label", default="V0-TIME-MACHINE")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument(
        "--timestamp",
        action="append",
        default=[],
        help="Audit an exact ISO-8601 candle timestamp; may be supplied multiple times",
    )
    parser.add_argument(
        "--start-timestamp",
        help="Inclusive UTC/ISO-8601 start for deterministic window sampling",
    )
    parser.add_argument(
        "--end-timestamp",
        help="Inclusive UTC/ISO-8601 end for deterministic window sampling",
    )
    parser.add_argument("--horizon-bars", type=int, default=6)
    parser.add_argument("--lookback-bars", type=int, default=24)
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    parser.add_argument("--take-profit-bps", type=float, default=20.0)
    parser.add_argument("--stop-loss-bps", type=float, default=10.0)
    parser.add_argument("--round-trip-cost-bps", type=float, default=2.0)
    parser.add_argument("--output-json", help="Optional path for complete audit JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bars = load_ohlc_csv(args.csv)
    predictor = MomentumBaselinePredictor(lookback_bars=args.lookback_bars)
    config = ReplayConfig(
        horizon_bars=args.horizon_bars,
        confidence_threshold=args.confidence_threshold,
        take_profit_bps=args.take_profit_bps,
        stop_loss_bps=args.stop_loss_bps,
        round_trip_cost_bps=args.round_trip_cost_bps,
    )

    has_window_arg = args.start_timestamp is not None or args.end_timestamp is not None
    if args.timestamp and has_window_arg:
        raise SystemExit("--timestamp cannot be combined with --start-timestamp/--end-timestamp")
    if has_window_arg and not (args.start_timestamp and args.end_timestamp):
        raise SystemExit("--start-timestamp and --end-timestamp must be supplied together")

    window_start = None
    window_end = None
    if args.timestamp:
        indices = indices_for_timestamps(
            bars,
            args.timestamp,
            min_history_bars=args.lookback_bars,
            horizon_bars=args.horizon_bars,
        )
        selection = "explicit historical timestamps"
    elif has_window_arg:
        indices = evenly_spaced_window_indices(
            bars,
            start_timestamp=args.start_timestamp,
            end_timestamp=args.end_timestamp,
            min_history_bars=args.lookback_bars,
            horizon_bars=args.horizon_bars,
            samples=args.samples,
        )
        window_start = _parse_timestamp(args.start_timestamp)
        window_end = _parse_timestamp(args.end_timestamp)
        selection = "evenly spaced valid candles inside frozen historical window (no cherry-picking)"
    else:
        indices = evenly_spaced_indices(
            len(bars),
            min_history_bars=args.lookback_bars,
            horizon_bars=args.horizon_bars,
            samples=args.samples,
        )
        selection = "evenly spaced historical timestamps (no cherry-picking)"

    rows = [
        audit_at_index(
            bars,
            index,
            predictor=predictor,
            config=config,
            min_history_bars=args.lookback_bars,
        )
        for index in indices
    ]

    print(f"Multi-Market time-machine audit | {args.symbol} | {args.version_label}")
    print("=" * 92)
    print(f"Dataset SHA-256 : {sha256_file(args.csv)}")
    print(f"Selection       : {selection}")
    if window_start is not None and window_end is not None:
        print(
            "Frozen window   : "
            f"{window_start.isoformat().replace('+00:00', 'Z')} .. "
            f"{window_end.isoformat().replace('+00:00', 'Z')}"
        )
    print("History visible : only candles at/before each decision timestamp")
    print(f"Future horizon  : {args.horizon_bars} bars")
    print()
    print(
        "decision time       prediction  conf    actual   move(bps)  verdict    trade outcome"
    )
    print("-" * 92)

    for row in rows:
        trade_outcome = "-"
        if row.trade_exit_reason is not None and row.trade_net_return_bps is not None:
            trade_outcome = f"{row.trade_exit_reason} {row.trade_net_return_bps:+.2f}bp"
        print(
            f"{_format_timestamp(row.decision_timestamp):19s} "
            f"{row.prediction_direction.value:9s} "
            f"{row.confidence:6.1%}  "
            f"{row.actual_direction.value:6s} "
            f"{row.actual_return_bps:+9.2f}  "
            f"{row.verdict:9s}  {trade_outcome}"
        )

    correct = sum(row.verdict == "CORRECT" for row in rows)
    wrong = sum(row.verdict == "WRONG" for row in rows)
    no_trade = sum(row.verdict == "NO_TRADE" for row in rows)
    actionable = correct + wrong
    accuracy = correct / actionable if actionable else 0.0

    print()
    print("Audit summary")
    print("=" * 42)
    print(f"Historical samples : {len(rows)}")
    print(f"Actionable         : {actionable}")
    print(f"Correct            : {correct}")
    print(f"Wrong              : {wrong}")
    print(f"NO_TRADE           : {no_trade}")
    print(f"Selected accuracy  : {accuracy:.2%}" if actionable else "Selected accuracy  : n/a")
    print("Leakage rule       : prediction is frozen before future bars are revealed")

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "symbol": args.symbol,
            "version": args.version_label,
            "dataset_sha256": sha256_file(args.csv),
            "selection": selection,
            "window_start": (
                window_start.isoformat().replace("+00:00", "Z") if window_start else None
            ),
            "window_end": (
                window_end.isoformat().replace("+00:00", "Z") if window_end else None
            ),
            "horizon_bars": args.horizon_bars,
            "lookback_bars": args.lookback_bars,
            "confidence_threshold": args.confidence_threshold,
            "samples": len(rows),
            "actionable": actionable,
            "correct": correct,
            "wrong": wrong,
            "no_trade": no_trade,
            "selected_accuracy": accuracy if actionable else None,
            "rows": [_serialize_row(row) for row in rows],
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Audit JSON         : {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
