from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import timezone
from math import inf
from pathlib import Path
from typing import Sequence

from .data import load_ohlc_csv
from .metrics import calculate_metrics
from .models import Direction, Prediction, TradeResult
from .portfolio import calculate_portfolio_metrics
from .reporting import sha256_file
from .v1_cli import _evaluate
from .v1_model import V1Config


def threshold_predictions(
    predictions: Sequence[Prediction],
    hits: Sequence[bool],
    trades: Sequence[TradeResult],
    threshold: float,
):
    if len(predictions) != len(hits):
        raise ValueError("predictions and hits must align")
    if not 0.5 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0.5, 1]")

    trade_by_timestamp = {trade.prediction.timestamp: trade for trade in trades}
    filtered_predictions: list[Prediction] = []
    filtered_hits: list[bool] = []
    filtered_trades: list[TradeResult] = []

    for prediction, hit in zip(predictions, hits):
        if prediction.confidence >= threshold:
            filtered_predictions.append(prediction)
            filtered_hits.append(hit)
            trade = trade_by_timestamp.get(prediction.timestamp)
            if trade is not None:
                filtered_trades.append(trade)
        else:
            filtered_predictions.append(
                Prediction(
                    timestamp=prediction.timestamp,
                    direction=Direction.NO_TRADE,
                    confidence=prediction.confidence,
                    reference_price=prediction.reference_price,
                    horizon_bars=prediction.horizon_bars,
                )
            )
            filtered_hits.append(False)

    return filtered_predictions, filtered_hits, filtered_trades


def confidence_buckets(predictions: Sequence[Prediction], hits: Sequence[bool]) -> list[dict[str, object]]:
    buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.80), (0.80, 1.01)]
    result: list[dict[str, object]] = []
    for lower, upper in buckets:
        selected = [
            hit
            for prediction, hit in zip(predictions, hits)
            if lower <= prediction.confidence < upper
        ]
        result.append(
            {
                "lower": lower,
                "upper": min(1.0, upper),
                "count": len(selected),
                "accuracy": (sum(selected) / len(selected)) if selected else None,
            }
        )
    return result


def _feature_importance(predictor) -> list[dict[str, object]]:
    model = getattr(predictor, "_model", None)
    values = getattr(model, "feature_importances_", None) if model is not None else None
    if values is None:
        return []
    rows = [
        {"feature": name, "importance": float(value)}
        for name, value in zip(predictor.feature_names, values)
    ]
    rows.sort(key=lambda row: row["importance"], reverse=True)
    return rows


def _read_v0_baseline(path: str | None) -> dict[str, object] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "path": path,
        "version": payload.get("version"),
        "dataset_sha256": payload.get("dataset_sha256"),
        "samples": payload.get("samples", payload.get("frozen_samples")),
        "actionable": payload.get("actionable"),
        "correct": payload.get("correct"),
        "wrong": payload.get("wrong"),
        "no_trade": payload.get("no_trade"),
        "selected_accuracy": payload.get("selected_accuracy"),
        "timestamps": [row.get("decision_timestamp") for row in payload.get("rows", [])],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Comprehensive V1 walk-forward diagnostics across horizons and confidence thresholds"
    )
    parser.add_argument("csv")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=[3, 6, 12])
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=[0.55, 0.60, 0.65, 0.70, 0.75]
    )
    parser.add_argument("--min-train-rows", type=int, default=5000)
    parser.add_argument("--retrain-every", type=int, default=500)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--take-profit-bps", type=float, default=20.0)
    parser.add_argument("--stop-loss-bps", type=float, default=10.0)
    parser.add_argument("--round-trip-cost-bps", type=float, default=2.0)
    parser.add_argument("--v0-baseline-json")
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if any(not 0.5 <= threshold <= 1.0 for threshold in args.thresholds):
        raise SystemExit("all --thresholds must be in [0.5, 1]")

    bars = load_ohlc_csv(args.csv)
    dataset_hash = sha256_file(args.csv)
    v0_baseline = _read_v0_baseline(args.v0_baseline_json)
    horizon_rows: list[dict[str, object]] = []

    print(f"Multi-Market V1 research diagnostics | {args.symbol}")
    print("=" * 104)
    print(f"Dataset SHA-256 : {dataset_hash}")
    print(f"Bars            : {len(bars)}")
    print(
        f"Range           : {bars[0].timestamp.astimezone(timezone.utc).isoformat()} .. "
        f"{bars[-1].timestamp.astimezone(timezone.utc).isoformat()}"
    )
    if v0_baseline:
        accuracy = v0_baseline.get("selected_accuracy")
        print(
            "Frozen V0       : "
            + (f"{float(accuracy):.2%}" if accuracy is not None else "n/a")
            + f" ({v0_baseline.get('actionable')} actionable)"
        )
    print()

    for horizon in args.horizons:
        # Evaluate once at 0.50 so threshold sweeps reuse identical walk-forward predictions.
        config = V1Config(
            horizon_bars=horizon,
            confidence_threshold=0.50,
            min_train_rows=args.min_train_rows,
            retrain_every=args.retrain_every,
            validation_fraction=args.validation_fraction,
        )
        predictor, predictions, hits, trades, _ = _evaluate(
            bars,
            symbol=args.symbol,
            version=f"V1-XGB-{horizon}B-RESEARCH",
            config=config,
            take_profit_bps=args.take_profit_bps,
            stop_loss_bps=args.stop_loss_bps,
            round_trip_cost_bps=args.round_trip_cost_bps,
        )

        threshold_rows: list[dict[str, object]] = []
        print(f"Horizon {horizon} bars")
        print("threshold coverage accuracy   PF   exp(bp) execTrades execPF execExp net% maxDD%")
        print("-" * 88)
        for threshold in args.thresholds:
            p2, h2, t2 = threshold_predictions(predictions, hits, trades, threshold)
            metrics = calculate_metrics(p2, h2, t2)
            portfolio = calculate_portfolio_metrics(t2)
            row = {
                "threshold": threshold,
                "coverage": metrics.coverage,
                "directional_accuracy": metrics.directional_accuracy,
                "signal_trades": metrics.trades,
                "signal_profit_factor": metrics.profit_factor,
                "signal_expectancy_bps": metrics.expectancy_bps,
                "execution": asdict(portfolio),
            }
            threshold_rows.append(row)
            pf = "inf" if metrics.profit_factor == inf else f"{metrics.profit_factor:.2f}"
            exec_pf = "inf" if portfolio.profit_factor == inf else f"{portfolio.profit_factor:.2f}"
            print(
                f"{threshold:8.2f} {metrics.coverage:7.2%} {metrics.directional_accuracy:7.2%} "
                f"{pf:>5s} {metrics.expectancy_bps:+8.2f} {portfolio.trades:10d} "
                f"{exec_pf:>6s} {portfolio.expectancy_bps:+7.2f} "
                f"{portfolio.net_return_pct:+5.2f} {portfolio.max_drawdown_pct:6.2f}"
            )

        # This is an exploratory ranking only; it must not be treated as an untouched test selection.
        eligible = [
            row for row in threshold_rows
            if row["execution"]["trades"] >= 20
        ] or threshold_rows
        best = max(
            eligible,
            key=lambda row: (
                row["execution"]["expectancy_bps"],
                row["directional_accuracy"],
            ),
        )
        horizon_rows.append(
            {
                "horizon_bars": horizon,
                "prediction_count": len(predictions),
                "confidence_buckets": confidence_buckets(predictions, hits),
                "thresholds": threshold_rows,
                "exploratory_best_threshold": best["threshold"],
                "top_feature_importance": _feature_importance(predictor)[:20],
            }
        )
        print(f"Exploratory best threshold: {best['threshold']:.2f} (not an untouched-test choice)\n")

    payload = {
        "symbol": args.symbol,
        "version": "V1-XGB-RESEARCH",
        "dataset": {
            "path": args.csv,
            "sha256": dataset_hash,
            "bars": len(bars),
            "start": bars[0].timestamp.astimezone(timezone.utc).isoformat(),
            "end": bars[-1].timestamp.astimezone(timezone.utc).isoformat(),
        },
        "protocol": {
            "horizons": args.horizons,
            "thresholds": args.thresholds,
            "min_train_rows": args.min_train_rows,
            "retrain_every": args.retrain_every,
            "validation_fraction": args.validation_fraction,
            "take_profit_bps": args.take_profit_bps,
            "stop_loss_bps": args.stop_loss_bps,
            "round_trip_cost_bps": args.round_trip_cost_bps,
            "selection_warning": (
                "Threshold ranking is exploratory on this evaluation history; freeze a threshold "
                "before claiming performance on a later untouched period."
            ),
        },
        "v0_time_machine_baseline": v0_baseline,
        "horizons": horizon_rows,
    }

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Research manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
