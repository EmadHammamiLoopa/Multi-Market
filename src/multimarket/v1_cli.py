from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from math import inf
from pathlib import Path

from .data import load_ohlc_csv
from .execution import simulate_barrier_trade
from .metrics import Metrics, calculate_metrics
from .models import Direction, Prediction, TradeResult
from .portfolio import calculate_portfolio_metrics
from .reporting import sha256_file
from .v1_model import V1Config, WalkForwardXGBoostPredictor


def _evaluate(
    bars,
    *,
    symbol: str,
    version: str,
    config: V1Config,
    take_profit_bps: float,
    stop_loss_bps: float,
    round_trip_cost_bps: float,
    progress_every: int | None = None,
):
    predictor = WalkForwardXGBoostPredictor(bars, config)
    predictions: list[Prediction] = []
    directional_hits: list[bool] = []
    trades: list[TradeResult] = []

    first_index = max(48, config.min_train_rows + config.horizon_bars)
    last_index = len(bars) - config.horizon_bars - 1
    total = max(0, last_index - first_index + 1)
    for offset, index in enumerate(range(first_index, last_index + 1), start=1):
        try:
            prediction = predictor.predict_at_index(index)
        except ValueError as exc:
            if "labeled historical rows" in str(exc):
                continue
            raise

        future = bars[index + 1 : index + 1 + config.horizon_bars]
        actual_return = future[-1].close / bars[index].close - 1.0
        actual_direction = Direction.LONG if actual_return > 0 else Direction.SHORT
        hit = (
            prediction.direction is not Direction.NO_TRADE
            and prediction.direction is actual_direction
        )

        predictions.append(prediction)
        directional_hits.append(hit)
        if prediction.direction is not Direction.NO_TRADE:
            trades.append(
                simulate_barrier_trade(
                    prediction,
                    future,
                    take_profit_bps=take_profit_bps,
                    stop_loss_bps=stop_loss_bps,
                    round_trip_cost_bps=round_trip_cost_bps,
                )
            )

        if progress_every and (offset % progress_every == 0 or offset == total):
            print(
                f"[{symbol} {version}] progress {offset}/{total} "
                f"({offset / total:.1%}) fits={predictor._fit_count}",
                flush=True,
            )

    metrics = calculate_metrics(predictions, directional_hits, trades)
    return predictor, predictions, directional_hits, trades, metrics


def _serialize_metrics(metrics: Metrics) -> dict[str, object]:
    result = asdict(metrics)
    if result["profit_factor"] == inf:
        result["profit_factor"] = "inf"
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V1 leakage-safe expanding-window XGBoost intraday evaluator"
    )
    parser.add_argument("csv", help="OHLC CSV path")
    parser.add_argument("--symbol", default="UNKNOWN")
    parser.add_argument("--version-label", default="V1-XGB")
    parser.add_argument("--horizon-bars", type=int, default=6)
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    parser.add_argument("--min-train-rows", type=int, default=1000)
    parser.add_argument("--retrain-every", type=int, default=250)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--take-profit-bps", type=float, default=20.0)
    parser.add_argument("--stop-loss-bps", type=float, default=10.0)
    parser.add_argument("--round-trip-cost-bps", type=float, default=2.0)
    parser.add_argument("--output-json", help="Optional complete V1 metrics JSON path")
    parser.add_argument("--output-jsonl", help="Optional prediction/trade JSONL path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bars = load_ohlc_csv(args.csv)
    config = V1Config(
        horizon_bars=args.horizon_bars,
        confidence_threshold=args.confidence_threshold,
        min_train_rows=args.min_train_rows,
        retrain_every=args.retrain_every,
        validation_fraction=args.validation_fraction,
    )
    predictor, predictions, hits, trades, metrics = _evaluate(
        bars,
        symbol=args.symbol,
        version=args.version_label,
        config=config,
        take_profit_bps=args.take_profit_bps,
        stop_loss_bps=args.stop_loss_bps,
        round_trip_cost_bps=args.round_trip_cost_bps,
    )
    portfolio = calculate_portfolio_metrics(trades)

    dataset_hash = sha256_file(args.csv)
    payload = _serialize_metrics(metrics)
    portfolio_payload = asdict(portfolio)
    if portfolio_payload["profit_factor"] == inf:
        portfolio_payload["profit_factor"] = "inf"
    payload.update(
        {
            "symbol": args.symbol,
            "version": args.version_label,
            "dataset_sha256": dataset_hash,
            "horizon_bars": args.horizon_bars,
            "confidence_threshold": args.confidence_threshold,
            "min_train_rows": args.min_train_rows,
            "retrain_every": args.retrain_every,
            "validation_fraction": args.validation_fraction,
            "feature_count": len(predictor.feature_names),
            "feature_names": list(predictor.feature_names),
            "portfolio_non_overlapping": portfolio_payload,
        }
    )

    print(f"Multi-Market {args.version_label} | {args.symbol}")
    print("=" * 58)
    print(f"Dataset SHA-256       : {dataset_hash}")
    print(f"Predictions            : {metrics.predictions}")
    print(f"Trades (signals)       : {metrics.trades}")
    print(f"Coverage               : {metrics.coverage:.2%}")
    print(f"Directional accuracy   : {metrics.directional_accuracy:.2%}")
    print(f"Signal win rate        : {metrics.win_rate:.2%}")
    print(f"Signal profit factor   : {'inf' if metrics.profit_factor == inf else f'{metrics.profit_factor:.3f}'}")
    print(f"Signal expectancy      : {metrics.expectancy_bps:+.3f} bps/trade")
    print("\nNon-overlapping execution")
    print("-" * 58)
    print(f"Executed trades        : {portfolio.trades}")
    print(f"Win rate               : {portfolio.win_rate:.2%}")
    print(f"Profit factor          : {'inf' if portfolio.profit_factor == inf else f'{portfolio.profit_factor:.3f}'}")
    print(f"Expectancy             : {portfolio.expectancy_bps:+.3f} bps/trade")
    print(f"Net return             : {portfolio.net_return_pct:+.3f}%")
    print(f"Max drawdown           : {portfolio.max_drawdown_pct:.3f}%")

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Metrics JSON           : {output}")

    if args.output_jsonl:
        trade_by_timestamp = {trade.prediction.timestamp: trade for trade in trades}
        output = Path(args.output_jsonl)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for prediction, hit in zip(predictions, hits):
                row: dict[str, object] = {
                    "version": args.version_label,
                    "symbol": args.symbol,
                    "prediction": asdict(prediction),
                    "directional_hit": hit,
                }
                trade = trade_by_timestamp.get(prediction.timestamp)
                if trade is not None:
                    row["trade"] = asdict(trade)
                handle.write(json.dumps(row, default=str) + "\n")
        print(f"Predictions JSONL      : {output}")

    print("\nMachine-readable metrics")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
