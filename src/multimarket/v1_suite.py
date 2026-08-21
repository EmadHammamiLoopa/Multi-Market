from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .data import load_ohlc_csv
from .portfolio import calculate_portfolio_metrics
from .reporting import sha256_file
from .v1_cli import _evaluate
from .v1_model import V1Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run V1 walk-forward XGBoost across 15m/30m/60m horizons on 5m bars"
    )
    parser.add_argument("csv")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=[3, 6, 12])
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
    dataset_hash = sha256_file(args.csv)
    rows = []

    print(f"Multi-Market V1 multi-horizon suite | {args.symbol}")
    print("=" * 84)
    print(f"Dataset SHA-256: {dataset_hash}")
    print("horizon  predictions trades coverage accuracy   PF    exp(bp)  exec  execPF execExp")
    print("-" * 84)

    for horizon in args.horizons:
        config = V1Config(
            horizon_bars=horizon,
            confidence_threshold=args.confidence_threshold,
            min_train_rows=args.min_train_rows,
            retrain_every=args.retrain_every,
            validation_fraction=args.validation_fraction,
        )
        _, predictions, hits, trades, metrics = _evaluate(
            bars,
            symbol=args.symbol,
            version=f"V1-XGB-{horizon}B",
            config=config,
            take_profit_bps=args.take_profit_bps,
            stop_loss_bps=args.stop_loss_bps,
            round_trip_cost_bps=args.round_trip_cost_bps,
        )
        portfolio = calculate_portfolio_metrics(trades)
        row = {
            "horizon_bars": horizon,
            "predictions": metrics.predictions,
            "signal_trades": metrics.trades,
            "coverage": metrics.coverage,
            "directional_accuracy": metrics.directional_accuracy,
            "profit_factor": metrics.profit_factor,
            "expectancy_bps": metrics.expectancy_bps,
            "portfolio_non_overlapping": asdict(portfolio),
        }
        rows.append(row)
        print(
            f"{horizon:7d}B {metrics.predictions:11d} {metrics.trades:6d} "
            f"{metrics.coverage:7.2%} {metrics.directional_accuracy:7.2%} "
            f"{metrics.profit_factor:5.2f} {metrics.expectancy_bps:+8.2f} "
            f"{portfolio.trades:5d} {portfolio.profit_factor:6.2f} {portfolio.expectancy_bps:+7.2f}"
        )

    best_accuracy = max(rows, key=lambda row: row["directional_accuracy"])
    best_expectancy = max(
        rows,
        key=lambda row: row["portfolio_non_overlapping"]["expectancy_bps"],
    )
    print("\nBest by selected accuracy :", f"{best_accuracy['horizon_bars']} bars")
    print("Best by exec expectancy   :", f"{best_expectancy['horizon_bars']} bars")

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "symbol": args.symbol,
                    "version": "V1-XGB-SUITE",
                    "dataset_sha256": dataset_hash,
                    "confidence_threshold": args.confidence_threshold,
                    "rows": rows,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"Suite JSON              : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
