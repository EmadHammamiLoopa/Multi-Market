from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from math import isinf
from pathlib import Path

from .data import load_ohlc_csv
from .replay import ReplayConfig, ReplayEngine
from .reporting import (
    append_benchmark_history,
    load_benchmark_history,
    ranking_lines,
    save_dashboard_plot,
    save_metrics_json,
    sha256_file,
    summary_lines,
)
from .strategy import MomentumBaselinePredictor

DEFAULT_VERSION = "V0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Point-in-time historical market replay")
    parser.add_argument("csv", help="OHLC CSV path")
    parser.add_argument("--symbol", default="UNKNOWN")
    parser.add_argument("--version-label", default=DEFAULT_VERSION)
    parser.add_argument("--horizon-bars", type=int, default=4)
    parser.add_argument("--lookback-bars", type=int, default=20)
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    parser.add_argument("--take-profit-bps", type=float, default=20.0)
    parser.add_argument("--stop-loss-bps", type=float, default=10.0)
    parser.add_argument("--round-trip-cost-bps", type=float, default=2.0)
    parser.add_argument("--output-jsonl", help="Optional path for prediction/trade records")
    parser.add_argument("--report-dir", default="reports", help="Directory for metrics JSON and optional plots (default: reports)")
    parser.add_argument("--history-csv", default="results/benchmark_history.csv", help="Append this run for V0/V1/... comparison; use empty string to disable")
    parser.add_argument("--plot", action="store_true", help="Generate a PNG dashboard (requires optional matplotlib dependency)")
    parser.add_argument("--json-only", action="store_true", help="Print only machine-readable metrics JSON")
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
    report = ReplayEngine(predictor, config).run(bars, min_history_bars=args.lookback_bars)

    dataset_hash = sha256_file(args.csv)
    result = report.as_dict()
    if isinf(result["profit_factor"]):
        result["profit_factor"] = "inf"
    result.update({"version": args.version_label, "symbol": args.symbol, "dataset_sha256": dataset_hash})

    report_dir = Path(args.report_dir)
    safe_symbol = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.symbol)
    safe_version = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.version_label)
    stem = f"{safe_symbol}_{safe_version}"
    metrics_path = save_metrics_json(report, report_dir / f"{stem}_metrics.json", symbol=args.symbol, version=args.version_label, dataset_sha256=dataset_hash)

    history_rows: list[dict[str, str]] = []
    if args.history_csv:
        history_path = append_benchmark_history(report, args.history_csv, symbol=args.symbol, version=args.version_label, dataset_sha256=dataset_hash)
        history_rows = load_benchmark_history(history_path)

    plot_path: Path | None = None
    if args.plot:
        plot_path = save_dashboard_plot(report, report_dir / f"{stem}_dashboard.png", symbol=args.symbol, version=args.version_label, history_rows=history_rows, dataset_sha256=dataset_hash)

    if args.json_only:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("\n".join(summary_lines(report, symbol=args.symbol, version=args.version_label)))
        print(f"Metrics JSON          : {metrics_path}")
        if args.history_csv:
            print(f"Benchmark history     : {args.history_csv}")
        if plot_path is not None:
            print(f"Dashboard plot        : {plot_path}")
        if history_rows:
            print("\n".join(ranking_lines(history_rows, dataset_sha256=dataset_hash)))
        print("\nMachine-readable metrics")
        print(json.dumps(result, indent=2, sort_keys=True))

    if args.output_jsonl:
        output = Path(args.output_jsonl)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            trade_by_timestamp = {trade.prediction.timestamp: trade for trade in report.trades}
            for prediction, hit in zip(report.predictions, report.directional_hits):
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
