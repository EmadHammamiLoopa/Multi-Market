from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .replay import ReplayReport

BENCHMARK_FIELDS = [
    "run_utc",
    "version",
    "symbol",
    "dataset_sha256",
    "predictions",
    "trades",
    "coverage",
    "directional_accuracy",
    "win_rate",
    "profit_factor",
    "expectancy_bps",
    "net_return_pct",
    "max_drawdown_pct",
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary_lines(report: ReplayReport, *, symbol: str, version: str) -> list[str]:
    m = report.metrics
    pf = "inf" if m.profit_factor == float("inf") else f"{m.profit_factor:.3f}"
    return [
        f"Multi-Market {version} | {symbol}",
        "=" * 42,
        f"Predictions          : {m.predictions}",
        f"Trades               : {m.trades}",
        f"Coverage             : {m.coverage * 100:.2f}%",
        f"Directional accuracy : {m.directional_accuracy * 100:.2f}%",
        f"Trade win rate       : {m.win_rate * 100:.2f}%",
        f"Profit factor        : {pf}",
        f"Expectancy           : {m.expectancy_bps:.3f} bps/trade",
        f"Net return           : {m.net_return_pct:.3f}%",
        f"Max drawdown         : {m.max_drawdown_pct:.3f}%",
    ]


def save_metrics_json(report: ReplayReport, path: str | Path, *, symbol: str, version: str, dataset_sha256: str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = report.as_dict()
    if payload["profit_factor"] == float("inf"):
        payload["profit_factor"] = "inf"
    payload.update({
        "version": version,
        "symbol": symbol,
        "dataset_sha256": dataset_sha256,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    })
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def append_benchmark_history(report: ReplayReport, history_csv: str | Path, *, symbol: str, version: str, dataset_sha256: str) -> Path:
    output = Path(history_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists() and output.stat().st_size > 0
    m = report.metrics
    row = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "version": version,
        "symbol": symbol,
        "dataset_sha256": dataset_sha256,
        "predictions": m.predictions,
        "trades": m.trades,
        "coverage": m.coverage,
        "directional_accuracy": m.directional_accuracy,
        "win_rate": m.win_rate,
        "profit_factor": m.profit_factor,
        "expectancy_bps": m.expectancy_bps,
        "net_return_pct": m.net_return_pct,
        "max_drawdown_pct": m.max_drawdown_pct,
    }
    with output.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return output


def load_benchmark_history(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.exists():
        return []
    with source.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ranking_lines(rows: Iterable[dict[str, str]], *, dataset_sha256: str) -> list[str]:
    matching = [r for r in rows if r.get("dataset_sha256") == dataset_sha256]
    if not matching:
        return []
    ranked = sorted(
        matching,
        key=lambda r: (float(r.get("directional_accuracy", 0.0)), float(r.get("net_return_pct", 0.0))),
        reverse=True,
    )
    lines = ["", "Version ranking on this exact dataset", "=" * 42]
    for i, row in enumerate(ranked, start=1):
        lines.append(
            f"{i:>2}. {row['version']:<8} "
            f"accuracy={float(row['directional_accuracy']) * 100:6.2f}%  "
            f"win={float(row['win_rate']) * 100:6.2f}%  "
            f"net={float(row['net_return_pct']):8.3f}%  "
            f"maxDD={float(row['max_drawdown_pct']):7.3f}%"
        )
    return lines


def save_dashboard_plot(report: ReplayReport, path: str | Path, *, symbol: str, version: str, history_rows: Iterable[dict[str, str]] = (), dataset_sha256: str | None = None) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib. Install with: python -m pip install -e '.[plot]'") from exc

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    actionable_pairs = [
        (prediction, hit)
        for prediction, hit in zip(report.predictions, report.directional_hits)
        if prediction.direction.value != "NO_TRADE"
    ]

    running_accuracy: list[float] = []
    correct = 0
    for i, (_, hit) in enumerate(actionable_pairs, start=1):
        correct += int(hit)
        running_accuracy.append(correct / i * 100.0)

    equity_curve = [0.0]
    equity = 1.0
    for trade in report.trades:
        equity *= 1.0 + trade.net_return_bps / 10_000.0
        equity_curve.append((equity - 1.0) * 100.0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Multi-Market {version} — {symbol} backtest dashboard")

    ax = axes[0, 0]
    ax.plot(range(1, len(running_accuracy) + 1), running_accuracy)
    ax.axhline(50.0, linestyle="--", linewidth=1)
    ax.set_title("Running directional accuracy")
    ax.set_xlabel("Actionable predictions")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(range(len(equity_curve)), equity_curve)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_title("Simulated cumulative net return")
    ax.set_xlabel("Trades")
    ax.set_ylabel("Net return (%)")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    wins = sum(t.net_return_bps > 0 for t in report.trades)
    losses = sum(t.net_return_bps <= 0 for t in report.trades)
    ax.bar(["Wins", "Losses"], [wins, losses])
    ax.set_title("Trade outcomes")
    ax.set_ylabel("Count")

    ax = axes[1, 1]
    comparison = [row for row in history_rows if dataset_sha256 is not None and row.get("dataset_sha256") == dataset_sha256]
    if comparison:
        by_version: dict[str, float] = {}
        for row in comparison:
            by_version[row["version"]] = max(
                by_version.get(row["version"], 0.0),
                float(row["directional_accuracy"]) * 100.0,
            )
        labels = list(by_version)
        values = [by_version[label] for label in labels]
        ax.bar(labels, values)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Best recorded accuracy by version")
    else:
        m = report.metrics
        ax.bar(["Accuracy", "Win rate", "Coverage"], [m.directional_accuracy * 100, m.win_rate * 100, m.coverage * 100])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percent")
        ax.set_title("V0 headline metrics")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
