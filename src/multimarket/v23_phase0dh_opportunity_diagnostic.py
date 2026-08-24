from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np

from .v23_phase0dh_opportunity import (
    SYMBOLS,
    HORIZONS,
    ALPHAS,
    GATE_QUANTILES,
    FOLDS,
    _utc,
    _load_numeric,
    _gross_return_bps,
    _opportunity_target,
    _fit_predict,
    _trade_metrics,
)

DIAGNOSTIC_COSTS = (0.0, 5.0, 8.0, 10.0, 12.0, 15.0)


def _reasons(m12: dict[str, object], m15: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    if int(m12["trades"]) == 0:
        reasons.append("NO_TRADES")
    if float(m12["total_net_bps"]) <= 0:
        reasons.append("NONPOSITIVE_TOTAL_12")
    if float(m12["net_bps_trade"]) <= 0:
        reasons.append("NONPOSITIVE_EXPECTANCY_12")
    if float(m12["profit_factor"]) <= 1.0:
        reasons.append("PF_LE_1_12")
    if float(m12["median_trades_day_active"]) < 5.0:
        reasons.append("MEDIAN_TRADES_DAY_LT_5")
    if float(m15["total_net_bps"]) <= 0:
        reasons.append("NONPOSITIVE_TOTAL_15")
    if float(m15["net_bps_trade"]) <= 0:
        reasons.append("NONPOSITIVE_EXPECTANCY_15")
    return reasons


def diagnose_symbol(path: Path, symbol: str) -> dict[str, object]:
    timestamps, price, X = _load_numeric(path)
    folds_out: list[dict[str, object]] = []
    reason_counts: dict[str, int] = {}

    for fold_idx, (eval_start_d, eval_end_d) in enumerate(FOLDS, 1):
        outer_train_end = int(np.searchsorted(timestamps, _utc(eval_start_d) - max(HORIZONS), side="left"))
        cut = int(outer_train_end * 0.8)
        inner_start_ts = int(timestamps[cut])
        inner_end_exclusive_ts = int(timestamps[outer_train_end - 1]) + 1
        candidates: list[dict[str, object]] = []

        for horizon in HORIZONS:
            gross = _gross_return_bps(price, horizon)
            y = _opportunity_target(gross)
            train_end_h = cut - horizon
            val_end_h = outer_train_end - horizon
            if train_end_h < 10000 or val_end_h <= cut:
                continue
            Xtr = X[:train_end_h]
            ytr = y[:train_end_h]
            Xv = X[cut:val_end_h]

            for alpha in ALPHAS:
                train_pred, val_pred = _fit_predict(Xtr, ytr, Xv, alpha)
                abs_train = np.abs(train_pred)
                for q in GATE_QUANTILES:
                    gate = float(np.quantile(abs_train, q))
                    costs: dict[str, object] = {}
                    for cost in DIAGNOSTIC_COSTS:
                        costs[str(int(cost))] = _trade_metrics(
                            timestamps,
                            gross,
                            val_pred,
                            gate,
                            horizon,
                            cut,
                            val_end_h,
                            inner_start_ts,
                            inner_end_exclusive_ts,
                            cost,
                        )
                    m12 = costs["12"]
                    m15 = costs["15"]
                    reasons = _reasons(m12, m15)
                    for reason in set(reasons):
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    m0 = costs["0"]
                    candidate = {
                        "config": {"horizon": horizon, "alpha": alpha, "gate_quantile": q},
                        "absolute_prediction_gate": gate,
                        "gross_expectancy_bps": float(m0["gross_bps_trade"]),
                        "break_even_round_trip_cost_bps": float(m0["gross_bps_trade"]),
                        "trades": int(m0["trades"]),
                        "median_trades_day_active": float(m0["median_trades_day_active"]),
                        "mean_trades_day_all": float(m0["mean_trades_day_all"]),
                        "costs": costs,
                        "failed_filters": reasons,
                    }
                    candidates.append(candidate)

        def top(key, reverse=True, n=10):
            return sorted(candidates, key=key, reverse=reverse)[:n]

        folds_out.append({
            "fold": fold_idx,
            "eval_start": eval_start_d.isoformat(),
            "eval_end": eval_end_d.isoformat(),
            "tested": len(candidates),
            "survivors_under_original_gate": sum(1 for c in candidates if not c["failed_filters"]),
            "top_by_gross_expectancy": top(lambda c: float(c["gross_expectancy_bps"])),
            "top_by_net_expectancy_12bps": top(lambda c: float(c["costs"]["12"]["net_bps_trade"])),
            "top_by_total_net_12bps": top(lambda c: float(c["costs"]["12"]["total_net_bps"])),
            "top_by_trade_count": top(lambda c: int(c["trades"])),
        })

    all_top = [c for fold in folds_out for c in fold["top_by_gross_expectancy"]]
    return {
        "phase": "V2.3-PHASE0DH-OPPORTUNITY-DIAGNOSTIC",
        "symbol": symbol,
        "exploratory_only": True,
        "promotion_decision_changed": False,
        "historical_holdout_opened": False,
        "diagnostic_costs_bps_round_trip": list(DIAGNOSTIC_COSTS),
        "failure_reason_counts": dict(sorted(reason_counts.items())),
        "folds": folds_out,
        "best_gross_expectancy_seen_among_fold_top10": max(
            (float(c["gross_expectancy_bps"]) for c in all_top), default=0.0
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Explore why Phase 0D-H opportunity configurations failed without opening holdout")
    p.add_argument("--work-dir", default="evidence/v23/phase0dh_tf")
    p.add_argument("--output-dir", default="evidence/v23/phase0dh_opportunity_diagnostic")
    args = p.parse_args(argv)

    work = Path(args.work_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "phase": "V2.3-PHASE0DH-OPPORTUNITY-DIAGNOSTIC",
        "exploratory_only": True,
        "historical_holdout_opened": False,
        "symbols": {},
    }

    for symbol in SYMBOLS:
        print(f"[{symbol}] diagnostic sweep on development only", flush=True)
        result = diagnose_symbol(work / f"{symbol}_DEV.csv", symbol)
        (out / f"{symbol}_OPPORTUNITY_DIAGNOSTIC.json").write_text(json.dumps(result, indent=2) + "\n")
        summary["symbols"][symbol] = {
            "failure_reason_counts": result["failure_reason_counts"],
            "best_gross_expectancy_seen_among_fold_top10": result["best_gross_expectancy_seen_among_fold_top10"],
        }
        print(
            f"[{symbol}] best_gross_expectancy={result['best_gross_expectancy_seen_among_fold_top10']:.4f} bps/trade",
            flush=True,
        )

    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("PHASE0DH_OPPORTUNITY_DIAGNOSTIC=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
