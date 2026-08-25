from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from multimarket.v23_phase0dl_score import (
    ALPHAS,
    BLOCKS,
    DAYS,
    HORIZONS_S,
    QUANTILES,
    SYMBOLS,
    Config,
    _fit_from_stats,
    _labels,
    _load_day,
    _predict,
    _ridge_stats,
    _score_day,
    _stack,
)

COSTS = (0.0, 5.0, 8.0, 12.0)


def _one_config(inner_train, inner_val, block: str, horizon: int, alpha: float, q: float) -> dict:
    Xtr, ytr = _stack(inner_train, block, horizon)
    stats = _ridge_stats(Xtr, ytr)
    model = _fit_from_stats(stats, alpha)
    train_pred = _predict(Xtr, model)
    gate = float(np.quantile(np.abs(train_pred), q))
    ok, _, _, _ = _labels(inner_val, horizon, block)
    idx = np.flatnonzero(ok)
    pred = _predict(inner_val.X[block][idx], model)
    metrics = {
        str(int(c)): _score_day(inner_val, block, horizon, pred, idx, gate, c)
        for c in COSTS
    }
    m0 = metrics["0"]
    m8 = metrics["8"]
    gross = float(m0["net_bps_trade"])
    return {
        "config": Config(block, horizon, alpha, q).__dict__,
        "gate": gate,
        "metrics": metrics,
        "gross_bps_trade": gross,
        "break_even_additional_cost_bps": gross,
        "primary_gate_components": {
            "trades_ge_20": int(m8["trades"]) >= 20,
            "exp8_positive": float(m8["net_bps_trade"]) > 0,
            "total8_positive": float(m8["total_net_bps"]) > 0,
            "pf8_gt_1": float(m8["profit_factor"]) > 1,
            "exp12_positive": float(metrics["12"]["net_bps_trade"]) > 0,
            "total12_positive": float(metrics["12"]["total_net_bps"]) > 0,
        },
    }


def _rank_key(r: dict):
    m0 = r["metrics"]["0"]
    return (
        float(m0["net_bps_trade"]),
        float(m0["total_net_bps"]),
        int(m0["trades"]),
    )


def _fold(inner_train, inner_val, blocks: tuple[str, ...]) -> dict:
    rows = []
    for block in blocks:
        for horizon in HORIZONS_S:
            for alpha in ALPHAS:
                for q in QUANTILES:
                    rows.append(_one_config(inner_train, inner_val, block, horizon, alpha, q))
    rows.sort(key=_rank_key, reverse=True)
    top = rows[:10]
    best = top[0] if top else None
    return {
        "tested": len(rows),
        "best_by_gross_expectancy": best,
        "top10_by_gross_expectancy": top,
        "count_positive_gross": sum(float(r["metrics"]["0"]["net_bps_trade"]) > 0 for r in rows),
        "count_positive_after_5": sum(float(r["metrics"]["5"]["net_bps_trade"]) > 0 for r in rows),
        "count_positive_after_8": sum(float(r["metrics"]["8"]["net_bps_trade"]) > 0 for r in rows),
        "count_positive_after_12": sum(float(r["metrics"]["12"]["net_bps_trade"]) > 0 for r in rows),
        "max_break_even_additional_cost_bps": max((float(r["break_even_additional_cost_bps"]) for r in rows), default=0.0),
    }


def _symbol(feature_dir: Path, symbol: str) -> dict:
    days = [_load_day(feature_dir / symbol / f"{d.isoformat()}_FEATURES250.csv", d) for d in DAYS]
    folds = []
    for eval_i in range(2, 7):
        outer_train = days[:eval_i]
        inner_train = outer_train[:-1]
        inner_val = outer_train[-1]
        rec = {
            "evaluation_day": days[eval_i].day.isoformat(),
            "inner_validation_day": inner_val.day.isoformat(),
            "l0": _fold(inner_train, inner_val, ("L0",)),
            "dynamic": _fold(inner_train, inner_val, ("L1", "L2")),
        }
        folds.append(rec)
        b0 = rec["l0"]["best_by_gross_expectancy"]
        bd = rec["dynamic"]["best_by_gross_expectancy"]
        print(
            f"{symbol} inner={inner_val.day} "
            f"L0_best_gross={b0['gross_bps_trade']:.4f} trades={b0['metrics']['0']['trades']} "
            f"DYN_best_gross={bd['gross_bps_trade']:.4f} trades={bd['metrics']['0']['trades']}",
            flush=True,
        )
    return {"symbol": symbol, "folds": folds}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Diagnostic-only Phase 0D-L postmortem; cannot promote and never reads Aug 1")
    ap.add_argument("--feature-dir", default="evidence/v23/phase0dl_features250")
    ap.add_argument("--output", default="reports/V23_PHASE0DL_POSTMORTEM.json")
    a = ap.parse_args(argv)
    payload = {
        "phase": "V2.3-PHASE0DL-L2-MECHANISM",
        "stage": "POSTMORTEM_DIAGNOSTIC_ONLY",
        "promotion_allowed": False,
        "confirmation_analytically_opened": False,
        "costs_bps": list(COSTS),
        "symbols": [_symbol(Path(a.feature_dir), s) for s in SYMBOLS],
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    print("PHASE0DL_POSTMORTEM=COMPLETE_DIAGNOSTIC_ONLY")
    print("CONFIRMATION_2026_08_01=KEEP_SEALED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
