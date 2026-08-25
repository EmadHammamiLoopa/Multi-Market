from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

SYMBOLS = ("BTCUSDT", "ETHUSDT")
DAYS = tuple(date(2026, m, 1) for m in range(1, 8))
HORIZONS_S = (1, 3, 10, 30)
ALPHAS = (0.1, 1.0, 10.0, 100.0)
QUANTILES = (0.990, 0.995, 0.9975, 0.999)
GRID_S = 0.25
ENTRY_STEPS = 1
COST_PRIMARY = 8.0
COST_STRESS = 12.0
EXPECTED_ROWS = 345_600

L0_NAMES = (
    "spread_bps", "microprice_minus_mid_bps", "obi_l1", "obi_l5", "obi_l10",
    "log_bid_qty_l1", "log_ask_qty_l1", "log_bid_depth_l5", "log_ask_depth_l5",
    "log_bid_depth_l10", "log_ask_depth_l10",
)
L1_EXTRA_NAMES = (
    "ofi_l1_250ms", "ofi_l1_1s", "ofi_l1_3s", "mlofi_l5_250ms", "mlofi_l5_1s", "mlofi_l5_3s",
    "mlofi_l10_250ms", "mlofi_l10_1s", "mlofi_l10_3s", "trade_qty_imbalance_250ms",
    "trade_qty_imbalance_1s", "trade_qty_imbalance_3s", "trade_count_imbalance_250ms",
    "trade_count_imbalance_1s", "trade_count_imbalance_3s",
)
L2_EXTRA_NAMES = (
    "d_obi_l1_250ms", "d_obi_l1_1s", "d_obi_l5_250ms", "d_obi_l5_1s", "d_obi_l10_250ms",
    "d_obi_l10_1s", "d_spread_bps_250ms", "d_spread_bps_1s", "d_microprice_minus_mid_bps_250ms",
    "d_microprice_minus_mid_bps_1s", "bid_replenish_l5_1s", "ask_replenish_l5_1s",
    "bid_deplete_l5_1s", "ask_deplete_l5_1s", "trade_qty_imbalance_1s_x_obi_l5",
    "trade_qty_imbalance_1s_x_microprice_minus_mid_bps", "mlofi_l5_1s_x_spread_bps",
)
BLOCKS = {
    "L0": L0_NAMES,
    "L1": L0_NAMES + L1_EXTRA_NAMES,
    "L2": L0_NAMES + L1_EXTRA_NAMES + L2_EXTRA_NAMES,
}
VALID_COL = {"L0": "l0_valid", "L1": "l1_valid", "L2": "l2_valid"}


@dataclass(frozen=True)
class Config:
    block: str
    horizon_s: int
    alpha: float
    quantile: float


@dataclass
class DayData:
    day: date
    ts: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    mid: np.ndarray
    book_valid: np.ndarray
    valid: dict[str, np.ndarray]
    X: dict[str, np.ndarray]


def _header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return f.readline().rstrip("\n\r").split(",")


def _load_day(path: Path, day: date) -> DayData:
    names = _header(path)
    pos = {n: i for i, n in enumerate(names)}
    required = {"local_timestamp_us", "best_bid", "best_ask", "mid", "book_valid", "l0_valid", "l1_valid", "l2_valid", *L0_NAMES, *L1_EXTRA_NAMES, *L2_EXTRA_NAMES}
    missing = sorted(required - set(pos))
    if missing:
        raise RuntimeError(f"missing feature columns in {path}: {missing}")
    ordered = ["local_timestamp_us", "best_bid", "best_ask", "mid", "book_valid", "l0_valid", "l1_valid", "l2_valid", *L0_NAMES, *L1_EXTRA_NAMES, *L2_EXTRA_NAMES]
    use = [pos[n] for n in ordered]
    a = np.loadtxt(path, delimiter=",", skiprows=1, usecols=use, dtype=np.float64, ndmin=2)
    if len(a) != EXPECTED_ROWS:
        raise RuntimeError(f"{path}: expected {EXPECTED_ROWS} rows, got {len(a)}")
    ts = a[:, 0].astype(np.int64, copy=False)
    if np.any(np.diff(ts) != 250_000):
        raise RuntimeError(f"{path}: non-250ms timestamp grid")
    bid, ask, mid = a[:, 1], a[:, 2], a[:, 3]
    book_valid = a[:, 4].astype(bool)
    valid = {"L0": a[:, 5].astype(bool), "L1": a[:, 6].astype(bool), "L2": a[:, 7].astype(bool)}
    base = 8
    n0, n1, n2 = len(L0_NAMES), len(L1_EXTRA_NAMES), len(L2_EXTRA_NAMES)
    x0 = a[:, base:base+n0].astype(np.float32)
    x1 = a[:, base:base+n0+n1].astype(np.float32)
    x2 = a[:, base:base+n0+n1+n2].astype(np.float32)
    return DayData(day, ts, bid, ask, mid, book_valid, valid, {"L0": x0, "L1": x1, "L2": x2})


def _labels(day: DayData, horizon_s: int, block: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h = int(round(horizon_s / GRID_S))
    n = len(day.ts)
    entry = np.arange(n, dtype=np.int64) + ENTRY_STEPS
    exit_ = entry + h
    ok = day.valid[block].copy()
    ok &= exit_ < n
    safe_entry = np.minimum(entry, n-1)
    safe_exit = np.minimum(exit_, n-1)
    ok &= day.book_valid[safe_entry] & day.book_valid[safe_exit]
    y = np.full(n, np.nan, dtype=np.float64)
    gl = np.full(n, np.nan, dtype=np.float64)
    gs = np.full(n, np.nan, dtype=np.float64)
    idx = np.flatnonzero(ok)
    e = entry[idx]; x = exit_[idx]
    y[idx] = 10000.0 * np.log(day.mid[x] / day.mid[e])
    gl[idx] = 10000.0 * np.log(day.bid[x] / day.ask[e])
    gs[idx] = 10000.0 * np.log(day.bid[e] / day.ask[x])
    ok &= np.isfinite(y) & np.isfinite(gl) & np.isfinite(gs)
    return ok, y, gl, gs


def _stack(days: list[DayData], block: str, horizon_s: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for d in days:
        ok, y, _, _ = _labels(d, horizon_s, block)
        idx = np.flatnonzero(ok)
        xs.append(d.X[block][idx])
        ys.append(y[idx])
    if not xs:
        return np.empty((0, len(BLOCKS[block])), dtype=np.float32), np.empty(0)
    return np.concatenate(xs, axis=0), np.concatenate(ys)


def _ridge_stats(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    if len(X) < 2:
        raise RuntimeError("insufficient training rows")
    xd = X.astype(np.float64, copy=False)
    mu = xd.mean(axis=0)
    scale = xd.std(axis=0)
    scale[scale == 0] = 1.0
    ymu = float(y.mean())
    xtx = xd.T @ xd
    centered = xtx - len(xd) * np.outer(mu, mu)
    gram = centered / np.outer(scale, scale)
    xty = xd.T @ y
    cov = (xty - len(xd) * mu * ymu) / scale
    return mu, scale, ymu, gram, cov


def _fit_from_stats(stats, alpha: float) -> tuple[np.ndarray, float]:
    mu, scale, ymu, gram, cov = stats
    coef_z = np.linalg.solve(gram + float(alpha) * np.eye(len(mu)), cov)
    w = coef_z / scale
    intercept = ymu - float(mu @ w)
    return w, intercept


def _predict(X: np.ndarray, model: tuple[np.ndarray, float]) -> np.ndarray:
    w, b = model
    return X.astype(np.float64, copy=False) @ w + b


def _greedy_signals(indices: np.ndarray, horizon_s: int) -> np.ndarray:
    h = int(round(horizon_s / GRID_S))
    next_allowed = -1
    chosen: list[int] = []
    for i in indices.tolist():
        if i >= next_allowed:
            chosen.append(i)
            next_allowed = i + ENTRY_STEPS + h
    return np.asarray(chosen, dtype=np.int64)


def _maxdd(net: np.ndarray) -> float:
    if not len(net):
        return 0.0
    eq = np.cumsum(net)
    peak = np.maximum.accumulate(np.concatenate(([0.0], eq)))[:-1]
    return float(np.max(peak - eq))


def _score_day(day: DayData, block: str, horizon_s: int, pred: np.ndarray, pred_idx: np.ndarray, gate: float, cost: float, arrays: bool=False) -> dict[str, object]:
    ok, _, gl, gs = _labels(day, horizon_s, block)
    eligible = ok[pred_idx] & np.isfinite(pred) & (np.abs(pred) >= gate) & (pred != 0)
    local = np.flatnonzero(eligible)
    chosen_local = _greedy_signals(pred_idx[local], horizon_s)
    if len(chosen_local):
        loc_map = {int(g): j for j, g in enumerate(pred_idx)}
        ppos = np.asarray([loc_map[int(g)] for g in chosen_local], dtype=np.int64)
        directions = np.sign(pred[ppos])
        gross = np.where(directions > 0, gl[chosen_local], gs[chosen_local])
    else:
        directions = np.empty(0); gross = np.empty(0)
    net = gross - cost
    gp = float(net[net > 0].sum()) if np.any(net > 0) else 0.0
    glos = float(-net[net < 0].sum()) if np.any(net < 0) else 0.0
    pf = gp / glos if glos > 0 else (float("inf") if gp > 0 else 0.0)
    dd = _maxdd(net)
    total = float(net.sum())
    hours = (day.ts[chosen_local] // 3_600_000_000).astype(np.int64) if len(chosen_local) else np.empty(0, dtype=np.int64)
    hour_pnl: dict[int, float] = {}
    for h, p in zip(hours.tolist(), net.tolist()):
        hour_pnl[h] = hour_pnl.get(h, 0.0) + float(p)
    out: dict[str, object] = {
        "trades": int(len(net)),
        "gross_bps_trade": float(gross.mean()) if len(gross) else 0.0,
        "net_bps_trade": float(net.mean()) if len(net) else 0.0,
        "total_net_bps": total,
        "profit_factor": float(pf),
        "max_drawdown_bps": dd,
        "pnl_to_drawdown": float(total/dd) if dd > 0 else (float("inf") if total > 0 else 0.0),
        "active_hours": int(len(hour_pnl)),
        "positive_active_hour_fraction": float(np.mean(np.asarray(list(hour_pnl.values())) > 0)) if hour_pnl else 0.0,
    }
    if arrays:
        out["trade_net_bps"] = net.tolist()
        out["trade_gross_bps"] = gross.tolist()
        out["trade_signal_ts_us"] = day.ts[chosen_local].astype(np.int64).tolist()
        out["hour_pnl"] = {str(k): v for k, v in sorted(hour_pnl.items())}
    return out


def _survives(m8: dict[str, object], m12: dict[str, object]) -> bool:
    return (
        float(m8["net_bps_trade"]) > 0 and float(m8["total_net_bps"]) > 0 and float(m8["profit_factor"]) > 1
        and int(m8["trades"]) >= 20 and float(m12["net_bps_trade"]) > 0 and float(m12["total_net_bps"]) > 0
    )


def _within_1pct(a: float, b: float) -> bool:
    return abs(a-b) <= 0.01 * max(abs(a), abs(b), 1e-12)


def _better(a: dict, b: dict | None) -> bool:
    if b is None:
        return True
    ae, be = float(a["m8"]["net_bps_trade"]), float(b["m8"]["net_bps_trade"])
    if not _within_1pct(ae, be):
        return ae > be
    at, bt = float(a["m8"]["total_net_bps"]), float(b["m8"]["total_net_bps"])
    if at != bt:
        return at > bt
    ap, bp = float(a["m8"]["profit_factor"]), float(b["m8"]["profit_factor"])
    if ap != bp:
        return ap > bp
    ad, bd = float(a["m8"]["max_drawdown_bps"]), float(b["m8"]["max_drawdown_bps"])
    if ad != bd:
        return ad < bd
    ac, bc = a["cfg"], b["cfg"]
    rank = {"L0": 0, "L1": 1, "L2": 2}
    if ac.block != bc.block:
        return rank[ac.block] < rank[bc.block]
    if ac.horizon_s != bc.horizon_s:
        return ac.horizon_s < bc.horizon_s
    if ac.quantile != bc.quantile:
        return ac.quantile > bc.quantile
    return ac.alpha < bc.alpha


def _select(inner_train: list[DayData], inner_val: DayData, blocks: tuple[str, ...]) -> tuple[Config | None, dict[str, object]]:
    best = None
    tested = 0; survivors = 0
    for block in blocks:
        for horizon in HORIZONS_S:
            Xtr, ytr = _stack(inner_train, block, horizon)
            if not len(Xtr):
                continue
            stats = _ridge_stats(Xtr, ytr)
            val_ok, _, _, _ = _labels(inner_val, horizon, block)
            val_idx = np.flatnonzero(val_ok)
            Xv = inner_val.X[block][val_idx]
            for alpha in ALPHAS:
                model = _fit_from_stats(stats, alpha)
                train_pred = _predict(Xtr, model)
                val_pred = _predict(Xv, model)
                abs_train = np.abs(train_pred)
                for q in QUANTILES:
                    tested += 1
                    gate = float(np.quantile(abs_train, q))
                    m8 = _score_day(inner_val, block, horizon, val_pred, val_idx, gate, COST_PRIMARY)
                    m12 = _score_day(inner_val, block, horizon, val_pred, val_idx, gate, COST_STRESS)
                    if not _survives(m8, m12):
                        continue
                    survivors += 1
                    cfg = Config(block, horizon, alpha, q)
                    cand = {"cfg": cfg, "gate": gate, "m8": m8, "m12": m12}
                    if _better(cand, best):
                        best = cand
    if best is None:
        return None, {"tested": tested, "survivors": 0, "reason": "NO_CONFIGURATION"}
    cfg = best["cfg"]
    return cfg, {"tested": tested, "survivors": survivors, "selected": cfg.__dict__, "selected_inner_8bps": best["m8"], "selected_inner_12bps": best["m12"]}


def _outer_fit_score(train_days: list[DayData], eval_day: DayData, cfg: Config) -> dict[str, object]:
    Xtr, ytr = _stack(train_days, cfg.block, cfg.horizon_s)
    stats = _ridge_stats(Xtr, ytr)
    model = _fit_from_stats(stats, cfg.alpha)
    train_pred = _predict(Xtr, model)
    gate = float(np.quantile(np.abs(train_pred), cfg.quantile))
    ok, _, _, _ = _labels(eval_day, cfg.horizon_s, cfg.block)
    idx = np.flatnonzero(ok)
    pred = _predict(eval_day.X[cfg.block][idx], model)
    return {
        "config": cfg.__dict__,
        "absolute_prediction_gate": gate,
        "cost_8": _score_day(eval_day, cfg.block, cfg.horizon_s, pred, idx, gate, COST_PRIMARY, arrays=True),
        "cost_12": _score_day(eval_day, cfg.block, cfg.horizon_s, pred, idx, gate, COST_STRESS, arrays=True),
    }


def _pool(folds: list[dict[str, object]], key: str) -> dict[str, object]:
    nets8: list[float] = []; nets12: list[float] = []; fold_exp8: list[float] = []
    hour_pnl: dict[str, float] = {}
    configs = 0
    for f in folds:
        o = f.get(key)
        if not o:
            fold_exp8.append(float("-inf")); continue
        configs += 1
        m8, m12 = o["cost_8"], o["cost_12"]
        nets8.extend(float(x) for x in m8["trade_net_bps"])
        nets12.extend(float(x) for x in m12["trade_net_bps"])
        fold_exp8.append(float(m8["net_bps_trade"]))
        for h, p in m8["hour_pnl"].items():
            hour_pnl[h] = hour_pnl.get(h, 0.0) + float(p)
    a8 = np.asarray(nets8, dtype=np.float64); a12 = np.asarray(nets12, dtype=np.float64)
    gp = float(a8[a8 > 0].sum()) if np.any(a8 > 0) else 0.0
    loss = float(-a8[a8 < 0].sum()) if np.any(a8 < 0) else 0.0
    pf = gp/loss if loss > 0 else (float("inf") if gp > 0 else 0.0)
    dd = _maxdd(a8); total8 = float(a8.sum()); total12 = float(a12.sum())
    return {
        "config_folds": configs,
        "trades": int(len(a8)),
        "positive_expectancy_folds_8": int(sum(x > 0 for x in fold_exp8 if np.isfinite(x))),
        "fold_expectancy_8": fold_exp8,
        "net_bps_trade_8": float(a8.mean()) if len(a8) else 0.0,
        "total_net_bps_8": total8,
        "profit_factor_8": float(pf),
        "max_drawdown_bps_8": dd,
        "pnl_to_drawdown_8": float(total8/dd) if dd > 0 else (float("inf") if total8 > 0 else 0.0),
        "net_bps_trade_12": float(a12.mean()) if len(a12) else 0.0,
        "total_net_bps_12": total12,
        "positive_active_hour_fraction_8": float(np.mean(np.asarray(list(hour_pnl.values())) > 0)) if hour_pnl else 0.0,
        "active_hours": len(hour_pnl),
    }


def _structural_pass(p: dict[str, object]) -> bool:
    fexp = [float(x) for x in p["fold_expectancy_8"]]
    return (
        int(p["config_folds"]) == 5
        and int(p["positive_expectancy_folds_8"]) >= 4
        and float(p["net_bps_trade_8"]) >= 1.0
        and float(p["total_net_bps_8"]) > 0
        and float(p["profit_factor_8"]) >= 1.20
        and float(p["net_bps_trade_12"]) > 0
        and float(p["total_net_bps_12"]) > 0
        and all(x >= -2.0 for x in fexp)
        and int(p["trades"]) >= 100
        and float(p["positive_active_hour_fraction_8"]) >= 0.55
        and float(p["pnl_to_drawdown_8"]) >= 2.0
    )


def _score_symbol(feature_dir: Path, symbol: str) -> dict[str, object]:
    days = [_load_day(feature_dir / symbol / f"{d.isoformat()}_FEATURES250.csv", d) for d in DAYS]
    folds: list[dict[str, object]] = []
    for eval_i in range(2, 7):
        outer_train = days[:eval_i]
        eval_day = days[eval_i]
        inner_train = outer_train[:-1]
        inner_val = outer_train[-1]
        l0_cfg, l0_sel = _select(inner_train, inner_val, ("L0",))
        dyn_cfg, dyn_sel = _select(inner_train, inner_val, ("L1", "L2"))
        rec: dict[str, object] = {
            "fold": eval_i - 1,
            "outer_train_days": [d.day.isoformat() for d in outer_train],
            "inner_train_days": [d.day.isoformat() for d in inner_train],
            "inner_validation_day": inner_val.day.isoformat(),
            "evaluation_day": eval_day.day.isoformat(),
            "l0_selection": l0_sel,
            "dynamic_selection": dyn_sel,
        }
        if l0_cfg is not None:
            rec["l0_outer"] = _outer_fit_score(outer_train, eval_day, l0_cfg)
        if dyn_cfg is not None:
            rec["dynamic_outer"] = _outer_fit_score(outer_train, eval_day, dyn_cfg)
        folds.append(rec)
        print(symbol, eval_day.day, "L0=", l0_cfg, "DYN=", dyn_cfg, flush=True)
    l0_pool = _pool(folds, "l0_outer")
    dyn_pool = _pool(folds, "dynamic_outer")
    l0_pass = _structural_pass(l0_pool)
    dyn_structural = _structural_pass(dyn_pool)
    incremental = dyn_structural and dyn_pool["net_bps_trade_8"] > l0_pool["net_bps_trade_8"] and dyn_pool["total_net_bps_8"] > l0_pool["total_net_bps_8"]
    passed = bool(dyn_structural and incremental)
    if passed:
        status = "PASS_OPEN_CONFIRMATION_ALLOWED"
    elif l0_pass:
        status = "STATIC_BOOK_SIGNAL_ONLY_KEEP_CONFIRMATION_SEALED"
    else:
        status = "FAIL_KEEP_CONFIRMATION_SEALED"
    return {
        "symbol": symbol,
        "folds": folds,
        "l0_pooled": l0_pool,
        "dynamic_pooled": dyn_pool,
        "l0_structural_pass": l0_pass,
        "dynamic_structural_pass": dyn_structural,
        "incremental_information_pass": bool(incremental),
        "development_pass": passed,
        "status": status,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Frozen Phase 0D-L development scorer; never reads Aug 1")
    ap.add_argument("--feature-dir", default="evidence/v23/phase0dl_features250")
    ap.add_argument("--output", default="reports/V23_PHASE0DL_DEVELOPMENT_RESULT.json")
    a = ap.parse_args(argv)
    feature_dir = Path(a.feature_dir)
    results = [_score_symbol(feature_dir, s) for s in SYMBOLS]
    payload = {
        "phase": "V2.3-PHASE0DL-L2-MECHANISM",
        "stage": "DEVELOPMENT_ONLY",
        "confirmation_day": "2026-08-01",
        "confirmation_analytically_opened": False,
        "older_holdout_analytically_opened": False,
        "reaction_latency_ms": 250,
        "cost_primary_bps": COST_PRIMARY,
        "cost_stress_bps": COST_STRESS,
        "symbols": results,
        "any_development_pass": any(r["development_pass"] for r in results),
    }
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    for r in results:
        p = r["dynamic_pooled"]
        print(
            f"{r['symbol']} status={r['status']} trades={p['trades']} "
            f"exp8={p['net_bps_trade_8']:.6f} total8={p['total_net_bps_8']:.3f} "
            f"pf8={p['profit_factor_8']:.4f} exp12={p['net_bps_trade_12']:.6f} "
            f"incremental={r['incremental_information_pass']}", flush=True,
        )
    print("PHASE0DL_DEVELOPMENT=" + ("PASS" if payload["any_development_pass"] else "FAIL"))
    print("CONFIRMATION_2026_08_01=" + ("MAY_OPEN_FOR_PASSING_SYMBOLS_ONLY" if payload["any_development_pass"] else "KEEP_SEALED"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
