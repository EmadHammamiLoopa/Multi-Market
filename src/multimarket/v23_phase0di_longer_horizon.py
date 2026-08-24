from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

SYMBOLS = ("BTCUSDT", "ETHUSDT")
HORIZONS = (600, 1800, 3600)
ALPHAS = (0.1, 1.0, 10.0, 100.0)
GATE_QUANTILES = (0.9900, 0.9950, 0.9975, 0.9990, 0.9995)
COSTS = (5.0, 8.0, 10.0, 12.0, 15.0)
PRIMARY_COST = 12.0
STRESS_COST = 15.0
MAX_TRAIL = 300

FOLDS = (
    (date(2026, 6, 15), date(2026, 6, 24)),
    (date(2026, 6, 25), date(2026, 7, 4)),
    (date(2026, 7, 5), date(2026, 7, 14)),
    (date(2026, 7, 15), date(2026, 7, 24)),
    (date(2026, 7, 25), date(2026, 8, 3)),
)

BASE_FEATURES = (
    "ret1", "ret3", "qfi1", "cfi1", "qfi3", "qfi5", "qfi10",
    "cfi3", "cfi5", "cfi10", "log_qty1", "log_qty5",
    "log_count1", "log_count5", "vwap_pressure_bps", "buy_present", "sell_present",
)
LONG_WINDOWS = (30, 60, 120, 300)


@dataclass(frozen=True)
class Config:
    horizon: int
    alpha: float
    gate_quantile: float


def _utc(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


def _load_base(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    pos = {name: i for i, name in enumerate(header)}
    required = ("timestamp", "price", *BASE_FEATURES)
    missing = [n for n in required if n not in pos]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    usecols = tuple(pos[n] for n in required)
    matrix = np.loadtxt(path, delimiter=",", skiprows=1, usecols=usecols, dtype=np.float64, ndmin=2)
    ts = matrix[:, 0].astype(np.int64, copy=False)
    price = matrix[:, 1]
    Xbase = matrix[:, 2:]
    if np.any(np.diff(ts) != 1):
        raise ValueError("development dataset must be contiguous one-second grid")
    fmap = {name: i for i, name in enumerate(BASE_FEATURES)}
    return ts, price, Xbase, fmap


def _rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    cs = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    out = np.full(len(x), np.nan, dtype=np.float64)
    out[w - 1:] = cs[w:] - cs[:-w]
    return out


def _rolling_return(price: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(price), np.nan, dtype=np.float64)
    out[w:] = np.log(price[w:] / price[:-w]) * 10000.0
    return out


def _derive_long_features(price: np.ndarray, Xbase: np.ndarray, fmap: dict[str, int]) -> np.ndarray:
    qty = np.expm1(Xbase[:, fmap["log_qty1"]])
    cnt = np.expm1(Xbase[:, fmap["log_count1"]])
    qfi1 = Xbase[:, fmap["qfi1"]]
    cfi1 = Xbase[:, fmap["cfi1"]]

    buy_qty = 0.5 * qty * (1.0 + qfi1)
    sell_qty = qty - buy_qty
    buy_cnt = 0.5 * cnt * (1.0 + cfi1)
    sell_cnt = cnt - buy_cnt

    extra: list[np.ndarray] = []
    for w in LONG_WINDOWS:
        bq = _rolling_sum(buy_qty, w)
        sq = _rolling_sum(sell_qty, w)
        bc = _rolling_sum(buy_cnt, w)
        sc = _rolling_sum(sell_cnt, w)
        tq = bq + sq
        tc = bc + sc
        qfi = np.divide(bq - sq, tq, out=np.zeros_like(tq), where=tq > 0)
        cfi = np.divide(bc - sc, tc, out=np.zeros_like(tc), where=tc > 0)
        extra.extend([
            qfi,
            cfi,
            np.log1p(tq),
            np.log1p(tc),
            _rolling_return(price, w),
        ])
    return np.column_stack([Xbase, *extra])


def _gross_return(price: np.ndarray, h: int) -> np.ndarray:
    out = np.full(len(price), np.nan, dtype=np.float64)
    out[:-h] = np.log(price[h:] / price[:-h]) * 10000.0
    return out


def _fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xev: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(Xtr)
    Ztr = scaler.transform(Xtr)
    model = Ridge(alpha=alpha).fit(Ztr, ytr)
    return model.predict(Ztr), model.predict(scaler.transform(Xev))


def _greedy(indices: np.ndarray, horizon: int) -> np.ndarray:
    selected: list[int] = []
    next_allowed = -1
    for idx in indices.tolist():
        if idx >= next_allowed:
            selected.append(idx)
            next_allowed = idx + horizon + 1
    return np.asarray(selected, dtype=np.int64)


def _max_drawdown(pnls: np.ndarray) -> float:
    if len(pnls) == 0:
        return 0.0
    eq = np.cumsum(pnls)
    peaks = np.maximum.accumulate(np.concatenate(([0.0], eq)))[:-1]
    return float(np.max(peaks - eq))


def _metrics(ts: np.ndarray, gross: np.ndarray, pred: np.ndarray, gate: float, horizon: int,
             start_idx: int, end_idx: int, start_ts: int, end_ts_excl: int, cost: float,
             include_arrays: bool = False) -> dict[str, object]:
    eligible = np.flatnonzero(np.abs(pred) >= gate)
    local = _greedy(eligible, horizon)
    selected = local + start_idx
    direction = np.sign(pred[local]) if len(local) else np.empty(0)
    keep = direction != 0
    selected = selected[keep]
    direction = direction[keep]
    gross_trade = direction * gross[selected] if len(selected) else np.empty(0)
    net = gross_trade - cost

    wins = net[net > 0]
    losses = net[net < 0]
    gp = float(np.sum(wins)) if len(wins) else 0.0
    gl = float(-np.sum(losses)) if len(losses) else 0.0
    pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
    dd = _max_drawdown(net)
    total = float(np.sum(net))

    day0 = start_ts // 86400
    day1 = (end_ts_excl - 1) // 86400
    days = np.arange(day0, day1 + 1, dtype=np.int64)
    counts = np.zeros(len(days), dtype=np.int64)
    daily = np.zeros(len(days), dtype=np.float64)
    if len(selected):
        offsets = ts[selected] // 86400 - day0
        valid = (offsets >= 0) & (offsets < len(days))
        np.add.at(counts, offsets[valid], 1)
        np.add.at(daily, offsets[valid], net[valid])
    active = counts > 0
    rolling5 = np.convolve(daily, np.ones(5), mode="valid") if len(daily) >= 5 else np.asarray([np.sum(daily)])

    out: dict[str, object] = {
        "trades": int(len(selected)),
        "gross_bps_trade": float(np.mean(gross_trade)) if len(gross_trade) else 0.0,
        "net_bps_trade": float(np.mean(net)) if len(net) else 0.0,
        "total_net_bps": total,
        "profit_factor": float(pf),
        "max_drawdown_bps": dd,
        "pnl_to_drawdown": float(total / dd) if dd > 0 else (float("inf") if total > 0 else 0.0),
        "median_trades_day_active": float(np.median(counts[active])) if np.any(active) else 0.0,
        "mean_trades_day_all": float(np.mean(counts)) if len(counts) else 0.0,
        "positive_active_day_fraction": float(np.mean(daily[active] > 0)) if np.any(active) else 0.0,
        "median_net_bps_day_all": float(np.median(daily)) if len(daily) else 0.0,
        "worst_5day_rolling_net_bps": float(np.min(rolling5)) if len(rolling5) else 0.0,
    }
    if include_arrays:
        out["trade_net_bps"] = net.tolist()
        out["daily_counts"] = counts.tolist()
        out["daily_pnl_bps"] = daily.tolist()
    return out


def _survives(m12: dict[str, object], m15: dict[str, object]) -> bool:
    return bool(
        float(m12["net_bps_trade"]) > 0
        and float(m12["total_net_bps"]) > 0
        and float(m12["profit_factor"]) > 1.0
        and float(m15["net_bps_trade"]) > 0
        and float(m15["total_net_bps"]) > 0
        and float(m12["median_trades_day_active"]) >= 2.0
    )


def _better(a: dict[str, object], b: dict[str, object] | None) -> bool:
    if b is None:
        return True
    am = float(a["m12"]["median_net_bps_day_all"])
    bm = float(b["m12"]["median_net_bps_day_all"])
    if am != bm:
        return am > bm
    aw = float(a["m12"]["worst_5day_rolling_net_bps"])
    bw = float(b["m12"]["worst_5day_rolling_net_bps"])
    if aw != bw:
        return aw > bw
    at = float(a["m12"]["median_trades_day_active"])
    bt = float(b["m12"]["median_trades_day_active"])
    if at != bt:
        return at > bt
    ad = float(a["m12"]["max_drawdown_bps"])
    bd = float(b["m12"]["max_drawdown_bps"])
    if ad != bd:
        return ad < bd
    ac: Config = a["cfg"]  # type: ignore[assignment]
    bc: Config = b["cfg"]  # type: ignore[assignment]
    if ac.horizon != bc.horizon:
        return ac.horizon < bc.horizon
    return ac.gate_quantile > bc.gate_quantile


def _select_inner(ts: np.ndarray, price: np.ndarray, X: np.ndarray, outer_train_end_maxh: int) -> tuple[Config | None, dict[str, object]]:
    cut = int(outer_train_end_maxh * 0.8)
    best: dict[str, object] | None = None
    survivors = 0
    tested = 0
    start_ts = int(ts[cut])
    end_ts = int(ts[outer_train_end_maxh - 1]) + 1

    for h in HORIZONS:
        gross = _gross_return(price, h)
        train_end = min(cut - h, cut)
        val_end = outer_train_end_maxh - h
        if train_end <= MAX_TRAIL or val_end <= cut:
            continue
        valid_train_start = MAX_TRAIL
        Xtr = X[valid_train_start:train_end]
        ytr = gross[valid_train_start:train_end]
        Xv = X[cut:val_end]
        for alpha in ALPHAS:
            train_pred, val_pred = _fit_predict(Xtr, ytr, Xv, alpha)
            abs_train = np.abs(train_pred)
            for q in GATE_QUANTILES:
                tested += 1
                gate = float(np.quantile(abs_train, q))
                m12 = _metrics(ts, gross, val_pred, gate, h, cut, val_end, start_ts, end_ts, 12.0)
                m15 = _metrics(ts, gross, val_pred, gate, h, cut, val_end, start_ts, end_ts, 15.0)
                if not _survives(m12, m15):
                    continue
                survivors += 1
                cand = {"cfg": Config(h, alpha, q), "m12": m12, "m15": m15}
                if _better(cand, best):
                    best = cand

    if best is None:
        return None, {"tested": tested, "survivors": 0, "reason": "NO_CONFIGURATION"}
    cfg: Config = best["cfg"]  # type: ignore[assignment]
    return cfg, {
        "tested": tested,
        "survivors": survivors,
        "selected": {"horizon": cfg.horizon, "alpha": cfg.alpha, "gate_quantile": cfg.gate_quantile},
        "selected_inner_12bps": best["m12"],
        "selected_inner_15bps": best["m15"],
    }


def _outer(ts: np.ndarray, price: np.ndarray, X: np.ndarray, cfg: Config, start_d: date, end_d: date) -> dict[str, object]:
    start_ts = _utc(start_d)
    end_ts_excl = _utc(end_d + timedelta(days=1))
    train_end = int(np.searchsorted(ts, start_ts - cfg.horizon, side="left"))
    eval_begin = int(np.searchsorted(ts, start_ts, side="left"))
    eval_end = int(np.searchsorted(ts, end_ts_excl - cfg.horizon, side="left"))
    gross = _gross_return(price, cfg.horizon)
    Xtr = X[MAX_TRAIL:train_end]
    ytr = gross[MAX_TRAIL:train_end]
    train_pred, pred = _fit_predict(Xtr, ytr, X[eval_begin:eval_end], cfg.alpha)
    gate = float(np.quantile(np.abs(train_pred), cfg.gate_quantile))
    costs = {
        str(int(c)): _metrics(ts, gross, pred, gate, cfg.horizon, eval_begin, eval_end,
                              start_ts, end_ts_excl, c, include_arrays=True)
        for c in COSTS
    }
    return {
        "config": {"horizon": cfg.horizon, "alpha": cfg.alpha, "gate_quantile": cfg.gate_quantile},
        "absolute_prediction_gate": gate,
        "costs": costs,
    }


def _pool(folds: list[dict[str, object]], cost: str) -> dict[str, object]:
    pnls: list[float] = []
    daily_counts: list[int] = []
    daily_pnl: list[float] = []
    positive_folds = 0
    fold_exp: list[float] = []
    for f in folds:
        outer = f.get("outer")
        if not outer:
            fold_exp.append(float("-inf"))
            continue
        m = outer["costs"][cost]
        p = [float(x) for x in m.get("trade_net_bps", [])]
        pnls.extend(p)
        daily_counts.extend(int(x) for x in m.get("daily_counts", []))
        daily_pnl.extend(float(x) for x in m.get("daily_pnl_bps", []))
        fold_exp.append(float(m["net_bps_trade"]))
        if float(m["total_net_bps"]) > 0:
            positive_folds += 1
    a = np.asarray(pnls, dtype=float)
    dc = np.asarray(daily_counts, dtype=int)
    dp = np.asarray(daily_pnl, dtype=float)
    wins = a[a > 0]
    losses = a[a < 0]
    gp = float(np.sum(wins)) if len(wins) else 0.0
    gl = float(-np.sum(losses)) if len(losses) else 0.0
    pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
    dd = _max_drawdown(a)
    total = float(np.sum(a))
    active = dc > 0
    return {
        "trades": int(len(a)),
        "net_bps_trade": float(np.mean(a)) if len(a) else 0.0,
        "total_net_bps": total,
        "profit_factor": float(pf),
        "max_drawdown_bps": dd,
        "pnl_to_drawdown": float(total / dd) if dd > 0 else (float("inf") if total > 0 else 0.0),
        "positive_outer_folds": positive_folds,
        "fold_expectancies": fold_exp,
        "median_trades_day_active": float(np.median(dc[active])) if np.any(active) else 0.0,
        "positive_active_day_fraction": float(np.mean(dp[active] > 0)) if np.any(active) else 0.0,
    }


def _passes(p12: dict[str, object], p15: dict[str, object], folds: list[dict[str, object]]) -> bool:
    scored = sum(1 for f in folds if f.get("outer"))
    return bool(
        scored == 5
        and int(p12["positive_outer_folds"]) >= 4
        and float(p12["net_bps_trade"]) >= 1.0
        and float(p12["total_net_bps"]) > 0
        and float(p15["net_bps_trade"]) > 0
        and float(p15["total_net_bps"]) > 0
        and float(p12["profit_factor"]) >= 1.15
        and float(p12["positive_active_day_fraction"]) >= 0.55
        and float(p12["pnl_to_drawdown"]) >= 2.0
        and min(float(x) for x in p12["fold_expectancies"]) >= -2.0
        and float(p12["median_trades_day_active"]) >= 2.0
    )


def score_symbol(path: Path, symbol: str) -> dict[str, object]:
    ts, price, Xbase, fmap = _load_base(path)
    print(f"[{symbol}] deriving causal 30/60/120/300s features", flush=True)
    X = _derive_long_features(price, Xbase, fmap)
    if not np.all(np.isfinite(X[MAX_TRAIL:])):
        raise ValueError("non-finite longer-horizon features after 300s warmup")

    folds: list[dict[str, object]] = []
    for i, (sd, ed) in enumerate(FOLDS, 1):
        outer_train_end_maxh = int(np.searchsorted(ts, _utc(sd) - max(HORIZONS), side="left"))
        cfg, inner = _select_inner(ts, price, X, outer_train_end_maxh)
        rec: dict[str, object] = {"fold": i, "eval_start": sd.isoformat(), "eval_end": ed.isoformat(), "inner_selection": inner}
        if cfg is None:
            rec["status"] = "NO_CONFIGURATION"
        else:
            rec["status"] = "SCORED"
            rec["outer"] = _outer(ts, price, X, cfg, sd, ed)
        folds.append(rec)

    p12 = _pool(folds, "12")
    p15 = _pool(folds, "15")
    passed = _passes(p12, p15, folds)
    return {
        "phase": "V2.3-PHASE0DI-LONGER-HORIZON-FLOW",
        "symbol": symbol,
        "development_only": True,
        "historical_holdout_opened": False,
        "folds": folds,
        "pooled_12bps": p12,
        "pooled_15bps": p15,
        "development_pass": passed,
        "decision": "CANDIDATE_FREEZE_BEFORE_CONFIRMATION" if passed else "FAIL_KEEP_HOLDOUT_SEALED",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score frozen V2.3 Phase 0D-I longer-horizon trade-flow audit")
    p.add_argument("--work-dir", default="evidence/v23/phase0dh_tf")
    p.add_argument("--output-dir", default="evidence/v23/phase0di_longer_horizon")
    args = p.parse_args(argv)
    work = Path(args.work_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for symbol in SYMBOLS:
        print(f"[{symbol}] Phase 0D-I nested scoring", flush=True)
        result = score_symbol(work / f"{symbol}_DEV.csv", symbol)
        results.append(result)
        (out / f"{symbol}_PHASE0DI.json").write_text(json.dumps(result, indent=2) + "\n")
        p12 = result["pooled_12bps"]
        print(f"[{symbol}] pass={result['development_pass']} trades={p12['trades']} expectancy12={p12['net_bps_trade']:.6f}", flush=True)

    candidates = [r["symbol"] for r in results if r["development_pass"]]
    summary = {
        "phase": "V2.3-PHASE0DI-LONGER-HORIZON-FLOW",
        "development_only": True,
        "historical_holdout_opened": False,
        "candidate_targets": candidates,
        "decision": "CANDIDATE_FOUND_FREEZE_BEFORE_CONFIRMATION" if candidates else "FAIL_KEEP_HOLDOUT_SEALED",
    }
    (out / "V23_PHASE0DI_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"candidate_targets={','.join(candidates) if candidates else 'NONE'}")
    print(f"decision={summary['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
