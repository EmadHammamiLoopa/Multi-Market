from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

SYMBOLS = ("BTCUSDT", "ETHUSDT")
HORIZONS = (10, 30, 60, 120, 300)
ALPHAS = (0.1, 1.0, 10.0, 100.0)
GATE_QUANTILES = (0.9900, 0.9950, 0.9975, 0.9990, 0.9995)
COSTS = (10.0, 12.0, 15.0)
PRIMARY_COST = 12.0
STRESS_COST = 15.0
TARGET_COST = 12.0

FOLDS = (
    (date(2026, 6, 15), date(2026, 6, 24)),
    (date(2026, 6, 25), date(2026, 7, 4)),
    (date(2026, 7, 5), date(2026, 7, 14)),
    (date(2026, 7, 15), date(2026, 7, 24)),
    (date(2026, 7, 25), date(2026, 8, 3)),
)

FEATURES = (
    "ret1", "ret3", "qfi1", "cfi1", "qfi3", "qfi5", "qfi10",
    "cfi3", "cfi5", "cfi10", "log_qty1", "log_qty5",
    "log_count1", "log_count5", "vwap_pressure_bps", "buy_present", "sell_present",
)


@dataclass(frozen=True)
class Config:
    horizon: int
    alpha: float
    gate_quantile: float


def _utc(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


def _load_numeric(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    pos = {name: i for i, name in enumerate(header)}
    required = ("timestamp", "price", *FEATURES)
    missing = [name for name in required if name not in pos]
    if missing:
        raise ValueError(f"missing development columns: {missing}")
    usecols = tuple(pos[name] for name in required)
    matrix = np.loadtxt(
        path, delimiter=",", skiprows=1, usecols=usecols,
        dtype=np.float64, ndmin=2,
    )
    ts = matrix[:, 0].astype(np.int64, copy=False)
    price = matrix[:, 1]
    X = matrix[:, 2:]
    if len(ts) < 2 or np.any(np.diff(ts) != 1):
        raise ValueError("development dataset must be a contiguous one-second grid")
    return ts, price, X


def _gross_return_bps(price: np.ndarray, horizon: int) -> np.ndarray:
    out = np.full(len(price), np.nan, dtype=np.float64)
    if len(price) <= horizon:
        return out
    out[:-horizon] = np.log(price[horizon:] / price[:-horizon]) * 10000.0
    return out


def _opportunity_target(gross: np.ndarray) -> np.ndarray:
    return np.sign(gross) * np.maximum(np.abs(gross) - TARGET_COST, 0.0)


def _fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(X_train)
    Xt = scaler.transform(X_train)
    model = Ridge(alpha=alpha).fit(Xt, y_train)
    train_pred = model.predict(Xt)
    pred = model.predict(scaler.transform(X_pred))
    return train_pred, pred


def _greedy_trade_indices(indices: np.ndarray, horizon: int) -> np.ndarray:
    if len(indices) == 0:
        return indices
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
    equity = np.cumsum(pnls)
    peaks = np.maximum.accumulate(np.concatenate(([0.0], equity)))[:-1]
    dd = peaks - equity
    return float(np.max(dd)) if len(dd) else 0.0


def _daily_stats(
    trade_ts: np.ndarray,
    pnls: np.ndarray,
    eval_start: int,
    eval_end_exclusive: int,
) -> dict[str, object]:
    day0 = eval_start // 86400
    day1 = (eval_end_exclusive - 1) // 86400
    days = np.arange(day0, day1 + 1, dtype=np.int64)
    daily_count = np.zeros(len(days), dtype=np.int64)
    daily_pnl = np.zeros(len(days), dtype=np.float64)
    if len(trade_ts):
        trade_days = trade_ts // 86400
        offsets = trade_days - day0
        valid = (offsets >= 0) & (offsets < len(days))
        np.add.at(daily_count, offsets[valid], 1)
        np.add.at(daily_pnl, offsets[valid], pnls[valid])
    active = daily_count > 0
    active_counts = daily_count[active]
    active_pnl = daily_pnl[active]
    if len(daily_pnl) >= 5:
        rolling5 = np.convolve(daily_pnl, np.ones(5), mode="valid")
        worst5 = float(np.min(rolling5))
    else:
        worst5 = float(np.sum(daily_pnl))
    return {
        "mean_trades_day_all": float(np.mean(daily_count)) if len(days) else 0.0,
        "median_trades_day_active": float(np.median(active_counts)) if len(active_counts) else 0.0,
        "active_days": int(np.sum(active)),
        "positive_active_day_fraction": float(np.mean(active_pnl > 0)) if len(active_pnl) else 0.0,
        "median_net_bps_day_all": float(np.median(daily_pnl)) if len(daily_pnl) else 0.0,
        "mean_net_bps_day_all": float(np.mean(daily_pnl)) if len(daily_pnl) else 0.0,
        "worst_5day_rolling_net_bps": worst5,
        "daily_counts": daily_count.tolist(),
        "daily_pnl_bps": daily_pnl.tolist(),
    }


def _trade_metrics(
    timestamps: np.ndarray,
    gross: np.ndarray,
    pred: np.ndarray,
    absolute_gate: float,
    horizon: int,
    eval_start_idx: int,
    eval_end_idx: int,
    eval_start_ts: int,
    eval_end_exclusive_ts: int,
    cost: float,
    include_daily_arrays: bool = False,
) -> dict[str, object]:
    local_pred = pred
    eligible_local = np.flatnonzero(np.abs(local_pred) >= absolute_gate)
    selected_local = _greedy_trade_indices(eligible_local, horizon)
    selected = selected_local + eval_start_idx
    if len(selected):
        direction = np.sign(local_pred[selected_local])
        valid_direction = direction != 0
        selected = selected[valid_direction]
        direction = direction[valid_direction]
    else:
        direction = np.empty(0, dtype=float)
    gross_trade = direction * gross[selected] if len(selected) else np.empty(0, dtype=float)
    net = gross_trade - cost
    wins = net[net > 0]
    losses = net[net < 0]
    gross_profit = float(np.sum(wins)) if len(wins) else 0.0
    gross_loss = float(-np.sum(losses)) if len(losses) else 0.0
    if gross_loss > 0:
        pf = gross_profit / gross_loss
    elif gross_profit > 0:
        pf = float("inf")
    else:
        pf = 0.0
    dd = _max_drawdown(net)
    total = float(np.sum(net))
    daily = _daily_stats(
        timestamps[selected] if len(selected) else np.empty(0, dtype=np.int64),
        net,
        eval_start_ts,
        eval_end_exclusive_ts,
    )
    result: dict[str, object] = {
        "trades": int(len(selected)),
        "long_trades": int(np.sum(direction > 0)),
        "short_trades": int(np.sum(direction < 0)),
        "gross_bps_trade": float(np.mean(gross_trade)) if len(gross_trade) else 0.0,
        "net_bps_trade": float(np.mean(net)) if len(net) else 0.0,
        "total_net_bps": total,
        "win_rate": float(np.mean(net > 0)) if len(net) else 0.0,
        "average_winner_bps": float(np.mean(wins)) if len(wins) else 0.0,
        "average_loser_bps": float(np.mean(losses)) if len(losses) else 0.0,
        "profit_factor": float(pf),
        "max_drawdown_bps": dd,
        "pnl_to_drawdown": float(total / dd) if dd > 0 else (float("inf") if total > 0 else 0.0),
        **{k: v for k, v in daily.items() if include_daily_arrays or k not in {"daily_counts", "daily_pnl_bps"}},
    }
    if include_daily_arrays:
        result["daily_counts"] = daily["daily_counts"]
        result["daily_pnl_bps"] = daily["daily_pnl_bps"]
        result["trade_timestamps"] = timestamps[selected].tolist() if len(selected) else []
        result["trade_net_bps"] = net.tolist()
    return result


def _inner_survives(m12: dict[str, object], m15: dict[str, object]) -> bool:
    return bool(
        m12["total_net_bps"] > 0
        and m12["net_bps_trade"] > 0
        and m12["profit_factor"] > 1.0
        and m12["median_trades_day_active"] >= 5
        and m15["total_net_bps"] > 0
        and m15["net_bps_trade"] > 0
    )


def _better(a: dict[str, object], b: dict[str, object] | None) -> bool:
    if b is None:
        return True
    am = float(a["metrics12"]["median_net_bps_day_all"])
    bm = float(b["metrics12"]["median_net_bps_day_all"])
    scale = max(abs(am), abs(bm), 1e-12)
    if abs(am - bm) / scale > 0.01:
        return am > bm
    aw = float(a["metrics12"]["worst_5day_rolling_net_bps"])
    bw = float(b["metrics12"]["worst_5day_rolling_net_bps"])
    if aw != bw:
        return aw > bw
    at = float(a["metrics12"]["median_trades_day_active"])
    bt = float(b["metrics12"]["median_trades_day_active"])
    if at != bt:
        return at > bt
    ad = float(a["metrics12"]["max_drawdown_bps"])
    bd = float(b["metrics12"]["max_drawdown_bps"])
    if ad != bd:
        return ad < bd
    ac: Config = a["config"]  # type: ignore[assignment]
    bc: Config = b["config"]  # type: ignore[assignment]
    if ac.horizon != bc.horizon:
        return ac.horizon < bc.horizon
    return ac.gate_quantile > bc.gate_quantile


def _fit_config_and_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    alpha: float,
    gate_quantile: float,
) -> tuple[np.ndarray, float]:
    train_pred, eval_pred = _fit_predict(X_train, y_train, X_eval, alpha)
    gate = float(np.quantile(np.abs(train_pred), gate_quantile))
    return eval_pred, gate


def _select_inner(
    timestamps: np.ndarray,
    price: np.ndarray,
    X: np.ndarray,
    outer_train_end: int,
) -> tuple[Config | None, dict[str, object]]:
    cut = int(outer_train_end * 0.8)
    if cut < 10000 or outer_train_end - cut < 1000:
        return None, {"survivors": 0, "reason": "INSUFFICIENT_INNER_ROWS"}

    best: dict[str, object] | None = None
    survivors = 0
    tested = 0

    inner_start_ts = int(timestamps[cut])
    inner_end_exclusive_ts = int(timestamps[outer_train_end - 1]) + 1

    for horizon in HORIZONS:
        gross = _gross_return_bps(price, horizon)
        y = _opportunity_target(gross)
        train_end_h = min(cut - horizon, cut)
        val_end_h = outer_train_end - horizon
        if train_end_h < 10000 or val_end_h <= cut:
            continue
        finite_train = np.isfinite(y[:train_end_h])
        if not np.all(finite_train):
            raise ValueError("unexpected non-finite inner training target")
        Xtr = X[:train_end_h]
        ytr = y[:train_end_h]
        Xv = X[cut:val_end_h]

        for alpha in ALPHAS:
            train_pred, val_pred = _fit_predict(Xtr, ytr, Xv, alpha)
            abs_train = np.abs(train_pred)
            for q in GATE_QUANTILES:
                tested += 1
                gate = float(np.quantile(abs_train, q))
                m12 = _trade_metrics(
                    timestamps, gross, val_pred, gate, horizon,
                    cut, val_end_h, inner_start_ts, inner_end_exclusive_ts,
                    PRIMARY_COST,
                )
                m15 = _trade_metrics(
                    timestamps, gross, val_pred, gate, horizon,
                    cut, val_end_h, inner_start_ts, inner_end_exclusive_ts,
                    STRESS_COST,
                )
                if not _inner_survives(m12, m15):
                    continue
                survivors += 1
                candidate = {
                    "config": Config(horizon, alpha, q),
                    "metrics12": m12,
                    "metrics15": m15,
                }
                if _better(candidate, best):
                    best = candidate

    if best is None:
        return None, {"survivors": 0, "tested": tested, "reason": "NO_CONFIGURATION"}
    cfg: Config = best["config"]  # type: ignore[assignment]
    return cfg, {
        "survivors": survivors,
        "tested": tested,
        "selected": {"horizon": cfg.horizon, "alpha": cfg.alpha, "gate_quantile": cfg.gate_quantile},
        "selected_inner_metrics_12bps": best["metrics12"],
        "selected_inner_metrics_15bps": best["metrics15"],
    }


def _outer_eval(
    timestamps: np.ndarray,
    price: np.ndarray,
    X: np.ndarray,
    cfg: Config,
    eval_start_d: date,
    eval_end_d: date,
) -> dict[str, object]:
    eval_start_ts = _utc(eval_start_d)
    eval_end_exclusive_ts = _utc(eval_end_d + timedelta(days=1))
    train_end = int(np.searchsorted(timestamps, eval_start_ts - cfg.horizon, side="left"))
    eval_begin = int(np.searchsorted(timestamps, eval_start_ts, side="left"))
    eval_end = int(np.searchsorted(timestamps, eval_end_exclusive_ts - cfg.horizon, side="left"))

    gross = _gross_return_bps(price, cfg.horizon)
    y = _opportunity_target(gross)
    Xtr = X[:train_end]
    ytr = y[:train_end]
    Xev = X[eval_begin:eval_end]
    train_pred, pred = _fit_predict(Xtr, ytr, Xev, cfg.alpha)
    gate = float(np.quantile(np.abs(train_pred), cfg.gate_quantile))

    costs: dict[str, object] = {}
    for cost in COSTS:
        costs[str(int(cost))] = _trade_metrics(
            timestamps, gross, pred, gate, cfg.horizon,
            eval_begin, eval_end, eval_start_ts, eval_end_exclusive_ts,
            cost, include_daily_arrays=True,
        )
    return {
        "config": {"horizon": cfg.horizon, "alpha": cfg.alpha, "gate_quantile": cfg.gate_quantile},
        "absolute_prediction_gate": gate,
        "train_rows": train_end,
        "eval_rows": eval_end - eval_begin,
        "costs": costs,
    }


def _pool_outer(folds: list[dict[str, object]], cost: str) -> dict[str, object]:
    trade_pnls: list[float] = []
    trade_ts: list[int] = []
    daily_counts: list[int] = []
    daily_pnls: list[float] = []
    positive_folds = 0
    fold_expectancies: list[float] = []
    total_long = total_short = 0

    for fold in folds:
        outer = fold.get("outer")
        if not outer:
            fold_expectancies.append(float("-inf"))
            continue
        m = outer["costs"][cost]
        pnl = [float(x) for x in m.get("trade_net_bps", [])]
        ts = [int(x) for x in m.get("trade_timestamps", [])]
        trade_pnls.extend(pnl)
        trade_ts.extend(ts)
        daily_counts.extend(int(x) for x in m.get("daily_counts", []))
        daily_pnls.extend(float(x) for x in m.get("daily_pnl_bps", []))
        total_long += int(m["long_trades"])
        total_short += int(m["short_trades"])
        expectancy = float(m["net_bps_trade"])
        fold_expectancies.append(expectancy)
        if float(m["total_net_bps"]) > 0:
            positive_folds += 1

    pnl_arr = np.asarray(trade_pnls, dtype=float)
    daily_counts_arr = np.asarray(daily_counts, dtype=int)
    daily_pnls_arr = np.asarray(daily_pnls, dtype=float)
    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr < 0]
    gp = float(np.sum(wins)) if len(wins) else 0.0
    gl = float(-np.sum(losses)) if len(losses) else 0.0
    pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
    dd = _max_drawdown(pnl_arr)
    total = float(np.sum(pnl_arr))
    active = daily_counts_arr > 0
    active_pnl = daily_pnls_arr[active]
    return {
        "trades": int(len(pnl_arr)),
        "long_trades": total_long,
        "short_trades": total_short,
        "net_bps_trade": float(np.mean(pnl_arr)) if len(pnl_arr) else 0.0,
        "total_net_bps": total,
        "profit_factor": float(pf),
        "max_drawdown_bps": dd,
        "pnl_to_drawdown": float(total / dd) if dd > 0 else (float("inf") if total > 0 else 0.0),
        "positive_outer_folds": positive_folds,
        "fold_expectancies": fold_expectancies,
        "median_trades_day_active": float(np.median(daily_counts_arr[active])) if np.any(active) else 0.0,
        "mean_trades_day_all": float(np.mean(daily_counts_arr)) if len(daily_counts_arr) else 0.0,
        "positive_active_day_fraction": float(np.mean(active_pnl > 0)) if len(active_pnl) else 0.0,
    }


def _development_pass(p12: dict[str, object], p15: dict[str, object], folds: list[dict[str, object]]) -> bool:
    scored = sum(1 for f in folds if f.get("outer"))
    return bool(
        scored == 5
        and p12["positive_outer_folds"] >= 4
        and p12["net_bps_trade"] >= 0.50
        and p12["total_net_bps"] > 0
        and p15["net_bps_trade"] > 0
        and p15["total_net_bps"] > 0
        and p12["profit_factor"] >= 1.10
        and p12["positive_active_day_fraction"] >= 0.55
        and p12["pnl_to_drawdown"] >= 2.0
        and min(float(x) for x in p12["fold_expectancies"]) >= -1.0
        and p12["median_trades_day_active"] >= 10
    )


def score_symbol(path: Path, symbol: str) -> dict[str, object]:
    timestamps, price, X = _load_numeric(path)
    folds: list[dict[str, object]] = []

    for fold_idx, (eval_start_d, eval_end_d) in enumerate(FOLDS, 1):
        outer_train_end_maxh = int(np.searchsorted(timestamps, _utc(eval_start_d) - max(HORIZONS), side="left"))
        cfg, inner = _select_inner(timestamps, price, X, outer_train_end_maxh)
        record: dict[str, object] = {
            "fold": fold_idx,
            "eval_start": eval_start_d.isoformat(),
            "eval_end": eval_end_d.isoformat(),
            "inner_selection": inner,
        }
        if cfg is None:
            record["status"] = "NO_CONFIGURATION"
        else:
            record["status"] = "SCORED"
            record["outer"] = _outer_eval(timestamps, price, X, cfg, eval_start_d, eval_end_d)
        folds.append(record)

    pooled12 = _pool_outer(folds, "12")
    pooled15 = _pool_outer(folds, "15")
    passed = _development_pass(pooled12, pooled15, folds)
    return {
        "phase": "V2.3-PHASE0DH-OPPORTUNITY",
        "symbol": symbol,
        "development_only": True,
        "historical_holdout_opened": False,
        "objective_order": ["profitability", "stability", "opportunity_count", "raw_accuracy"],
        "folds": folds,
        "pooled_12bps": pooled12,
        "pooled_15bps": pooled15,
        "development_pass": passed,
        "decision": "CANDIDATE_FREEZE_BEFORE_HOLDOUT" if passed else "FAIL_KEEP_HOLDOUT_SEALED",
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score frozen Phase 0D-H opportunity development audit")
    p.add_argument("--work-dir", default="evidence/v23/phase0dh_tf")
    p.add_argument("--output-dir", default="evidence/v23/phase0dh_opportunity")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    work = Path(args.work_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for symbol in SYMBOLS:
        path = work / f"{symbol}_DEV.csv"
        if not path.exists():
            raise SystemExit(f"missing frozen development dataset: {path}")
        print(f"[{symbol}] nested opportunity scoring", flush=True)
        result = score_symbol(path, symbol)
        results.append(result)
        (out / f"{symbol}_PHASE0DH_OPPORTUNITY.json").write_text(json.dumps(result, indent=2) + "\n")
        p12 = result["pooled_12bps"]
        print(
            f"[{symbol}] pass={result['development_pass']} trades={p12['trades']} "
            f"expectancy12={p12['net_bps_trade']:.6f} median_trades_day={p12['median_trades_day_active']:.2f}",
            flush=True,
        )

    candidates = [r["symbol"] for r in results if r["development_pass"]]
    summary = {
        "phase": "V2.3-PHASE0DH-OPPORTUNITY",
        "development_only": True,
        "historical_holdout_opened": False,
        "candidate_targets": candidates,
        "decision": "CANDIDATE_FREEZE_BEFORE_HOLDOUT" if candidates else "FAIL_KEEP_HOLDOUT_SEALED",
    }
    (out / "V23_PHASE0DH_OPPORTUNITY_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"candidate_targets={','.join(candidates) if candidates else 'NONE'}")
    print(f"decision={summary['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
