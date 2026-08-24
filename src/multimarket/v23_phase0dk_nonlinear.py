from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from . import v23_phase0dj_score as j

SEED = 20260824
MODEL_CONFIGS = (
    ("X1", 3, 0.05, 300),
    ("X2", 3, 0.05, 600),
    ("X3", 5, 0.03, 300),
    ("X4", 5, 0.03, 600),
)
BLOCKS = ("K1", "K2")


@dataclass(frozen=True)
class KConfig:
    block: str
    horizon: int
    model_name: str
    max_depth: int
    learning_rate: float
    n_estimators: int
    quantile: float


def _model(cfg: KConfig) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device="cuda",
        random_state=SEED,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=5.0,
        n_jobs=1,
        verbosity=0,
    )


def cuda_preflight() -> dict[str, object]:
    """Fail before official scoring if CUDA XGBoost cannot train/predict."""
    x = np.arange(64, dtype=np.float32).reshape(32, 2)
    y = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    cfg = KConfig("K1", 5, "X1", 3, 0.05, 2, 0.99)
    try:
        m = _model(cfg)
        m.fit(x, y)
        p = m.predict(x[:4])
    except Exception as exc:
        raise RuntimeError(
            "PHASE0DK_CUDA_PREFLIGHT=FAIL: XGBoost CUDA training/prediction unavailable"
        ) from exc
    if len(p) != 4 or not np.all(np.isfinite(p)):
        raise RuntimeError("PHASE0DK_CUDA_PREFLIGHT=FAIL: invalid CUDA predictions")
    booster_cfg = json.loads(m.get_booster().save_config())
    device = booster_cfg.get("learner", {}).get("generic_param", {}).get("device")
    if device is None or not str(device).startswith("cuda"):
        raise RuntimeError(f"PHASE0DK_CUDA_PREFLIGHT=FAIL: resolved device={device!r}")
    return {"pass": True, "resolved_device": str(device)}


def _cfg_dict(c: KConfig) -> dict[str, object]:
    return {
        "block": c.block,
        "horizon": c.horizon,
        "model_name": c.model_name,
        "max_depth": c.max_depth,
        "learning_rate": c.learning_rate,
        "n_estimators": c.n_estimators,
        "quantile": c.quantile,
    }


def _better(a: dict[str, object], b: dict[str, object] | None) -> bool:
    if b is None:
        return True
    am = float(a["m12"]["median_net_bps_day_all"])
    bm = float(b["m12"]["median_net_bps_day_all"])
    scale = max(abs(am), abs(bm), 1e-12)
    if abs(am - bm) / scale > 0.01:
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
    ac: KConfig = a["cfg"]  # type: ignore[assignment]
    bc: KConfig = b["cfg"]  # type: ignore[assignment]
    if ac.block != bc.block:
        return ac.block == "K1"
    if ac.horizon != bc.horizon:
        return ac.horizon < bc.horizon
    if ac.quantile != bc.quantile:
        return ac.quantile > bc.quantile
    ai = next(i for i, x in enumerate(MODEL_CONFIGS) if x[0] == ac.model_name)
    bi = next(i for i, x in enumerate(MODEL_CONFIGS) if x[0] == bc.model_name)
    return ai < bi


def _survives(m12: dict[str, object], m15: dict[str, object]) -> bool:
    return bool(
        float(m12["net_bps_trade"]) > 0
        and float(m12["total_net_bps"]) > 0
        and float(m12["profit_factor"]) > 1.0
        and float(m12["median_trades_day_active"]) >= 2.0
        and float(m15["net_bps_trade"]) > 0
        and float(m15["total_net_bps"]) > 0
    )


def _select(
    decision: np.ndarray,
    blocks: dict[str, np.ndarray],
    gross_by_h: dict[int, np.ndarray],
    eval_start_ts: int,
) -> tuple[KConfig | None, dict[str, object]]:
    best: dict[str, object] | None = None
    tested = 0
    survivors = 0
    for h in j.HORIZONS:
        outer_end = int(np.searchsorted(decision, eval_start_ts - h * 60, side="left"))
        cut = int(outer_end * 0.8)
        cut_ts = int(decision[cut])
        train_end = int(np.searchsorted(decision, cut_ts - h * 60, side="left"))
        if train_end <= j.WARMUP or outer_end <= cut:
            continue
        y = gross_by_h[h]
        for block in BLOCKS:
            X = blocks[block]
            Xtr = X[j.WARMUP:train_end]
            ytr = y[j.WARMUP:train_end]
            Xv = X[cut:outer_end]
            for model_name, depth, lr, trees in MODEL_CONFIGS:
                base = KConfig(block, h, model_name, depth, lr, trees, j.QUANTILES[0])
                model = _model(base)
                model.fit(Xtr, ytr)
                train_pred = model.predict(Xtr)
                val_pred = model.predict(Xv)
                abs_train = np.abs(train_pred)
                for q in j.QUANTILES:
                    tested += 1
                    cfg = KConfig(block, h, model_name, depth, lr, trees, q)
                    gate = float(np.quantile(abs_train, q))
                    m12 = j._metrics(
                        decision, y, val_pred, gate, h, cut, outer_end,
                        int(decision[cut]), eval_start_ts, 12.0,
                    )
                    m15 = j._metrics(
                        decision, y, val_pred, gate, h, cut, outer_end,
                        int(decision[cut]), eval_start_ts, 15.0,
                    )
                    if not _survives(m12, m15):
                        continue
                    survivors += 1
                    cand = {"cfg": cfg, "m12": m12, "m15": m15}
                    if _better(cand, best):
                        best = cand
    if best is None:
        return None, {"tested": tested, "survivors": 0, "reason": "NO_CONFIGURATION"}
    cfg: KConfig = best["cfg"]  # type: ignore[assignment]
    return cfg, {
        "tested": tested,
        "survivors": survivors,
        "selected": _cfg_dict(cfg),
        "selected_inner_12bps": best["m12"],
        "selected_inner_15bps": best["m15"],
    }


def _outer(
    decision: np.ndarray,
    blocks: dict[str, np.ndarray],
    gross_by_h: dict[int, np.ndarray],
    cfg: KConfig,
    sd,
    ed,
) -> dict[str, object]:
    st = j._utc(sd)
    et = j._utc(ed + j.timedelta(days=1))
    tr_end = int(np.searchsorted(decision, st - cfg.horizon * 60, side="left"))
    eb = int(np.searchsorted(decision, st, side="left"))
    ee = int(np.searchsorted(decision, et - cfg.horizon * 60, side="left"))
    X = blocks[cfg.block]
    y = gross_by_h[cfg.horizon]
    model = _model(cfg)
    model.fit(X[j.WARMUP:tr_end], y[j.WARMUP:tr_end])
    train_pred = model.predict(X[j.WARMUP:tr_end])
    pred = model.predict(X[eb:ee])
    gate = float(np.quantile(np.abs(train_pred), cfg.quantile))
    return {
        "config": _cfg_dict(cfg),
        "absolute_prediction_gate": gate,
        "costs": {
            str(int(c)): j._metrics(
                decision, y, pred, gate, cfg.horizon, eb, ee, st, et, c, arrays=True
            )
            for c in j.COSTS
        },
    }


def _load_j_reference(path: Path, symbol: str) -> dict[str, float]:
    d = json.loads((path / f"{symbol}_PHASE0DJ.json").read_text())
    x = d["pooled_candidate_12bps"]
    return {
        "net_bps_trade": float(x["net_bps_trade"]),
        "total_net_bps": float(x["total_net_bps"]),
    }


def score_symbol(raw: Path, work: Path, j_dir: Path, symbol: str) -> dict[str, object]:
    state_open, mark, index, premium = j._load_state(raw, symbol)
    state_keep, decision, trade_price, f0 = j._load_trade_minute(
        work / f"{symbol}_DEV.csv", state_open
    )
    mark = mark[state_keep]
    index = index[state_keep]
    premium = premium[state_keep]
    jb = j._build_blocks(mark, index, premium, f0)
    blocks = {"K1": jb["J1"], "K2": jb["J2"]}
    if not all(np.all(np.isfinite(v[j.WARMUP:])) for v in blocks.values()):
        raise ValueError("non-finite Phase 0D-K features after warmup")
    gross = {h: j._gross(trade_price, h) for h in j.HORIZONS}
    folds: list[dict[str, object]] = []
    for i, (sd, ed) in enumerate(j.FOLDS, 1):
        cfg, inner = _select(decision, blocks, gross, j._utc(sd))
        rec: dict[str, object] = {
            "fold": i,
            "eval_start": sd.isoformat(),
            "eval_end": ed.isoformat(),
            "inner_selection": inner,
        }
        if cfg is None:
            rec["status"] = "NO_CONFIGURATION"
        else:
            rec["status"] = "SCORED"
            rec["outer"] = _outer(decision, blocks, gross, cfg, sd, ed)
        folds.append(rec)
    p12 = j._pool(folds, "outer", "12")
    p15 = j._pool(folds, "outer", "15")
    structural = bool(j._gate(p12, p15))
    jref = _load_j_reference(j_dir, symbol)
    incremental = bool(
        float(p12["net_bps_trade"]) > jref["net_bps_trade"]
        and float(p12["total_net_bps"]) > jref["total_net_bps"]
    )
    passed = bool(structural and incremental)
    return {
        "phase": "V2.3-PHASE0DK-NONLINEAR-FUTURES-STATE",
        "symbol": symbol,
        "development_only": True,
        "historical_holdout_opened": False,
        "boundary_state_minutes_trimmed": int(len(state_open) - np.count_nonzero(state_keep)),
        "phase0dj_ridge_reference_12bps": jref,
        "folds": folds,
        "pooled_candidate_12bps": p12,
        "pooled_candidate_15bps": p15,
        "candidate_structural_gate": structural,
        "incremental_vs_phase0dj_ridge": incremental,
        "development_pass": passed,
        "decision": "CANDIDATE_FREEZE_BEFORE_CONFIRMATION" if passed else "FAIL_KEEP_HOLDOUT_SEALED",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score frozen Phase 0D-K CUDA XGBoost audit")
    p.add_argument("--raw-dir", default="data/v23_phase0dj_state_raw")
    p.add_argument("--work-dir", default="evidence/v23/phase0dh_tf")
    p.add_argument("--phase0dj-dir", default="evidence/v23/phase0dj_score")
    p.add_argument("--output-dir", default="evidence/v23/phase0dk_nonlinear")
    p.add_argument("--preflight-only", action="store_true")
    a = p.parse_args(argv)

    pf = cuda_preflight()
    print(f"PHASE0DK_CUDA_PREFLIGHT=PASS device={pf['resolved_device']}", flush=True)
    if a.preflight_only:
        return 0

    raw = Path(a.raw_dir)
    work = Path(a.work_dir)
    j_dir = Path(a.phase0dj_dir)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for symbol in j.SYMBOLS:
        print(f"[{symbol}] Phase 0D-K nested CUDA XGBoost scoring", flush=True)
        r = score_symbol(raw, work, j_dir, symbol)
        results.append(r)
        (out / f"{symbol}_PHASE0DK.json").write_text(json.dumps(r, indent=2) + "\n")
        p12 = r["pooled_candidate_12bps"]
        print(
            f"[{symbol}] pass={r['development_pass']} trades={p12['trades']} "
            f"expectancy12={p12['net_bps_trade']:.6f} "
            f"incremental={r['incremental_vs_phase0dj_ridge']}",
            flush=True,
        )

    candidates = [r["symbol"] for r in results if r["development_pass"]]
    summary = {
        "phase": "V2.3-PHASE0DK-NONLINEAR-FUTURES-STATE",
        "development_only": True,
        "historical_holdout_opened": False,
        "cuda_preflight": pf,
        "candidate_targets": candidates,
        "decision": "CANDIDATE_FOUND_FREEZE_BEFORE_CONFIRMATION" if candidates else "FAIL_KEEP_HOLDOUT_SEALED",
    }
    (out / "V23_PHASE0DK_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("candidate_targets=" + (",".join(candidates) if candidates else "NONE"))
    print("decision=" + summary["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
