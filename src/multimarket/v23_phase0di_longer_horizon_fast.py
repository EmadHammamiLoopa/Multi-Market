from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import v23_phase0di_longer_horizon as ref


def _scale_once(
    Xtr: np.ndarray,
    Xev: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the frozen StandardScaler once and transform train/eval once.

    The reference implementation repeated these exact operations for every
    Ridge alpha. Reusing the identical scaled matrices changes execution cost,
    not the experiment or numerical inputs supplied to Ridge.
    """
    scaler = StandardScaler().fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xev)


def _fit_scaled(
    Ztr: np.ndarray,
    ytr: np.ndarray,
    Zev: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    model = Ridge(alpha=alpha).fit(Ztr, ytr)
    return model.predict(Ztr), model.predict(Zev)


def _select_inner_fast(
    ts: np.ndarray,
    X: np.ndarray,
    gross_by_h: dict[int, np.ndarray],
    outer_train_end_maxh: int,
) -> tuple[ref.Config | None, dict[str, object]]:
    cut = int(outer_train_end_maxh * 0.8)
    best: dict[str, object] | None = None
    survivors = 0
    tested = 0
    start_ts = int(ts[cut])
    end_ts = int(ts[outer_train_end_maxh - 1]) + 1

    for h in ref.HORIZONS:
        gross = gross_by_h[h]
        train_end = min(cut - h, cut)
        val_end = outer_train_end_maxh - h
        if train_end <= ref.MAX_TRAIL or val_end <= cut:
            continue

        Xtr = X[ref.MAX_TRAIL:train_end]
        ytr = gross[ref.MAX_TRAIL:train_end]
        Xv = X[cut:val_end]

        # Major optimization: StandardScaler and matrix transforms are
        # independent of alpha, so compute them once per (fold, horizon).
        Ztr, Zv = _scale_once(Xtr, Xv)

        for alpha in ref.ALPHAS:
            train_pred, val_pred = _fit_scaled(Ztr, ytr, Zv, alpha)
            abs_train = np.abs(train_pred)

            for q in ref.GATE_QUANTILES:
                tested += 1
                gate = float(np.quantile(abs_train, q))
                m12 = ref._metrics(
                    ts, gross, val_pred, gate, h,
                    cut, val_end, start_ts, end_ts, 12.0,
                )
                m15 = ref._metrics(
                    ts, gross, val_pred, gate, h,
                    cut, val_end, start_ts, end_ts, 15.0,
                )
                if not ref._survives(m12, m15):
                    continue

                survivors += 1
                cand = {"cfg": ref.Config(h, alpha, q), "m12": m12, "m15": m15}
                if ref._better(cand, best):
                    best = cand

            # Release large prediction vectors before the next alpha.
            del train_pred, val_pred, abs_train

        del Ztr, Zv

    if best is None:
        return None, {"tested": tested, "survivors": 0, "reason": "NO_CONFIGURATION"}

    cfg: ref.Config = best["cfg"]  # type: ignore[assignment]
    return cfg, {
        "tested": tested,
        "survivors": survivors,
        "selected": {
            "horizon": cfg.horizon,
            "alpha": cfg.alpha,
            "gate_quantile": cfg.gate_quantile,
        },
        "selected_inner_12bps": best["m12"],
        "selected_inner_15bps": best["m15"],
    }


def _outer_fast(
    ts: np.ndarray,
    X: np.ndarray,
    gross_by_h: dict[int, np.ndarray],
    cfg: ref.Config,
    start_d: date,
    end_d: date,
) -> dict[str, object]:
    start_ts = ref._utc(start_d)
    end_ts_excl = ref._utc(end_d + timedelta(days=1))
    train_end = int(np.searchsorted(ts, start_ts - cfg.horizon, side="left"))
    eval_begin = int(np.searchsorted(ts, start_ts, side="left"))
    eval_end = int(np.searchsorted(ts, end_ts_excl - cfg.horizon, side="left"))

    gross = gross_by_h[cfg.horizon]
    Xtr = X[ref.MAX_TRAIL:train_end]
    ytr = gross[ref.MAX_TRAIL:train_end]
    Xev = X[eval_begin:eval_end]

    Ztr, Zev = _scale_once(Xtr, Xev)
    train_pred, pred = _fit_scaled(Ztr, ytr, Zev, cfg.alpha)
    gate = float(np.quantile(np.abs(train_pred), cfg.gate_quantile))

    costs = {
        str(int(c)): ref._metrics(
            ts,
            gross,
            pred,
            gate,
            cfg.horizon,
            eval_begin,
            eval_end,
            start_ts,
            end_ts_excl,
            c,
            include_arrays=True,
        )
        for c in ref.COSTS
    }
    return {
        "config": {
            "horizon": cfg.horizon,
            "alpha": cfg.alpha,
            "gate_quantile": cfg.gate_quantile,
        },
        "absolute_prediction_gate": gate,
        "costs": costs,
    }


def score_symbol_fast(path: Path, symbol: str) -> dict[str, object]:
    ts, price, Xbase, fmap = ref._load_base(path)
    print(f"[{symbol}] deriving causal 30/60/120/300s features", flush=True)
    X = ref._derive_long_features(price, Xbase, fmap)
    if not np.all(np.isfinite(X[ref.MAX_TRAIL:])):
        raise ValueError("non-finite longer-horizon features after 300s warmup")

    # Gross-return labels are identical across all folds; compute once per
    # horizon rather than rebuilding these 6M-row arrays repeatedly.
    gross_by_h = {h: ref._gross_return(price, h) for h in ref.HORIZONS}

    folds: list[dict[str, object]] = []
    for i, (sd, ed) in enumerate(ref.FOLDS, 1):
        outer_train_end_maxh = int(
            np.searchsorted(ts, ref._utc(sd) - max(ref.HORIZONS), side="left")
        )
        cfg, inner = _select_inner_fast(ts, X, gross_by_h, outer_train_end_maxh)
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
            rec["outer"] = _outer_fast(ts, X, gross_by_h, cfg, sd, ed)
        folds.append(rec)

    p12 = ref._pool(folds, "12")
    p15 = ref._pool(folds, "15")
    passed = ref._passes(p12, p15, folds)
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


def _score_and_write(work: Path, out: Path, symbol: str) -> dict[str, object]:
    print(f"[{symbol}] Phase 0D-I nested scoring", flush=True)
    result = score_symbol_fast(work / f"{symbol}_DEV.csv", symbol)
    (out / f"{symbol}_PHASE0DI.json").write_text(json.dumps(result, indent=2) + "\n")
    p12 = result["pooled_12bps"]
    print(
        f"[{symbol}] pass={result['development_pass']} trades={p12['trades']} "
        f"expectancy12={p12['net_bps_trade']:.6f}",
        flush=True,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Fast, result-equivalent execution of frozen V2.3 Phase 0D-I"
    )
    p.add_argument("--work-dir", default="evidence/v23/phase0dh_tf")
    p.add_argument("--output-dir", default="evidence/v23/phase0di_longer_horizon")
    p.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Concurrent symbols; keep at 2 on a 32 GB machine",
    )
    args = p.parse_args(argv)

    work = Path(args.work_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    workers = max(1, min(int(args.workers), len(ref.SYMBOLS)))
    cpu_count = os.cpu_count() or 2
    blas_threads = max(1, cpu_count // workers)
    print(
        f"execution_workers={workers} cpu_count={cpu_count} "
        f"blas_threads_per_worker={blas_threads}",
        flush=True,
    )

    result_map: dict[str, dict[str, object]] = {}
    # Limit each independent symbol's BLAS work so two symbols together can
    # use the CPU without severe nested-thread oversubscription.
    with threadpool_limits(limits=blas_threads):
        if workers == 1:
            for symbol in ref.SYMBOLS:
                result_map[symbol] = _score_and_write(work, out, symbol)
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="phase0di") as ex:
                futures = {
                    ex.submit(_score_and_write, work, out, symbol): symbol
                    for symbol in ref.SYMBOLS
                }
                for future in as_completed(futures):
                    symbol = futures[future]
                    result_map[symbol] = future.result()

    results = [result_map[s] for s in ref.SYMBOLS]
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
