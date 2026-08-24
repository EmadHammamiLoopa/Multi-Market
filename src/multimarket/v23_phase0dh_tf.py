from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

SYMBOLS = ("BTCUSDT", "ETHUSDT")
DEV_START = date(2026, 5, 26)
DEV_END = date(2026, 8, 3)
HOLDOUT_START = date(2026, 8, 4)
HOLDOUT_END = date(2026, 8, 23)
HORIZON_SECONDS = 10
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)

FOLDS = (
    (date(2026, 6, 15), date(2026, 6, 24)),
    (date(2026, 6, 25), date(2026, 7, 4)),
    (date(2026, 7, 5), date(2026, 7, 14)),
    (date(2026, 7, 15), date(2026, 7, 24)),
    (date(2026, 7, 25), date(2026, 8, 3)),
)

T0_FEATURES = ("ret1", "ret3")
T1_FEATURES = (
    "ret1", "ret3", "qfi1", "cfi1", "qfi3", "qfi5", "qfi10",
    "cfi3", "cfi5", "cfi10", "log_qty1", "log_qty5",
    "log_count1", "log_count5", "vwap_pressure_bps", "buy_present", "sell_present",
)

ALIASES = {
    "price": {"price", "p"},
    "quantity": {"quantity", "qty", "q"},
    "trade_time": {"transact_time", "transaction_time", "trade_time", "time", "T"},
    "buyer_maker": {"is_buyer_maker", "buyer_is_maker", "m"},
}


def _days(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _find_col(fieldnames: list[str], aliases: set[str]) -> str:
    lower = {name.lower(): name for name in fieldnames}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    raise ValueError(f"missing required column; aliases={sorted(aliases)} fields={fieldnames}")


def _bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"true", "1", "t"}:
        return True
    if v in {"false", "0", "f"}:
        return False
    raise ValueError(f"invalid boolean {value!r}")


@dataclass
class Bucket:
    buy_qty: float = 0.0
    sell_qty: float = 0.0
    buy_count: int = 0
    sell_count: int = 0
    buy_notional: float = 0.0
    sell_notional: float = 0.0
    last_price: float | None = None

    def add(self, price: float, qty: float, buyer_maker: bool) -> None:
        if buyer_maker:
            self.sell_qty += qty
            self.sell_count += 1
            self.sell_notional += price * qty
        else:
            self.buy_qty += qty
            self.buy_count += 1
            self.buy_notional += price * qty
        self.last_price = price


def _archive(root: Path, symbol: str, d: date) -> Path:
    return root / "aggTrades" / symbol / f"{symbol}-aggTrades-{d.isoformat()}.zip"


def iter_day_trades(path: Path):
    with zipfile.ZipFile(path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected one csv member in {path}, got {members}")
        with zf.open(members[0], "r") as raw:
            import io
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            reader = csv.DictReader(text)
            if not reader.fieldnames:
                raise ValueError(f"missing CSV header in {path}")
            price_col = _find_col(reader.fieldnames, ALIASES["price"])
            qty_col = _find_col(reader.fieldnames, ALIASES["quantity"])
            time_col = _find_col(reader.fieldnames, ALIASES["trade_time"])
            maker_col = _find_col(reader.fieldnames, ALIASES["buyer_maker"])
            prev_ms = None
            for row in reader:
                ts_ms = int(row[time_col])
                if prev_ms is not None and ts_ms < prev_ms:
                    raise ValueError(f"non-monotonic trade time in {path}: {ts_ms} < {prev_ms}")
                prev_ms = ts_ms
                yield ts_ms, float(row[price_col]), float(row[qty_col]), _bool(row[maker_col])


def _imb(buy: float, sell: float) -> float:
    total = buy + sell
    return (buy - sell) / total if total > 0 else 0.0


def build_symbol_dataset(raw_root: Path, symbol: str, output: Path) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    windows: deque[dict[str, float]] = deque(maxlen=10)
    price_history: deque[float] = deque(maxlen=31)
    rows_written = 0
    first_ts = None
    last_ts = None

    with output.open("w", newline="", encoding="utf-8") as handle:
        fields = ["timestamp", "date", "price", *T1_FEATURES, "label10"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        last_known_price: float | None = None
        pending: deque[dict[str, object]] = deque()

        for d in _days(DEV_START, DEV_END):
            path = _archive(raw_root, symbol, d)
            if not path.exists():
                raise FileNotFoundError(path)
            buckets: dict[int, Bucket] = {}
            for ts_ms, price, qty, maker in iter_day_trades(path):
                sec = ts_ms // 1000
                b = buckets.setdefault(sec, Bucket())
                b.add(price, qty, maker)

            day_start = int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
            for sec in range(day_start, day_start + 86400):
                b = buckets.get(sec, Bucket())
                if b.last_price is not None:
                    last_known_price = b.last_price
                if last_known_price is None:
                    continue

                current = {
                    "buy_qty": b.buy_qty, "sell_qty": b.sell_qty,
                    "buy_count": float(b.buy_count), "sell_count": float(b.sell_count),
                }
                windows.append(current)
                price_history.append(last_known_price)
                if len(price_history) < 4:
                    continue

                def sums(n: int, key: str) -> float:
                    return sum(x[key] for x in list(windows)[-n:])

                def win_imb(n: int, buy_key: str, sell_key: str) -> float:
                    return _imb(sums(n, buy_key), sums(n, sell_key))

                buy_vwap = b.buy_notional / b.buy_qty if b.buy_qty > 0 else 0.0
                sell_vwap = b.sell_notional / b.sell_qty if b.sell_qty > 0 else 0.0
                both = b.buy_qty > 0 and b.sell_qty > 0
                pressure = ((buy_vwap - sell_vwap) / last_known_price * 10000.0) if both else 0.0
                ret1 = math.log(price_history[-1] / price_history[-2]) * 10000.0
                ret3 = math.log(price_history[-1] / price_history[-4]) * 10000.0

                features = {
                    "ret1": ret1,
                    "ret3": ret3,
                    "qfi1": _imb(b.buy_qty, b.sell_qty),
                    "cfi1": _imb(float(b.buy_count), float(b.sell_count)),
                    "qfi3": win_imb(3, "buy_qty", "sell_qty"),
                    "qfi5": win_imb(5, "buy_qty", "sell_qty"),
                    "qfi10": win_imb(10, "buy_qty", "sell_qty"),
                    "cfi3": win_imb(3, "buy_count", "sell_count"),
                    "cfi5": win_imb(5, "buy_count", "sell_count"),
                    "cfi10": win_imb(10, "buy_count", "sell_count"),
                    "log_qty1": math.log1p(b.buy_qty + b.sell_qty),
                    "log_qty5": math.log1p(sums(5, "buy_qty") + sums(5, "sell_qty")),
                    "log_count1": math.log1p(b.buy_count + b.sell_count),
                    "log_count5": math.log1p(sums(5, "buy_count") + sums(5, "sell_count")),
                    "vwap_pressure_bps": pressure,
                    "buy_present": 1.0 if b.buy_qty > 0 else 0.0,
                    "sell_present": 1.0 if b.sell_qty > 0 else 0.0,
                }
                rec = {"timestamp": sec, "date": d.isoformat(), "price": last_known_price, **features}
                pending.append(rec)

                while pending and int(pending[0]["timestamp"]) <= sec - HORIZON_SECONDS:
                    old = pending.popleft()
                    if int(old["timestamp"]) != sec - HORIZON_SECONDS:
                        continue
                    old["label10"] = math.log(last_known_price / float(old["price"])) * 10000.0
                    writer.writerow(old)
                    rows_written += 1
                    first_ts = first_ts or int(old["timestamp"])
                    last_ts = int(old["timestamp"])

    return {"symbol": symbol, "rows": rows_written, "first_timestamp": first_ts, "last_timestamp": last_ts}


def _load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load only numeric scoring columns from the development CSV.

    The date and raw price columns are deliberately skipped. This is an I/O and
    memory optimization only; the frozen rows, feature values, labels, folds,
    models, and promotion gates are unchanged.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    positions = {name: idx for idx, name in enumerate(header)}
    required = ("timestamp", *T1_FEATURES, "label10")
    missing = [name for name in required if name not in positions]
    if missing:
        raise ValueError(f"development dataset missing required columns: {missing}")
    usecols = tuple(positions[name] for name in required)
    matrix = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=usecols,
        dtype=np.float64,
        ndmin=2,
    )
    timestamps = matrix[:, 0].astype(np.int64, copy=False)
    X = matrix[:, 1:1 + len(T1_FEATURES)]
    y = matrix[:, -1]
    return timestamps, X, y


def _select_alpha(X: np.ndarray, y: np.ndarray) -> float:
    cut = max(1, int(len(y) * 0.8))
    Xtr, Xv = X[:cut], X[cut:]
    ytr, yv = y[:cut], y[cut:]
    best_alpha, best_mse = RIDGE_ALPHAS[0], float("inf")
    for alpha in RIDGE_ALPHAS:
        scaler = StandardScaler().fit(Xtr)
        model = Ridge(alpha=alpha).fit(scaler.transform(Xtr), ytr)
        pred = model.predict(scaler.transform(Xv))
        mse = float(np.mean((pred - yv) ** 2))
        if mse < best_mse:
            best_alpha, best_mse = alpha, mse
    return best_alpha


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rho = spearmanr(y, pred).statistic
    return {
        "r2": float(r2_score(y, pred)),
        "spearman": float(rho if np.isfinite(rho) else 0.0),
        "directional_accuracy": float(np.mean(np.sign(y) == np.sign(pred))),
    }


def score_symbol(dataset: Path, symbol: str) -> dict[str, object]:
    timestamps, X_all, y_all = _load_dataset(dataset)

    fold_results = []
    pooled = {"T0": [[], []], "T1": [[], []], "T2": [[], []]}
    positive_delta = {"T1": 0, "T2": 0}

    for idx, (eval_start_d, eval_end_d) in enumerate(FOLDS, 1):
        eval_start = int(datetime(eval_start_d.year, eval_start_d.month, eval_start_d.day, tzinfo=timezone.utc).timestamp())
        eval_end_exclusive = int((datetime(eval_end_d.year, eval_end_d.month, eval_end_d.day, tzinfo=timezone.utc) + timedelta(days=1)).timestamp())

        # Frozen rules: train label endpoint must be strictly before eval start;
        # eval label endpoint must be strictly before eval end-exclusive.
        train_end = int(np.searchsorted(timestamps, eval_start - HORIZON_SECONDS, side="left"))
        eval_begin = int(np.searchsorted(timestamps, eval_start, side="left"))
        eval_end = int(np.searchsorted(timestamps, eval_end_exclusive - HORIZON_SECONDS, side="left"))
        if train_end < 10000 or eval_end - eval_begin < 1000:
            continue

        ytr = y_all[:train_end]
        yev = y_all[eval_begin:eval_end]
        fold_models = {}
        fold_metrics = {}

        for name, n_features in (("T0", len(T0_FEATURES)), ("T1", len(T1_FEATURES))):
            Xtr = X_all[:train_end, :n_features]
            Xev = X_all[eval_begin:eval_end, :n_features]
            alpha = _select_alpha(Xtr, ytr)
            scaler = StandardScaler().fit(Xtr)
            model = Ridge(alpha=alpha).fit(scaler.transform(Xtr), ytr)
            pred = model.predict(scaler.transform(Xev))
            fold_models[name] = {"alpha": alpha}
            fold_metrics[name] = _metrics(yev, pred)
            pooled[name][0].append(yev)
            pooled[name][1].append(pred)

        model2 = HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
            min_samples_leaf=100, l2_regularization=1.0, random_state=0,
        ).fit(X_all[:train_end], ytr)
        pred2 = model2.predict(X_all[eval_begin:eval_end])
        fold_metrics["T2"] = _metrics(yev, pred2)
        pooled["T2"][0].append(yev)
        pooled["T2"][1].append(pred2)

        base_r2 = fold_metrics["T0"]["r2"]
        for candidate in ("T1", "T2"):
            delta = fold_metrics[candidate]["r2"] - base_r2
            fold_metrics[candidate]["delta_r2_vs_t0"] = delta
            if delta > 0:
                positive_delta[candidate] += 1

        fold_results.append({
            "fold": idx,
            "eval_start": eval_start_d.isoformat(),
            "eval_end": eval_end_d.isoformat(),
            "train_rows": train_end,
            "eval_rows": eval_end - eval_begin,
            "models": fold_models,
            "metrics": fold_metrics,
        })

    pooled_metrics = {}
    for name in ("T0", "T1", "T2"):
        if not pooled[name][0]:
            continue
        yy = np.concatenate(pooled[name][0])
        pp = np.concatenate(pooled[name][1])
        pooled_metrics[name] = _metrics(yy, pp)
    if "T0" in pooled_metrics:
        for candidate in ("T1", "T2"):
            if candidate in pooled_metrics:
                pooled_metrics[candidate]["delta_r2_vs_t0"] = pooled_metrics[candidate]["r2"] - pooled_metrics["T0"]["r2"]
                pooled_metrics[candidate]["positive_delta_folds"] = positive_delta[candidate]
                pooled_metrics[candidate]["scored_folds"] = len(fold_results)
                pooled_metrics[candidate]["pass"] = bool(
                    pooled_metrics[candidate]["delta_r2_vs_t0"] > 0
                    and pooled_metrics[candidate]["spearman"] > 0
                    and len(fold_results) >= 4
                    and positive_delta[candidate] >= 3
                    and pooled_metrics[candidate]["directional_accuracy"] > 0.50
                )

    candidate = None
    if pooled_metrics.get("T1", {}).get("pass"):
        candidate = "T1"
    elif pooled_metrics.get("T2", {}).get("pass"):
        candidate = "T2"

    return {
        "phase": "V2.3-PHASE0DH-TF-HISTORICAL",
        "symbol": symbol,
        "holdout_opened": False,
        "folds": fold_results,
        "pooled": pooled_metrics,
        "signal_candidate": candidate,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build and score frozen Phase 0D-H-TF development data")
    p.add_argument("--raw-dir", default="data/v23_phase0dh_tf_raw")
    p.add_argument("--work-dir", default="evidence/v23/phase0dh_tf")
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--score-only", action="store_true")
    args = p.parse_args(argv)
    if args.build_only and args.score_only:
        raise SystemExit("choose only one of --build-only or --score-only")

    raw = Path(args.raw_dir)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((raw / "ACQUISITION_MANIFEST.json").read_text())
    if not manifest.get("official_acquisition_complete"):
        raise SystemExit("acquisition manifest is not complete")

    if not args.score_only:
        builds = []
        for symbol in SYMBOLS:
            dataset = work / f"{symbol}_DEV.csv"
            print(f"[{symbol}] building causal development dataset", flush=True)
            builds.append(build_symbol_dataset(raw, symbol, dataset))
            print(f"[{symbol}] rows={builds[-1]['rows']}", flush=True)
        (work / "BUILD_SUMMARY.json").write_text(json.dumps(builds, indent=2) + "\n")
        if args.build_only:
            print("PHASE0DH_TF_BUILD=PASS")
            return 0
    else:
        if not (work / "BUILD_SUMMARY.json").exists():
            raise SystemExit("--score-only requires an existing BUILD_SUMMARY.json")
        for symbol in SYMBOLS:
            dataset = work / f"{symbol}_DEV.csv"
            if not dataset.exists() or dataset.stat().st_size == 0:
                raise SystemExit(f"--score-only missing development dataset: {dataset}")

    results = []
    for symbol in SYMBOLS:
        print(f"[{symbol}] scoring T0/T1/T2 on development folds", flush=True)
        result = score_symbol(work / f"{symbol}_DEV.csv", symbol)
        results.append(result)
        (work / f"{symbol}_PHASE0DH_TF.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"[{symbol}] signal_candidate={result['signal_candidate']}", flush=True)

    candidates = [r["symbol"] for r in results if r["signal_candidate"]]
    summary = {
        "phase": "V2.3-PHASE0DH-TF-HISTORICAL",
        "development_only": True,
        "historical_holdout_opened": False,
        "candidate_targets": candidates,
        "decision": "CANDIDATE_FOUND_FREEZE_BEFORE_HOLDOUT" if candidates else "FAIL_KEEP_HOLDOUT_SEALED",
    }
    (work / "V23_PHASE0DH_TF_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"candidate_targets={','.join(candidates) if candidates else 'NONE'}")
    print(f"decision={summary['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
