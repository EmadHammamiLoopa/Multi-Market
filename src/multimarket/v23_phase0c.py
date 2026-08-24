from __future__ import annotations

import argparse
import json
from bisect import bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime
from math import cos, log, pi, sin, sqrt
from pathlib import Path
from statistics import fmean
from typing import Collection, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_ohlc_csv
from .models import MarketBar
from .v21_common import hard_eligible_indices, load_peer_markets
from .v21_features import PeerMarket
from .v23_phase0 import (
    JUMP_SIGMA_MULTIPLIER,
    MIN_TRAIN_ROWS,
    PRIMARY_HORIZON,
    RESERVED_WINDOWS,
    SECONDARY_HORIZON,
    _contiguous,
    _corr,
    _fold_ranges,
    _forward_return,
    _is_reserved,
    _one_bar_log_return_bps,
    _rank,
    _sigma48_series,
)


OWN_RETURN_HORIZONS = (1, 3, 6, 12, 24)
OWN_RV_WINDOWS = (6, 24, 72)
PEER_RETURN_HORIZONS = (1, 6, 24)
VOL_PERCENTILE_LOOKBACK = 288
MAX_PEER_STALENESS_SECONDS = 900
RIDGE_ALPHA = 10.0

HGBR_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 0,
}

EXPECTED_SENSORS: dict[str, tuple[str, ...]] = {
    "EURUSD": ("HYG", "TLT", "UUP"),
    "XAUUSD": ("HYG", "TLT", "UUP"),
    "BTCUSD": ("ETHUSD", "HYG", "QQQ"),
    "ETHUSD": ("BTCUSD", "HYG", "QQQ"),
    "QQQ": ("HYG", "TLT", "UUP", "XLP"),
}

REPRESENTATIONS = {
    "C0": ("c0_features", "ridge"),
    "C1": ("c1_features", "ridge"),
    "C2": ("c2_features", "ridge"),
    "C3": ("c2_features", "hgbr"),
}


@dataclass(frozen=True, slots=True)
class Phase0CRow:
    timestamp: datetime
    label_end_timestamp: datetime
    c0_features: tuple[float, ...]
    c1_features: tuple[float, ...]
    c2_features: tuple[float, ...]
    forward_1_bps: float
    forward_6_bps: float
    jump_state: int


@dataclass(frozen=True, slots=True)
class FitMetrics:
    rows: int
    r2: float
    mse: float
    rmse: float
    mae: float
    spearman: float | None
    pearson: float | None
    sign_accuracy: float


@dataclass(frozen=True, slots=True)
class PreparedPeer:
    bars: tuple[MarketBar, ...]
    timestamps: tuple[datetime, ...]
    eligible: frozenset[int]


def _prepare_peer(peer: PeerMarket) -> PreparedPeer:
    return PreparedPeer(
        bars=peer.bars,
        timestamps=tuple(bar.timestamp for bar in peer.bars),
        eligible=peer.eligible_indices,
    )


def _validate_sensor_set(symbol: str, peers: dict[str, PeerMarket]) -> None:
    target = symbol.upper()
    if target not in EXPECTED_SENSORS:
        raise ValueError(f"unsupported Phase 0C target: {target}")
    expected = set(EXPECTED_SENSORS[target])
    actual = set(peers)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "Phase 0C peer set must exactly match preregistration "
            f"for {target}; missing={missing}, extra={extra}"
        )


def _log_return_bps(
    bars: Sequence[MarketBar],
    index: int,
    horizon: int,
    eligible: Collection[int],
) -> float | None:
    start = index - horizon
    if start < 0:
        return None
    if any(i not in eligible for i in range(start, index + 1)):
        return None
    if not _contiguous(bars, start, index):
        return None
    return log(bars[index].close / bars[start].close) * 10_000.0


def _realized_vol_bps(
    bars: Sequence[MarketBar],
    index: int,
    window: int,
    eligible: Collection[int],
) -> float | None:
    start = index - window
    if start < 0:
        return None
    if any(i not in eligible for i in range(start, index + 1)):
        return None
    if not _contiguous(bars, start, index):
        return None
    returns = [_one_bar_log_return_bps(bars, i) for i in range(start + 1, index + 1)]
    return sqrt(fmean(value * value for value in returns))


def _zprice_24(
    bars: Sequence[MarketBar],
    index: int,
    eligible: Collection[int],
) -> float | None:
    start = index - 23
    if start < 0:
        return None
    if any(i not in eligible for i in range(start, index + 1)):
        return None
    if not _contiguous(bars, start, index):
        return None
    values = [log(bars[i].close) for i in range(start, index + 1)]
    mean = fmean(values)
    sigma = sqrt(fmean((value - mean) ** 2 for value in values))
    if sigma <= 0.0:
        return None
    return (values[-1] - mean) / sigma


def _target_own_features(
    bars: Sequence[MarketBar],
    index: int,
    eligible: Collection[int],
) -> tuple[tuple[float, ...], float, float] | None:
    returns: list[float] = []
    for horizon in OWN_RETURN_HORIZONS:
        value = _log_return_bps(bars, index, horizon, eligible)
        if value is None:
            return None
        returns.append(value)

    vols: list[float] = []
    for window in OWN_RV_WINDOWS:
        value = _realized_vol_bps(bars, index, window, eligible)
        if value is None:
            return None
        vols.append(value)

    zprice = _zprice_24(bars, index, eligible)
    if zprice is None:
        return None
    current_range = log(bars[index].high / bars[index].low) * 10_000.0

    own = tuple(returns + vols + [current_range, zprice])
    r24 = returns[OWN_RETURN_HORIZONS.index(24)]
    rv24 = vols[OWN_RV_WINDOWS.index(24)]
    return own, r24, rv24


def _find_latest_eligible_asof(peer: PreparedPeer, decision_timestamp: datetime) -> int | None:
    index = bisect_right(peer.timestamps, decision_timestamp) - 1
    while index >= 0 and index not in peer.eligible:
        index -= 1
    if index < 0:
        return None
    age = (decision_timestamp - peer.bars[index].timestamp).total_seconds()
    if age < 0 or age > MAX_PEER_STALENESS_SECONDS:
        return None
    return index


def _peer_packet(peer: PreparedPeer, decision_timestamp: datetime) -> tuple[float, ...]:
    asof = _find_latest_eligible_asof(peer, decision_timestamp)
    numeric: list[float] = []
    available: list[float] = []
    if asof is None:
        return tuple([0.0] * len(PEER_RETURN_HORIZONS) * 2)

    for horizon in PEER_RETURN_HORIZONS:
        value = _log_return_bps(peer.bars, asof, horizon, peer.eligible)
        if value is None:
            numeric.append(0.0)
            available.append(0.0)
        else:
            numeric.append(value)
            available.append(1.0)
    return tuple(numeric + available)


def _regime_features(
    bars: Sequence[MarketBar],
    index: int,
    eligible: Collection[int],
    sigma48: Sequence[float | None],
    rv24_series: Sequence[float | None],
    *,
    r24: float,
    rv24: float,
) -> tuple[tuple[float, ...], int] | None:
    sigma = sigma48[index]
    if sigma is None or sigma <= 0.0:
        return None
    if index < 6 or any(i not in eligible for i in range(index - 6, index + 1)):
        return None
    if not _contiguous(bars, index - 6, index):
        return None

    prior_rv24 = [
        value
        for value in rv24_series[max(0, index - VOL_PERCENTILE_LOOKBACK) : index]
        if value is not None
    ]
    vol_pct = (
        sum(float(value) <= rv24 for value in prior_rv24) / len(prior_rv24)
        if prior_rv24
        else 0.5
    )

    recent_returns = [_one_bar_log_return_bps(bars, i) for i in range(index - 5, index + 1)]
    recent_jump_ratio = max(abs(value) for value in recent_returns) / sigma
    current = _one_bar_log_return_bps(bars, index)
    jump = int(abs(current) > JUMP_SIGMA_MULTIPLIER * sigma)

    ts = bars[index].timestamp
    minute = ts.hour * 60 + ts.minute
    weekday = ts.weekday()
    trend_strength = abs(r24) / max(rv24 * sqrt(24.0), 1e-12)

    regime = (
        sin(2.0 * pi * minute / 1440.0),
        cos(2.0 * pi * minute / 1440.0),
        sin(2.0 * pi * weekday / 7.0),
        cos(2.0 * pi * weekday / 7.0),
        vol_pct,
        trend_strength,
        recent_jump_ratio,
    )
    return regime, jump


def build_phase0c_rows(
    bars: Sequence[MarketBar],
    *,
    symbol: str,
    peers: dict[str, PeerMarket],
) -> list[Phase0CRow]:
    target = symbol.upper()
    _validate_sensor_set(target, peers)
    eligible = hard_eligible_indices(bars, target)
    sigma48 = _sigma48_series(bars, eligible)
    rv24_series = tuple(_realized_vol_bps(bars, i, 24, eligible) for i in range(len(bars)))
    prepared_peers = {name: _prepare_peer(peer) for name, peer in peers.items()}

    result: list[Phase0CRow] = []
    for index, bar in enumerate(bars):
        if index not in eligible or _is_reserved(bar.timestamp):
            continue

        own_result = _target_own_features(bars, index, eligible)
        if own_result is None:
            continue
        own, r24, rv24 = own_result

        regime_result = _regime_features(
            bars,
            index,
            eligible,
            sigma48,
            rv24_series,
            r24=r24,
            rv24=rv24,
        )
        if regime_result is None:
            continue
        regime, jump = regime_result

        y1 = _forward_return(bars, index, SECONDARY_HORIZON, eligible)
        y6 = _forward_return(bars, index, PRIMARY_HORIZON, eligible)
        if y1 is None or y6 is None:
            continue

        peer_values: list[float] = []
        for name in EXPECTED_SENSORS[target]:
            peer_values.extend(_peer_packet(prepared_peers[name], bar.timestamp))

        linked = tuple(peer_values)
        result.append(
            Phase0CRow(
                timestamp=bar.timestamp,
                label_end_timestamp=bars[index + PRIMARY_HORIZON].timestamp,
                c0_features=own,
                c1_features=own + linked,
                c2_features=own + linked + regime,
                forward_1_bps=y1,
                forward_6_bps=y6,
                jump_state=jump,
            )
        )
    return result


def _metrics(y: Sequence[float], pred: Sequence[float]) -> FitMetrics:
    mse = float(mean_squared_error(y, pred))
    return FitMetrics(
        rows=len(y),
        r2=float(r2_score(y, pred)),
        mse=mse,
        rmse=sqrt(mse),
        mae=float(mean_absolute_error(y, pred)),
        spearman=_corr(_rank(pred), _rank(y)),
        pearson=_corr(pred, y),
        sign_accuracy=float(np.mean(np.sign(pred) == np.sign(y))),
    )


def _make_model(kind: str):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    if kind == "hgbr":
        return HistGradientBoostingRegressor(**HGBR_PARAMS)
    raise ValueError(kind)


def _incremental_payload(reps: dict[str, object]) -> dict[str, float]:
    result: dict[str, float] = {}
    c0 = reps["C0"]
    assert isinstance(c0, dict)
    base_r2 = float(c0["r2"])
    for candidate in ("C1", "C2", "C3"):
        payload = reps[candidate]
        assert isinstance(payload, dict)
        result[f"{candidate}_minus_C0"] = float(payload["r2"]) - base_r2
    c2 = reps["C2"]
    c3 = reps["C3"]
    assert isinstance(c2, dict) and isinstance(c3, dict)
    result["C3_minus_C2"] = float(c3["r2"]) - float(c2["r2"])
    return result


def evaluate_rows(rows: Sequence[Phase0CRow], *, symbol: str) -> dict[str, object]:
    if not rows:
        raise ValueError("Phase 0C has no eligible rows")

    folds: list[dict[str, object]] = []
    pooled: dict[tuple[str, int], dict[str, list[float]]] = {}

    for fold_number, (start, end) in enumerate(_fold_ranges(len(rows)), start=1):
        eval_rows = rows[start:end]
        if not eval_rows:
            continue
        eval_start = eval_rows[0].timestamp
        train_rows = [row for row in rows[:start] if row.label_end_timestamp < eval_start]
        if len(train_rows) < MIN_TRAIN_ROWS:
            folds.append(
                {
                    "fold": fold_number,
                    "status": "SKIP_MIN_TRAIN_ROWS",
                    "train_rows": len(train_rows),
                    "eval_rows": len(eval_rows),
                }
            )
            continue

        fold_payload: dict[str, object] = {
            "fold": fold_number,
            "status": "SCORED",
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "eval_start": eval_start.isoformat(),
            "horizons": {},
        }

        for horizon, y_field in ((1, "forward_1_bps"), (6, "forward_6_bps")):
            y_train = np.asarray([getattr(row, y_field) for row in train_rows], dtype=float)
            y_eval = np.asarray([getattr(row, y_field) for row in eval_rows], dtype=float)
            reps: dict[str, object] = {}

            for rep_name, (feature_field, model_kind) in REPRESENTATIONS.items():
                X_train = np.asarray([getattr(row, feature_field) for row in train_rows], dtype=float)
                X_eval = np.asarray([getattr(row, feature_field) for row in eval_rows], dtype=float)
                model = _make_model(model_kind)
                model.fit(X_train, y_train)
                pred = model.predict(X_eval)
                reps[rep_name] = asdict(_metrics(y_eval, pred))

                slot = pooled.setdefault((rep_name, horizon), {"y": [], "pred": [], "jump": []})
                slot["y"].extend(float(value) for value in y_eval)
                slot["pred"].extend(float(value) for value in pred)
                slot["jump"].extend(row.jump_state for row in eval_rows)

            fold_payload["horizons"][str(horizon)] = {
                "representations": reps,
                "incremental_r2": _incremental_payload(reps),
            }
        folds.append(fold_payload)

    pooled_payload: dict[str, object] = {}
    for horizon in (1, 6):
        reps: dict[str, object] = {}
        for rep_name in REPRESENTATIONS:
            slot = pooled.get((rep_name, horizon))
            if not slot:
                continue
            all_metrics = _metrics(slot["y"], slot["pred"])
            non_jump_idx = [i for i, flag in enumerate(slot["jump"]) if not flag]
            non_jump = (
                _metrics(
                    [slot["y"][i] for i in non_jump_idx],
                    [slot["pred"][i] for i in non_jump_idx],
                )
                if len(non_jump_idx) >= 2
                else None
            )
            reps[rep_name] = {
                "all": asdict(all_metrics),
                "non_jump": asdict(non_jump) if non_jump is not None else None,
            }

        incremental_all: dict[str, float] = {}
        incremental_non_jump: dict[str, float | None] = {}
        if "C0" in reps:
            base_all = reps["C0"]["all"]
            base_nj = reps["C0"]["non_jump"]
            for candidate in ("C1", "C2", "C3"):
                if candidate not in reps:
                    continue
                incremental_all[f"{candidate}_minus_C0"] = (
                    reps[candidate]["all"]["r2"] - base_all["r2"]
                )
                candidate_nj = reps[candidate]["non_jump"]
                incremental_non_jump[f"{candidate}_minus_C0"] = (
                    candidate_nj["r2"] - base_nj["r2"]
                    if candidate_nj is not None and base_nj is not None
                    else None
                )
            if "C2" in reps and "C3" in reps:
                incremental_all["C3_minus_C2"] = reps["C3"]["all"]["r2"] - reps["C2"]["all"]["r2"]
                c2_nj = reps["C2"]["non_jump"]
                c3_nj = reps["C3"]["non_jump"]
                incremental_non_jump["C3_minus_C2"] = (
                    c3_nj["r2"] - c2_nj["r2"]
                    if c2_nj is not None and c3_nj is not None
                    else None
                )

        pooled_payload[str(horizon)] = {
            "representations": reps,
            "incremental_r2": incremental_all,
            "incremental_non_jump_r2": incremental_non_jump,
        }

    scored_folds = [fold for fold in folds if fold["status"] == "SCORED"]
    candidate_status: dict[str, object] = {}
    six = pooled_payload.get("6", {})
    six_inc = six.get("incremental_r2", {}) if isinstance(six, dict) else {}
    six_nj = six.get("incremental_non_jump_r2", {}) if isinstance(six, dict) else {}

    for candidate in ("C1", "C2", "C3"):
        key = f"{candidate}_minus_C0"
        positive_folds = 0
        for fold in scored_folds:
            horizons = fold["horizons"]
            increment = horizons["6"]["incremental_r2"][key]
            if increment > 0.0:
                positive_folds += 1
        pooled_increment = six_inc.get(key)
        non_jump_increment = six_nj.get(key)
        conditions = {
            "pooled_incremental_r2_positive": pooled_increment is not None and pooled_increment > 0.0,
            "at_least_three_positive_folds": positive_folds >= 3,
            "non_jump_incremental_r2_positive": non_jump_increment is not None and non_jump_increment > 0.0,
            "at_least_three_scored_folds": len(scored_folds) >= 3,
        }
        candidate_status[candidate] = {
            "signal_pass": all(conditions.values()),
            "conditions": conditions,
            "scored_folds": len(scored_folds),
            "positive_folds": positive_folds,
            "pooled_incremental_r2": pooled_increment,
            "non_jump_incremental_r2": non_jump_increment,
        }

    passing = [name for name in ("C1", "C2", "C3") if candidate_status[name]["signal_pass"]]
    selected: str | None = None
    if passing:
        selected = passing[0]
        for candidate in passing[1:]:
            current = candidate_status[selected]
            challenger = candidate_status[candidate]
            if (
                challenger["pooled_incremental_r2"] > current["pooled_incremental_r2"]
                and challenger["non_jump_incremental_r2"] > current["non_jump_incremental_r2"]
                and challenger["positive_folds"] >= current["positive_folds"]
            ):
                selected = candidate

    target = symbol.upper()
    return {
        "version": "V2.3-PHASE0C-ASSET-SPECIFIC-REGIME",
        "symbol": target,
        "row_count": len(rows),
        "expected_sensors": list(EXPECTED_SENSORS[target]),
        "own_return_horizons": list(OWN_RETURN_HORIZONS),
        "own_rv_windows": list(OWN_RV_WINDOWS),
        "peer_return_horizons": list(PEER_RETURN_HORIZONS),
        "max_peer_staleness_seconds": MAX_PEER_STALENESS_SECONDS,
        "primary_horizon_bars": PRIMARY_HORIZON,
        "secondary_horizon_bars": SECONDARY_HORIZON,
        "min_train_rows": MIN_TRAIN_ROWS,
        "jump_sigma_multiplier": JUMP_SIGMA_MULTIPLIER,
        "reserved_windows": [[start.isoformat(), end.isoformat()] for start, end in RESERVED_WINDOWS],
        "models": {
            "ridge": {"alpha": RIDGE_ALPHA, "standard_scaler_train_only": True},
            "hgbr": dict(HGBR_PARAMS),
        },
        "representations": {
            "C0": "own -> Ridge",
            "C1": "own + linked -> Ridge",
            "C2": "own + linked + regime -> Ridge",
            "C3": "own + linked + regime -> HistGradientBoostingRegressor",
        },
        "folds": folds,
        "pooled": pooled_payload,
        "candidate_status": candidate_status,
        "target_signal_pass": bool(passing),
        "selected_representation": selected,
        "forbidden_outputs": [
            "pnl",
            "trade_direction",
            "long_short_no_trade",
            "transaction_cost_gate",
            "take_profit",
            "stop_loss",
            "decision_threshold",
            "position_size",
            "leverage",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V2.3 Phase 0C asset-specific causal signal audit; no trading policy or PnL"
    )
    parser.add_argument("csv")
    parser.add_argument("--symbol", required=True, choices=sorted(EXPECTED_SENSORS))
    parser.add_argument("--peer", action="append", default=[], metavar="SYMBOL=CSV")
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target = args.symbol.upper()
    bars = load_ohlc_csv(args.csv)
    peers = load_peer_markets(args.peer, target_symbol=target)
    _validate_sensor_set(target, peers)

    print(
        f"Building V2.3 Phase 0C rows | {target} | peers={','.join(EXPECTED_SENSORS[target])}",
        flush=True,
    )
    rows = build_phase0c_rows(bars, symbol=target, peers=peers)
    print(f"eligible_development_rows={len(rows)}", flush=True)
    payload = evaluate_rows(rows, symbol=target)

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print("\nPhase 0C six-bar signal status", flush=True)
    print("-" * 78, flush=True)
    for candidate in ("C1", "C2", "C3"):
        status = payload["candidate_status"][candidate]
        print(
            f"{candidate}: pass={status['signal_pass']} "
            f"pooled_dR2={status['pooled_incremental_r2']} "
            f"non_jump_dR2={status['non_jump_incremental_r2']} "
            f"positive_folds={status['positive_folds']}/{status['scored_folds']}",
            flush=True,
        )
    print(f"target_signal_pass={payload['target_signal_pass']}", flush=True)
    print(f"selected_representation={payload['selected_representation']}", flush=True)
    print(f"output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
