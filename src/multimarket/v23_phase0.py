from __future__ import annotations

import argparse
import json
from bisect import bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import cos, log, pi, sin, sqrt
from pathlib import Path
from statistics import fmean
from typing import Collection, Sequence

import numpy as np
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_ohlc_csv
from .models import MarketBar
from .v21_common import hard_eligible_indices, load_peer_markets
from .v21_features import PeerMarket


LAGS = (1, 2, 3, 4, 6, 9, 12)
PRIMARY_HORIZON = 6
SECONDARY_HORIZON = 1
EXPECTED_SECONDS = 300
MIN_TRAIN_ROWS = 5000
JUMP_SIGMA_MULTIPLIER = 4.0
RIDGE_ALPHA = 10.0
ELASTIC_ALPHA = 0.0005
ELASTIC_L1_RATIO = 0.25

RESERVED_WINDOWS = (
    (
        datetime(2025, 10, 6, 0, 0, tzinfo=timezone.utc),
        datetime(2025, 10, 24, 23, 59, 59, tzinfo=timezone.utc),
    ),
    (
        datetime(2026, 2, 2, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 2, 20, 23, 59, 59, tzinfo=timezone.utc),
    ),
    (
        datetime(2026, 7, 6, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 7, 24, 23, 59, 59, tzinfo=timezone.utc),
    ),
)


@dataclass(frozen=True, slots=True)
class Phase0Row:
    timestamp: datetime
    label_end_timestamp: datetime
    base_features: tuple[float, ...]
    cross_features: tuple[float, ...]
    forward_1_bps: float
    forward_6_bps: float
    jump_state: int


@dataclass(frozen=True, slots=True)
class FitMetrics:
    rows: int
    r2: float
    mse: float
    spearman: float | None
    pearson: float | None
    sign_accuracy: float


@dataclass(frozen=True, slots=True)
class PreparedPeer:
    bars: tuple[MarketBar, ...]
    timestamps: tuple[datetime, ...]
    eligible: frozenset[int]
    sigma48: tuple[float | None, ...]


def _is_reserved(ts: datetime) -> bool:
    stamp = ts.astimezone(timezone.utc)
    return any(start <= stamp <= end for start, end in RESERVED_WINDOWS)


def _contiguous(bars: Sequence[MarketBar], start: int, end: int) -> bool:
    if start < 0 or end >= len(bars) or start > end:
        return False
    return all(
        (bars[i].timestamp - bars[i - 1].timestamp).total_seconds() == EXPECTED_SECONDS
        for i in range(start + 1, end + 1)
    )


def _one_bar_log_return_bps(bars: Sequence[MarketBar], index: int) -> float:
    return log(bars[index].close / bars[index - 1].close) * 10_000.0


def _sigma48_series(
    bars: Sequence[MarketBar],
    eligible: Collection[int],
) -> tuple[float | None, ...]:
    """Causal sigma for index i using 48 returns ending at i-1, never return i."""
    eligible_set = eligible if isinstance(eligible, (set, frozenset)) else set(eligible)
    one_bar: list[float | None] = [None] * len(bars)
    for i in range(1, len(bars)):
        if i not in eligible_set or i - 1 not in eligible_set:
            continue
        if (bars[i].timestamp - bars[i - 1].timestamp).total_seconds() != EXPECTED_SECONDS:
            continue
        one_bar[i] = _one_bar_log_return_bps(bars, i)

    result: list[float | None] = [None] * len(bars)
    for index in range(49, len(bars)):
        values = one_bar[index - 48 : index]
        if any(value is None for value in values):
            continue
        clean = [float(value) for value in values if value is not None]
        mean = fmean(clean)
        sigma = sqrt(fmean((x - mean) ** 2 for x in clean))
        if sigma > 0.0:
            result[index] = sigma
    return tuple(result)


def _prepare_peer(peer: PeerMarket) -> PreparedPeer:
    return PreparedPeer(
        bars=peer.bars,
        timestamps=tuple(bar.timestamp for bar in peer.bars),
        eligible=peer.eligible_indices,
        sigma48=_sigma48_series(peer.bars, peer.eligible_indices),
    )


def _target_lag_features(
    bars: Sequence[MarketBar],
    index: int,
    eligible: Collection[int],
) -> tuple[float, ...] | None:
    values: list[float] = []
    for lag in LAGS:
        end = index - (lag - 1)
        start = end - 1
        if start < 0 or start not in eligible or end not in eligible:
            return None
        if not _contiguous(bars, start, end):
            return None
        values.append((bars[end].close / bars[start].close - 1.0) * 10_000.0)
    return tuple(values)


def _peer_lag_packet(
    peer: PreparedPeer,
    decision_timestamp: datetime,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    asof = bisect_right(peer.timestamps, decision_timestamp) - 1
    numeric: list[float] = []
    available: list[float] = []

    for lag in LAGS:
        # lag=1 is the latest completed one-bar peer return known at decision time.
        end = asof - (lag - 1)
        start = end - 1
        ok = (
            start >= 0
            and start in peer.eligible
            and end in peer.eligible
            and peer.bars[end].timestamp <= decision_timestamp
            and _contiguous(peer.bars, start, end)
            and peer.sigma48[end] is not None
        )
        if not ok:
            numeric.append(0.0)
            available.append(0.0)
            continue

        raw = (peer.bars[end].close / peer.bars[start].close - 1.0) * 10_000.0
        sigma = peer.sigma48[end]
        assert sigma is not None and sigma > 0.0
        numeric.append(raw / sigma)
        available.append(1.0)

    return tuple(numeric), tuple(available)


def _cross_summary(
    peer_packets: Sequence[tuple[tuple[float, ...], tuple[float, ...]]],
) -> tuple[float, ...]:
    summary: list[float] = []
    for lag_position in range(len(LAGS)):
        values = [
            packet[0][lag_position]
            for packet in peer_packets
            if packet[1][lag_position] > 0.5
        ]
        if len(values) < 2:
            # summary_available, breadth, mean, dispersion, max_abs
            summary.extend((0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        mean = fmean(values)
        dispersion = sqrt(fmean((x - mean) ** 2 for x in values))
        breadth = sum(x > 0.0 for x in values) / len(values)
        summary.extend((1.0, breadth, mean, dispersion, max(abs(x) for x in values)))
    return tuple(summary)


def _intraday_state(
    bars: Sequence[MarketBar],
    index: int,
    eligible: Collection[int],
    sigma48: Sequence[float | None],
) -> tuple[tuple[float, ...], int] | None:
    sigma = sigma48[index]
    if sigma is None or index < 1 or index not in eligible or index - 1 not in eligible:
        return None
    if not _contiguous(bars, index - 1, index):
        return None

    ts = bars[index].timestamp.astimezone(timezone.utc)
    minute = ts.hour * 60 + ts.minute
    weekday = ts.weekday()
    current = _one_bar_log_return_bps(bars, index)
    jump = int(abs(current) > JUMP_SIGMA_MULTIPLIER * sigma)

    prior_sigmas = [
        value
        for value in sigma48[max(0, index - 288) : index]
        if value is not None
    ]
    vol_pct = (
        sum(float(value) <= sigma for value in prior_sigmas) / len(prior_sigmas)
        if prior_sigmas
        else 0.5
    )

    state = (
        sin(2.0 * pi * minute / 1440.0),
        cos(2.0 * pi * minute / 1440.0),
        sin(2.0 * pi * weekday / 7.0),
        cos(2.0 * pi * weekday / 7.0),
        vol_pct,
        abs(current) / sigma,
        float(jump),
    )
    return state, jump


def _forward_return(
    bars: Sequence[MarketBar],
    index: int,
    horizon: int,
    eligible: Collection[int],
) -> float | None:
    end = index + horizon
    if end >= len(bars):
        return None
    if any(i not in eligible for i in range(index, end + 1)):
        return None
    if not _contiguous(bars, index, end):
        return None
    if any(_is_reserved(bars[i].timestamp) for i in range(index, end + 1)):
        return None
    return (bars[end].close / bars[index].close - 1.0) * 10_000.0


def build_phase0_rows(
    bars: Sequence[MarketBar],
    *,
    symbol: str,
    peers: dict[str, PeerMarket],
) -> list[Phase0Row]:
    eligible = hard_eligible_indices(bars, symbol)
    target_sigma = _sigma48_series(bars, eligible)
    prepared_peers = {name: _prepare_peer(peer) for name, peer in peers.items()}
    result: list[Phase0Row] = []

    for index, bar in enumerate(bars):
        if index not in eligible or _is_reserved(bar.timestamp):
            continue
        target_lags = _target_lag_features(bars, index, eligible)
        state_result = _intraday_state(bars, index, eligible, target_sigma)
        if target_lags is None or state_result is None:
            continue
        state, jump = state_result
        y1 = _forward_return(bars, index, SECONDARY_HORIZON, eligible)
        y6 = _forward_return(bars, index, PRIMARY_HORIZON, eligible)
        if y1 is None or y6 is None:
            continue

        peer_packets = [
            _peer_lag_packet(prepared_peers[name], bar.timestamp)
            for name in sorted(prepared_peers)
        ]
        peer_values: list[float] = []
        for numeric, available in peer_packets:
            peer_values.extend(numeric)
            peer_values.extend(available)
        summaries = _cross_summary(peer_packets)

        result.append(
            Phase0Row(
                timestamp=bar.timestamp,
                label_end_timestamp=bars[index + PRIMARY_HORIZON].timestamp,
                base_features=target_lags + state,
                cross_features=target_lags + state + tuple(peer_values) + summaries,
                forward_1_bps=y1,
                forward_6_bps=y6,
                jump_state=jump,
            )
        )
    return result


def _rank(values: Sequence[float]) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0
        ranks[order[i : j + 1]] = mean_rank
        i = j + 1
    return ranks


def _corr(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) < 2:
        return None
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _metrics(y: Sequence[float], pred: Sequence[float]) -> FitMetrics:
    return FitMetrics(
        rows=len(y),
        r2=float(r2_score(y, pred)),
        mse=float(mean_squared_error(y, pred)),
        spearman=_corr(_rank(pred), _rank(y)),
        pearson=_corr(pred, y),
        sign_accuracy=float(np.mean(np.sign(pred) == np.sign(y))),
    )


def _make_model(name: str):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    if name == "elasticnet":
        return make_pipeline(
            StandardScaler(),
            ElasticNet(
                alpha=ELASTIC_ALPHA,
                l1_ratio=ELASTIC_L1_RATIO,
                max_iter=10000,
                random_state=0,
            ),
        )
    raise ValueError(name)


def _fold_ranges(row_count: int) -> list[tuple[int, int]]:
    bounds = [round(i * row_count / 5) for i in range(6)]
    return [(bounds[i], bounds[i + 1]) for i in range(5)]


def evaluate_rows(rows: Sequence[Phase0Row], *, symbol: str) -> dict[str, object]:
    if not rows:
        raise ValueError("Phase 0 has no eligible rows")
    folds: list[dict[str, object]] = []
    pooled: dict[tuple[str, str, int], dict[str, list[float]]] = {}

    for fold_number, (start, end) in enumerate(_fold_ranges(len(rows)), start=1):
        eval_rows = rows[start:end]
        if not eval_rows:
            continue
        eval_start = eval_rows[0].timestamp
        train_rows = [
            row
            for row in rows[:start]
            if row.label_end_timestamp < eval_start
        ]
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

        fold_result: dict[str, object] = {
            "fold": fold_number,
            "status": "SCORED",
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "eval_start": eval_start.isoformat(),
            "models": {},
        }

        for model_name in ("ridge", "elasticnet"):
            model_payload: dict[str, object] = {}
            for horizon, y_field in ((1, "forward_1_bps"), (6, "forward_6_bps")):
                y_train = np.asarray([getattr(row, y_field) for row in train_rows], dtype=float)
                y_eval = np.asarray([getattr(row, y_field) for row in eval_rows], dtype=float)
                representations: dict[str, object] = {}
                for rep_name, feature_field in (("base", "base_features"), ("cross", "cross_features")):
                    X_train = np.asarray([getattr(row, feature_field) for row in train_rows], dtype=float)
                    X_eval = np.asarray([getattr(row, feature_field) for row in eval_rows], dtype=float)
                    model = _make_model(model_name)
                    model.fit(X_train, y_train)
                    pred = model.predict(X_eval)
                    metrics = _metrics(y_eval, pred)
                    representations[rep_name] = asdict(metrics)

                    key = (model_name, rep_name, horizon)
                    slot = pooled.setdefault(key, {"y": [], "pred": [], "jump": []})
                    slot["y"].extend(float(x) for x in y_eval)
                    slot["pred"].extend(float(x) for x in pred)
                    slot["jump"].extend(row.jump_state for row in eval_rows)

                representations["incremental_r2"] = (
                    representations["cross"]["r2"] - representations["base"]["r2"]
                )
                model_payload[str(horizon)] = representations
            fold_result["models"][model_name] = model_payload
        folds.append(fold_result)

    pooled_payload: dict[str, object] = {}
    for model_name in ("ridge", "elasticnet"):
        model_payload: dict[str, object] = {}
        for horizon in (1, 6):
            reps: dict[str, object] = {}
            for rep_name in ("base", "cross"):
                slot = pooled.get((model_name, rep_name, horizon))
                if not slot:
                    continue
                metrics = _metrics(slot["y"], slot["pred"])
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
                    "all": asdict(metrics),
                    "non_jump": asdict(non_jump) if non_jump is not None else None,
                }
            if "base" in reps and "cross" in reps:
                reps["incremental_r2"] = reps["cross"]["all"]["r2"] - reps["base"]["all"]["r2"]
                base_nj = reps["base"]["non_jump"]
                cross_nj = reps["cross"]["non_jump"]
                reps["incremental_non_jump_r2"] = (
                    cross_nj["r2"] - base_nj["r2"]
                    if base_nj is not None and cross_nj is not None
                    else None
                )
            model_payload[str(horizon)] = reps
        pooled_payload[model_name] = model_payload

    return {
        "version": "V2.3-PHASE0-INFORMATION-DIFFUSION",
        "symbol": symbol,
        "row_count": len(rows),
        "lags": list(LAGS),
        "primary_horizon_bars": PRIMARY_HORIZON,
        "secondary_horizon_bars": SECONDARY_HORIZON,
        "min_train_rows": MIN_TRAIN_ROWS,
        "jump_sigma_multiplier": JUMP_SIGMA_MULTIPLIER,
        "reserved_windows": [[start.isoformat(), end.isoformat()] for start, end in RESERVED_WINDOWS],
        "models": {
            "ridge": {"alpha": RIDGE_ALPHA},
            "elasticnet": {
                "alpha": ELASTIC_ALPHA,
                "l1_ratio": ELASTIC_L1_RATIO,
                "max_iter": 10000,
            },
        },
        "folds": folds,
        "pooled": pooled_payload,
        "forbidden_outputs": ["pnl", "trade_direction", "take_profit", "stop_loss", "decision_threshold"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V2.3 Phase 0 causal cross-sectional information audit; no trading policy or PnL"
    )
    parser.add_argument("csv")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--peer", action="append", default=[], metavar="SYMBOL=CSV")
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bars = load_ohlc_csv(args.csv)
    peers = load_peer_markets(args.peer, target_symbol=args.symbol)

    print(
        f"Building V2.3 Phase 0 rows | {args.symbol.upper()} | peers={','.join(sorted(peers))}",
        flush=True,
    )
    rows = build_phase0_rows(bars, symbol=args.symbol, peers=peers)
    print(f"eligible_development_rows={len(rows)}", flush=True)
    payload = evaluate_rows(rows, symbol=args.symbol.upper())

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print("\nPooled six-bar incremental R2", flush=True)
    for model_name in ("ridge", "elasticnet"):
        block = payload["pooled"][model_name]["6"]
        value = block.get("incremental_r2")
        non_jump = block.get("incremental_non_jump_r2")
        print(
            f"{model_name}: incremental_r2={value:+.8f} "
            f"non_jump_incremental_r2={non_jump:+.8f}",
            flush=True,
        )
    print(f"Output: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
