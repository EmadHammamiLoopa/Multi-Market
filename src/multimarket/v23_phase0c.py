from __future__ import annotations

import argparse
import json
from bisect import bisect_right
from dataclasses import asdict, dataclass
from hashlib import sha256
from datetime import datetime, timezone
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
from .v23_phase0 import EXPECTED_SECONDS, MIN_TRAIN_ROWS, PRIMARY_HORIZON, RESERVED_WINDOWS


OWN_RETURN_WINDOWS = (1, 3, 6, 12, 24)
OWN_RV_WINDOWS = (6, 24, 72)
SENSOR_RETURN_WINDOWS = (1, 6, 24)
VOL_PERCENTILE_LOOKBACK = 120
RECENT_JUMP_LOOKBACK = 6
JUMP_SIGMA_MULTIPLIER = 4.0
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
INNER_VALIDATION_FRACTION = 0.20
SENSOR_MAX_STALENESS_SECONDS = EXPECTED_SECONDS
HGBR_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "random_state": 0,
}
EXPERIMENTS = {
    "C0": ("own", "ridge"),
    "C1": ("linked", "ridge"),
    "C2": ("regime", "ridge"),
    "C3": ("regime", "hgbr"),
}


@dataclass(frozen=True, slots=True)
class Phase0CRow:
    timestamp: datetime
    label_end_timestamp: datetime
    execution_exit_timestamp: datetime
    own_features: tuple[float, ...]
    linked_features: tuple[float, ...]
    regime_features: tuple[float, ...]
    volatility_percentile: float
    forward_6_bps: float
    executable_forward_6_bps: float
    jump_state: int


@dataclass(frozen=True, slots=True)
class FitMetrics:
    rows: int
    r2: float
    mae: float
    rmse: float
    spearman: float | None
    pearson: float | None
    sign_accuracy: float


@dataclass(frozen=True, slots=True)
class PreparedMarket:
    bars: tuple[MarketBar, ...]
    timestamps: tuple[datetime, ...]
    eligible: frozenset[int]


@dataclass(frozen=True, slots=True)
class FoldWindow:
    fold: int
    eval_start: datetime
    eval_end: datetime | None


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


def _window_return_bps(
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
    if any(_is_reserved(bars[i].timestamp) for i in range(start, index + 1)):
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
    if any(_is_reserved(bars[i].timestamp) for i in range(start, index + 1)):
        return None
    if not _contiguous(bars, start, index):
        return None
    returns = [_one_bar_log_return_bps(bars, i) for i in range(start + 1, index + 1)]
    return sqrt(sum(value * value for value in returns))


def _sigma48_before(
    bars: Sequence[MarketBar],
    index: int,
    eligible: Collection[int],
) -> float | None:
    start = index - 49
    if start < 0:
        return None
    if any(i not in eligible for i in range(start, index + 1)):
        return None
    if any(_is_reserved(bars[i].timestamp) for i in range(start, index + 1)):
        return None
    if not _contiguous(bars, start, index):
        return None
    returns = [_one_bar_log_return_bps(bars, i) for i in range(start + 1, index)]
    if len(returns) < 2:
        return None
    mean = fmean(returns)
    sigma = sqrt(fmean((value - mean) ** 2 for value in returns))
    return sigma if sigma > 0.0 else None


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
    if any(_is_reserved(bars[i].timestamp) for i in range(start, index + 1)):
        return None
    if not _contiguous(bars, start, index):
        return None
    closes = [bar.close for bar in bars[start : index + 1]]
    mean = fmean(closes)
    variance = fmean((value - mean) ** 2 for value in closes)
    std = sqrt(variance)
    return (bars[index].close - mean) / std if std > 0.0 else 0.0


def _rv24_series(
    bars: Sequence[MarketBar],
    eligible: Collection[int],
) -> tuple[float | None, ...]:
    return tuple(
        _realized_vol_bps(bars, index, 24, eligible)
        for index in range(len(bars))
    )


def _volatility_percentile(
    rv24: Sequence[float | None],
    index: int,
    *,
    lookback: int = VOL_PERCENTILE_LOOKBACK,
) -> float | None:
    current = rv24[index]
    if current is None:
        return None
    prior: list[float] = []
    prior_index = index - 1
    while prior_index >= 0 and len(prior) < lookback:
        value = rv24[prior_index]
        if value is not None:
            prior.append(float(value))
        prior_index -= 1
    if not prior:
        return None
    return sum(value <= current for value in prior) / len(prior)


def _own_and_regime_features(
    bars: Sequence[MarketBar],
    index: int,
    eligible: Collection[int],
    *,
    rv24: Sequence[float | None],
) -> tuple[tuple[float, ...], tuple[float, ...], float, int] | None:
    returns: list[float] = []
    for window in OWN_RETURN_WINDOWS:
        value = _window_return_bps(bars, index, window, eligible)
        if value is None:
            return None
        returns.append(value)

    rvs: list[float] = []
    for window in OWN_RV_WINDOWS:
        value = rv24[index] if window == 24 else _realized_vol_bps(bars, index, window, eligible)
        if value is None:
            return None
        rvs.append(float(value))

    zprice = _zprice_24(bars, index, eligible)
    vol_pct = _volatility_percentile(rv24, index)
    sigma48 = _sigma48_before(bars, index, eligible)
    if zprice is None or vol_pct is None or sigma48 is None:
        return None

    one_bar_values: list[float] = []
    jump_start = index - RECENT_JUMP_LOOKBACK + 1
    if jump_start < 1:
        return None
    for i in range(jump_start, index + 1):
        if i not in eligible or i - 1 not in eligible:
            return None
        if _is_reserved(bars[i - 1].timestamp) or _is_reserved(bars[i].timestamp):
            return None
        if not _contiguous(bars, i - 1, i):
            return None
        one_bar_values.append(abs(_one_bar_log_return_bps(bars, i)))

    current_r1 = returns[0]
    jump_state = int(abs(current_r1) > JUMP_SIGMA_MULTIPLIER * sigma48)
    recent_jump = max(one_bar_values) / sigma48
    trend_strength = abs(returns[-1]) / (rvs[1] + 1e-12)
    ts = bars[index].timestamp.astimezone(timezone.utc)
    minute = ts.hour * 60 + ts.minute

    own = tuple(
        returns
        + rvs
        + [log(bars[index].high / bars[index].low) * 10_000.0, zprice]
    )
    regime = (
        vol_pct,
        trend_strength,
        recent_jump,
        sin(2.0 * pi * minute / 1440.0),
        cos(2.0 * pi * minute / 1440.0),
    )
    return own, regime, vol_pct, jump_state


def _prepare_market(peer: PeerMarket) -> PreparedMarket:
    return PreparedMarket(
        bars=peer.bars,
        timestamps=tuple(bar.timestamp for bar in peer.bars),
        eligible=peer.eligible_indices,
    )


def _sensor_packet(
    peer: PreparedMarket,
    decision_timestamp: datetime,
) -> tuple[float, ...] | None:
    asof = bisect_right(peer.timestamps, decision_timestamp) - 1
    if asof < 0:
        return None
    age = (decision_timestamp - peer.bars[asof].timestamp).total_seconds()
    if age < 0 or age > SENSOR_MAX_STALENESS_SECONDS:
        return None
    values: list[float] = []
    for window in SENSOR_RETURN_WINDOWS:
        value = _window_return_bps(peer.bars, asof, window, peer.eligible)
        if value is None:
            return None
        values.append(value)
    return tuple(values)


def _forward_close_return_bps(
    bars: Sequence[MarketBar],
    index: int,
    eligible: Collection[int],
) -> float | None:
    end = index + PRIMARY_HORIZON
    if end >= len(bars):
        return None
    if any(i not in eligible for i in range(index, end + 1)):
        return None
    if not _contiguous(bars, index, end):
        return None
    if any(_is_reserved(bars[i].timestamp) for i in range(index, end + 1)):
        return None
    return log(bars[end].close / bars[index].close) * 10_000.0


def _forward_executable_return_bps(
    bars: Sequence[MarketBar],
    index: int,
    eligible: Collection[int],
) -> tuple[float, datetime] | None:
    entry = index + 1
    end = index + PRIMARY_HORIZON
    if end >= len(bars) or entry >= len(bars):
        return None
    if any(i not in eligible for i in range(index, end + 1)):
        return None
    if not _contiguous(bars, index, end):
        return None
    if any(_is_reserved(bars[i].timestamp) for i in range(index, end + 1)):
        return None
    return log(bars[end].close / bars[entry].open) * 10_000.0, bars[end].timestamp


def build_phase0c_rows(
    bars: Sequence[MarketBar],
    *,
    symbol: str,
    linked_peers: dict[str, PeerMarket],
) -> list[Phase0CRow]:
    eligible = hard_eligible_indices(bars, symbol)
    prepared = {name: _prepare_market(peer) for name, peer in linked_peers.items()}
    rv24 = _rv24_series(bars, eligible)
    result: list[Phase0CRow] = []

    for index, bar in enumerate(bars):
        if index not in eligible or _is_reserved(bar.timestamp):
            continue

        # Check the sparse linked block first.  For 24/7 targets this avoids
        # computing expensive target features when RTH sensors are unavailable.
        linked: list[float] = []
        available = True
        for name in prepared:  # dict insertion order is the frozen manifest order
            packet = _sensor_packet(prepared[name], bar.timestamp)
            if packet is None:
                available = False
                break
            linked.extend(packet)
        if not available:
            continue

        own_state = _own_and_regime_features(bars, index, eligible, rv24=rv24)
        if own_state is None:
            continue
        own, regime, vol_pct, jump_state = own_state

        forward = _forward_close_return_bps(bars, index, eligible)
        executable = _forward_executable_return_bps(bars, index, eligible)
        if forward is None or executable is None:
            continue
        executable_return, exit_timestamp = executable
        result.append(
            Phase0CRow(
                timestamp=bar.timestamp,
                label_end_timestamp=bars[index + PRIMARY_HORIZON].timestamp,
                execution_exit_timestamp=exit_timestamp,
                own_features=own,
                linked_features=own + tuple(linked),
                regime_features=own + tuple(linked) + regime,
                volatility_percentile=vol_pct,
                forward_6_bps=forward,
                executable_forward_6_bps=executable_return,
                jump_state=jump_state,
            )
        )
    return result


def load_phase0b_fold_windows(path: str | Path) -> list[FoldWindow]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    scored = [fold for fold in payload.get("folds", []) if fold.get("status") == "SCORED"]
    if not scored:
        raise ValueError("Phase 0B evidence has no scored folds")
    starts: list[tuple[int, datetime]] = []
    for fold in scored:
        raw = fold.get("eval_start")
        if not raw:
            raise ValueError("scored Phase 0B fold is missing eval_start")
        starts.append((int(fold["fold"]), datetime.fromisoformat(raw.replace("Z", "+00:00"))))
    starts.sort(key=lambda item: item[1])
    windows: list[FoldWindow] = []
    for i, (fold_number, start) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else None
        windows.append(FoldWindow(fold=fold_number, eval_start=start, eval_end=end))
    return windows


def _rank(values: Sequence[float]) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0
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
    mse = float(mean_squared_error(y, pred))
    return FitMetrics(
        rows=len(y),
        r2=float(r2_score(y, pred)),
        mae=float(mean_absolute_error(y, pred)),
        rmse=sqrt(mse),
        spearman=_corr(_rank(pred), _rank(y)),
        pearson=_corr(pred, y),
        sign_accuracy=float(np.mean(np.sign(pred) == np.sign(y))),
    )


def _features(rows: Sequence[Phase0CRow], representation: str) -> np.ndarray:
    field = {
        "own": "own_features",
        "linked": "linked_features",
        "regime": "regime_features",
    }[representation]
    return np.asarray([getattr(row, field) for row in rows], dtype=float)


def _ridge_model(alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def _select_ridge_alpha(
    rows: Sequence[Phase0CRow],
    representation: str,
) -> float:
    if len(rows) < 1000:
        return 10.0
    split = max(1, int(round(len(rows) * (1.0 - INNER_VALIDATION_FRACTION))))
    if split >= len(rows):
        return 10.0
    eval_start = rows[split].timestamp
    inner_train = [row for row in rows[:split] if row.label_end_timestamp < eval_start]
    inner_eval = list(rows[split:])
    if len(inner_train) < 500 or len(inner_eval) < 100:
        return 10.0

    X_train = _features(inner_train, representation)
    y_train = np.asarray([row.forward_6_bps for row in inner_train], dtype=float)
    X_eval = _features(inner_eval, representation)
    y_eval = np.asarray([row.forward_6_bps for row in inner_eval], dtype=float)
    scored: list[tuple[float, float]] = []
    for alpha in RIDGE_ALPHAS:
        model = _ridge_model(alpha)
        model.fit(X_train, y_train)
        scored.append((float(r2_score(y_eval, model.predict(X_eval))), alpha))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def _fit_model(
    rows: Sequence[Phase0CRow],
    *,
    representation: str,
    model_name: str,
):
    X = _features(rows, representation)
    y = np.asarray([row.forward_6_bps for row in rows], dtype=float)
    if model_name == "ridge":
        alpha = _select_ridge_alpha(rows, representation)
        model = _ridge_model(alpha)
        model.fit(X, y)
        return model, {"alpha": alpha}
    if model_name == "hgbr":
        model = HistGradientBoostingRegressor(**HGBR_PARAMS)
        model.fit(X, y)
        return model, dict(HGBR_PARAMS)
    raise ValueError(model_name)


def _trade_metrics(returns_bps: Sequence[float]) -> dict[str, float | int | bool | None]:
    values = np.asarray(list(returns_bps), dtype=float)
    if len(values) == 0:
        return {
            "trades": 0,
            "expectancy_bps": 0.0,
            "pnl_bps": 0.0,
            "profit_factor": 0.0,
            "profit_factor_infinite": False,
            "sharpe": None,
            "sortino": None,
            "max_drawdown_pct": 0.0,
            "win_rate": 0.0,
            "average_win_bps": 0.0,
            "average_loss_bps": 0.0,
            "sum_sq_bps": 0.0,
            "downside_count": 0,
            "downside_sum_bps": 0.0,
            "downside_sum_sq_bps": 0.0,
            "gross_profit_bps": 0.0,
            "gross_loss_bps": 0.0,
            "wins_count": 0,
            "wins_sum_bps": 0.0,
            "losses_count": 0,
            "losses_sum_bps": 0.0,
        }
    wins = values[values > 0]
    losses = values[values < 0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    pf_infinite = gross_loss == 0.0 and gross_profit > 0.0
    profit_factor = None if pf_infinite else (gross_profit / gross_loss if gross_loss > 0.0 else 0.0)
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    sharpe = float(np.mean(values) / std) if std > 0.0 else None
    downside_std = float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0
    sortino = float(np.mean(values) / downside_std) if downside_std > 0.0 else None

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for value in values:
        equity *= 1.0 + value / 10_000.0
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)

    return {
        "trades": int(len(values)),
        "expectancy_bps": float(np.mean(values)),
        "pnl_bps": float(np.sum(values)),
        "profit_factor": profit_factor,
        "profit_factor_infinite": pf_infinite,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_dd * 100.0,
        "win_rate": float(np.mean(values > 0)),
        "average_win_bps": float(np.mean(wins)) if len(wins) else 0.0,
        "average_loss_bps": float(np.mean(losses)) if len(losses) else 0.0,
        "sum_sq_bps": float(np.sum(values * values)),
        "downside_count": int(len(losses)),
        "downside_sum_bps": float(np.sum(losses)) if len(losses) else 0.0,
        "downside_sum_sq_bps": float(np.sum(losses * losses)) if len(losses) else 0.0,
        "gross_profit_bps": gross_profit,
        "gross_loss_bps": gross_loss,
        "wins_count": int(len(wins)),
        "wins_sum_bps": float(np.sum(wins)) if len(wins) else 0.0,
        "losses_count": int(len(losses)),
        "losses_sum_bps": float(np.sum(losses)) if len(losses) else 0.0,
    }


def _combine_trade_metrics(metrics: Sequence[dict[str, object]]) -> dict[str, object]:
    n = sum(int(item["trades"]) for item in metrics)
    if n == 0:
        return _trade_metrics([])
    total = sum(float(item["pnl_bps"]) for item in metrics)
    sum_sq = sum(float(item["sum_sq_bps"]) for item in metrics)
    mean = total / n
    variance = (sum_sq - n * mean * mean) / (n - 1) if n > 1 else 0.0
    std = sqrt(max(variance, 0.0))
    downside_n = sum(int(item["downside_count"]) for item in metrics)
    downside_sum = sum(float(item["downside_sum_bps"]) for item in metrics)
    downside_sum_sq = sum(float(item["downside_sum_sq_bps"]) for item in metrics)
    downside_variance = (
        (downside_sum_sq - downside_sum * downside_sum / downside_n) / (downside_n - 1)
        if downside_n > 1 else 0.0
    )
    downside_std = sqrt(max(downside_variance, 0.0))
    gross_profit = sum(float(item["gross_profit_bps"]) for item in metrics)
    gross_loss = sum(float(item["gross_loss_bps"]) for item in metrics)
    pf_infinite = gross_loss == 0.0 and gross_profit > 0.0
    profit_factor = None if pf_infinite else (gross_profit / gross_loss if gross_loss > 0.0 else 0.0)
    wins_count = sum(int(item.get("wins_count", 0)) for item in metrics)
    wins_sum = sum(float(item.get("wins_sum_bps", 0.0)) for item in metrics)
    losses_count = sum(int(item.get("losses_count", 0)) for item in metrics)
    losses_sum = sum(float(item.get("losses_sum_bps", 0.0)) for item in metrics)
    return {
        "trades": n,
        "expectancy_bps": mean,
        "pnl_bps": total,
        "profit_factor": profit_factor,
        "profit_factor_infinite": pf_infinite,
        "sharpe": mean / std if std > 0.0 else None,
        "sortino": mean / downside_std if downside_std > 0.0 else None,
        "win_rate": wins_count / n,
        "average_win_bps": wins_sum / wins_count if wins_count else 0.0,
        "average_loss_bps": losses_sum / losses_count if losses_count else 0.0,
        "max_drawdown_pct": None,
        "sum_sq_bps": sum_sq,
        "downside_count": downside_n,
        "downside_sum_bps": downside_sum,
        "downside_sum_sq_bps": downside_sum_sq,
        "gross_profit_bps": gross_profit,
        "gross_loss_bps": gross_loss,
        "wins_count": wins_count,
        "wins_sum_bps": wins_sum,
        "losses_count": losses_count,
        "losses_sum_bps": losses_sum,
    }


def _profit_factor_above(metric: dict[str, object], threshold: float) -> bool:
    if bool(metric.get("profit_factor_infinite")):
        return True
    value = metric.get("profit_factor")
    return value is not None and float(value) > threshold


def _economic_evaluation(
    train_rows: Sequence[Phase0CRow],
    eval_rows: Sequence[Phase0CRow],
    train_pred: Sequence[float],
    eval_pred: Sequence[float],
    *,
    round_trip_cost_bps: float | None,
    initial_last_exit: datetime | None = None,
) -> dict[str, object]:
    if round_trip_cost_bps is None:
        return {"status": "NOT_EVALUATED_NO_COST_MODEL"}
    if round_trip_cost_bps < 0:
        raise ValueError("round_trip_cost_bps must be non-negative")

    confidence_threshold = float(np.percentile(np.abs(np.asarray(train_pred, dtype=float)), 75.0))
    accepted: list[tuple[int, int, float, float]] = []
    last_exit = initial_last_exit
    for row, pred in zip(eval_rows, eval_pred):
        if last_exit is not None and row.timestamp <= last_exit:
            continue
        if row.volatility_percentile < 0.60:
            continue
        if abs(float(pred)) < confidence_threshold:
            continue
        if abs(float(pred)) <= 1.5 * round_trip_cost_bps:
            continue
        direction = 1 if pred > 0 else -1
        gross = direction * row.executable_forward_6_bps
        accepted.append((direction, row.jump_state, gross, float(pred)))
        last_exit = row.execution_exit_timestamp

    gross_returns = [item[2] for item in accepted]
    base = _trade_metrics([value - round_trip_cost_bps for value in gross_returns])
    stress = {
        str(multiplier): _trade_metrics(
            [value - round_trip_cost_bps * multiplier for value in gross_returns]
        )
        for multiplier in (1.0, 1.5, 2.0)
    }
    net_returns = [value - round_trip_cost_bps for value in gross_returns]
    long_values = [gross - round_trip_cost_bps for direction, _, gross, _ in accepted if direction > 0]
    short_values = [gross - round_trip_cost_bps for direction, _, gross, _ in accepted if direction < 0]
    stress_sequences = {
        str(multiplier): [value - round_trip_cost_bps * multiplier for value in gross_returns]
        for multiplier in (1.0, 1.5, 2.0)
    }
    return {
        "status": "SCORED",
        "round_trip_cost_bps": round_trip_cost_bps,
        "confidence_threshold_bps": confidence_threshold,
        "volatility_percentile_min": 0.60,
        "minimum_edge_vs_cost": 1.5,
        "non_overlapping_positions": True,
        "coverage": len(accepted) / len(eval_rows) if eval_rows else 0.0,
        "gross": _trade_metrics(gross_returns),
        "net": base,
        "cost_stress": stress,
        "long": _trade_metrics(long_values),
        "short": _trade_metrics(short_values),
        "_last_exit": last_exit,
        "_sequences": {
            "gross": gross_returns,
            "net": net_returns,
            "long": long_values,
            "short": short_values,
            "cost_stress": stress_sequences,
        },
    }


def evaluate_rows(
    rows: Sequence[Phase0CRow],
    *,
    symbol: str,
    fold_windows: Sequence[FoldWindow],
    round_trip_cost_bps: float | None,
) -> dict[str, object]:
    if not rows:
        raise ValueError("Phase 0C has no eligible rows")

    folds: list[dict[str, object]] = []
    pooled: dict[str, dict[str, list[float] | list[int]]] = {
        name: {"y": [], "pred": [], "jump": []} for name in EXPERIMENTS
    }
    fold_economic: dict[str, list[dict[str, object]]] = {name: [] for name in EXPERIMENTS}
    economic_last_exit: dict[str, datetime | None] = {name: None for name in EXPERIMENTS}

    for window in fold_windows:
        eval_rows = [
            row for row in rows
            if row.timestamp >= window.eval_start
            and (window.eval_end is None or row.timestamp < window.eval_end)
        ]
        train_rows = [
            row for row in rows
            if row.timestamp < window.eval_start and row.label_end_timestamp < window.eval_start
        ]
        if len(train_rows) < MIN_TRAIN_ROWS:
            folds.append({
                "fold": window.fold,
                "status": "SKIP_MIN_TRAIN_ROWS",
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "eval_start": window.eval_start.isoformat(),
            })
            continue
        if not eval_rows:
            folds.append({
                "fold": window.fold,
                "status": "SKIP_NO_EVAL_ROWS",
                "train_rows": len(train_rows),
                "eval_rows": 0,
                "eval_start": window.eval_start.isoformat(),
            })
            continue

        fold_payload: dict[str, object] = {
            "fold": window.fold,
            "status": "SCORED",
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "eval_start": window.eval_start.isoformat(),
            "eval_end": window.eval_end.isoformat() if window.eval_end else None,
            "experiments": {},
        }
        predictions: dict[str, np.ndarray] = {}

        for experiment, (representation, model_name) in EXPERIMENTS.items():
            model, model_config = _fit_model(
                train_rows,
                representation=representation,
                model_name=model_name,
            )
            X_train = _features(train_rows, representation)
            X_eval = _features(eval_rows, representation)
            train_pred = np.asarray(model.predict(X_train), dtype=float)
            pred = np.asarray(model.predict(X_eval), dtype=float)
            predictions[experiment] = pred
            y_eval = np.asarray([row.forward_6_bps for row in eval_rows], dtype=float)
            metrics = asdict(_metrics(y_eval, pred))
            non_jump_idx = [i for i, row in enumerate(eval_rows) if not row.jump_state]
            non_jump = (
                asdict(_metrics(y_eval[non_jump_idx], pred[non_jump_idx]))
                if len(non_jump_idx) >= 2
                else None
            )
            economic = _economic_evaluation(
                train_rows,
                eval_rows,
                train_pred,
                pred,
                round_trip_cost_bps=round_trip_cost_bps,
                initial_last_exit=economic_last_exit[experiment],
            )
            if economic.get("status") == "SCORED":
                economic_last_exit[experiment] = economic.get("_last_exit")
            fold_economic[experiment].append({"fold": window.fold, "economic": economic})
            public_economic = {
                key: value for key, value in economic.items() if not key.startswith("_")
            }
            fold_payload["experiments"][experiment] = {
                "representation": representation,
                "model": model_name,
                "model_config": model_config,
                "all": metrics,
                "non_jump": non_jump,
                "economic": public_economic,
            }
            slot = pooled[experiment]
            slot["y"].extend(float(value) for value in y_eval)
            slot["pred"].extend(float(value) for value in pred)
            slot["jump"].extend(row.jump_state for row in eval_rows)

        c0_r2 = fold_payload["experiments"]["C0"]["all"]["r2"]
        c0_nj = fold_payload["experiments"]["C0"]["non_jump"]
        for experiment in ("C1", "C2", "C3"):
            block = fold_payload["experiments"][experiment]
            block["delta_r2_vs_c0"] = block["all"]["r2"] - c0_r2
            block_nj = block["non_jump"]
            block["delta_non_jump_r2_vs_c0"] = (
                block_nj["r2"] - c0_nj["r2"]
                if c0_nj is not None and block_nj is not None
                else None
            )
        folds.append(fold_payload)

    pooled_payload: dict[str, object] = {}
    for experiment in EXPERIMENTS:
        slot = pooled[experiment]
        if not slot["y"]:
            continue
        all_metrics = asdict(_metrics(slot["y"], slot["pred"]))
        non_jump_idx = [i for i, flag in enumerate(slot["jump"]) if not flag]
        non_jump = (
            asdict(_metrics(
                [slot["y"][i] for i in non_jump_idx],
                [slot["pred"][i] for i in non_jump_idx],
            )) if len(non_jump_idx) >= 2 else None
        )
        pooled_payload[experiment] = {
            "representation": EXPERIMENTS[experiment][0],
            "model": EXPERIMENTS[experiment][1],
            "all": all_metrics,
            "non_jump": non_jump,
        }

    if "C0" in pooled_payload:
        c0 = pooled_payload["C0"]
        for experiment in ("C1", "C2", "C3"):
            if experiment not in pooled_payload:
                continue
            block = pooled_payload[experiment]
            block["delta_r2_vs_c0"] = block["all"]["r2"] - c0["all"]["r2"]
            block["delta_non_jump_r2_vs_c0"] = (
                block["non_jump"]["r2"] - c0["non_jump"]["r2"]
                if block["non_jump"] is not None and c0["non_jump"] is not None
                else None
            )

    candidate_status: dict[str, object] = {}
    for experiment in ("C2", "C3"):
        scored_folds = [fold for fold in folds if fold.get("status") == "SCORED"]
        deltas = [
            float(fold["experiments"][experiment]["delta_r2_vs_c0"])
            for fold in scored_folds
        ]
        pooled_block = pooled_payload.get(experiment)
        statistical_pass = bool(
            pooled_block
            and pooled_block.get("delta_r2_vs_c0", 0.0) > 0.0
            and pooled_block.get("delta_non_jump_r2_vs_c0") is not None
            and pooled_block["delta_non_jump_r2_vs_c0"] > 0.0
            and len(deltas) >= 4
            and sum(value > 0.0 for value in deltas) >= 3
        )

        economic_scored = [
            item for item in fold_economic[experiment]
            if item["economic"].get("status") == "SCORED"
        ]
        economic_evaluated = bool(economic_scored) and len(economic_scored) == len(scored_folds)
        fold_net_pnls: list[float] = []
        if economic_evaluated:
            fold_net = [item["economic"]["net"] for item in economic_scored]
            def _concat(sequence_key: str) -> list[float]:
                values: list[float] = []
                for item in economic_scored:
                    values.extend(float(value) for value in item["economic"]["_sequences"][sequence_key])
                return values

            pooled_net = _trade_metrics(_concat("net"))
            pooled_gross = _trade_metrics(_concat("gross"))
            pooled_long = _trade_metrics(_concat("long"))
            pooled_short = _trade_metrics(_concat("short"))
            pooled_stress = {}
            for multiplier in ("1.0", "1.5", "2.0"):
                values: list[float] = []
                for item in economic_scored:
                    values.extend(
                        float(value)
                        for value in item["economic"]["_sequences"]["cost_stress"][multiplier]
                    )
                pooled_stress[multiplier] = _trade_metrics(values)
            fold_net_pnls = [float(item["pnl_bps"]) for item in fold_net]
            total_net = float(pooled_net["pnl_bps"])
            total_trades = int(pooled_net["trades"])
            max_share = (
                max((max(value, 0.0) for value in fold_net_pnls), default=0.0) / total_net
                if total_net > 0.0 else None
            )
            economic_conditions = {
                "net_expectancy_positive": float(pooled_net["expectancy_bps"]) > 0.0,
                "profit_factor_above_1_10": _profit_factor_above(pooled_net, 1.10),
                "sharpe_above_0_50": pooled_net["sharpe"] is not None and float(pooled_net["sharpe"]) > 0.50,
                "stress_1_5x_net_pnl_positive": float(pooled_stress["1.5"]["pnl_bps"]) > 0.0,
                "minimum_30_trades": total_trades >= 30,
                "max_fold_profit_share_le_60pct": max_share is not None and max_share <= 0.60,
            }
            economic_pass = all(economic_conditions.values())
            economic_summary: dict[str, object] = {
                "status": "SCORED",
                "gross": pooled_gross,
                "net": pooled_net,
                "long": pooled_long,
                "short": pooled_short,
                "cost_stress": pooled_stress,
                "fold_net_pnl_bps": fold_net_pnls,
                "max_fold_profit_share": max_share,
                "conditions": economic_conditions,
            }
        else:
            economic_pass = False
            economic_summary = {"status": "NOT_EVALUATED_NO_COST_MODEL"}

        candidate_status[experiment] = {
            "statistical_pass": statistical_pass,
            "economic_evaluated": economic_evaluated,
            "economic_pass": economic_pass,
            "promotion_pass": statistical_pass and economic_pass,
            "positive_folds": sum(value > 0.0 for value in deltas),
            "scored_folds": len(deltas),
            "economic": economic_summary,
        }

    selected = None
    if candidate_status["C2"]["promotion_pass"]:
        selected = "C2"
    elif candidate_status["C3"]["promotion_pass"]:
        selected = "C3"

    signal_candidate = None
    if candidate_status["C2"]["statistical_pass"]:
        signal_candidate = "C2"
    elif candidate_status["C3"]["statistical_pass"]:
        signal_candidate = "C3"

    return {
        "version": "V2.3-PHASE0C-ASSET-SPECIFIC-REGIME-GATED",
        "symbol": symbol.upper(),
        "evaluation_status": "SCORED",
        "row_count": len(rows),
        "primary_horizon_bars": PRIMARY_HORIZON,
        "min_train_rows": MIN_TRAIN_ROWS,
        "purge_rule": "label_end_timestamp < eval_start",
        "fold_source": "PHASE0B_EVAL_START_TIMESTAMPS",
        "reserved_windows": [[start.isoformat(), end.isoformat()] for start, end in RESERVED_WINDOWS],
        "own_return_windows": list(OWN_RETURN_WINDOWS),
        "own_rv_windows": list(OWN_RV_WINDOWS),
        "sensor_return_windows": list(SENSOR_RETURN_WINDOWS),
        "experiments": {
            key: {"representation": value[0], "model": value[1]}
            for key, value in EXPERIMENTS.items()
        },
        "ridge_alpha_grid": list(RIDGE_ALPHAS),
        "hgbr": HGBR_PARAMS,
        "folds": folds,
        "pooled": pooled_payload,
        "candidates": candidate_status,
        "signal_candidate": signal_candidate,
        "promoted_candidate": selected,
        "promotion_pass": selected is not None,
        "cost_model_status": "SUPPLIED" if round_trip_cost_bps is not None else "MISSING",
    }


def _sha256_file(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def load_phase0c_manifest(path: str | Path, *, symbol: str) -> tuple[dict[str, object], tuple[str, ...]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    targets = payload.get("targets")
    if not isinstance(targets, dict) or symbol.upper() not in targets:
        raise ValueError(f"Phase 0C manifest has no target {symbol.upper()}")
    block = targets[symbol.upper()]
    linked = block.get("linked_sensors") if isinstance(block, dict) else None
    if not isinstance(linked, list) or not linked:
        raise ValueError(f"Phase 0C manifest target {symbol.upper()} has no linked_sensors")
    symbols: list[str] = []
    for item in linked:
        if not isinstance(item, dict) or not item.get("symbol"):
            raise ValueError("invalid linked_sensors entry in Phase 0C manifest")
        symbols.append(str(item["symbol"]).upper())
    if len(symbols) != len(set(symbols)):
        raise ValueError("duplicate linked sensor in Phase 0C manifest")
    return payload, tuple(symbols)


def validate_linked_peers(peers: dict[str, PeerMarket], expected: Sequence[str]) -> None:
    actual = tuple(peers.keys())
    if set(actual) != set(expected):
        raise ValueError(
            f"linked peers do not match frozen manifest: expected={list(expected)} actual={sorted(actual)}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V2.3 Phase 0C asset-specific, regime-aware causal prediction audit"
    )
    parser.add_argument("csv")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--peer", action="append", default=[], metavar="SYMBOL=CSV")
    parser.add_argument("--phase0b-json", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--round-trip-cost-bps", type=float, default=None)
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bars = load_ohlc_csv(args.csv)
    peers = load_peer_markets(args.peer, target_symbol=args.symbol)
    manifest, linked_order = load_phase0c_manifest(args.manifest, symbol=args.symbol)
    validate_linked_peers(peers, linked_order)
    peers = {name: peers[name] for name in linked_order}
    fold_windows = load_phase0b_fold_windows(args.phase0b_json)
    print(
        f"Building V2.3 Phase 0C rows | {args.symbol.upper()} | linked={','.join(sorted(peers))}",
        flush=True,
    )
    rows = build_phase0c_rows(bars, symbol=args.symbol, linked_peers=peers)
    print(f"eligible_phase0c_rows={len(rows)}", flush=True)
    payload = evaluate_rows(
        rows,
        symbol=args.symbol,
        fold_windows=fold_windows,
        round_trip_cost_bps=args.round_trip_cost_bps,
    )
    payload["frozen_manifest"] = {
        "path": str(Path(args.manifest)),
        "sha256": _sha256_file(args.manifest),
        "linked_sensor_order": list(linked_order),
        "manifest_version": manifest.get("version"),
    }
    payload["phase0b_boundary_source"] = {
        "path": str(Path(args.phase0b_json)),
        "sha256": _sha256_file(args.phase0b_json),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"signal_candidate={payload['signal_candidate']}", flush=True)
    print(f"promoted_candidate={payload['promoted_candidate']}", flush=True)
    print(f"Output: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
