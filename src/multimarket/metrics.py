from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Sequence

from .models import Direction, Prediction, TradeResult


@dataclass(frozen=True, slots=True)
class Metrics:
    predictions: int
    trades: int
    coverage: float
    directional_accuracy: float
    win_rate: float
    profit_factor: float
    expectancy_bps: float
    net_return_pct: float
    max_drawdown_pct: float


def _max_drawdown(returns_bps: Sequence[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for value in returns_bps:
        equity *= 1.0 + value / 10_000.0
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak
        max_dd = max(max_dd, drawdown)
    return max_dd * 100.0


def calculate_metrics(
    predictions: Sequence[Prediction],
    directional_hits: Sequence[bool],
    trades: Sequence[TradeResult],
) -> Metrics:
    if len(predictions) != len(directional_hits):
        raise ValueError("directional_hits must align with predictions")

    actionable = [p for p in predictions if p.direction is not Direction.NO_TRADE]
    hit_values = [
        hit for prediction, hit in zip(predictions, directional_hits)
        if prediction.direction is not Direction.NO_TRADE
    ]

    wins = [trade.net_return_bps for trade in trades if trade.net_return_bps > 0]
    losses = [-trade.net_return_bps for trade in trades if trade.net_return_bps < 0]
    gross_profit = sum(wins)
    gross_loss = sum(losses)

    if gross_loss == 0:
        profit_factor = inf if gross_profit > 0 else 0.0
    else:
        profit_factor = gross_profit / gross_loss

    equity = 1.0
    for trade in trades:
        equity *= 1.0 + trade.net_return_bps / 10_000.0

    return Metrics(
        predictions=len(predictions),
        trades=len(trades),
        coverage=(len(actionable) / len(predictions)) if predictions else 0.0,
        directional_accuracy=(sum(hit_values) / len(hit_values)) if hit_values else 0.0,
        win_rate=(sum(t.net_return_bps > 0 for t in trades) / len(trades)) if trades else 0.0,
        profit_factor=profit_factor,
        expectancy_bps=(sum(t.net_return_bps for t in trades) / len(trades)) if trades else 0.0,
        net_return_pct=(equity - 1.0) * 100.0,
        max_drawdown_pct=_max_drawdown([t.net_return_bps for t in trades]),
    )
