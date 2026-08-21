from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Sequence

from .models import TradeResult


@dataclass(frozen=True, slots=True)
class PortfolioMetrics:
    trades: int
    win_rate: float
    profit_factor: float
    expectancy_bps: float
    net_return_pct: float
    max_drawdown_pct: float


def select_non_overlapping_trades(trades: Sequence[TradeResult]) -> list[TradeResult]:
    selected: list[TradeResult] = []
    last_exit = None
    for trade in sorted(trades, key=lambda item: item.prediction.timestamp):
        if last_exit is not None and trade.prediction.timestamp < last_exit:
            continue
        selected.append(trade)
        last_exit = trade.exit_timestamp
    return selected


def calculate_portfolio_metrics(trades: Sequence[TradeResult]) -> PortfolioMetrics:
    selected = select_non_overlapping_trades(trades)
    if not selected:
        return PortfolioMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    positives = [trade.net_return_bps for trade in selected if trade.net_return_bps > 0]
    negatives = [-trade.net_return_bps for trade in selected if trade.net_return_bps < 0]
    gross_profit = sum(positives)
    gross_loss = sum(negatives)
    profit_factor = inf if gross_loss == 0 and gross_profit > 0 else (
        gross_profit / gross_loss if gross_loss else 0.0
    )

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for trade in selected:
        equity *= 1.0 + trade.net_return_bps / 10_000.0
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)

    return PortfolioMetrics(
        trades=len(selected),
        win_rate=sum(trade.net_return_bps > 0 for trade in selected) / len(selected),
        profit_factor=profit_factor,
        expectancy_bps=sum(trade.net_return_bps for trade in selected) / len(selected),
        net_return_pct=(equity - 1.0) * 100.0,
        max_drawdown_pct=max_dd * 100.0,
    )
