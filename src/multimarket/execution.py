from __future__ import annotations

from typing import Sequence

from .models import Direction, MarketBar, Prediction, TradeResult


def simulate_barrier_trade(
    prediction: Prediction,
    future_bars: Sequence[MarketBar],
    *,
    take_profit_bps: float,
    stop_loss_bps: float,
    round_trip_cost_bps: float = 0.0,
) -> TradeResult:
    if prediction.direction is Direction.NO_TRADE:
        raise ValueError("NO_TRADE predictions cannot be executed")
    if not future_bars:
        raise ValueError("future_bars cannot be empty")
    if min(take_profit_bps, stop_loss_bps) <= 0:
        raise ValueError("barriers must be positive")
    if round_trip_cost_bps < 0:
        raise ValueError("round_trip_cost_bps cannot be negative")

    entry = prediction.reference_price
    tp = take_profit_bps / 10_000.0
    sl = stop_loss_bps / 10_000.0

    if prediction.direction is Direction.LONG:
        tp_price = entry * (1 + tp)
        sl_price = entry * (1 - sl)
    else:
        tp_price = entry * (1 - tp)
        sl_price = entry * (1 + sl)

    for bar in future_bars:
        hit_tp = bar.high >= tp_price if prediction.direction is Direction.LONG else bar.low <= tp_price
        hit_sl = bar.low <= sl_price if prediction.direction is Direction.LONG else bar.high >= sl_price
        if hit_tp and hit_sl:
            gross_bps = -stop_loss_bps
            return TradeResult(prediction, bar.timestamp, sl_price, gross_bps, gross_bps - round_trip_cost_bps, False, "AMBIGUOUS_BAR_STOP_FIRST")
        if hit_tp:
            return TradeResult(prediction, bar.timestamp, tp_price, take_profit_bps, take_profit_bps - round_trip_cost_bps, True, "TAKE_PROFIT")
        if hit_sl:
            return TradeResult(prediction, bar.timestamp, sl_price, -stop_loss_bps, -stop_loss_bps - round_trip_cost_bps, False, "STOP_LOSS")

    final = future_bars[-1]
    raw_return = final.close / entry - 1.0
    if prediction.direction is Direction.SHORT:
        raw_return = -raw_return
    gross_bps = raw_return * 10_000.0
    return TradeResult(prediction, final.timestamp, final.close, gross_bps, gross_bps - round_trip_cost_bps, gross_bps > 0, "HORIZON")
