from __future__ import annotations

from datetime import timedelta
from math import log, sqrt
from typing import Collection, Sequence

from .models import MarketBar


def build_causal_ewma_volatility_bps(
    bars: Sequence[MarketBar],
    *,
    eligible_indices: Collection[int] | None = None,
    span: int = 48,
    min_observations: int = 24,
    expected_interval: timedelta = timedelta(minutes=5),
) -> list[float | None]:
    """Build a causal EWMA volatility series from valid one-bar log returns.

    Only returns formed by two eligible bars separated by exactly the expected
    interval update the estimate. Gaps and ineligible bars therefore cannot inject
    overnight/weekend/closure jumps into the volatility state. At a bar following
    a gap, the most recent historical volatility estimate is carried forward until
    a new valid return arrives.

    The estimator uses exponentially weighted first and second moments. Every
    output at index i depends only on bars at or before i.
    """
    if span <= 1:
        raise ValueError("span must be greater than 1")
    if min_observations <= 0:
        raise ValueError("min_observations must be positive")
    if expected_interval <= timedelta(0):
        raise ValueError("expected_interval must be positive")

    eligible = None
    if eligible_indices is not None:
        eligible = eligible_indices if isinstance(eligible_indices, set) else set(eligible_indices)

    alpha = 2.0 / (span + 1.0)
    mean: float | None = None
    second_moment: float | None = None
    observations = 0
    result: list[float | None] = [None] * len(bars)

    for index in range(1, len(bars)):
        previous = bars[index - 1]
        current = bars[index]
        contiguous = current.timestamp - previous.timestamp == expected_interval
        pair_eligible = eligible is None or (index - 1 in eligible and index in eligible)

        if contiguous and pair_eligible:
            if previous.close <= 0 or current.close <= 0:
                raise ValueError("close prices must be positive for log-return volatility")
            value = log(current.close / previous.close) * 10_000.0
            if mean is None:
                mean = value
                second_moment = value * value
            else:
                assert second_moment is not None
                mean = (1.0 - alpha) * mean + alpha * value
                second_moment = (1.0 - alpha) * second_moment + alpha * value * value
            observations += 1

        if observations >= min_observations and mean is not None and second_moment is not None:
            variance = max(0.0, second_moment - mean * mean)
            result[index] = sqrt(variance)

    return result


def volatility_scaled_barriers_bps(
    one_bar_volatility_bps: float,
    *,
    horizon_bars: int,
    take_profit_vol_multiplier: float = 1.0,
    stop_loss_vol_multiplier: float = 1.0,
) -> tuple[float, float]:
    """Convert one-bar volatility into horizon-scaled TP/SL distances.

    The default is symmetric 1.0 x horizon volatility. Multipliers are explicit
    research parameters and must be frozen before untouched validation.
    """
    if one_bar_volatility_bps < 0:
        raise ValueError("one_bar_volatility_bps cannot be negative")
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    if take_profit_vol_multiplier <= 0 or stop_loss_vol_multiplier <= 0:
        raise ValueError("volatility multipliers must be positive")

    horizon_vol = one_bar_volatility_bps * sqrt(float(horizon_bars))
    return (
        horizon_vol * take_profit_vol_multiplier,
        horizon_vol * stop_loss_vol_multiplier,
    )
