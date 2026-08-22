from __future__ import annotations

from datetime import timedelta
from typing import Collection, Sequence

from .features import WARMUP_BARS
from .models import MarketBar


def feature_history_is_clean(
    bars: Sequence[MarketBar],
    decision_index: int,
    *,
    eligible_indices: Collection[int] | None = None,
    lookback_bars: int = WARMUP_BARS,
    expected_interval: timedelta = timedelta(minutes=5),
) -> bool:
    """Require the full causal feature window to be contiguous and eligible.

    Existing feature calculations remain unchanged for reproducibility, but V2 only
    consumes a feature vector when every raw bar required by its maximum lookback is
    structurally valid. This prevents rolling features from silently bridging market
    closures or data gaps.
    """
    if lookback_bars <= 0:
        raise ValueError("lookback_bars must be positive")
    if expected_interval <= timedelta(0):
        raise ValueError("expected_interval must be positive")
    start = decision_index - lookback_bars
    if start < 0 or decision_index >= len(bars):
        return False

    eligible = None
    if eligible_indices is not None:
        eligible = eligible_indices if isinstance(eligible_indices, set) else set(eligible_indices)
        if any(index not in eligible for index in range(start, decision_index + 1)):
            return False

    for index in range(start + 1, decision_index + 1):
        if bars[index].timestamp - bars[index - 1].timestamp != expected_interval:
            return False
    return True


def clean_feature_indices(
    bars: Sequence[MarketBar],
    *,
    eligible_indices: Collection[int] | None = None,
    lookback_bars: int = WARMUP_BARS,
    expected_interval: timedelta = timedelta(minutes=5),
) -> set[int]:
    return {
        index
        for index in range(lookback_bars, len(bars))
        if feature_history_is_clean(
            bars,
            index,
            eligible_indices=eligible_indices,
            lookback_bars=lookback_bars,
            expected_interval=expected_interval,
        )
    }
