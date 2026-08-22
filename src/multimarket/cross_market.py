from __future__ import annotations

from bisect import bisect_right
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import log, sqrt
from statistics import fmean
from typing import Collection, Sequence

from .models import MarketBar


@dataclass(frozen=True, slots=True)
class PeerSnapshot:
    peer_index: int
    peer_timestamp: datetime
    staleness: timedelta
    ret_1_bps: float | None
    ret_6_bps: float | None
    ret_12_bps: float | None
    vol_12_bps: float | None


def _history_is_usable(
    bars: Sequence[MarketBar],
    index: int,
    lookback_bars: int,
    *,
    expected_interval: timedelta,
    eligible_indices: Collection[int] | None,
) -> bool:
    if lookback_bars <= 0:
        raise ValueError("lookback_bars must be positive")
    if expected_interval <= timedelta(0):
        raise ValueError("expected_interval must be positive")
    start = index - lookback_bars
    if start < 0:
        return False
    if eligible_indices is not None:
        # set and frozenset are already O(1)-membership containers. In particular,
        # V2.1 stores peer eligibility as a frozenset; copying that ~100k-element
        # collection for every feature lookup was the dominant evaluation cost.
        eligible = eligible_indices if isinstance(eligible_indices, AbstractSet) else set(eligible_indices)
        if any(j not in eligible for j in range(start, index + 1)):
            return False
    return all(
        bars[j].timestamp - bars[j - 1].timestamp == expected_interval
        for j in range(start + 1, index + 1)
    )


def _return_bps(
    bars: Sequence[MarketBar],
    index: int,
    lag: int,
    *,
    expected_interval: timedelta,
    eligible_indices: Collection[int] | None,
) -> float | None:
    if not _history_is_usable(
        bars,
        index,
        lag,
        expected_interval=expected_interval,
        eligible_indices=eligible_indices,
    ):
        return None
    return (bars[index].close / bars[index - lag].close - 1.0) * 10_000.0


def _vol_bps(
    bars: Sequence[MarketBar],
    index: int,
    window: int,
    *,
    expected_interval: timedelta,
    eligible_indices: Collection[int] | None,
) -> float | None:
    if not _history_is_usable(
        bars,
        index,
        window,
        expected_interval=expected_interval,
        eligible_indices=eligible_indices,
    ):
        return None
    values = [
        log(bars[j].close / bars[j - 1].close) * 10_000.0
        for j in range(index - window + 1, index + 1)
    ]
    mean = fmean(values)
    return sqrt(fmean((value - mean) ** 2 for value in values))


@dataclass(frozen=True, slots=True)
class CausalPeerIndex:
    """Reusable causal as-of index for one peer market.

    Timestamp extraction is intentionally done once. Peer return/volatility features
    are emitted only when every raw bar required by that feature is both contiguous
    at the expected interval and, when supplied, present in the peer hard-eligibility
    set. This prevents weekend/session-invalid, repeated-OHLC, zero-range, and gap
    history from silently entering cross-market features.
    """

    bars: tuple[MarketBar, ...]
    timestamps: tuple[datetime, ...]

    @classmethod
    def build(cls, peer_bars: Sequence[MarketBar]) -> "CausalPeerIndex":
        bars = tuple(peer_bars)
        return cls(bars=bars, timestamps=tuple(bar.timestamp for bar in bars))

    def snapshot(
        self,
        decision_timestamp: datetime,
        *,
        max_staleness: timedelta = timedelta(minutes=15),
        expected_interval: timedelta = timedelta(minutes=5),
        eligible_indices: Collection[int] | None = None,
    ) -> PeerSnapshot | None:
        if max_staleness < timedelta(0):
            raise ValueError("max_staleness must be non-negative")
        if expected_interval <= timedelta(0):
            raise ValueError("expected_interval must be positive")
        index = bisect_right(self.timestamps, decision_timestamp) - 1
        if index < 0:
            return None
        peer_timestamp = self.bars[index].timestamp
        if peer_timestamp > decision_timestamp:
            raise AssertionError("causal alignment selected a future peer bar")
        staleness = decision_timestamp - peer_timestamp
        if staleness > max_staleness:
            return None
        return PeerSnapshot(
            peer_index=index,
            peer_timestamp=peer_timestamp,
            staleness=staleness,
            ret_1_bps=_return_bps(
                self.bars,
                index,
                1,
                expected_interval=expected_interval,
                eligible_indices=eligible_indices,
            ),
            ret_6_bps=_return_bps(
                self.bars,
                index,
                6,
                expected_interval=expected_interval,
                eligible_indices=eligible_indices,
            ),
            ret_12_bps=_return_bps(
                self.bars,
                index,
                12,
                expected_interval=expected_interval,
                eligible_indices=eligible_indices,
            ),
            vol_12_bps=_vol_bps(
                self.bars,
                index,
                12,
                expected_interval=expected_interval,
                eligible_indices=eligible_indices,
            ),
        )


def causal_peer_snapshot(
    peer_bars: Sequence[MarketBar],
    decision_timestamp: datetime,
    *,
    max_staleness: timedelta = timedelta(minutes=15),
    expected_interval: timedelta = timedelta(minutes=5),
    eligible_indices: Collection[int] | None = None,
) -> PeerSnapshot | None:
    """Return one causal peer snapshot; bulk callers should reuse CausalPeerIndex."""
    return CausalPeerIndex.build(peer_bars).snapshot(
        decision_timestamp,
        max_staleness=max_staleness,
        expected_interval=expected_interval,
        eligible_indices=eligible_indices,
    )
