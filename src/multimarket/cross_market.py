from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import log, sqrt
from statistics import fmean
from typing import Sequence

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


def _return_bps(bars: Sequence[MarketBar], index: int, lag: int) -> float | None:
    if index - lag < 0:
        return None
    return (bars[index].close / bars[index - lag].close - 1.0) * 10_000.0


def _vol_bps(bars: Sequence[MarketBar], index: int, window: int) -> float | None:
    if index - window < 0:
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

    Timestamp extraction is intentionally done once. Reusing this object avoids the
    O(target_rows * peer_rows) behavior that results from rebuilding the timestamp
    list for every target lookup.
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
    ) -> PeerSnapshot | None:
        if max_staleness < timedelta(0):
            raise ValueError("max_staleness must be non-negative")
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
            ret_1_bps=_return_bps(self.bars, index, 1),
            ret_6_bps=_return_bps(self.bars, index, 6),
            ret_12_bps=_return_bps(self.bars, index, 12),
            vol_12_bps=_vol_bps(self.bars, index, 12),
        )


def causal_peer_snapshot(
    peer_bars: Sequence[MarketBar],
    decision_timestamp: datetime,
    *,
    max_staleness: timedelta = timedelta(minutes=15),
) -> PeerSnapshot | None:
    """Return one causal peer snapshot.

    This compatibility helper builds an index for a single lookup. Bulk callers must
    construct ``CausalPeerIndex`` once and reuse it.
    """
    return CausalPeerIndex.build(peer_bars).snapshot(
        decision_timestamp,
        max_staleness=max_staleness,
    )
