from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Collection, Mapping, Sequence

from .cross_market import CausalPeerIndex
from .features import FEATURE_NAMES, FeaturePoint, build_feature_points
from .models import MarketBar
from .v2_features import clean_feature_indices


REGIME_FEATURE_NAMES = (
    "regime_vol48_percentile_288",
    "regime_abs_ret12_percentile_288",
    "regime_trend_consistency_12",
)
PEER_FIELD_NAMES = (
    "available",
    "staleness_fraction",
    "ret_1_bps",
    "ret_6_bps",
    "ret_12_bps",
    "vol_12_bps",
)
REGIME_LOOKBACK = 288
PEER_MISSING_STALENESS_SENTINEL = 1.25


@dataclass(frozen=True, slots=True)
class PeerMarket:
    bars: tuple[MarketBar, ...]
    eligible_indices: frozenset[int]

    @classmethod
    def build(
        cls,
        bars: Sequence[MarketBar],
        *,
        eligible_indices: Collection[int],
    ) -> "PeerMarket":
        return cls(tuple(bars), frozenset(eligible_indices))


@dataclass(frozen=True, slots=True)
class V21FeaturePoint:
    bar_index: int
    timestamp: object
    values: tuple[float, ...]


def v21_feature_names(peer_symbols: Collection[str]) -> tuple[str, ...]:
    peer_names = tuple(
        f"{symbol.upper()}_{field}"
        for symbol in sorted({symbol.upper() for symbol in peer_symbols})
        for field in PEER_FIELD_NAMES
    )
    return FEATURE_NAMES + REGIME_FEATURE_NAMES + peer_names


def _percentile_rank(current: float, history: Sequence[float]) -> float:
    if not history:
        return 0.5
    return sum(value <= current for value in history) / len(history)


def _trend_consistency_12(bars: Sequence[MarketBar], index: int) -> float:
    cumulative = bars[index].close / bars[index - 12].close - 1.0
    if cumulative == 0.0:
        return 0.5
    desired_positive = cumulative > 0.0
    agreeing = 0
    for j in range(index - 11, index + 1):
        one_bar = bars[j].close / bars[j - 1].close - 1.0
        if (one_bar > 0.0) == desired_positive and one_bar != 0.0:
            agreeing += 1
    return agreeing / 12.0


def _usable_peer_values(
    peer_index: CausalPeerIndex,
    eligible_indices: Collection[int],
    decision_timestamp,
    *,
    max_staleness: timedelta,
) -> tuple[float, ...]:
    snapshot = peer_index.snapshot(
        decision_timestamp,
        max_staleness=max_staleness,
        eligible_indices=eligible_indices,
    )
    if snapshot is None or None in (
        snapshot.ret_1_bps,
        snapshot.ret_6_bps,
        snapshot.ret_12_bps,
        snapshot.vol_12_bps,
    ):
        return (0.0, PEER_MISSING_STALENESS_SENTINEL, 0.0, 0.0, 0.0, 0.0)
    staleness_fraction = snapshot.staleness.total_seconds() / max_staleness.total_seconds() if max_staleness else 0.0
    return (
        1.0,
        staleness_fraction,
        float(snapshot.ret_1_bps),
        float(snapshot.ret_6_bps),
        float(snapshot.ret_12_bps),
        float(snapshot.vol_12_bps),
    )


def build_v21_feature_points(
    bars: Sequence[MarketBar],
    *,
    eligible_indices: Collection[int],
    peers: Mapping[str, PeerMarket],
    max_peer_staleness: timedelta = timedelta(minutes=15),
) -> tuple[list[V21FeaturePoint], set[int]]:
    """Build causal V2.1 features without requiring simultaneous peer availability."""
    if max_peer_staleness <= timedelta(0):
        raise ValueError("max_peer_staleness must be positive")

    base_points = build_feature_points(bars)
    clean_indices = clean_feature_indices(bars, eligible_indices=eligible_indices)
    base_by_index: dict[int, FeaturePoint] = {point.bar_index: point for point in base_points}
    peer_symbols = sorted(symbol.upper() for symbol in peers)
    normalized_peers = {symbol.upper(): peers[symbol] for symbol in peers}
    peer_indices = {symbol: CausalPeerIndex.build(normalized_peers[symbol].bars) for symbol in peer_symbols}

    ret12_position = FEATURE_NAMES.index("ret_12_bps")
    vol48_position = FEATURE_NAMES.index("vol_48_bps")
    prior_ret12_abs: list[float] = []
    prior_vol48: list[float] = []
    result: list[V21FeaturePoint] = []

    for index in sorted(base_by_index):
        base = base_by_index[index]
        if index not in clean_indices:
            continue

        recent_ret = prior_ret12_abs[-REGIME_LOOKBACK:]
        recent_vol = prior_vol48[-REGIME_LOOKBACK:]
        current_abs_ret12 = abs(base.values[ret12_position])
        current_vol48 = base.values[vol48_position]
        regime = (
            _percentile_rank(current_vol48, recent_vol),
            _percentile_rank(current_abs_ret12, recent_ret),
            _trend_consistency_12(bars, index),
        )

        peer_values: list[float] = []
        for symbol in peer_symbols:
            peer = normalized_peers[symbol]
            peer_values.extend(
                _usable_peer_values(
                    peer_indices[symbol],
                    peer.eligible_indices,
                    base.timestamp,
                    max_staleness=max_peer_staleness,
                )
            )

        values = tuple(base.values) + regime + tuple(peer_values)
        expected = v21_feature_names(peer_symbols)
        if len(values) != len(expected):
            raise AssertionError("V2.1 feature vector length does not match feature names")
        if any(value != value or value in (float("inf"), float("-inf")) for value in values):
            raise ValueError(f"non-finite V2.1 feature at {base.timestamp.isoformat()}")
        result.append(V21FeaturePoint(index, base.timestamp, values))
        prior_ret12_abs.append(current_abs_ret12)
        prior_vol48.append(current_vol48)

    return result, clean_indices
