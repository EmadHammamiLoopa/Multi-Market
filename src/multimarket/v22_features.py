from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Collection, Mapping, Sequence

from .models import MarketBar
from .v21_features import PeerMarket, build_v21_feature_points, v21_feature_names
from .v22_macro import MacroLedgerIndex, build_macro_features, macro_feature_names


@dataclass(frozen=True, slots=True)
class V22FeaturePoint:
    bar_index: int
    timestamp: object
    values: tuple[float, ...]


def v22_feature_names(peer_symbols: Collection[str]) -> tuple[str, ...]:
    return v21_feature_names(peer_symbols) + macro_feature_names()


def build_v22_feature_points(
    bars: Sequence[MarketBar],
    *,
    eligible_indices: Collection[int],
    peers: Mapping[str, PeerMarket],
    macro_ledger: MacroLedgerIndex,
    max_peer_staleness: timedelta = timedelta(minutes=15),
) -> tuple[list[V22FeaturePoint], set[int]]:
    """Append frozen point-in-time macro packets to every causal V2.1 feature row.

    Missing macro context is represented by the preregistered masks and never
    deletes an otherwise eligible target row.
    """
    v21_points, clean_indices = build_v21_feature_points(
        bars,
        eligible_indices=eligible_indices,
        peers=peers,
        max_peer_staleness=max_peer_staleness,
    )
    expected_length = len(v22_feature_names(peers))
    result: list[V22FeaturePoint] = []
    for point in v21_points:
        macro = build_macro_features(macro_ledger, point.timestamp)
        values = tuple(point.values) + macro
        if len(values) != expected_length:
            raise AssertionError("V2.2 feature vector length does not match frozen feature names")
        if any(value != value or value in (float("inf"), float("-inf")) for value in values):
            raise ValueError(f"non-finite V2.2 feature at {point.timestamp.isoformat()}")
        result.append(V22FeaturePoint(point.bar_index, point.timestamp, values))
    return result, clean_indices
