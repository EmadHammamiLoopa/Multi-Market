from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from multimarket.features import FEATURE_NAMES
from multimarket.models import MarketBar
from multimarket.v21_features import (
    PEER_MISSING_STALENESS_SENTINEL,
    REGIME_FEATURE_NAMES,
    PeerMarket,
    build_v21_feature_points,
    v21_feature_names,
)


def _bars(count: int, *, offset_minutes: int = 0) -> list[MarketBar]:
    start = datetime(2026, 1, 5, tzinfo=timezone.utc) + timedelta(minutes=offset_minutes)
    result = []
    for i in range(count):
        close = 100.0 + i * 0.05 + (0.02 if i % 3 == 0 else 0.0)
        result.append(
            MarketBar(
                timestamp=start + timedelta(minutes=5 * i),
                open=close - 0.01,
                high=close + 0.02,
                low=close - 0.02,
                close=close,
            )
        )
    return result


class V21FeatureTests(unittest.TestCase):
    def test_feature_names_use_deterministic_peer_order(self) -> None:
        names = v21_feature_names({"QQQ", "BTCUSD"})
        self.assertEqual(names[: len(FEATURE_NAMES)], FEATURE_NAMES)
        self.assertEqual(
            names[len(FEATURE_NAMES) : len(FEATURE_NAMES) + len(REGIME_FEATURE_NAMES)],
            REGIME_FEATURE_NAMES,
        )
        peer_names = names[len(FEATURE_NAMES) + len(REGIME_FEATURE_NAMES) :]
        self.assertTrue(peer_names[0].startswith("BTCUSD_"))
        self.assertTrue(peer_names[6].startswith("QQQ_"))

    def test_unavailable_peer_is_masked_without_dropping_target_row(self) -> None:
        target = _bars(80)
        peer_bars = _bars(55)
        peers = {
            "QQQ": PeerMarket.build(peer_bars, eligible_indices=set(range(len(peer_bars))))
        }
        points, clean = build_v21_feature_points(
            target,
            eligible_indices=set(range(len(target))),
            peers=peers,
        )
        self.assertIn(79, clean)
        point = next(point for point in points if point.bar_index == 79)
        names = v21_feature_names(peers)
        mask_index = names.index("QQQ_available")
        stale_index = names.index("QQQ_staleness_fraction")
        self.assertEqual(point.values[mask_index], 0.0)
        self.assertEqual(point.values[stale_index], PEER_MISSING_STALENESS_SENTINEL)
        self.assertEqual(point.values[names.index("QQQ_ret_12_bps")], 0.0)

    def test_complete_peer_has_mask_and_bounded_staleness(self) -> None:
        target = _bars(80, offset_minutes=2)
        peer_bars = _bars(80)
        peers = {
            "BTCUSD": PeerMarket.build(peer_bars, eligible_indices=set(range(len(peer_bars))))
        }
        points, _ = build_v21_feature_points(
            target,
            eligible_indices=set(range(len(target))),
            peers=peers,
        )
        point = next(point for point in points if point.bar_index == 70)
        names = v21_feature_names(peers)
        self.assertEqual(point.values[names.index("BTCUSD_available")], 1.0)
        stale = point.values[names.index("BTCUSD_staleness_fraction")]
        self.assertGreaterEqual(stale, 0.0)
        self.assertLessEqual(stale, 1.0)

    def test_future_mutation_does_not_change_past_v21_features(self) -> None:
        target = _bars(90)
        peer_bars = _bars(90)
        peers = {
            "BTCUSD": PeerMarket.build(peer_bars, eligible_indices=set(range(len(peer_bars))))
        }
        before, _ = build_v21_feature_points(
            target,
            eligible_indices=set(range(len(target))),
            peers=peers,
        )
        before_point = next(point for point in before if point.bar_index == 70)

        mutated_target = list(target)
        future = mutated_target[85]
        mutated_target[85] = MarketBar(
            timestamp=future.timestamp,
            open=500.0,
            high=501.0,
            low=499.0,
            close=500.5,
        )
        mutated_peer = list(peer_bars)
        future_peer = mutated_peer[88]
        mutated_peer[88] = MarketBar(
            timestamp=future_peer.timestamp,
            open=700.0,
            high=701.0,
            low=699.0,
            close=700.5,
        )
        after, _ = build_v21_feature_points(
            mutated_target,
            eligible_indices=set(range(len(mutated_target))),
            peers={"BTCUSD": PeerMarket.build(mutated_peer, eligible_indices=set(range(len(mutated_peer))))},
        )
        after_point = next(point for point in after if point.bar_index == 70)
        self.assertEqual(before_point.values, after_point.values)

    def test_peer_hard_quality_failure_masks_entire_peer_snapshot(self) -> None:
        target = _bars(80)
        peer_bars = _bars(80)
        eligible = set(range(len(peer_bars)))
        eligible.remove(65)
        peers = {"BTCUSD": PeerMarket.build(peer_bars, eligible_indices=eligible)}
        points, _ = build_v21_feature_points(
            target,
            eligible_indices=set(range(len(target))),
            peers=peers,
        )
        point = next(point for point in points if point.bar_index == 70)
        names = v21_feature_names(peers)
        self.assertEqual(point.values[names.index("BTCUSD_available")], 0.0)
        self.assertEqual(point.values[names.index("BTCUSD_ret_1_bps")], 0.0)
        self.assertEqual(point.values[names.index("BTCUSD_vol_12_bps")], 0.0)


if __name__ == "__main__":
    unittest.main()
