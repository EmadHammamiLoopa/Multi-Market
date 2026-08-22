from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from multimarket.cross_market import CausalPeerIndex, causal_peer_snapshot
from multimarket.models import MarketBar


def _bar(minute: int, close: float) -> MarketBar:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * minute)
    return MarketBar(timestamp=ts, open=close, high=close, low=close, close=close)


class CrossMarketAlignmentTests(unittest.TestCase):
    def test_never_uses_future_peer_bar(self) -> None:
        bars = [_bar(i, 100.0 + i) for i in range(20)]
        decision = bars[10].timestamp + timedelta(minutes=2)
        snapshot = causal_peer_snapshot(bars, decision, max_staleness=timedelta(minutes=15))
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.peer_index, 10)
        self.assertLessEqual(snapshot.peer_timestamp, decision)

    def test_rejects_peer_data_beyond_staleness_limit(self) -> None:
        bars = [_bar(i, 100.0 + i) for i in range(20)]
        decision = bars[-1].timestamp + timedelta(minutes=16)
        snapshot = causal_peer_snapshot(bars, decision, max_staleness=timedelta(minutes=15))
        self.assertIsNone(snapshot)

    def test_exact_timestamp_is_allowed(self) -> None:
        bars = [_bar(i, 100.0 + i) for i in range(20)]
        snapshot = causal_peer_snapshot(bars, bars[15].timestamp, max_staleness=timedelta(minutes=0))
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.peer_index, 15)
        self.assertEqual(snapshot.staleness, timedelta(0))
        self.assertIsNotNone(snapshot.ret_12_bps)
        self.assertIsNotNone(snapshot.vol_12_bps)

    def test_future_mutation_does_not_change_snapshot(self) -> None:
        bars = [_bar(i, 100.0 + i) for i in range(20)]
        decision = bars[15].timestamp
        before = causal_peer_snapshot(bars, decision)
        mutated = list(bars)
        mutated[19] = _bar(19, 9999.0)
        after = causal_peer_snapshot(mutated, decision)
        self.assertEqual(before, after)

    def test_reusable_index_matches_single_lookup_helper(self) -> None:
        bars = [_bar(i, 100.0 + i) for i in range(30)]
        index = CausalPeerIndex.build(bars)
        for minute in (12, 15, 20, 29):
            decision = bars[minute].timestamp + timedelta(minutes=2)
            expected = causal_peer_snapshot(bars, decision, max_staleness=timedelta(minutes=15))
            actual = index.snapshot(decision, max_staleness=timedelta(minutes=15))
            self.assertEqual(actual, expected)

    def test_reusable_index_copies_source_sequence(self) -> None:
        bars = [_bar(i, 100.0 + i) for i in range(20)]
        index = CausalPeerIndex.build(bars)
        decision = bars[15].timestamp
        before = index.snapshot(decision)
        bars[15] = _bar(15, 9999.0)
        after = index.snapshot(decision)
        self.assertEqual(before, after)

    def test_gap_inside_peer_history_blocks_affected_features(self) -> None:
        bars = [_bar(i, 100.0 + i) for i in range(20)]
        del bars[8]
        snapshot = causal_peer_snapshot(bars, bars[13].timestamp, max_staleness=timedelta(minutes=0))
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertIsNotNone(snapshot.ret_1_bps)
        self.assertIsNone(snapshot.ret_6_bps)
        self.assertIsNone(snapshot.ret_12_bps)
        self.assertIsNone(snapshot.vol_12_bps)

    def test_gap_before_required_window_does_not_block_features(self) -> None:
        bars = [_bar(i, 100.0 + i) for i in range(25)]
        del bars[2]
        snapshot = causal_peer_snapshot(bars, bars[-1].timestamp, max_staleness=timedelta(minutes=0))
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertIsNotNone(snapshot.ret_1_bps)
        self.assertIsNotNone(snapshot.ret_6_bps)
        self.assertIsNotNone(snapshot.ret_12_bps)
        self.assertIsNotNone(snapshot.vol_12_bps)

    def test_custom_expected_interval_is_respected(self) -> None:
        bars = [_bar(i * 3, 100.0 + i) for i in range(20)]
        default_snapshot = causal_peer_snapshot(bars, bars[-1].timestamp, max_staleness=timedelta(minutes=0))
        custom_snapshot = causal_peer_snapshot(
            bars,
            bars[-1].timestamp,
            max_staleness=timedelta(minutes=0),
            expected_interval=timedelta(minutes=15),
        )
        self.assertIsNotNone(default_snapshot)
        self.assertIsNotNone(custom_snapshot)
        assert default_snapshot is not None and custom_snapshot is not None
        self.assertIsNone(default_snapshot.ret_1_bps)
        self.assertIsNotNone(custom_snapshot.ret_12_bps)
        self.assertIsNotNone(custom_snapshot.vol_12_bps)

    def test_ineligible_peer_bar_inside_window_blocks_affected_features(self) -> None:
        bars = [_bar(i, 100.0 + i) for i in range(25)]
        eligible = set(range(len(bars)))
        eligible.remove(18)
        snapshot = causal_peer_snapshot(
            bars,
            bars[24].timestamp,
            max_staleness=timedelta(minutes=0),
            eligible_indices=eligible,
        )
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertIsNotNone(snapshot.ret_1_bps)
        self.assertIsNone(snapshot.ret_6_bps)
        self.assertIsNone(snapshot.ret_12_bps)
        self.assertIsNone(snapshot.vol_12_bps)

    def test_ineligible_peer_bar_before_window_does_not_block_features(self) -> None:
        bars = [_bar(i, 100.0 + i) for i in range(25)]
        eligible = set(range(len(bars)))
        eligible.remove(5)
        snapshot = causal_peer_snapshot(
            bars,
            bars[24].timestamp,
            max_staleness=timedelta(minutes=0),
            eligible_indices=eligible,
        )
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertIsNotNone(snapshot.ret_1_bps)
        self.assertIsNotNone(snapshot.ret_6_bps)
        self.assertIsNotNone(snapshot.ret_12_bps)
        self.assertIsNotNone(snapshot.vol_12_bps)


if __name__ == "__main__":
    unittest.main()
