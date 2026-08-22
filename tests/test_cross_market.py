from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from multimarket.cross_market import causal_peer_snapshot
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


if __name__ == "__main__":
    unittest.main()
