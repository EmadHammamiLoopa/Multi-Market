from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from multimarket.models import Direction, MarketBar
from multimarket.v2_labels import build_economic_event, event_is_eligible, horizon_is_contiguous


UTC = timezone.utc


def _bar(ts: datetime, close: float, *, high: float | None = None, low: float | None = None) -> MarketBar:
    return MarketBar(
        timestamp=ts,
        open=close,
        high=high if high is not None else close,
        low=low if low is not None else close,
        close=close,
    )


class V2LabelTests(unittest.TestCase):
    def test_contiguous_horizon_rejects_gap(self) -> None:
        t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
        bars = [
            _bar(t0, 100.0),
            _bar(t0 + timedelta(minutes=5), 100.0),
            _bar(t0 + timedelta(minutes=15), 100.0),
        ]
        self.assertFalse(horizon_is_contiguous(bars, 0, 2))

    def test_event_requires_every_bar_to_be_eligible(self) -> None:
        t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
        bars = [_bar(t0 + timedelta(minutes=5 * i), 100.0) for i in range(4)]
        self.assertTrue(event_is_eligible(bars, 0, 3, eligible_indices={0, 1, 2, 3}))
        self.assertFalse(event_is_eligible(bars, 0, 3, eligible_indices={0, 1, 3}))

    def test_long_and_short_outcomes_include_costs(self) -> None:
        t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
        bars = [
            _bar(t0, 100.0),
            _bar(t0 + timedelta(minutes=5), 100.1, high=100.25, low=100.05),
            _bar(t0 + timedelta(minutes=10), 100.2, high=100.25, low=100.15),
        ]
        event = build_economic_event(
            bars,
            0,
            horizon_bars=2,
            take_profit_bps=20,
            stop_loss_bps=10,
            round_trip_cost_bps=2,
            eligible_indices={0, 1, 2},
        )
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.long.exit_reason, "TAKE_PROFIT")
        self.assertAlmostEqual(event.long.net_return_bps, 18.0)
        self.assertEqual(event.short.exit_reason, "STOP_LOSS")
        self.assertAlmostEqual(event.short.net_return_bps, -12.0)
        self.assertEqual(event.best_realized_direction, Direction.LONG)

    def test_same_bar_ambiguity_is_conservative_for_each_side(self) -> None:
        t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
        bars = [
            _bar(t0, 100.0),
            MarketBar(
                timestamp=t0 + timedelta(minutes=5),
                open=100.0,
                high=100.30,
                low=99.70,
                close=100.0,
            ),
        ]
        event = build_economic_event(
            bars,
            0,
            horizon_bars=1,
            take_profit_bps=20,
            stop_loss_bps=10,
            round_trip_cost_bps=2,
            eligible_indices={0, 1},
        )
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.long.exit_reason, "AMBIGUOUS_BAR_STOP_FIRST")
        self.assertEqual(event.short.exit_reason, "AMBIGUOUS_BAR_STOP_FIRST")
        self.assertAlmostEqual(event.long.net_return_bps, -12.0)
        self.assertAlmostEqual(event.short.net_return_bps, -12.0)
        self.assertEqual(event.best_realized_direction, Direction.NO_TRADE)


if __name__ == "__main__":
    unittest.main()
