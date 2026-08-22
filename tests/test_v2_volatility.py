from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from multimarket.models import MarketBar
from multimarket.v2_volatility import (
    build_causal_ewma_volatility_bps,
    volatility_scaled_barriers_bps,
)


UTC = timezone.utc


def _bar(ts: datetime, close: float) -> MarketBar:
    return MarketBar(timestamp=ts, open=close, high=close, low=close, close=close)


class V2VolatilityTests(unittest.TestCase):
    def test_barrier_scaling_uses_sqrt_horizon(self) -> None:
        tp, sl = volatility_scaled_barriers_bps(
            2.0,
            horizon_bars=9,
            take_profit_vol_multiplier=1.5,
            stop_loss_vol_multiplier=0.5,
        )
        self.assertAlmostEqual(tp, 9.0)
        self.assertAlmostEqual(sl, 3.0)

    def test_gap_does_not_update_volatility(self) -> None:
        t0 = datetime(2026, 1, 2, 20, 0, tzinfo=UTC)
        bars = [
            _bar(t0, 100.0),
            _bar(t0 + timedelta(minutes=5), 100.1),
            _bar(t0 + timedelta(days=2), 120.0),
            _bar(t0 + timedelta(days=2, minutes=5), 120.12),
        ]
        vol = build_causal_ewma_volatility_bps(
            bars,
            span=4,
            min_observations=1,
        )
        self.assertIsNotNone(vol[1])
        self.assertEqual(vol[2], vol[1])
        self.assertIsNotNone(vol[3])
        assert vol[3] is not None
        self.assertLess(vol[3], 20.0)

    def test_ineligible_pair_does_not_update_state(self) -> None:
        t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
        bars = [_bar(t0 + timedelta(minutes=5 * i), 100.0 + i * 0.1) for i in range(4)]
        vol = build_causal_ewma_volatility_bps(
            bars,
            eligible_indices={0, 1, 3},
            span=4,
            min_observations=1,
        )
        self.assertIsNotNone(vol[1])
        self.assertEqual(vol[2], vol[1])
        self.assertEqual(vol[3], vol[1])

    def test_outputs_are_causal_under_future_change(self) -> None:
        t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
        first = [_bar(t0 + timedelta(minutes=5 * i), 100.0 + i * 0.1) for i in range(6)]
        second = list(first)
        second[-1] = _bar(t0 + timedelta(minutes=25), 150.0)
        vol_a = build_causal_ewma_volatility_bps(first, span=4, min_observations=1)
        vol_b = build_causal_ewma_volatility_bps(second, span=4, min_observations=1)
        self.assertEqual(vol_a[:5], vol_b[:5])


if __name__ == "__main__":
    unittest.main()
