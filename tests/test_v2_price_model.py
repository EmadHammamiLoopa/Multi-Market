from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from multimarket.models import MarketBar
from multimarket.v2_features import feature_history_is_clean
from multimarket.v2_model import V2Config


UTC = timezone.utc


def _bar(ts: datetime, close: float = 100.0) -> MarketBar:
    return MarketBar(timestamp=ts, open=close, high=close + 0.1, low=close - 0.1, close=close)


class V2PriceModelTests(unittest.TestCase):
    def test_clean_feature_history_requires_full_contiguous_eligible_window(self) -> None:
        t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
        bars = [_bar(t0 + timedelta(minutes=5 * i)) for i in range(50)]
        eligible = set(range(50))
        self.assertTrue(feature_history_is_clean(bars, 48, eligible_indices=eligible))
        self.assertFalse(feature_history_is_clean(bars, 48, eligible_indices=eligible - {12}))

    def test_clean_feature_history_rejects_gap_inside_lookback(self) -> None:
        t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
        bars = [_bar(t0 + timedelta(minutes=5 * i)) for i in range(50)]
        bars[20] = _bar(bars[19].timestamp + timedelta(minutes=10))
        self.assertFalse(feature_history_is_clean(bars, 48, eligible_indices=set(range(50))))

    def test_frozen_v2_baseline_defaults(self) -> None:
        config = V2Config()
        self.assertEqual(config.horizon_bars, 6)
        self.assertEqual(config.volatility_span, 48)
        self.assertEqual(config.take_profit_vol_multiplier, 1.0)
        self.assertEqual(config.stop_loss_vol_multiplier, 1.0)
        self.assertEqual(config.round_trip_cost_bps, 2.0)
        self.assertEqual(config.min_positive_probability, 0.55)
        self.assertEqual(config.min_predicted_ev_bps, 0.0)
        self.assertEqual(config.embargo_bars, 6)

    def test_embargo_cannot_be_shorter_than_label_horizon(self) -> None:
        with self.assertRaises(ValueError):
            V2Config(horizon_bars=6, embargo_bars=5)


if __name__ == "__main__":
    unittest.main()
