import unittest
from datetime import datetime, timezone

from multimarket.data_quality import audit_bars
from multimarket.market_calendar import tradability_at
from multimarket.models import MarketBar


class MarketCalendarTests(unittest.TestCase):
    def test_crypto_is_eligible_on_weekend(self):
        stamp = datetime(2026, 6, 6, 12, 0, tzinfo=timezone.utc)  # Saturday
        self.assertTrue(tradability_at("BTCUSD", stamp).eligible)

    def test_fx_and_gold_reject_weekend(self):
        stamp = datetime(2026, 6, 6, 12, 0, tzinfo=timezone.utc)  # Saturday
        self.assertFalse(tradability_at("EURUSD", stamp).eligible)
        self.assertFalse(tradability_at("XAUUSD", stamp).eligible)

    def test_qqq_regular_hours_are_dst_aware(self):
        # June is EDT: 09:30 New York == 13:30 UTC.
        open_stamp = datetime(2026, 6, 1, 13, 30, tzinfo=timezone.utc)
        before_open = datetime(2026, 6, 1, 13, 25, tzinfo=timezone.utc)
        self.assertTrue(tradability_at("QQQ", open_stamp).eligible)
        self.assertFalse(tradability_at("QQQ", before_open).eligible)


class DataQualityTests(unittest.TestCase):
    def test_stale_and_tiny_bars_are_flagged_without_mutation(self):
        stamp = datetime(2026, 6, 6, 12, 0, tzinfo=timezone.utc)
        bars = [
            MarketBar(stamp, 100.0, 100.0, 100.0, 100.0),
            MarketBar(stamp.replace(minute=5), 100.0, 100.0, 100.0, 100.0),
        ]
        original = tuple(bars)
        rows = audit_bars(bars, symbol="EURUSD", tiny_range_bps=0.10)
        self.assertEqual(tuple(bars), original)
        self.assertFalse(rows[0].session_eligible)
        self.assertTrue(rows[0].zero_range)
        self.assertTrue(rows[0].tiny_range)
        self.assertTrue(rows[1].repeated_close)
        self.assertTrue(rows[1].repeated_ohlc)

    def test_normal_crypto_bar_not_session_rejected(self):
        stamp = datetime(2026, 6, 6, 12, 0, tzinfo=timezone.utc)
        bars = [MarketBar(stamp, 100.0, 101.0, 99.0, 100.5)]
        row = audit_bars(bars, symbol="BTCUSD")[0]
        self.assertTrue(row.session_eligible)
        self.assertFalse(row.zero_range)
        self.assertFalse(row.tiny_range)


if __name__ == "__main__":
    unittest.main()
