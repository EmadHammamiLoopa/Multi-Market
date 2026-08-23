import unittest
from datetime import datetime, timezone

from multimarket.v23_phase0b_acquisition_audit import _coverage_flags


class TestV23Phase0BRthCoverage(unittest.TestCase):
    def test_rth_first_and_last_bars_cover_frozen_trading_dates(self):
        first = datetime(2025, 8, 1, 13, 30, tzinfo=timezone.utc)
        last = datetime(2026, 8, 21, 19, 55, tzinfo=timezone.utc)
        self.assertEqual(_coverage_flags(first, last), (True, True))

    def test_late_start_or_early_end_fails(self):
        late_first = datetime(2025, 8, 4, 13, 30, tzinfo=timezone.utc)
        early_last = datetime(2026, 8, 20, 19, 55, tzinfo=timezone.utc)
        self.assertEqual(_coverage_flags(late_first, early_last), (False, False))


if __name__ == "__main__":
    unittest.main()
