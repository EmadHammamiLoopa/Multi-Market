from __future__ import annotations

import unittest
from datetime import date

from multimarket.v22_bls_schedule import frozen_bls_schedule


class V22FrozenBLSScheduleTests(unittest.TestCase):
    def test_expected_release_counts_for_frozen_window_years(self) -> None:
        schedule = frozen_bls_schedule(2024, 2026)
        self.assertEqual(len(schedule["CPI"]), 31)
        self.assertEqual(len(schedule["EMPLOYMENT"]), 31)

    def test_2025_lapse_exceptions_are_frozen(self) -> None:
        schedule = frozen_bls_schedule(2025, 2025)
        self.assertIn(date(2025, 10, 24), schedule["CPI"])
        self.assertNotIn(date(2025, 11, 13), schedule["CPI"])
        self.assertIn(date(2025, 11, 20), schedule["EMPLOYMENT"])
        self.assertNotIn(date(2025, 10, 3), schedule["EMPLOYMENT"])

    def test_release_time_is_0830_eastern_with_dst_conversion(self) -> None:
        schedule = frozen_bls_schedule(2024, 2024)
        january = schedule["CPI"][date(2024, 1, 11)]
        july = schedule["CPI"][date(2024, 7, 11)]
        self.assertEqual((january.hour, january.minute), (13, 30))
        self.assertEqual((july.hour, july.minute), (12, 30))

    def test_year_filter_does_not_leak_adjacent_years(self) -> None:
        schedule = frozen_bls_schedule(2026, 2026)
        self.assertTrue(schedule["CPI"])
        self.assertTrue(schedule["EMPLOYMENT"])
        self.assertTrue(all(day.year == 2026 for day in schedule["CPI"]))
        self.assertTrue(all(day.year == 2026 for day in schedule["EMPLOYMENT"]))


if __name__ == "__main__":
    unittest.main()
