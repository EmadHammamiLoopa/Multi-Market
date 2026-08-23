from __future__ import annotations

import unittest
from datetime import date

from multimarket.v22_bea_schedule import frozen_bea_pce_schedule


class V22FrozenBeaScheduleTests(unittest.TestCase):
    def test_counts_match_frozen_evidence_interval(self) -> None:
        schedule = frozen_bea_pce_schedule(2024, 2026)
        self.assertEqual(len(schedule), 31)
        by_year = {year: 0 for year in (2024, 2025, 2026)}
        for day in schedule:
            by_year[day.year] += 1
        self.assertEqual(by_year, {2024: 12, 2025: 11, 2026: 8})

    def test_special_release_times_are_preserved(self) -> None:
        schedule = frozen_bea_pce_schedule(2025, 2026)

        # March 2025 PIO: April 30, 2025 at 10:00 EDT = 14:00Z.
        april = schedule[date(2025, 4, 30)]
        self.assertEqual((april.hour, april.minute), (14, 0))

        # September 2025 delayed release: Dec 5 at 10:00 EST = 15:00Z.
        december = schedule[date(2025, 12, 5)]
        self.assertEqual((december.hour, december.minute), (15, 0))

        # Oct/Nov 2025 combined release: Jan 22 at 10:00 EST = 15:00Z.
        january = schedule[date(2026, 1, 22)]
        self.assertEqual((january.hour, january.minute), (15, 0))

    def test_normal_release_respects_dst(self) -> None:
        schedule = frozen_bea_pce_schedule(2024, 2024)
        winter = schedule[date(2024, 1, 26)]
        summer = schedule[date(2024, 7, 26)]
        self.assertEqual((winter.hour, winter.minute), (13, 30))
        self.assertEqual((summer.hour, summer.minute), (12, 30))


if __name__ == "__main__":
    unittest.main()
