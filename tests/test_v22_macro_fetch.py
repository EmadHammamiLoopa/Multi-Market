from __future__ import annotations

import unittest
from datetime import date, datetime, timezone

from multimarket.v22_macro_fetch import (
    normalize_ledger_rows,
    parse_bea_schedule,
    parse_bls_schedule,
)


class V22MacroFetchTests(unittest.TestCase):
    def test_parse_bls_schedule_extracts_cpi_and_employment(self) -> None:
        html = """
        <html><body>
        Friday, January 10, 2025 08:30 AM Employment Situation for December 2024
        Wednesday, January 15, 2025 08:30 AM Consumer Price Index for December 2024
        </body></html>
        """
        parsed = parse_bls_schedule(html, 2025)
        self.assertEqual(len(parsed["EMPLOYMENT"]), 1)
        self.assertEqual(len(parsed["CPI"]), 1)
        employment = parsed["EMPLOYMENT"][date(2025, 1, 10)]
        cpi = parsed["CPI"][date(2025, 1, 15)]
        self.assertEqual(employment.tzinfo, timezone.utc)
        self.assertEqual(cpi.tzinfo, timezone.utc)
        self.assertEqual(employment.hour, 13)
        self.assertEqual(employment.minute, 30)

    def test_parse_bea_schedule_preserves_nonstandard_release_time(self) -> None:
        html = """
        <html><body>
        January 22 10:00 AM News Personal Income and Outlays, October and November 2025 View
        February 20 8:30 AM News Personal Income and Outlays, December 2025 View
        </body></html>
        """
        parsed = parse_bea_schedule(html, 2026)
        january = parsed[date(2026, 1, 22)]
        february = parsed[date(2026, 2, 20)]
        self.assertEqual((january.hour, january.minute), (15, 0))
        self.assertEqual((february.hour, february.minute), (13, 30))

    def test_unverified_vintage_date_is_excluded_not_guessed(self) -> None:
        schedule = {
            "CPI": {date(2025, 1, 15): datetime(2025, 1, 15, 13, 30, tzinfo=timezone.utc)},
            "EMPLOYMENT": {},
            "PCE": {},
        }
        raw = {
            "CPIAUCSL": [
                {"realtime_start": "2025-01-15", "date": "2024-12-01", "value": "315.0"},
                {"realtime_start": "2025-01-16", "date": "2024-12-01", "value": "999.0"},
            ]
        }
        ledger, excluded = normalize_ledger_rows(raw, schedule)
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0].value, 315.0)
        self.assertEqual(excluded["CPIAUCSL"], 1)


if __name__ == "__main__":
    unittest.main()
