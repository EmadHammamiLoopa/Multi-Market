from __future__ import annotations

import unittest
from datetime import date
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

from multimarket.v22_macro_fetch_realtime import fetch_alfred_changes


class V22MacroFetchRealtimeTests(unittest.TestCase):
    def test_uses_output_type_one_and_keeps_realtime_rows(self) -> None:
        payload = {
            "observations": [
                {
                    "realtime_start": "2024-01-11",
                    "realtime_end": "2024-02-12",
                    "date": "2023-12-01",
                    "value": "308.742",
                },
                {
                    "realtime_start": "2024-02-13",
                    "realtime_end": "9999-12-31",
                    "date": "2023-12-01",
                    "value": "308.744",
                },
            ]
        }
        captured: list[str] = []

        def fake_fetch(url: str):
            captured.append(url)
            return payload

        with patch("multimarket.v22_macro_fetch_realtime._base._fetch_json", fake_fetch):
            rows = fetch_alfred_changes(
                "CPIAUCSL",
                api_key="secret",
                realtime_start=date(2024, 1, 1),
                realtime_end=date(2024, 12, 31),
                observation_start=date(2023, 1, 1),
            )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["realtime_start"], "2024-01-11")
        query = parse_qs(urlparse(captured[0]).query)
        self.assertEqual(query["output_type"], ["1"])
        self.assertEqual(query["realtime_start"], ["2024-01-01"])
        self.assertEqual(query["realtime_end"], ["2024-12-31"])

    def test_filters_carry_in_rows_outside_frozen_realtime_window(self) -> None:
        payload = {
            "observations": [
                {"realtime_start": "2023-12-20", "date": "2023-11-01", "value": "1.0"},
                {"realtime_start": "2024-01-11", "date": "2023-12-01", "value": "2.0"},
                {"realtime_start": "2025-01-01", "date": "2024-12-01", "value": "3.0"},
            ]
        }
        with patch(
            "multimarket.v22_macro_fetch_realtime._base._fetch_json",
            return_value=payload,
        ):
            rows = fetch_alfred_changes(
                "CPIAUCSL",
                api_key="secret",
                realtime_start=date(2024, 1, 1),
                realtime_end=date(2024, 12, 31),
                observation_start=date(2023, 1, 1),
            )
        self.assertEqual(rows, [
            {"realtime_start": "2024-01-11", "date": "2023-12-01", "value": "2.0"}
        ])

    def test_deduplicates_identical_rows_and_skips_missing_values(self) -> None:
        payload = {
            "observations": [
                {"realtime_start": "2024-01-11", "date": "2023-12-01", "value": "2.0"},
                {"realtime_start": "2024-01-11", "date": "2023-12-01", "value": "2.0"},
                {"realtime_start": "2024-02-13", "date": "2024-01-01", "value": "."},
            ]
        }
        with patch(
            "multimarket.v22_macro_fetch_realtime._base._fetch_json",
            return_value=payload,
        ):
            rows = fetch_alfred_changes(
                "CPIAUCSL",
                api_key="secret",
                realtime_start=date(2024, 1, 1),
                realtime_end=date(2024, 12, 31),
                observation_start=date(2023, 1, 1),
            )
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
