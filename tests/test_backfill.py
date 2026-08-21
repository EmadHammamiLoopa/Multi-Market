import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from multimarket.backfill import backfill_symbol, build_backfill_url


class BackfillTests(unittest.TestCase):
    def test_url_uses_bounded_dates_without_outputsize(self):
        url = build_backfill_url(
            "EUR/USD",
            "5m",
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 15, tzinfo=timezone.utc),
            "secret",
        )
        self.assertIn("start_date=2026-01-01T00%3A00%3A00", url)
        self.assertIn("end_date=2026-01-15T00%3A00%3A00", url)
        self.assertNotIn("outputsize", url)

    def test_chunks_are_merged_and_deduplicated(self):
        payloads = [
            {
                "status": "ok",
                "values": [
                    {"datetime": "2026-01-01 00:00:00", "open": "1", "high": "2", "low": "0.5", "close": "1.5"},
                    {"datetime": "2026-01-01 00:05:00", "open": "1.5", "high": "2", "low": "1", "close": "1.6"},
                ],
            },
            {
                "status": "ok",
                "values": [
                    {"datetime": "2026-01-01 00:05:00", "open": "1.5", "high": "2", "low": "1", "close": "1.6"},
                    {"datetime": "2026-01-01 00:10:00", "open": "1.6", "high": "2", "low": "1", "close": "1.7"},
                ],
            },
        ]

        def getter(_url):
            return payloads.pop(0)

        with tempfile.TemporaryDirectory() as tmp:
            path = backfill_symbol(
                "EURUSD",
                interval="5m",
                start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end=datetime(2026, 1, 3, tzinfo=timezone.utc),
                chunk_days=1,
                output_dir=tmp,
                api_key="secret",
                getter=getter,
            )
            lines = Path(path).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 4)  # header + three unique rows


if __name__ == "__main__":
    unittest.main()
