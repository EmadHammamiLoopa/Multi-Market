import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError

from multimarket.backfill import RequestPacer, backfill_symbol, build_backfill_url


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

    def test_request_pacer_spaces_requests(self):
        clock = [100.0]
        sleeps = []

        def monotonic():
            return clock[0]

        def sleep(seconds):
            sleeps.append(seconds)
            clock[0] += seconds

        pacer = RequestPacer(8, monotonic_fn=monotonic, sleep_fn=sleep)
        pacer.before_request()
        clock[0] += 1.0
        pacer.before_request()
        self.assertEqual(len(sleeps), 1)
        self.assertAlmostEqual(sleeps[0], 6.5)

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
            self.assertEqual(len(lines), 4)
            state = Path(tmp) / ".backfill_state" / "EURUSD_5m.json"
            self.assertTrue(state.exists())

    def test_resume_skips_completed_chunks(self):
        first_payload = {
            "status": "ok",
            "values": [
                {"datetime": "2026-01-01 00:00:00", "open": "1", "high": "2", "low": "0.5", "close": "1.5"},
            ],
        }
        second_payload = {
            "status": "ok",
            "values": [
                {"datetime": "2026-01-02 00:00:00", "open": "1.5", "high": "2", "low": "1", "close": "1.6"},
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            calls = []

            def failing_getter(url):
                calls.append(url)
                if len(calls) == 1:
                    return first_payload
                raise RuntimeError("simulated credit interruption")

            with self.assertRaises(RuntimeError):
                backfill_symbol(
                    "EURUSD",
                    interval="5m",
                    start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    end=datetime(2026, 1, 3, tzinfo=timezone.utc),
                    chunk_days=1,
                    output_dir=tmp,
                    api_key="secret",
                    getter=failing_getter,
                )

            data_path = Path(tmp) / "EURUSD_5m.csv"
            self.assertTrue(data_path.exists())
            self.assertIn("2026-01-01T00:00:00Z", data_path.read_text(encoding="utf-8"))

            resume_calls = []

            def resume_getter(url):
                resume_calls.append(url)
                return second_payload

            backfill_symbol(
                "EURUSD",
                interval="5m",
                start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end=datetime(2026, 1, 3, tzinfo=timezone.utc),
                chunk_days=1,
                output_dir=tmp,
                api_key="secret",
                getter=resume_getter,
            )
            self.assertEqual(len(resume_calls), 1)
            text = data_path.read_text(encoding="utf-8")
            self.assertIn("2026-01-01T00:00:00Z", text)
            self.assertIn("2026-01-02T00:00:00Z", text)

    def test_transient_network_failure_retries_then_succeeds(self):
        payload = {
            "status": "ok",
            "values": [
                {"datetime": "2026-01-01 00:00:00", "open": "1", "high": "2", "low": "0.5", "close": "1.5"},
            ],
        }
        calls = []
        sleeps = []

        def flaky_getter(url):
            calls.append(url)
            if len(calls) < 3:
                raise URLError("Temporary failure in name resolution")
            return payload

        with tempfile.TemporaryDirectory() as tmp:
            path = backfill_symbol(
                "EURUSD",
                interval="5m",
                start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end=datetime(2026, 1, 2, tzinfo=timezone.utc),
                chunk_days=1,
                output_dir=tmp,
                api_key="secret",
                getter=flaky_getter,
                max_network_retries=5,
                retry_sleep_fn=sleeps.append,
            )
            self.assertTrue(Path(path).exists())
            self.assertEqual(len(calls), 3)
            self.assertEqual(sleeps, [5.0, 10.0])


if __name__ == "__main__":
    unittest.main()
