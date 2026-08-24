import tempfile
import unittest
import zipfile
from datetime import date
from pathlib import Path

from multimarket.v23_phase0dh_fetch import _days, archive_url, validate_zip


class V23Phase0DHFetchTests(unittest.TestCase):
    def test_frozen_daily_urls(self):
        day = date(2026, 8, 23)
        self.assertEqual(
            archive_url("aggTrades", "BTCUSDT", day),
            "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/"
            "BTCUSDT-aggTrades-2026-08-23.zip",
        )
        self.assertEqual(
            archive_url("bookTicker", "ETHUSDT", day),
            "https://data.binance.vision/data/futures/um/daily/bookTicker/ETHUSDT/"
            "ETHUSDT-bookTicker-2026-08-23.zip",
        )

    def test_date_range_is_inclusive(self):
        values = list(_days(date(2026, 5, 26), date(2026, 5, 28)))
        self.assertEqual(
            values,
            [date(2026, 5, 26), date(2026, 5, 27), date(2026, 5, 28)],
        )

    def test_zip_validation_detects_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("sample.csv", "a,b\n1,2\n")
            count, members = validate_zip(path)
            self.assertEqual(count, 1)
            self.assertEqual(members, ["sample.csv"])

    def test_zip_validation_rejects_empty_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.zip"
            with zipfile.ZipFile(path, "w"):
                pass
            with self.assertRaises(ValueError):
                validate_zip(path)


if __name__ == "__main__":
    unittest.main()
