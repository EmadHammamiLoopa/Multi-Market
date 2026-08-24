import unittest
from datetime import date

from multimarket.v23_phase0dj_fetch import archive_url, _official_checksum


class Phase0DJStateTests(unittest.TestCase):
    def test_archive_url_matches_binance_futures_state_layout(self):
        url = archive_url("markPriceKlines", "BTCUSDT", date(2026, 8, 23))
        self.assertEqual(
            url,
            "https://data.binance.vision/data/futures/um/daily/markPriceKlines/BTCUSDT/1m/BTCUSDT-1m-2026-08-23.zip",
        )

    def test_checksum_parser_accepts_standard_sidecar(self):
        h = "a" * 64
        self.assertEqual(
            _official_checksum(f"{h}  BTCUSDT-1m-2026-08-23.zip\n", "BTCUSDT-1m-2026-08-23.zip"),
            h,
        )

    def test_checksum_parser_rejects_wrong_filename(self):
        with self.assertRaises(ValueError):
            _official_checksum("%s  wrong.zip\n" % ("b" * 64), "expected.zip")


if __name__ == "__main__":
    unittest.main()
