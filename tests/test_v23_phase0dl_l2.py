import unittest
from datetime import date

from multimarket.v23_phase0dl_fetch import Item, SAMPLE_DAYS
from multimarket.v23_phase0dl_audit import L2_HEADER, TRADES_HEADER, DEV_DAYS


class Phase0DLL2Tests(unittest.TestCase):
    def test_frozen_days_exact(self):
        self.assertEqual(
            SAMPLE_DAYS,
            tuple(date(2026, m, 1) for m in range(1, 9)),
        )
        self.assertEqual(
            DEV_DAYS,
            tuple(date(2026, m, 1) for m in range(1, 8)),
        )

    def test_tardis_url_is_exact(self):
        x = Item(date(2026, 3, 1), "BTCUSDT", "incremental_book_L2")
        self.assertEqual(
            x.url,
            "https://datasets.tardis.dev/v1/binance-futures/"
            "incremental_book_L2/2026/03/01/BTCUSDT.csv.gz",
        )

    def test_frozen_headers_match_tardis_normalized_schema(self):
        self.assertEqual(
            L2_HEADER,
            ("exchange","symbol","timestamp","local_timestamp","is_snapshot","side","price","amount"),
        )
        self.assertEqual(
            TRADES_HEADER,
            ("exchange","symbol","timestamp","local_timestamp","id","side","price","amount"),
        )


if __name__ == "__main__":
    unittest.main()
