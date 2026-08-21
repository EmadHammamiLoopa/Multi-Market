import csv
import tempfile
import unittest
from pathlib import Path

from multimarket.fetcher import fetch_symbol, parse_chart, resolve_symbol


PAYLOAD = {
    "chart": {
        "result": [
            {
                "timestamp": [1704067200, 1704068100],
                "indicators": {
                    "quote": [
                        {
                            "open": [1.0, 2.0],
                            "high": [2.0, 3.0],
                            "low": [0.5, 1.5],
                            "close": [1.5, 2.5],
                        }
                    ]
                },
            }
        ],
        "error": None,
    }
}


class FetcherTests(unittest.TestCase):
    def test_known_symbol_mapping(self):
        self.assertEqual(resolve_symbol("EURUSD")[0], "EURUSD=X")
        self.assertEqual(resolve_symbol("XAUUSD")[0], "GC=F")
        self.assertEqual(resolve_symbol("NDX")[0], "^NDX")

    def test_parse_chart_normalizes_utc_ohlc(self):
        rows = parse_chart(PAYLOAD)
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["timestamp"].endswith("Z"))
        self.assertEqual(rows[1]["close"], 2.5)

    def test_fetch_symbol_writes_replay_compatible_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = fetch_symbol(
                "BTCUSD",
                output_dir=tmp,
                getter=lambda _url: PAYLOAD,
            )
            self.assertEqual(result.provider_symbol, "BTC-USD")
            self.assertEqual(result.rows, 2)
            self.assertTrue(Path(result.output_path).exists())

            with Path(result.output_path).open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                list(rows[0].keys()),
                ["timestamp", "open", "high", "low", "close"],
            )


if __name__ == "__main__":
    unittest.main()
