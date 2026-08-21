import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multimarket.fetcher import (
    build_twelve_url,
    fetch_symbol,
    fetch_symbol_auto,
    parse_chart,
    parse_twelve_time_series,
    resolve_symbol,
)


YAHOO_PAYLOAD = {
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

TWELVE_PAYLOAD = {
    "meta": {"symbol": "EUR/USD", "interval": "5min"},
    "values": [
        {
            "datetime": "2024-01-01 00:00:00",
            "open": "1.1000",
            "high": "1.1010",
            "low": "1.0990",
            "close": "1.1005",
        },
        {
            "datetime": "2024-01-01 00:05:00",
            "open": "1.1005",
            "high": "1.1020",
            "low": "1.1000",
            "close": "1.1015",
        },
    ],
    "status": "ok",
}


class FetcherTests(unittest.TestCase):
    def test_known_yahoo_symbol_mapping(self):
        self.assertEqual(resolve_symbol("EURUSD")[0], "EURUSD=X")
        self.assertEqual(resolve_symbol("XAUUSD")[0], "GC=F")
        self.assertEqual(resolve_symbol("NDX")[0], "^NDX")

    def test_known_twelve_symbol_mapping(self):
        self.assertEqual(resolve_symbol("EURUSD", "twelve-data")[0], "EUR/USD")
        self.assertEqual(resolve_symbol("XAUUSD", "twelve-data")[0], "XAU/USD")
        self.assertEqual(resolve_symbol("BTCUSD", "twelve-data")[0], "BTC/USD")

    def test_parse_chart_normalizes_utc_ohlc(self):
        rows = parse_chart(YAHOO_PAYLOAD)
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["timestamp"].endswith("Z"))
        self.assertEqual(rows[1]["close"], 2.5)

    def test_parse_twelve_time_series_normalizes_and_sorts(self):
        rows = parse_twelve_time_series(TWELVE_PAYLOAD)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["timestamp"], "2024-01-01T00:00:00Z")
        self.assertEqual(rows[1]["close"], 1.1015)

    def test_twelve_url_never_uses_yahoo_symbol(self):
        url = build_twelve_url("EUR/USD", "5m", 5000, "secret")
        self.assertIn("symbol=EUR%2FUSD", url)
        self.assertIn("interval=5min", url)
        self.assertIn("outputsize=5000", url)

    def test_fetch_symbol_writes_replay_compatible_yahoo_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = fetch_symbol(
                "BTCUSD",
                output_dir=tmp,
                getter=lambda _url: YAHOO_PAYLOAD,
            )
            self.assertEqual(result.provider, "yahoo")
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

    def test_fetch_symbol_reads_twelve_key_from_environment(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"TWELVE_DATA_API_KEY": "local-test-key"}
        ):
            result = fetch_symbol(
                "EURUSD",
                provider="twelve-data",
                interval="5m",
                output_dir=tmp,
                getter=lambda _url: TWELVE_PAYLOAD,
            )
            self.assertEqual(result.provider, "twelve-data")
            self.assertEqual(result.provider_symbol, "EUR/USD")
            self.assertEqual(result.rows, 2)

    def test_auto_falls_back_to_yahoo_when_twelve_fails(self):
        calls = []

        def getter(url):
            calls.append(url)
            if "twelvedata.com" in url:
                return {"status": "error", "message": "free-plan limit"}
            return YAHOO_PAYLOAD

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"TWELVE_DATA_API_KEY": "local-test-key"}
        ):
            result = fetch_symbol_auto(
                "EURUSD",
                interval="5m",
                range_="60d",
                output_dir=tmp,
                outputsize=5000,
                getter=getter,
            )
            self.assertEqual(result.provider, "yahoo")
            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
