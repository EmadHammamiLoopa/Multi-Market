import gzip
import json
import tempfile
import unittest
from pathlib import Path

from multimarket.v23_phase0d_book import DepthSequenceError, LocalOrderBook
from multimarket.v23_phase0d_collect import (
    DailyJsonlWriter,
    TradeBucket,
    _build_stream_urls,
    _classify_ws_event,
)


class V23Phase0DOrderBookTests(unittest.TestCase):
    def _snapshot(self):
        return {
            "lastUpdateId": 100,
            "bids": [["100.0", "2.0"], ["99.5", "3.0"]],
            "asks": [["100.5", "4.0"], ["101.0", "5.0"]],
        }

    def test_snapshot_requires_bridge_before_valid(self):
        book = LocalOrderBook()
        book.load_snapshot(self._snapshot())
        self.assertFalse(book.valid)
        bridged = book.bridge({
            "U": 99,
            "u": 101,
            "pu": 98,
            "b": [["100.0", "2.5"]],
            "a": [["100.5", "3.5"]],
        })
        self.assertTrue(bridged)
        self.assertTrue(book.valid)
        self.assertEqual(book.last_update_id, 101)

    def test_future_or_nonbridging_event_is_not_accepted(self):
        book = LocalOrderBook()
        book.load_snapshot(self._snapshot())
        self.assertFalse(book.bridge({
            "U": 102,
            "u": 103,
            "pu": 101,
            "b": [],
            "a": [],
        }))
        self.assertFalse(book.valid)

    def test_continuity_uses_previous_final_update_id(self):
        book = LocalOrderBook()
        book.load_snapshot(self._snapshot())
        self.assertTrue(book.bridge({
            "U": 100, "u": 101, "pu": 99,
            "b": [], "a": [],
        }))
        book.apply_diff({
            "U": 102, "u": 103, "pu": 101,
            "b": [["99.5", "0"]],
            "a": [["101.0", "6"]],
        })
        self.assertEqual(book.last_update_id, 103)
        self.assertNotIn(99.5, book.bids)
        self.assertEqual(book.asks[101.0], 6.0)

    def test_gap_invalidates_book(self):
        book = LocalOrderBook()
        book.load_snapshot(self._snapshot())
        self.assertTrue(book.bridge({
            "U": 100, "u": 101, "pu": 99,
            "b": [], "a": [],
        }))
        with self.assertRaises(DepthSequenceError):
            book.apply_diff({
                "U": 105, "u": 106, "pu": 104,
                "b": [], "a": [],
            })
        self.assertFalse(book.valid)

    def test_snapshot_metrics_are_causal_book_state(self):
        book = LocalOrderBook()
        book.load_snapshot(self._snapshot())
        self.assertTrue(book.bridge({
            "U": 100, "u": 101, "pu": 99,
            "b": [], "a": [],
        }))
        m = book.snapshot_metrics()
        self.assertEqual(m["best_bid"], 100.0)
        self.assertEqual(m["best_ask"], 100.5)
        self.assertGreater(m["spread_bps"], 0.0)
        self.assertTrue(m["depth_sequence_valid"])
        self.assertEqual(m["last_depth_update_id"], 101)


class V23Phase0DTradeTests(unittest.TestCase):
    def test_exchange_maker_flag_classifies_aggressor_side(self):
        bucket = TradeBucket()
        bucket.add_agg_trade({"q": "2.0", "m": False})  # aggressive buy
        bucket.add_agg_trade({"q": "1.0", "m": True})   # aggressive sell
        result = bucket.consume()
        self.assertEqual(result["agg_buy_qty_1s"], 2.0)
        self.assertEqual(result["agg_sell_qty_1s"], 1.0)
        self.assertAlmostEqual(result["trade_flow_imbalance_1s"], 1.0 / 3.0)
        self.assertEqual(bucket.buy_qty, 0.0)
        self.assertEqual(bucket.sell_qty, 0.0)

    def test_aggtrade_routing_is_case_insensitive(self):
        payload = {"e": "aggTrade", "s": "BTCUSDT", "q": "1.0", "m": False}
        self.assertEqual(_classify_ws_event("btcusdt@aggTrade", payload), "agg_trade")
        self.assertEqual(_classify_ws_event("btcusdt@aggtrade", payload), "agg_trade")

    def test_depth_routing_prefers_payload_event_type(self):
        payload = {"e": "depthUpdate", "s": "BTCUSDT"}
        self.assertEqual(_classify_ws_event("unexpected-name", payload), "depth")

    def test_stream_namespaces_follow_binance_migration(self):
        public_url, market_url = _build_stream_urls(("BTCUSDT", "ETHUSDT"))
        self.assertIn("/public/stream?streams=", public_url)
        self.assertIn("btcusdt@depth@100ms", public_url)
        self.assertIn("btcusdt@bookTicker", public_url)
        self.assertNotIn("aggTrade", public_url)
        self.assertIn("/market/stream?streams=", market_url)
        self.assertIn("btcusdt@aggTrade", market_url)
        self.assertIn("ethusdt@aggTrade", market_url)
        self.assertNotIn("@depth", market_url)


class V23Phase0DRawCaptureTests(unittest.TestCase):
    def test_raw_writer_uses_lossless_gzip_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = DailyJsonlWriter(root, "BTCUSDT")
            record = {
                "receive_time_utc": "2026-08-24T11:00:00+00:00",
                "receive_time_ns": 123,
                "stream": "btcusdt@aggTrade",
                "symbol": "BTCUSDT",
                "exchange_event_time_ms": 456,
                "payload": {"e": "aggTrade", "q": "1.25", "m": False},
            }
            writer.write(record)
            writer.close()
            files = list((root / "raw" / "BTCUSDT").glob("*.jsonl.gz"))
            self.assertEqual(len(files), 1)
            with gzip.open(files[0], "rt", encoding="utf-8") as handle:
                restored = json.loads(handle.readline())
            self.assertEqual(restored, record)


if __name__ == "__main__":
    unittest.main()
