import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

from multimarket.v23_phase0dh_tf import _bool, _imb, iter_day_trades


class Phase0DHTFTests(unittest.TestCase):
    def test_imbalance_definition(self):
        self.assertAlmostEqual(_imb(3.0, 1.0), 0.5)
        self.assertAlmostEqual(_imb(1.0, 3.0), -0.5)
        self.assertEqual(_imb(0.0, 0.0), 0.0)

    def test_buyer_maker_boolean(self):
        self.assertTrue(_bool("true"))
        self.assertTrue(_bool("1"))
        self.assertFalse(_bool("false"))
        self.assertFalse(_bool("0"))

    def test_archive_parser_preserves_exchange_time_order_and_side(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "BTCUSDT-aggTrades-2026-05-26.zip"
            csv_name = "BTCUSDT-aggTrades-2026-05-26.csv"
            rows = [
                [1, "100.0", "2.0", 10, 10, 1000, "true"],
                [2, "101.0", "1.5", 11, 11, 1500, "false"],
            ]
            with zipfile.ZipFile(path, "w") as zf:
                import io
                buf = io.StringIO()
                writer = csv.writer(buf)
                writer.writerow([
                    "agg_trade_id", "price", "quantity", "first_trade_id",
                    "last_trade_id", "transact_time", "is_buyer_maker",
                ])
                writer.writerows(rows)
                zf.writestr(csv_name, buf.getvalue())

            parsed = list(iter_day_trades(path))
            self.assertEqual(parsed[0], (1000, 100.0, 2.0, True))
            self.assertEqual(parsed[1], (1500, 101.0, 1.5, False))

    def test_archive_parser_rejects_non_monotonic_exchange_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "BTCUSDT-aggTrades-2026-05-26.zip"
            csv_name = "BTCUSDT-aggTrades-2026-05-26.csv"
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr(
                    csv_name,
                    "price,quantity,transact_time,is_buyer_maker\n"
                    "100,1,2000,true\n"
                    "101,1,1000,false\n",
                )
            with self.assertRaises(ValueError):
                list(iter_day_trades(path))


if __name__ == "__main__":
    unittest.main()
