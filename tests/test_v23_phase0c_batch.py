import unittest
from pathlib import Path

from multimarket.v23_phase0c_batch import target_command


class V23Phase0CBatchTests(unittest.TestCase):
    def test_btc_routes_core_and_canonical_peers_without_substitution(self):
        command = target_command(
            symbol="BTCUSD",
            data_dir=Path("data"),
            sensor_dir=Path("data/v23_phase0b_canonical"),
            output_dir=Path("evidence/v23/phase0c"),
        )
        rendered = " ".join(str(part) for part in command)
        self.assertIn("ETHUSD=data/ETHUSD_5m.csv", rendered)
        self.assertIn("QQQ=data/QQQ_5m.csv", rendered)
        self.assertIn("HYG=data/v23_phase0b_canonical/HYG_5m.csv", rendered)
        self.assertNotIn("UUP=", rendered)

    def test_xau_routes_exact_macro_sensor_set(self):
        command = target_command(
            symbol="XAUUSD",
            data_dir=Path("data"),
            sensor_dir=Path("data/v23_phase0b_canonical"),
            output_dir=Path("evidence/v23/phase0c"),
        )
        rendered = " ".join(str(part) for part in command)
        for peer in ("HYG", "TLT", "UUP"):
            self.assertIn(f"{peer}=data/v23_phase0b_canonical/{peer}_5m.csv", rendered)
        self.assertNotIn("GLD=", rendered)


if __name__ == "__main__":
    unittest.main()
