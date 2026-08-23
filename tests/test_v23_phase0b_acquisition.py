from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from multimarket.market_calendar import tradability_at
from multimarket.v23_phase0b_acquisition_audit import SENSORS, main


class V23Phase0BAcquisitionTests(unittest.TestCase):
    def test_expanded_etf_sensors_use_us_regular_hours(self):
        # 14:30 UTC is 09:30 New York during standard time.
        open_ts = datetime(2026, 1, 12, 14, 30, tzinfo=timezone.utc)
        before_ts = datetime(2026, 1, 12, 14, 25, tzinfo=timezone.utc)
        for symbol in ("SPY", "GLD", "XLK", "TLT", "UUP"):
            with self.subTest(symbol=symbol):
                self.assertTrue(tradability_at(symbol, open_ts).eligible)
                self.assertEqual(tradability_at(symbol, open_ts).reason, "us_regular_hours")
                self.assertFalse(tradability_at(symbol, before_ts).eligible)
                self.assertEqual(
                    tradability_at(symbol, before_ts).reason,
                    "outside_us_regular_hours",
                )

    def test_missing_any_frozen_sensor_blocks_phase0b_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "audit.json"
            code = main([
                "--data-dir", str(root),
                "--output-json", str(output),
            ])
            self.assertEqual(code, 1)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(payload["gate_pass"])
            self.assertEqual(payload["decision"], "BLOCK_PHASE0B_SCORING")
            self.assertEqual(payload["missing_sensors"], list(SENSORS))


if __name__ == "__main__":
    unittest.main()
