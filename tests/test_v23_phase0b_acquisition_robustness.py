from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from multimarket.v23_phase0b_acquisition_audit import audit_sensor


class V23Phase0BAcquisitionRobustnessTests(unittest.TestCase):
    def test_invalid_ohlc_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "SPY_5m.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["timestamp", "open", "high", "low", "close"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "timestamp": "2025-08-01T13:30:00Z",
                        "open": "100",
                        "high": "99",
                        "low": "98",
                        "close": "100",
                    }
                )

            result = audit_sensor(path, "SPY")
            self.assertEqual(result.symbol, "SPY")
            self.assertEqual(result.status, "FAIL_INVALID_OHLC")
            self.assertIsNotNone(result.load_error)
            self.assertIn("Invalid row 2", result.load_error or "")
            self.assertFalse(result.covers_required_start)
            self.assertFalse(result.covers_required_end)
            self.assertFalse(result.enough_hard_eligible_bars)


if __name__ == "__main__":
    unittest.main()
