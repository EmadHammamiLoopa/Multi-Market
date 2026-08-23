from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from multimarket.v23_phase0b_ohlc_canonicalize import canonicalize_file


class Phase0BCanonicalizationTests(unittest.TestCase):
    def _write(self, path: Path, row: list[str]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["timestamp", "open", "high", "low", "close"])
            writer.writerow(row)

    def test_small_boundary_violation_is_canonicalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "IWM_5m.csv"
            dst = root / "canonical" / "IWM_5m.csv"
            self._write(src, ["2025-11-28T14:35:00Z", "247.31", "247.7796", "247.089996", "247.78"])
            result = canonicalize_file(src, dst, "IWM")
            self.assertEqual(result["mutation_count"], 1)
            with dst.open("r", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(float(row["high"]), 247.78)
            self.assertEqual(float(row["low"]), 247.089996)

    def test_large_adjustment_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "BAD_5m.csv"
            dst = root / "canonical" / "BAD_5m.csv"
            self._write(src, ["2026-01-01T14:30:00Z", "100", "99", "98", "100"])
            with self.assertRaisesRegex(ValueError, "exceeds"):
                canonicalize_file(src, dst, "BAD")


if __name__ == "__main__":
    unittest.main()
