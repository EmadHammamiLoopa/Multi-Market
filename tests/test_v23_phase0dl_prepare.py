import shutil
import tempfile
import unittest
from datetime import date
from pathlib import Path

from multimarket import v23_phase0dl_prepare as p


class Phase0DLPrepareTests(unittest.TestCase):
    def test_development_days_exclude_confirmation(self):
        self.assertEqual(p.DEV_DAYS, tuple(date(2026, m, 1) for m in range(1, 8)))
        self.assertEqual(p.SEALED_CONFIRMATION_DAY, date(2026, 8, 1))
        self.assertNotIn(p.SEALED_CONFIRMATION_DAY, p.DEV_DAYS)

    def test_day_bounds_are_exact_utc_day(self):
        start, end = p._bounds_us(date(2026, 1, 1))
        self.assertEqual(end - start, 86_400_000_000)
        self.assertEqual(start % 1_000_000, 0)

    def test_grid_row_count_is_frozen(self):
        self.assertEqual(p.EXPECTED_ROWS, 345_600)
        self.assertEqual(86_400 * 4, p.EXPECTED_ROWS)

    @unittest.skipUnless(shutil.which("g++"), "g++ not installed")
    def test_native_reconstructor_compiles(self):
        with tempfile.TemporaryDirectory() as td:
            exe = p._build(p._repo_root(), Path(td))
            self.assertTrue(exe.exists())
            self.assertTrue(exe.is_file())


if __name__ == "__main__":
    unittest.main()
