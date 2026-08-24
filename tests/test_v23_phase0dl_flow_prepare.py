import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from multimarket import v23_phase0dl_prepare as book_prepare


class Phase0DLFlowPrepareTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "g++ not installed")
    def test_native_flow_extractor_compiles(self):
        root = book_prepare._repo_root()
        src = root / "tools" / "v23_phase0dl_flow250.cpp"
        self.assertTrue(src.exists())
        with tempfile.TemporaryDirectory() as td:
            exe = Path(td) / "v23_phase0dl_flow250"
            proc = subprocess.run(
                [
                    shutil.which("g++"),
                    "-std=c++17",
                    "-O2",
                    str(src),
                    "-lz",
                    "-o",
                    str(exe),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(exe.exists())


if __name__ == "__main__":
    unittest.main()
