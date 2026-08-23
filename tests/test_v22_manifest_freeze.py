from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from multimarket.v22_manifest_freeze import (
    SAMPLE_COUNT,
    WINDOWS,
    build_manifest,
    evenly_spaced_positions,
    select_timestamps,
)


class V22ManifestFreezeTests(unittest.TestCase):
    def test_evenly_spaced_positions_are_deterministic_and_endpoint_inclusive(self) -> None:
        positions = evenly_spaced_positions(100, 30)
        self.assertEqual(len(positions), 30)
        self.assertEqual(len(set(positions)), 30)
        self.assertEqual(positions[0], 0)
        self.assertEqual(positions[-1], 99)
        self.assertEqual(positions, evenly_spaced_positions(100, 30))

    def test_evenly_spaced_positions_reject_too_few_candidates(self) -> None:
        with self.assertRaises(ValueError):
            evenly_spaced_positions(29, SAMPLE_COUNT)

    def test_select_timestamps_uses_only_timestamps(self) -> None:
        start = datetime(2026, 1, 5, tzinfo=timezone.utc)
        candidates = [start + timedelta(minutes=5 * i) for i in range(100)]
        selected = select_timestamps(reversed(candidates))
        self.assertEqual(len(selected), 30)
        self.assertEqual(selected[0], candidates[0])
        self.assertEqual(selected[-1], candidates[-1])
        self.assertEqual(tuple(sorted(selected)), selected)

    def test_manifest_contains_no_outcome_or_price_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "EURUSD_5m.csv"
            source.write_text(
                "timestamp,open,high,low,close\n"
                "2026-01-05T00:00:00Z,1,1,1,1\n",
                encoding="utf-8",
            )
            start, _ = WINDOWS["E"]
            candidates = [start + timedelta(minutes=5 * i) for i in range(100)]
            manifest = build_manifest(
                source_csv=source,
                symbol="EURUSD",
                window_label="E",
                candidate_timestamps=candidates,
            )
            self.assertEqual(manifest["sample_count"], 30)
            self.assertEqual(len(manifest["rows"]), 30)
            forbidden = {
                "open", "high", "low", "close", "return", "returns",
                "label", "prediction", "probability", "ev", "outcome",
                "realized_net_bps", "verdict",
            }
            serialized = json.dumps(manifest).lower()
            for field in forbidden:
                self.assertNotIn(f'"{field}"', serialized)
            self.assertEqual(set(manifest["rows"][0]), {"decision_timestamp"})


if __name__ == "__main__":
    unittest.main()
