from __future__ import annotations

import unittest
from datetime import datetime, timezone

from multimarket.models import Direction, Prediction, TradeResult
from multimarket.v1_research import confidence_buckets, threshold_predictions
from multimarket.version_compare import compare_time_machine


class V1ResearchTests(unittest.TestCase):
    def _prediction(self, minute: int, direction: Direction, confidence: float) -> Prediction:
        return Prediction(
            timestamp=datetime(2026, 1, 1, 0, minute, tzinfo=timezone.utc),
            direction=direction,
            confidence=confidence,
            reference_price=100.0,
            horizon_bars=6,
        )

    def test_threshold_predictions_turns_low_confidence_into_no_trade(self):
        p1 = self._prediction(0, Direction.LONG, 0.58)
        p2 = self._prediction(5, Direction.SHORT, 0.72)
        t1 = TradeResult(p1, datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc), 101.0, 10, 8, True, "HORIZON")
        t2 = TradeResult(p2, datetime(2026, 1, 1, 0, 35, tzinfo=timezone.utc), 99.0, 10, 8, True, "HORIZON")

        predictions, hits, trades = threshold_predictions([p1, p2], [True, True], [t1, t2], 0.60)
        self.assertEqual(predictions[0].direction, Direction.NO_TRADE)
        self.assertEqual(predictions[1].direction, Direction.SHORT)
        self.assertEqual(hits, [False, True])
        self.assertEqual(trades, [t2])

    def test_confidence_buckets_cover_predictions(self):
        predictions = [
            self._prediction(0, Direction.LONG, 0.52),
            self._prediction(5, Direction.LONG, 0.57),
            self._prediction(10, Direction.SHORT, 0.83),
        ]
        buckets = confidence_buckets(predictions, [True, False, True])
        self.assertEqual(sum(row["count"] for row in buckets), 3)

    def test_paired_comparison_uses_shared_timestamps(self):
        v0 = {
            "rows": [
                {"decision_timestamp": "2026-01-01T00:00:00Z", "verdict": "WRONG", "prediction_direction": "LONG"},
                {"decision_timestamp": "2026-01-01T00:05:00Z", "verdict": "CORRECT", "prediction_direction": "SHORT"},
            ]
        }
        v1 = {
            "rows": [
                {"decision_timestamp": "2026-01-01T00:00:00Z", "verdict": "CORRECT", "prediction_direction": "SHORT", "actual_direction": "SHORT"},
                {"decision_timestamp": "2026-01-01T00:05:00Z", "verdict": "NO_TRADE", "prediction_direction": "NO_TRADE", "actual_direction": "SHORT"},
            ]
        }
        result = compare_time_machine(v0, v1)
        self.assertEqual(result["shared_timestamps"], 2)
        self.assertEqual(result["transitions"]["WRONG->CORRECT"], 1)
        self.assertEqual(result["transitions"]["CORRECT->NO_TRADE"], 1)
        self.assertEqual(result["v1"]["selected_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
