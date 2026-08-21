import unittest
from datetime import datetime, timedelta, timezone

from multimarket.models import Direction, MarketBar, Prediction
from multimarket.replay import ReplayConfig
from multimarket.time_machine import audit_at_index, evenly_spaced_indices


class SpyPredictor:
    def __init__(self):
        self.history_end = None
        self.history_len = None

    def predict(self, history, horizon_bars):
        self.history_end = history[-1].timestamp
        self.history_len = len(history)
        return Prediction(
            timestamp=history[-1].timestamp,
            direction=Direction.LONG,
            confidence=0.9,
            reference_price=history[-1].close,
            horizon_bars=horizon_bars,
        )


def make_bars(count=20):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = []
    price = 100.0
    for i in range(count):
        close = price + i
        bars.append(
            MarketBar(
                timestamp=start + timedelta(minutes=5 * i),
                open=close,
                high=close + 0.5,
                low=close - 0.5,
                close=close,
            )
        )
    return bars


class TimeMachineTests(unittest.TestCase):
    def test_predictor_sees_history_only(self):
        bars = make_bars(20)
        predictor = SpyPredictor()
        config = ReplayConfig(horizon_bars=3, confidence_threshold=0.6)
        row = audit_at_index(
            bars,
            10,
            predictor=predictor,
            config=config,
            min_history_bars=5,
        )
        self.assertEqual(predictor.history_len, 11)
        self.assertEqual(predictor.history_end, bars[10].timestamp)
        self.assertEqual(row.future_end_timestamp, bars[13].timestamp)
        self.assertEqual(row.verdict, "CORRECT")

    def test_evenly_spaced_selection_is_deterministic(self):
        first = evenly_spaced_indices(100, min_history_bars=10, horizon_bars=6, samples=5)
        second = evenly_spaced_indices(100, min_history_bars=10, horizon_bars=6, samples=5)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(first[0], 9)
        self.assertEqual(first[-1], 93)


if __name__ == "__main__":
    unittest.main()
