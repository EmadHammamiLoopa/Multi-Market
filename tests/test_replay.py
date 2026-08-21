import unittest
from datetime import datetime, timedelta, timezone

from multimarket.models import Direction, MarketBar, Prediction
from multimarket.replay import ReplayConfig, ReplayEngine


class RecordingPredictor:
    def __init__(self):
        self.last_seen = []

    def predict(self, history, horizon_bars):
        self.last_seen.append(history[-1].timestamp)
        return Prediction(
            timestamp=history[-1].timestamp,
            direction=Direction.LONG,
            confidence=0.9,
            reference_price=history[-1].close,
            horizon_bars=horizon_bars,
        )


class ReplayLeakageTest(unittest.TestCase):
    def setUp(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        self.bars = []
        for i in range(12):
            price = 100 + i
            self.bars.append(MarketBar(start + timedelta(minutes=15 * i), price, price + 1, price - 1, price + 0.5))

    def test_predictor_only_sees_current_or_past_bars(self):
        predictor = RecordingPredictor()
        engine = ReplayEngine(predictor, ReplayConfig(horizon_bars=2, confidence_threshold=0.0))
        report = engine.run(self.bars, min_history_bars=3)
        expected_decision_times = [bar.timestamp for bar in self.bars[2:-2]]
        self.assertEqual(predictor.last_seen, expected_decision_times)
        self.assertEqual(len(report.predictions), len(expected_decision_times))

    def test_threshold_converts_weak_signal_to_no_trade(self):
        class WeakPredictor(RecordingPredictor):
            def predict(self, history, horizon_bars):
                return Prediction(history[-1].timestamp, Direction.LONG, 0.55, history[-1].close, horizon_bars)

        report = ReplayEngine(WeakPredictor(), ReplayConfig(horizon_bars=2, confidence_threshold=0.60)).run(self.bars, min_history_bars=3)
        self.assertTrue(all(p.direction is Direction.NO_TRADE for p in report.predictions))
        self.assertEqual(report.metrics.trades, 0)


if __name__ == "__main__":
    unittest.main()
