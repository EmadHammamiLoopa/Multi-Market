import unittest
from datetime import datetime, timezone

from multimarket.metrics import calculate_metrics
from multimarket.models import Direction, Prediction, TradeResult


class MetricsTest(unittest.TestCase):
    def test_metrics_use_net_trade_returns(self):
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        predictions = [
            Prediction(now, Direction.LONG, 0.8, 100, 1),
            Prediction(now, Direction.SHORT, 0.8, 100, 1),
        ]
        trades = [
            TradeResult(predictions[0], now, 101, 20, 18, True, "TAKE_PROFIT"),
            TradeResult(predictions[1], now, 101, -10, -12, False, "STOP_LOSS"),
        ]
        metrics = calculate_metrics(predictions, [True, False], trades)
        self.assertAlmostEqual(metrics.directional_accuracy, 0.5)
        self.assertAlmostEqual(metrics.win_rate, 0.5)
        self.assertAlmostEqual(metrics.profit_factor, 1.5)
        self.assertAlmostEqual(metrics.expectancy_bps, 3.0)
        self.assertGreater(metrics.net_return_pct, 0)
        self.assertGreater(metrics.max_drawdown_pct, 0)


if __name__ == "__main__":
    unittest.main()
