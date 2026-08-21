import unittest
from datetime import datetime, timedelta, timezone

from multimarket.execution import simulate_barrier_trade
from multimarket.models import Direction, MarketBar, Prediction


class ExecutionTest(unittest.TestCase):
    def setUp(self):
        self.t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def test_long_take_profit_and_costs(self):
        prediction = Prediction(self.t0, Direction.LONG, 0.8, 100.0, 2)
        future = [MarketBar(self.t0 + timedelta(minutes=15), 100.0, 100.3, 99.95, 100.2)]
        trade = simulate_barrier_trade(prediction, future, take_profit_bps=20, stop_loss_bps=10, round_trip_cost_bps=2)
        self.assertEqual(trade.exit_reason, "TAKE_PROFIT")
        self.assertAlmostEqual(trade.net_return_bps, 18.0)
        self.assertTrue(trade.won)

    def test_ambiguous_bar_resolves_pessimistically(self):
        prediction = Prediction(self.t0, Direction.LONG, 0.8, 100.0, 1)
        future = [MarketBar(self.t0 + timedelta(minutes=15), 100.0, 100.3, 99.8, 100.1)]
        trade = simulate_barrier_trade(prediction, future, take_profit_bps=20, stop_loss_bps=10)
        self.assertEqual(trade.exit_reason, "AMBIGUOUS_BAR_STOP_FIRST")
        self.assertAlmostEqual(trade.gross_return_bps, -10.0)


if __name__ == "__main__":
    unittest.main()
