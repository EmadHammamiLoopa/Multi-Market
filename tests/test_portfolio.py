import unittest
from datetime import datetime, timedelta, timezone

from multimarket.models import Direction, Prediction, TradeResult
from multimarket.portfolio import calculate_portfolio_metrics, select_non_overlapping_trades


class PortfolioTests(unittest.TestCase):
    def _trade(self, minute, exit_minute, net_bps):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        prediction = Prediction(
            timestamp=start + timedelta(minutes=minute),
            direction=Direction.LONG,
            confidence=0.8,
            reference_price=100.0,
            horizon_bars=6,
        )
        return TradeResult(
            prediction=prediction,
            exit_timestamp=start + timedelta(minutes=exit_minute),
            exit_price=100.0,
            gross_return_bps=net_bps,
            net_return_bps=net_bps,
            won=net_bps > 0,
            exit_reason="TEST",
        )

    def test_overlapping_trade_is_skipped(self):
        trades = [
            self._trade(0, 30, 10),
            self._trade(5, 20, 50),
            self._trade(30, 60, -5),
        ]
        selected = select_non_overlapping_trades(trades)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0].net_return_bps, 10)
        self.assertEqual(selected[1].net_return_bps, -5)

    def test_portfolio_metrics_use_selected_trades_only(self):
        metrics = calculate_portfolio_metrics(
            [self._trade(0, 30, 10), self._trade(5, 20, 100), self._trade(30, 60, -5)]
        )
        self.assertEqual(metrics.trades, 2)
        self.assertAlmostEqual(metrics.expectancy_bps, 2.5)


if __name__ == "__main__":
    unittest.main()
