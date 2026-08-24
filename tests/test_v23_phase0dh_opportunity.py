import unittest

import numpy as np

from multimarket.v23_phase0dh_opportunity import (
    _greedy_trade_indices,
    _max_drawdown,
    _opportunity_target,
)


class Phase0DHOpportunityTests(unittest.TestCase):
    def test_cost_excess_target(self):
        gross = np.array([-20.0, -12.0, -5.0, 0.0, 8.0, 12.0, 20.0])
        got = _opportunity_target(gross)
        expected = np.array([-8.0, -0.0, -0.0, 0.0, 0.0, 0.0, 8.0])
        np.testing.assert_allclose(got, expected)

    def test_non_overlapping_trade_rule(self):
        indices = np.array([0, 1, 10, 11, 21, 22])
        got = _greedy_trade_indices(indices, horizon=10)
        np.testing.assert_array_equal(got, np.array([0, 11, 22]))

    def test_max_drawdown(self):
        pnl = np.array([5.0, -2.0, -4.0, 3.0])
        self.assertAlmostEqual(_max_drawdown(pnl), 6.0)


if __name__ == "__main__":
    unittest.main()
