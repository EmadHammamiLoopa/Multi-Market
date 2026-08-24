import unittest

import numpy as np

from multimarket.v23_phase0di_longer_horizon import _rolling_return, _rolling_sum, _greedy


class Phase0DILongerHorizonTests(unittest.TestCase):
    def test_rolling_sum_is_causal_and_right_closed(self):
        x = np.arange(1.0, 7.0)
        out = _rolling_sum(x, 3)
        self.assertTrue(np.isnan(out[0]))
        self.assertTrue(np.isnan(out[1]))
        self.assertEqual(out[2], 1.0 + 2.0 + 3.0)
        self.assertEqual(out[5], 4.0 + 5.0 + 6.0)

    def test_rolling_return_uses_only_past_endpoint(self):
        price = np.asarray([100.0, 101.0, 102.0, 103.0])
        out = _rolling_return(price, 2)
        self.assertTrue(np.isnan(out[0]))
        self.assertTrue(np.isnan(out[1]))
        expected = np.log(102.0 / 100.0) * 10000.0
        self.assertAlmostEqual(out[2], expected)

    def test_greedy_prevents_overlapping_trades(self):
        idx = np.asarray([0, 1, 2, 10, 11, 20], dtype=np.int64)
        selected = _greedy(idx, 10)
        self.assertEqual(selected.tolist(), [0, 11])


if __name__ == "__main__":
    unittest.main()
