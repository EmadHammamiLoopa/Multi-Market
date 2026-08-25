import unittest

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from multimarket.v23_phase0dl_score import _fit_from_stats, _greedy_signals, _predict, _ridge_stats


class Phase0DLScoreTests(unittest.TestCase):
    def test_sufficient_statistics_matches_standardscaler_ridge(self):
        rng = np.random.default_rng(20260825)
        X = rng.normal(size=(5000, 7)).astype(np.float32)
        X[:, 0] = 3.0  # constant-column StandardScaler semantics
        y = 0.4 * X[:, 1] - 0.2 * X[:, 2] + rng.normal(scale=0.3, size=len(X))
        Xt = rng.normal(size=(300, 7)).astype(np.float32)
        Xt[:, 0] = 3.0
        for alpha in (0.1, 1.0, 10.0, 100.0):
            scaler = StandardScaler().fit(X)
            ref = Ridge(alpha=alpha).fit(scaler.transform(X), y)
            expected = ref.predict(scaler.transform(Xt))
            got = _predict(Xt, _fit_from_stats(_ridge_stats(X, y), alpha))
            np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-8)

    def test_non_overlap_includes_latency_and_horizon(self):
        # 1-second horizon on a 250-ms grid: signal i enters at i+1 and exits at i+5.
        # A new signal at the exit timestamp is allowed; earlier ones are ignored.
        candidate = np.arange(0, 20, dtype=np.int64)
        chosen = _greedy_signals(candidate, 1)
        np.testing.assert_array_equal(chosen, np.asarray([0, 5, 10, 15], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
