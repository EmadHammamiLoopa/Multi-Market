import unittest

import numpy as np

from multimarket.v23_phase0di_longer_horizon import _fit_predict
from multimarket.v23_phase0di_longer_horizon_fast import _fit_scaled, _scale_once


class Phase0DIFastTests(unittest.TestCase):
    def test_scaled_reuse_matches_reference_ridge_predictions(self):
        rng = np.random.default_rng(7)
        Xtr = rng.normal(size=(500, 8))
        ytr = rng.normal(size=500)
        Xev = rng.normal(size=(120, 8))

        for alpha in (0.1, 1.0, 10.0, 100.0):
            ref_train, ref_eval = _fit_predict(Xtr, ytr, Xev, alpha)
            Ztr, Zev = _scale_once(Xtr, Xev)
            fast_train, fast_eval = _fit_scaled(Ztr, ytr, Zev, alpha)
            np.testing.assert_allclose(fast_train, ref_train, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(fast_eval, ref_eval, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
