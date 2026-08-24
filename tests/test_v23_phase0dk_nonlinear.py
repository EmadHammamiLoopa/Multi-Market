import json
import unittest

from multimarket.v23_phase0dk_nonlinear import (
    BLOCKS,
    MODEL_CONFIGS,
    KConfig,
    _better,
    _cfg_dict,
)


class Phase0DKNonlinearTests(unittest.TestCase):
    def test_frozen_model_grid_is_exact(self):
        self.assertEqual(BLOCKS, ("K1", "K2"))
        self.assertEqual(
            MODEL_CONFIGS,
            (
                ("X1", 3, 0.05, 300),
                ("X2", 3, 0.05, 600),
                ("X3", 5, 0.03, 300),
                ("X4", 5, 0.03, 600),
            ),
        )

    def test_config_payload_is_json_serializable(self):
        cfg = KConfig("K2", 10, "X3", 5, 0.03, 300, 0.9975)
        payload = _cfg_dict(cfg)
        encoded = json.dumps(payload)
        self.assertIn('"model_name": "X3"', encoded)

    def test_tie_prefers_simpler_k1(self):
        base_metrics = {
            "median_net_bps_day_all": 10.0,
            "worst_5day_rolling_net_bps": 5.0,
            "median_trades_day_active": 3.0,
            "max_drawdown_bps": 20.0,
        }
        a = {
            "cfg": KConfig("K1", 10, "X1", 3, 0.05, 300, 0.995),
            "m12": dict(base_metrics),
        }
        b = {
            "cfg": KConfig("K2", 10, "X1", 3, 0.05, 300, 0.995),
            "m12": dict(base_metrics),
        }
        self.assertTrue(_better(a, b))
        self.assertFalse(_better(b, a))


if __name__ == "__main__":
    unittest.main()
