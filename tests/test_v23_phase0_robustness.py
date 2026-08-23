from __future__ import annotations

import unittest

from multimarket.v23_phase0_robust import ELASTIC_MAX_ITER, _make_model
from multimarket.v23_phase0_summary_robust import summarize


class V23Phase0RobustnessTests(unittest.TestCase):
    def test_elasticnet_objective_is_unchanged_except_iteration_ceiling(self):
        pipeline = _make_model("elasticnet")
        model = pipeline.steps[-1][1]
        self.assertEqual(model.alpha, 0.0005)
        self.assertEqual(model.l1_ratio, 0.25)
        self.assertEqual(model.max_iter, ELASTIC_MAX_ITER)
        self.assertEqual(ELASTIC_MAX_ITER, 50_000)

    def test_unavailable_target_does_not_relax_three_of_five_rule(self):
        def scored(symbol: str, increment: float):
            def metrics(r2: float):
                rows = 100
                mse = 1.0
                return {
                    "rows": rows,
                    "r2": r2,
                    "mse": mse,
                    "spearman": 0.0,
                    "pearson": 0.0,
                    "sign_accuracy": 0.5,
                }

            pooled = {}
            for model in ("ridge", "elasticnet"):
                base = metrics(0.0)
                cross = metrics(increment)
                pooled[model] = {
                    "6": {
                        "base": {"all": base, "non_jump": base},
                        "cross": {"all": cross, "non_jump": cross},
                        "incremental_r2": increment,
                        "incremental_non_jump_r2": increment,
                    }
                }
            fold_models = {
                model: {"6": {"incremental_r2": increment}}
                for model in ("ridge", "elasticnet")
            }
            return {
                "symbol": symbol,
                "pooled": pooled,
                "folds": [
                    {
                        "fold": 1,
                        "status": "SCORED",
                        "models": fold_models,
                    }
                ],
            }

        unavailable = {
            "symbol": "QQQ",
            "pooled": {"ridge": {}, "elasticnet": {}},
            "folds": [
                {"fold": 1, "status": "SKIP_MIN_TRAIN_ROWS", "train_rows": 0, "eval_rows": 100}
            ],
        }

        payloads = [
            scored("EURUSD", -0.01),
            scored("XAUUSD", -0.01),
            scored("BTCUSD", -0.01),
            scored("ETHUSD", -0.01),
            unavailable,
        ]
        summary = summarize(payloads)

        self.assertEqual(summary["unavailable_targets"], 1)
        self.assertEqual(summary["positive_target_requirement_denominator"], 5)
        self.assertEqual(summary["positive_targets"], 0)
        self.assertFalse(summary["promotion_conditions"]["at_least_three_positive_targets"])
        self.assertFalse(summary["promotion_pass"])
        qqq = next(row for row in summary["target_results"] if row["symbol"] == "QQQ")
        self.assertEqual(qqq["status"], "UNAVAILABLE_NO_SCORED_FOLDS")
        self.assertFalse(qqq["positive"])


if __name__ == "__main__":
    unittest.main()
