from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from multimarket.v23_phase0_batch import resolve_parallel_plan


class V23Phase0BatchTests(unittest.TestCase):
    @patch.object(os, "cpu_count", return_value=24)
    def test_default_plan_uses_budget_without_oversubscription(self, _mock_cpu):
        plan = resolve_parallel_plan(
            symbols=("EURUSD", "XAUUSD", "BTCUSD", "ETHUSD", "QQQ"),
            cpu_budget=None,
            workers=None,
            threads_per_worker=None,
        )
        self.assertEqual(plan.logical_cpus, 24)
        self.assertEqual(plan.cpu_budget, 24)
        self.assertEqual(plan.workers, 5)
        self.assertEqual(plan.threads_per_worker, 4)
        self.assertEqual(plan.nominal_thread_slots, 20)
        self.assertLessEqual(plan.nominal_thread_slots, plan.cpu_budget)

    @patch.object(os, "cpu_count", return_value=24)
    def test_explicit_thread_request_is_capped_by_budget(self, _mock_cpu):
        plan = resolve_parallel_plan(
            symbols=("A", "B", "C", "D", "E"),
            cpu_budget=20,
            workers=5,
            threads_per_worker=12,
        )
        self.assertEqual(plan.workers, 5)
        self.assertEqual(plan.threads_per_worker, 4)
        self.assertEqual(plan.nominal_thread_slots, 20)


if __name__ == "__main__":
    unittest.main()
