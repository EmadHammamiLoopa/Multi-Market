from __future__ import annotations

import unittest

from multimarket.v2_ranking_audit import _average_ranks, _spearman, _top_fraction_mean


class V2RankingAuditTests(unittest.TestCase):
    def test_average_ranks_handles_ties(self) -> None:
        self.assertEqual(_average_ranks([10.0, 20.0, 20.0, 40.0]), [1.0, 2.5, 2.5, 4.0])

    def test_spearman_detects_monotonic_order(self) -> None:
        self.assertAlmostEqual(_spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]), 1.0)
        self.assertAlmostEqual(_spearman([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]), -1.0)

    def test_top_fraction_mean_selects_highest_scores(self) -> None:
        value = _top_fraction_mean([0.1, 0.9, 0.8, 0.2, 0.7], [-1.0, 5.0, 3.0, -2.0, 1.0], 0.40)
        self.assertAlmostEqual(value, 4.0)


if __name__ == "__main__":
    unittest.main()
