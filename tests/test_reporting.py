import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from multimarket.metrics import Metrics
from multimarket.models import Direction, Prediction, TradeResult
from multimarket.replay import ReplayReport
from multimarket.reporting import append_benchmark_history, load_benchmark_history, ranking_lines, summary_lines


class ReportingTest(unittest.TestCase):
    def _report(self, accuracy=0.625, net=4.2):
        prediction = Prediction(datetime(2024, 1, 1, tzinfo=timezone.utc), Direction.LONG, 0.8, 100.0, 4)
        trade = TradeResult(prediction, datetime(2024, 1, 1, 1, tzinfo=timezone.utc), 101.0, 100.0, 98.0, True, "take_profit")
        metrics = Metrics(10, 8, 0.8, accuracy, 0.625, 1.5, 4.0, net, 2.1)
        return ReplayReport([prediction], [True], [trade], metrics)

    def test_summary_prints_accuracy_as_percent(self):
        text = "\n".join(summary_lines(self._report(), symbol="TEST", version="V0"))
        self.assertIn("Directional accuracy : 62.50%", text)
        self.assertIn("Multi-Market V0", text)

    def test_history_keeps_versions_and_ranks_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            append_benchmark_history(self._report(0.60, 3.0), path, symbol="TEST", version="V0", dataset_sha256="abc")
            append_benchmark_history(self._report(0.70, 2.0), path, symbol="TEST", version="V1", dataset_sha256="abc")
            rows = load_benchmark_history(path)
            self.assertEqual(len(rows), 2)
            ranking = ranking_lines(rows, dataset_sha256="abc")
            self.assertIn("V1", ranking[3])
            self.assertIn("70.00%", ranking[3])


if __name__ == "__main__":
    unittest.main()
