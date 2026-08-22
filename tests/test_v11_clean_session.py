from datetime import datetime, timedelta, timezone
import unittest

from multimarket.models import MarketBar
from multimarket.v1_model import V1Config, WalkForwardXGBoostPredictor


class V11CleanSessionTests(unittest.TestCase):
    def _bars(self, count: int = 140) -> list[MarketBar]:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bars = []
        price = 100.0
        for i in range(count):
            price += 0.01 if i % 2 == 0 else -0.004
            bars.append(
                MarketBar(
                    timestamp=start + timedelta(minutes=5 * i),
                    open=price,
                    high=price + 0.02,
                    low=price - 0.02,
                    close=price + 0.005,
                )
            )
        return bars

    def test_training_mask_excludes_ineligible_decision_and_future_endpoint(self):
        bars = self._bars()
        cfg = V1Config(horizon_bars=6, min_train_rows=100)
        eligible = set(range(len(bars)))
        eligible.remove(80)
        predictor = WalkForwardXGBoostPredictor(bars, cfg, eligible_indices=eligible)
        labeled_indices = {point.bar_index for point in predictor.labeled}
        self.assertNotIn(80, labeled_indices)
        self.assertNotIn(74, labeled_indices)  # future endpoint 80 is not eligible

    def test_predict_rejects_ineligible_decision_index_before_training(self):
        bars = self._bars()
        cfg = V1Config(horizon_bars=6, min_train_rows=100)
        eligible = set(range(len(bars)))
        eligible.remove(120)
        predictor = WalkForwardXGBoostPredictor(bars, cfg, eligible_indices=eligible)
        with self.assertRaisesRegex(ValueError, "not eligible"):
            predictor.predict_at_index(120)


if __name__ == "__main__":
    unittest.main()
