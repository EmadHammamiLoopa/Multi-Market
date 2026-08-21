import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from multimarket.models import MarketBar
from multimarket.v1_model import V1Config, WalkForwardXGBoostPredictor


class FakeProbabilities:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, item):
        if isinstance(item, tuple):
            rows, column = item
            selected = self.rows[rows]
            return [row[column] for row in selected]
        return self.rows[item]


class FakeModel:
    def predict_proba(self, X):
        return FakeProbabilities([[0.4, 0.6] for _ in X])


def make_bars(count=220):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    price = 100.0
    for index in range(count):
        price *= 1.0005 if index % 4 else 0.999
        rows.append(
            MarketBar(
                timestamp=start + timedelta(minutes=index * 5),
                open=price * 0.999,
                high=price * 1.002,
                low=price * 0.998,
                close=price,
            )
        )
    return rows


class V1ModelTests(unittest.TestCase):
    def test_training_rows_have_completed_horizon_before_decision(self):
        data = make_bars()
        config = V1Config(
            horizon_bars=6,
            confidence_threshold=0.55,
            min_train_rows=100,
            retrain_every=1000,
            validation_fraction=0.20,
        )
        predictor = WalkForwardXGBoostPredictor(data, config)
        decision_index = 170

        captured = {}

        def fake_fit(X, y, random_state):
            captured["train_rows"] = len(X)
            return FakeModel()

        with patch("multimarket.v1_model._fit_xgb", side_effect=fake_fit):
            prediction = predictor.predict_at_index(decision_index)

        eligible = [
            point
            for point in predictor.labeled
            if point.bar_index + config.horizon_bars <= decision_index
        ]
        expected_split = max(
            config.min_train_rows,
            int(len(eligible) * (1.0 - config.validation_fraction)),
        )
        expected_split = min(expected_split, len(eligible) - 1)
        self.assertEqual(captured["train_rows"], expected_split)
        self.assertEqual(prediction.timestamp, data[decision_index].timestamp)
        self.assertTrue(
            all(
                point.bar_index + config.horizon_bars <= decision_index
                for point in eligible
            )
        )


if __name__ == "__main__":
    unittest.main()
