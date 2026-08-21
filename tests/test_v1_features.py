import unittest
from datetime import datetime, timedelta, timezone

from multimarket.features import FEATURE_NAMES, WARMUP_BARS, build_feature_points
from multimarket.models import MarketBar
from multimarket.v1_model import build_labeled_points


def bars(count=90):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = []
    price = 100.0
    for index in range(count):
        price *= 1.0 + (0.0005 if index % 3 else -0.0002)
        result.append(
            MarketBar(
                timestamp=start + timedelta(minutes=5 * index),
                open=price * 0.9998,
                high=price * 1.0005,
                low=price * 0.9995,
                close=price,
            )
        )
    return result


class V1FeatureTests(unittest.TestCase):
    def test_feature_vector_shape_and_warmup(self):
        points = build_feature_points(bars())
        self.assertEqual(points[0].bar_index, WARMUP_BARS)
        self.assertEqual(len(points[0].values), len(FEATURE_NAMES))

    def test_future_mutation_does_not_change_past_features(self):
        original = bars()
        before = build_feature_points(original)
        target = before[5]

        changed = list(original)
        future_index = target.bar_index + 10
        future = changed[future_index]
        changed[future_index] = MarketBar(
            timestamp=future.timestamp,
            open=future.open * 2,
            high=future.high * 2,
            low=future.low * 2,
            close=future.close * 2,
        )
        after = build_feature_points(changed)
        matching = next(point for point in after if point.bar_index == target.bar_index)
        self.assertEqual(target.values, matching.values)

    def test_label_uses_only_configured_future_horizon(self):
        data = bars()
        points = build_feature_points(data)
        labeled = build_labeled_points(data, points, 6)
        first = labeled[0]
        expected = 1 if data[first.bar_index + 6].close > data[first.bar_index].close else 0
        self.assertEqual(first.label, expected)


if __name__ == "__main__":
    unittest.main()
