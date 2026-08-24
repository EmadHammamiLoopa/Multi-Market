import unittest
from datetime import datetime, timedelta, timezone

from multimarket.models import MarketBar
from multimarket.v21_features import PeerMarket
from multimarket.v23_phase0cr import _async_sensor_packet, _prepare_async_sensor


def _bars(count: int, *, start: datetime, mutate_index: int | None = None):
    result = []
    price = 100.0
    for i in range(count):
        price *= 1.0005 if i % 4 else 0.9998
        if mutate_index == i:
            price *= 1.5
        close = price
        open_ = close * 0.9999
        result.append(MarketBar(
            timestamp=start + timedelta(minutes=5 * i),
            open=open_, high=close * 1.0002, low=open_ * 0.9998, close=close,
        ))
    return result


class Phase0CRAsyncSensorTests(unittest.TestCase):
    def test_last_known_packet_survives_sensor_gap_with_age_metadata(self):
        start = datetime(2026, 3, 2, 14, 30, tzinfo=timezone.utc)
        bars = _bars(40, start=start)
        peer = PeerMarket.build(bars, eligible_indices=set(range(40)))
        sensor = _prepare_async_sensor(peer)
        decision = bars[-1].timestamp + timedelta(hours=12)
        packet = _async_sensor_packet(sensor, decision)
        self.assertIsNotNone(packet)
        self.assertEqual(len(packet), 5)
        self.assertGreater(packet[3], 0.0)
        self.assertEqual(packet[4], 0.0)

    def test_five_minute_packet_is_marked_fresh(self):
        start = datetime(2026, 3, 2, 14, 30, tzinfo=timezone.utc)
        bars = _bars(40, start=start)
        peer = PeerMarket.build(bars, eligible_indices=set(range(40)))
        sensor = _prepare_async_sensor(peer)
        packet = _async_sensor_packet(sensor, bars[-1].timestamp + timedelta(minutes=5))
        self.assertIsNotNone(packet)
        self.assertEqual(packet[4], 1.0)

    def test_future_sensor_mutation_cannot_change_past_packet(self):
        start = datetime(2026, 3, 2, 14, 30, tzinfo=timezone.utc)
        original = _bars(80, start=start)
        changed = _bars(80, start=start, mutate_index=70)
        decision = original[60].timestamp
        a = _prepare_async_sensor(PeerMarket.build(original, eligible_indices=set(range(80))))
        b = _prepare_async_sensor(PeerMarket.build(changed, eligible_indices=set(range(80))))
        self.assertEqual(_async_sensor_packet(a, decision), _async_sensor_packet(b, decision))

    def test_no_packet_before_first_valid_24_bar_history(self):
        start = datetime(2026, 3, 2, 14, 30, tzinfo=timezone.utc)
        bars = _bars(40, start=start)
        sensor = _prepare_async_sensor(PeerMarket.build(bars, eligible_indices=set(range(40))))
        self.assertIsNone(_async_sensor_packet(sensor, bars[10].timestamp))


if __name__ == "__main__":
    unittest.main()
