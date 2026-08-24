import unittest
from datetime import datetime, timedelta, timezone

from multimarket.models import MarketBar
from multimarket.v21_features import PeerMarket
from multimarket.v23_phase0 import _forward_return
from multimarket.v23_phase0c import (
    EXPECTED_SENSORS,
    HGBR_PARAMS,
    MAX_PEER_STALENESS_SECONDS,
    Phase0CRow,
    _find_latest_eligible_asof,
    _peer_packet,
    _prepare_peer,
    _validate_sensor_set,
    evaluate_rows,
)


def _bars(count: int, *, start: datetime, future_bump_index: int | None = None):
    result = []
    price = 100.0
    for i in range(count):
        move = 0.0006 if i % 4 else -0.00035
        price *= 1.0 + move
        if future_bump_index is not None and i == future_bump_index:
            price *= 1.5
        close = price
        open_ = close * 0.9998
        result.append(
            MarketBar(
                timestamp=start + timedelta(minutes=5 * i),
                open=open_,
                high=max(open_, close) * 1.0003,
                low=min(open_, close) * 0.9997,
                close=close,
            )
        )
    return result


def _peer_market(bars):
    return PeerMarket.build(bars, eligible_indices=set(range(len(bars))))


class V23Phase0CCausalityTests(unittest.TestCase):
    def test_future_peer_mutation_cannot_change_past_packet(self):
        start = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
        original = _bars(100, start=start)
        mutated = _bars(100, start=start, future_bump_index=81)
        decision = original[80].timestamp

        packet_a = _peer_packet(_prepare_peer(_peer_market(original)), decision)
        packet_b = _peer_packet(_prepare_peer(_peer_market(mutated)), decision)
        self.assertEqual(packet_a, packet_b)

    def test_peer_older_than_fifteen_minutes_is_unavailable(self):
        start = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
        bars = _bars(80, start=start)
        prepared = _prepare_peer(_peer_market(bars))
        decision = bars[-1].timestamp + timedelta(seconds=MAX_PEER_STALENESS_SECONDS + 1)

        self.assertIsNone(_find_latest_eligible_asof(prepared, decision))
        self.assertEqual(_peer_packet(prepared, decision), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def test_exact_preregistered_peer_set_is_required(self):
        start = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
        peer = _peer_market(_bars(80, start=start))
        valid = {name: peer for name in EXPECTED_SENSORS["EURUSD"]}
        _validate_sensor_set("EURUSD", valid)

        invalid = dict(valid)
        invalid.pop("UUP")
        with self.assertRaises(ValueError):
            _validate_sensor_set("EURUSD", invalid)

    def test_forward_label_touching_reserved_g_window_is_excluded(self):
        start = datetime(2025, 10, 5, 19, 0, tzinfo=timezone.utc)
        bars = _bars(80, start=start)
        eligible = set(range(80))
        index = next(
            i for i, bar in enumerate(bars)
            if bar.timestamp == datetime(2025, 10, 5, 23, 35, tzinfo=timezone.utc)
        )
        self.assertIsNone(_forward_return(bars, index, 6, eligible))

    def test_hgbr_internal_random_validation_is_disabled(self):
        self.assertFalse(HGBR_PARAMS["early_stopping"])
        self.assertEqual(HGBR_PARAMS["random_state"], 0)


class V23Phase0CPromotionTests(unittest.TestCase):
    @staticmethod
    def _synthetic_rows(count: int = 100):
        start = datetime(2025, 8, 1, tzinfo=timezone.utc)
        rows = []
        for i in range(count):
            x = float(i % 17) / 17.0
            y = 3.0 * x - 1.0
            own = (x,)
            linked = (x, x * x)
            regime = (x, x * x, float(i % 2))
            rows.append(
                Phase0CRow(
                    timestamp=start + timedelta(minutes=5 * i),
                    label_end_timestamp=start + timedelta(minutes=5 * (i + 6)),
                    c0_features=own,
                    c1_features=linked,
                    c2_features=regime,
                    forward_1_bps=y,
                    forward_6_bps=y,
                    jump_state=0,
                )
            )
        return rows

    def test_insufficient_training_rows_never_becomes_signal_pass(self):
        payload = evaluate_rows(self._synthetic_rows(), symbol="EURUSD")
        self.assertFalse(payload["target_signal_pass"])
        self.assertIsNone(payload["selected_representation"])
        self.assertTrue(all(not status["signal_pass"] for status in payload["candidate_status"].values()))


if __name__ == "__main__":
    unittest.main()
