import unittest
from datetime import datetime, timedelta, timezone

from multimarket.models import MarketBar
from multimarket.v21_features import PeerMarket
from multimarket.v23_phase0 import (
    LAGS,
    _forward_return,
    _peer_lag_packet,
    _prepare_peer,
    _sigma48_series,
)
from multimarket.v23_phase0_summary import summarize


def _bars(count: int, *, start: datetime, bump_index: int | None = None, bump: float = 0.0):
    result = []
    price = 100.0
    for i in range(count):
        move = 0.0005 if i % 3 else -0.0003
        price *= 1.0 + move
        close = price * (1.0 + bump if i == bump_index else 1.0)
        open_ = close * 0.9999
        result.append(
            MarketBar(
                timestamp=start + timedelta(minutes=5 * i),
                open=open_,
                high=max(open_, close) * 1.0002,
                low=min(open_, close) * 0.9998,
                close=close,
            )
        )
    return result


class V23Phase0CausalityTests(unittest.TestCase):
    def test_sigma_at_index_excludes_current_return(self):
        start = datetime(2025, 8, 1, tzinfo=timezone.utc)
        original = _bars(80, start=start)
        mutated = _bars(80, start=start, bump_index=60, bump=0.50)
        eligible = set(range(80))
        a = _sigma48_series(original, eligible)
        b = _sigma48_series(mutated, eligible)
        self.assertIsNotNone(a[60])
        self.assertAlmostEqual(a[60], b[60], places=12)

    def test_future_peer_bar_cannot_change_past_packet(self):
        start = datetime(2025, 8, 1, tzinfo=timezone.utc)
        bars = _bars(80, start=start)
        decision = bars[65].timestamp

        peer_a = PeerMarket.build(bars[:66], eligible_indices=set(range(66)))
        peer_bars = list(bars)
        future = peer_bars[66]
        peer_bars[66] = MarketBar(
            timestamp=future.timestamp,
            open=future.open * 2.0,
            high=future.high * 2.0,
            low=future.low * 2.0,
            close=future.close * 2.0,
        )
        peer_b = PeerMarket.build(peer_bars, eligible_indices=set(range(80)))

        packet_a = _peer_lag_packet(_prepare_peer(peer_a), decision)
        packet_b = _peer_lag_packet(_prepare_peer(peer_b), decision)
        self.assertEqual(packet_a, packet_b)
        self.assertEqual(len(packet_a[0]), len(LAGS))

    def test_forward_label_touching_reserved_window_is_excluded(self):
        start = datetime(2025, 10, 5, 19, 0, tzinfo=timezone.utc)
        bars = _bars(80, start=start)
        eligible = set(range(80))
        index = next(
            i for i, bar in enumerate(bars)
            if bar.timestamp == datetime(2025, 10, 5, 23, 35, tzinfo=timezone.utc)
        )
        self.assertIsNone(_forward_return(bars, index, 6, eligible))


class V23PromotionRuleTests(unittest.TestCase):
    @staticmethod
    def _payload(symbol: str, *, positive: bool = True):
        cross_r2 = 0.10 if positive else -0.10
        cross_mse = 0.90 if positive else 1.10
        increment = cross_r2
        pooled = {}
        for model in ("ridge", "elasticnet"):
            pooled[model] = {
                "6": {
                    "base": {
                        "all": {"rows": 100, "r2": 0.0, "mse": 1.0},
                        "non_jump": {"rows": 80, "r2": 0.0, "mse": 1.0},
                    },
                    "cross": {
                        "all": {"rows": 100, "r2": cross_r2, "mse": cross_mse},
                        "non_jump": {"rows": 80, "r2": cross_r2, "mse": cross_mse},
                    },
                    "incremental_r2": increment,
                    "incremental_non_jump_r2": increment,
                }
            }
        folds = []
        for fold in range(1, 5):
            folds.append(
                {
                    "fold": fold,
                    "status": "SCORED",
                    "models": {
                        model: {"6": {"incremental_r2": increment}}
                        for model in ("ridge", "elasticnet")
                    },
                }
            )
        return {"symbol": symbol, "pooled": pooled, "folds": folds}

    def test_promotion_passes_only_when_all_frozen_conditions_pass(self):
        payloads = [self._payload(symbol) for symbol in ("EURUSD", "XAUUSD", "BTCUSD", "ETHUSD", "QQQ")]
        result = summarize(payloads)
        self.assertTrue(result["promotion_pass"])
        self.assertEqual(result["decision"], "PROMOTE_TO_V23_PHASE1")
        self.assertTrue(all(result["promotion_conditions"].values()))

    def test_one_failed_condition_rejects_whole_block(self):
        payloads = [
            self._payload("EURUSD", positive=True),
            self._payload("XAUUSD", positive=True),
            self._payload("BTCUSD", positive=False),
            self._payload("ETHUSD", positive=False),
            self._payload("QQQ", positive=False),
        ]
        result = summarize(payloads)
        self.assertFalse(result["promotion_pass"])
        self.assertEqual(result["decision"], "REJECT_DENSE_CROSS_SECTIONAL_BLOCK")
        self.assertFalse(result["promotion_conditions"]["at_least_three_positive_targets"])


if __name__ == "__main__":
    unittest.main()
