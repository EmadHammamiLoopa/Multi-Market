import json
import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from multimarket.models import MarketBar
from multimarket.v21_features import PeerMarket
from multimarket.v23_phase0c import (
    Phase0CRow,
    _economic_evaluation,
    _own_and_regime_features,
    _prepare_market,
    _rv24_series,
    _sensor_packet,
    _trade_metrics,
    _window_return_bps,
    load_phase0b_fold_windows,
    load_phase0c_manifest,
)
from multimarket.v23_phase0c_summary import summarize


def _bars(count: int, *, start: datetime, bump_index: int | None = None, bump: float = 0.0):
    result = []
    price = 100.0
    for i in range(count):
        move = 0.0008 if i % 5 else -0.00035
        price *= 1.0 + move
        close = price * (1.0 + bump if i == bump_index else 1.0)
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


def _row(timestamp: datetime, *, executable_bps: float = 10.0, vol_pct: float = 0.9):
    return Phase0CRow(
        timestamp=timestamp,
        label_end_timestamp=timestamp + timedelta(minutes=30),
        execution_exit_timestamp=timestamp + timedelta(minutes=30),
        own_features=(1.0,) * 10,
        linked_features=(1.0,) * 19,
        regime_features=(1.0,) * 24,
        volatility_percentile=vol_pct,
        forward_6_bps=executable_bps,
        executable_forward_6_bps=executable_bps,
        jump_state=0,
    )


def _candidate_block(*, scored_folds: int = 4):
    return {
        "C2": {"scored_folds": scored_folds},
        "C3": {"scored_folds": scored_folds},
    }


class V23Phase0CCausalityTests(unittest.TestCase):
    def test_future_target_bar_cannot_change_current_features(self):
        start = datetime(2025, 8, 1, tzinfo=timezone.utc)
        original = _bars(280, start=start)
        mutated = _bars(280, start=start, bump_index=250, bump=0.50)
        eligible = set(range(280))
        index = 220
        a = _own_and_regime_features(
            original, index, eligible, rv24=_rv24_series(original, eligible)
        )
        b = _own_and_regime_features(
            mutated, index, eligible, rv24=_rv24_series(mutated, eligible)
        )
        self.assertIsNotNone(a)
        self.assertEqual(a, b)

    def test_future_sensor_bar_cannot_change_current_packet(self):
        start = datetime(2025, 8, 1, tzinfo=timezone.utc)
        bars = _bars(100, start=start)
        decision = bars[70].timestamp
        peer_a = PeerMarket.build(bars, eligible_indices=set(range(100)))
        changed = list(bars)
        future = changed[80]
        changed[80] = MarketBar(
            timestamp=future.timestamp,
            open=future.open * 2.0,
            high=future.high * 2.0,
            low=future.low * 2.0,
            close=future.close * 2.0,
        )
        peer_b = PeerMarket.build(changed, eligible_indices=set(range(100)))
        self.assertEqual(
            _sensor_packet(_prepare_market(peer_a), decision),
            _sensor_packet(_prepare_market(peer_b), decision),
        )

    def test_feature_window_touching_reserved_holdout_is_rejected(self):
        start = datetime(2025, 10, 24, 22, 0, tzinfo=timezone.utc)
        bars = _bars(60, start=start)
        eligible = set(range(60))
        index = 40
        self.assertGreater(bars[index].timestamp, datetime(2025, 10, 24, 23, 59, 59, tzinfo=timezone.utc))
        self.assertIsNone(_window_return_bps(bars, index, 24, eligible))


class V23Phase0CFreezeTests(unittest.TestCase):
    def test_phase0b_fold_starts_are_reused(self):
        payload = {
            "folds": [
                {"fold": 1, "status": "SKIP_MIN_TRAIN_ROWS"},
                {"fold": 2, "status": "SCORED", "eval_start": "2025-10-25T04:45:00+00:00"},
                {"fold": 3, "status": "SCORED", "eval_start": "2025-12-29T14:50:00+00:00"},
                {"fold": 4, "status": "SCORED", "eval_start": "2026-03-21T12:00:00+00:00"},
                {"fold": 5, "status": "SCORED", "eval_start": "2026-05-20T12:00:00+00:00"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase0b.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            windows = load_phase0b_fold_windows(path)
        self.assertEqual([window.fold for window in windows], [2, 3, 4, 5])
        self.assertEqual(windows[0].eval_end, windows[1].eval_start)
        self.assertIsNone(windows[-1].eval_end)

    def test_manifest_preserves_declared_sensor_order(self):
        payload = {
            "version": "test",
            "targets": {
                "EURUSD": {
                    "linked_sensors": [
                        {"role": "USD", "symbol": "UUP"},
                        {"role": "RATES", "symbol": "TLT"},
                        {"role": "RISK", "symbol": "HYG"},
                    ]
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            _, order = load_phase0c_manifest(path, symbol="eurusd")
        self.assertEqual(order, ("UUP", "TLT", "HYG"))


class V23Phase0CEconomicTests(unittest.TestCase):
    def test_missing_cost_never_becomes_zero_cost(self):
        start = datetime(2025, 8, 1, tzinfo=timezone.utc)
        train = [_row(start + timedelta(minutes=5 * i)) for i in range(40)]
        eval_rows = [_row(start + timedelta(days=1, minutes=5 * i)) for i in range(20)]
        result = _economic_evaluation(
            train,
            eval_rows,
            [2.0] * len(train),
            [2.0] * len(eval_rows),
            round_trip_cost_bps=None,
        )
        self.assertEqual(result["status"], "NOT_EVALUATED_NO_COST_MODEL")

    def test_non_overlapping_gate_and_cost_are_applied(self):
        start = datetime(2025, 8, 1, tzinfo=timezone.utc)
        train = [_row(start + timedelta(minutes=5 * i)) for i in range(40)]
        eval_rows = [_row(start + timedelta(days=1, minutes=5 * i), executable_bps=10.0) for i in range(20)]
        result = _economic_evaluation(
            train,
            eval_rows,
            [2.0] * len(train),
            [2.0] * len(eval_rows),
            round_trip_cost_bps=1.0,
        )
        self.assertEqual(result["status"], "SCORED")
        self.assertLess(result["net"]["trades"], len(eval_rows))
        self.assertTrue(math.isclose(result["net"]["expectancy_bps"], 9.0))

    def test_profit_factor_infinity_is_json_safe(self):
        metrics = _trade_metrics([1.0, 2.0, 3.0])
        self.assertIsNone(metrics["profit_factor"])
        self.assertTrue(metrics["profit_factor_infinite"])
        json.dumps(metrics, allow_nan=False)


class V23Phase0CSummaryTests(unittest.TestCase):
    def test_insufficient_folds_are_inconclusive_not_rejected(self):
        result = summarize([
            {
                "symbol": "BTCUSD",
                "evaluation_status": "SCORED",
                "row_count": 8684,
                "signal_candidate": None,
                "promoted_candidate": None,
                "promotion_pass": False,
                "cost_model_status": "MISSING",
                "candidates": _candidate_block(scored_folds=2),
            }
        ])
        self.assertEqual(result["phase0c_promotion"], "INCONCLUSIVE")
        self.assertEqual(result["decision"], "REQUIRES_FEASIBILITY_REPAIR")
        self.assertEqual(result["inconclusive_targets"], ["BTCUSD"])
        self.assertEqual(result["scored_targets"], 0)

    def test_statistical_candidate_without_cost_is_pending(self):
        result = summarize([
            {
                "symbol": "XAUUSD",
                "evaluation_status": "SCORED",
                "signal_candidate": "C2",
                "promoted_candidate": None,
                "promotion_pass": False,
                "cost_model_status": "MISSING",
                "candidates": _candidate_block(),
            }
        ])
        self.assertEqual(result["phase0c_promotion"], "PENDING_COST_MODEL")
        self.assertEqual(result["pending_cost_targets"], ["XAUUSD"])

    def test_economic_failure_with_supplied_cost_is_not_pending(self):
        result = summarize([
            {
                "symbol": "XAUUSD",
                "evaluation_status": "SCORED",
                "signal_candidate": "C2",
                "promoted_candidate": None,
                "promotion_pass": False,
                "cost_model_status": "SUPPLIED",
                "candidates": _candidate_block(),
            }
        ])
        self.assertEqual(result["phase0c_promotion"], "FAIL")

    def test_one_promoted_target_is_partial_pass(self):
        result = summarize([
            {
                "symbol": "XAUUSD",
                "evaluation_status": "SCORED",
                "signal_candidate": "C2",
                "promoted_candidate": "C2",
                "promotion_pass": True,
                "cost_model_status": "SUPPLIED",
                "candidates": _candidate_block(),
            },
            {
                "symbol": "EURUSD",
                "evaluation_status": "SCORED",
                "signal_candidate": None,
                "promoted_candidate": None,
                "promotion_pass": False,
                "cost_model_status": "SUPPLIED",
                "candidates": _candidate_block(),
            },
        ])
        self.assertEqual(result["phase0c_promotion"], "PARTIAL_PASS")
        self.assertEqual(result["promoted_targets"], ["XAUUSD"])


if __name__ == "__main__":
    unittest.main()
