import json
import tempfile
import unittest
from pathlib import Path

from multimarket.v2_model import V2Config
from multimarket.v22_combined_audit import _load_rows, build_parser
from multimarket.v22_features import v22_feature_names
from multimarket.v21_features import PeerMarket


class V22CombinedAuditTests(unittest.TestCase):
    def test_manifest_rows_must_be_timestamp_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(
                json.dumps({"rows": [{"decision_timestamp": "2026-01-05T00:00:00Z"}]}),
                encoding="utf-8",
            )
            rows = _load_rows(path)
            self.assertEqual(rows, [{"decision_timestamp": "2026-01-05T00:00:00Z"}])

    def test_manifest_rejects_outcome_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "decision_timestamp": "2026-01-05T00:00:00Z",
                                "realized_net_bps": 1.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "only decision_timestamp"):
                _load_rows(path)

    def test_cli_requires_macro_ledger(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "data/EURUSD_5m.csv",
                    "--symbol",
                    "EURUSD",
                    "--manifest-json",
                    "manifest.json",
                ]
            )

    def test_v2_frozen_defaults_remain_unchanged(self):
        cfg = V2Config()
        self.assertEqual(cfg.horizon_bars, 6)
        self.assertEqual(cfg.volatility_span, 48)
        self.assertEqual(cfg.volatility_min_observations, 24)
        self.assertEqual(cfg.take_profit_vol_multiplier, 1.0)
        self.assertEqual(cfg.stop_loss_vol_multiplier, 1.0)
        self.assertEqual(cfg.round_trip_cost_bps, 2.0)
        self.assertEqual(cfg.min_train_rows, 5000)
        self.assertEqual(cfg.retrain_every, 500)
        self.assertEqual(cfg.validation_fraction, 0.20)
        self.assertEqual(cfg.min_positive_probability, 0.55)
        self.assertEqual(cfg.min_predicted_ev_bps, 0.0)
        self.assertEqual(cfg.embargo_bars, 6)

    def test_v22_feature_count_with_four_peers_is_86(self):
        peers = {
            symbol: PeerMarket(symbol=symbol, bars=(), eligible_indices=frozenset())
            for symbol in ("BTCUSD", "ETHUSD", "QQQ", "XAUUSD")
        }
        self.assertEqual(len(v22_feature_names(peers)), 86)


if __name__ == "__main__":
    unittest.main()
