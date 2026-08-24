import unittest

from multimarket.v23_phase0c_summary import summarize


class V23Phase0CSummaryTests(unittest.TestCase):
    @staticmethod
    def _payload(symbol: str, *, signal_pass: bool, scored: bool = True):
        return {
            "version": "V2.3-PHASE0C-ASSET-SPECIFIC-REGIME",
            "symbol": symbol,
            "folds": ([{"fold": 2, "status": "SCORED"}] if scored else [{"fold": 1, "status": "SKIP_MIN_TRAIN_ROWS"}]),
            "target_signal_pass": signal_pass,
            "selected_representation": "C2" if signal_pass else None,
            "candidate_status": {},
        }

    def test_partial_pass_promotes_only_passing_targets(self):
        payloads = [
            self._payload("EURUSD", signal_pass=False),
            self._payload("XAUUSD", signal_pass=True),
            self._payload("BTCUSD", signal_pass=False),
            self._payload("ETHUSD", signal_pass=False),
            self._payload("QQQ", signal_pass=False, scored=False),
        ]
        result = summarize(payloads)
        self.assertEqual(result["phase0c_promotion"], "PARTIAL_PASS")
        self.assertEqual(result["signal_pass_targets"], ["XAUUSD"])
        self.assertIn("QQQ", result["unavailable_targets"])

    def test_all_available_pass_is_pass_even_if_one_target_unavailable(self):
        payloads = [
            self._payload("EURUSD", signal_pass=True),
            self._payload("XAUUSD", signal_pass=True),
            self._payload("BTCUSD", signal_pass=True),
            self._payload("ETHUSD", signal_pass=True),
            self._payload("QQQ", signal_pass=False, scored=False),
        ]
        result = summarize(payloads)
        self.assertEqual(result["phase0c_promotion"], "PASS")
        self.assertEqual(set(result["signal_pass_targets"]), {"EURUSD", "XAUUSD", "BTCUSD", "ETHUSD"})

    def test_no_pass_is_fail(self):
        payloads = [
            self._payload("EURUSD", signal_pass=False),
            self._payload("XAUUSD", signal_pass=False),
            self._payload("BTCUSD", signal_pass=False),
            self._payload("ETHUSD", signal_pass=False),
            self._payload("QQQ", signal_pass=False, scored=False),
        ]
        result = summarize(payloads)
        self.assertEqual(result["phase0c_promotion"], "FAIL")


if __name__ == "__main__":
    unittest.main()
