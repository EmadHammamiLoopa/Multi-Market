from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta, timezone

from multimarket.v22_macro import (
    MACRO_MISSING_AGE_SENTINEL,
    MACRO_SERIES,
    MacroLedgerIndex,
    MacroLedgerRow,
    build_macro_features,
    macro_feature_names,
)


BASE = datetime(2026, 1, 1, 13, 30, tzinfo=timezone.utc)


def _row(series: str, days: int, observation_month: int, value: float) -> MacroLedgerRow:
    return MacroLedgerRow(
        series_id=series,
        known_at=BASE + timedelta(days=days),
        observation_date=date(2025, observation_month, 1),
        value=value,
    )


class V22MacroTests(unittest.TestCase):
    def test_feature_names_are_frozen_and_deterministic(self) -> None:
        names = macro_feature_names()
        self.assertEqual(len(names), 24)
        self.assertEqual(names[0], "CPIAUCSL_available")
        self.assertEqual(names[-1], "UNRATE_previous_change")

    def test_packet_is_masked_until_three_observations_are_known(self) -> None:
        ledger = MacroLedgerIndex([
            _row("CPIAUCSL", 0, 1, 300.0),
            _row("CPIAUCSL", 30, 2, 301.0),
        ])
        packet = ledger.packet("CPIAUCSL", BASE + timedelta(days=31))
        self.assertEqual(packet.available, 0.0)
        self.assertEqual(packet.age_days, MACRO_MISSING_AGE_SENTINEL)
        self.assertEqual(packet.latest_change, 0.0)
        self.assertEqual(packet.previous_change, 0.0)

    def test_percentage_change_packet_uses_only_known_rows(self) -> None:
        rows = [
            _row("CPIAUCSL", 0, 1, 300.0),
            _row("CPIAUCSL", 30, 2, 303.0),
            _row("CPIAUCSL", 60, 3, 306.03),
        ]
        ledger = MacroLedgerIndex(rows)
        packet = ledger.packet("CPIAUCSL", BASE + timedelta(days=61))
        self.assertEqual(packet.available, 1.0)
        self.assertAlmostEqual(packet.age_days, 1.0)
        self.assertAlmostEqual(packet.latest_change, 1.0)
        self.assertAlmostEqual(packet.previous_change, 1.0)

    def test_level_change_packet_preserves_units(self) -> None:
        rows = [
            _row("PAYEMS", 0, 1, 158000.0),
            _row("PAYEMS", 30, 2, 158125.0),
            _row("PAYEMS", 60, 3, 158325.0),
        ]
        ledger = MacroLedgerIndex(rows)
        packet = ledger.packet("PAYEMS", BASE + timedelta(days=60))
        self.assertEqual(packet.latest_change, 200.0)
        self.assertEqual(packet.previous_change, 125.0)

    def test_release_is_invisible_before_known_at_and_visible_at_boundary(self) -> None:
        rows = [
            _row("UNRATE", 0, 1, 4.0),
            _row("UNRATE", 30, 2, 4.1),
            _row("UNRATE", 60, 3, 4.2),
        ]
        ledger = MacroLedgerIndex(rows)
        before = ledger.packet("UNRATE", BASE + timedelta(days=60, seconds=-1))
        at_release = ledger.packet("UNRATE", BASE + timedelta(days=60))
        self.assertEqual(before.available, 0.0)
        self.assertEqual(at_release.available, 1.0)
        self.assertAlmostEqual(at_release.age_days, 0.0)

    def test_future_revision_does_not_change_earlier_feature(self) -> None:
        original = [
            _row("PAYEMS", 0, 1, 100.0),
            _row("PAYEMS", 30, 2, 110.0),
            _row("PAYEMS", 60, 3, 120.0),
        ]
        decision = BASE + timedelta(days=61)
        before = MacroLedgerIndex(original).packet("PAYEMS", decision)
        revised = original + [
            MacroLedgerRow(
                "PAYEMS",
                BASE + timedelta(days=90),
                date(2025, 3, 1),
                999.0,
            )
        ]
        after = MacroLedgerIndex(revised).packet("PAYEMS", decision)
        self.assertEqual(before, after)

    def test_revision_becomes_visible_only_after_known_at(self) -> None:
        rows = [
            _row("PAYEMS", 0, 1, 100.0),
            _row("PAYEMS", 30, 2, 110.0),
            _row("PAYEMS", 60, 3, 120.0),
            MacroLedgerRow("PAYEMS", BASE + timedelta(days=90), date(2025, 2, 1), 115.0),
        ]
        ledger = MacroLedgerIndex(rows)
        before = ledger.packet("PAYEMS", BASE + timedelta(days=89))
        after = ledger.packet("PAYEMS", BASE + timedelta(days=90))
        self.assertEqual(before.latest_change, 10.0)
        self.assertEqual(before.previous_change, 10.0)
        self.assertEqual(after.latest_change, 5.0)
        self.assertEqual(after.previous_change, 15.0)

    def test_future_mutation_does_not_change_full_feature_vector(self) -> None:
        rows: list[MacroLedgerRow] = []
        for series in MACRO_SERIES:
            if series in {"PAYEMS", "UNRATE"}:
                values = (100.0, 101.0, 102.0)
            else:
                values = (100.0, 101.0, 102.0)
            for month, (days, value) in enumerate(zip((0, 30, 60), values), start=1):
                rows.append(_row(series, days, month, value))
        decision = BASE + timedelta(days=61)
        before = build_macro_features(MacroLedgerIndex(rows), decision)
        rows.append(MacroLedgerRow("CPIAUCSL", BASE + timedelta(days=120), date(2025, 3, 1), 9999.0))
        after = build_macro_features(MacroLedgerIndex(rows), decision)
        self.assertEqual(before, after)
        self.assertEqual(len(before), 24)

    def test_naive_known_at_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MacroLedgerRow("UNRATE", datetime(2026, 1, 1, 8, 30), date(2025, 12, 1), 4.0)


if __name__ == "__main__":
    unittest.main()
