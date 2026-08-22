from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


MACRO_SERIES = (
    "CPIAUCSL",
    "CPILFESL",
    "PAYEMS",
    "PCEPI",
    "PCEPILFE",
    "UNRATE",
)

MACRO_PACKET_FIELDS = (
    "available",
    "age_days",
    "latest_change",
    "previous_change",
)

MACRO_MISSING_AGE_SENTINEL = 999.0
_PERCENT_CHANGE_SERIES = frozenset({"CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE"})
_LEVEL_CHANGE_SERIES = frozenset({"PAYEMS", "UNRATE"})


@dataclass(frozen=True, slots=True)
class MacroLedgerRow:
    series_id: str
    known_at: datetime
    observation_date: date
    value: float

    def __post_init__(self) -> None:
        if self.series_id not in MACRO_SERIES:
            raise ValueError(f"unsupported V2.2 macro series: {self.series_id}")
        if self.known_at.tzinfo is None:
            raise ValueError("known_at must be timezone-aware")
        if self.value != self.value or self.value in (float("inf"), float("-inf")):
            raise ValueError("macro value must be finite")


@dataclass(frozen=True, slots=True)
class MacroPacket:
    available: float
    age_days: float
    latest_change: float
    previous_change: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.available, self.age_days, self.latest_change, self.previous_change)


class MacroLedgerIndex:
    """Point-in-time macro ledger with revision-safe causal lookup.

    Rows are append-only facts: a value is invisible until ``known_at``. For each
    observation date, the latest revision knowable at the decision timestamp wins.
    """

    def __init__(self, rows: Iterable[MacroLedgerRow]) -> None:
        normalized = sorted(
            rows,
            key=lambda row: (
                row.series_id,
                row.known_at.astimezone(timezone.utc),
                row.observation_date,
            ),
        )
        by_series: dict[str, list[MacroLedgerRow]] = {series: [] for series in MACRO_SERIES}
        for row in normalized:
            by_series[row.series_id].append(row)
        self._by_series = {series: tuple(values) for series, values in by_series.items()}

    def point_in_time_values(self, series_id: str, decision_timestamp: datetime) -> list[MacroLedgerRow]:
        if series_id not in MACRO_SERIES:
            raise ValueError(f"unsupported V2.2 macro series: {series_id}")
        if decision_timestamp.tzinfo is None:
            raise ValueError("decision_timestamp must be timezone-aware")
        decision_utc = decision_timestamp.astimezone(timezone.utc)
        latest_by_observation: dict[date, MacroLedgerRow] = {}
        for row in self._by_series[series_id]:
            known_utc = row.known_at.astimezone(timezone.utc)
            if known_utc > decision_utc:
                break
            latest_by_observation[row.observation_date] = row
        return [latest_by_observation[key] for key in sorted(latest_by_observation)]

    def packet(self, series_id: str, decision_timestamp: datetime) -> MacroPacket:
        values = self.point_in_time_values(series_id, decision_timestamp)
        if len(values) < 3:
            return MacroPacket(0.0, MACRO_MISSING_AGE_SENTINEL, 0.0, 0.0)

        previous2, previous1, latest = values[-3:]
        age_days = (
            decision_timestamp.astimezone(timezone.utc)
            - latest.known_at.astimezone(timezone.utc)
        ).total_seconds() / 86400.0
        if age_days < 0.0:
            raise AssertionError("macro packet selected a future release")

        return MacroPacket(
            available=1.0,
            age_days=age_days,
            latest_change=_change(series_id, latest.value, previous1.value),
            previous_change=_change(series_id, previous1.value, previous2.value),
        )


def _change(series_id: str, new_value: float, prior_value: float) -> float:
    if series_id in _PERCENT_CHANGE_SERIES:
        if prior_value == 0.0:
            raise ValueError(f"cannot compute percentage change for zero prior value: {series_id}")
        return (new_value / prior_value - 1.0) * 100.0
    if series_id in _LEVEL_CHANGE_SERIES:
        return new_value - prior_value
    raise ValueError(f"unsupported V2.2 macro series: {series_id}")


def macro_feature_names() -> tuple[str, ...]:
    return tuple(
        f"{series}_{field}"
        for series in MACRO_SERIES
        for field in MACRO_PACKET_FIELDS
    )


def build_macro_features(
    ledger: MacroLedgerIndex,
    decision_timestamp: datetime,
) -> tuple[float, ...]:
    values: list[float] = []
    for series in MACRO_SERIES:
        values.extend(ledger.packet(series, decision_timestamp).as_tuple())
    result = tuple(values)
    if len(result) != 24:
        raise AssertionError("V2.2 macro feature vector must contain exactly 24 fields")
    if any(value != value or value in (float("inf"), float("-inf")) for value in result):
        raise ValueError("non-finite V2.2 macro feature")
    return result


def _parse_known_at(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("known_at must include an explicit timezone")
    return parsed.astimezone(timezone.utc)


def _parse_observation_date(value: str) -> date:
    return date.fromisoformat(value.strip())


def load_macro_ledger_csv(path: str | Path) -> tuple[MacroLedgerRow, ...]:
    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"series_id", "known_at", "observation_date", "value"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "macro ledger CSV requires columns: series_id,known_at,observation_date,value"
            )
        rows: list[MacroLedgerRow] = []
        for line_number, raw in enumerate(reader, start=2):
            try:
                series_id = str(raw["series_id"]).strip().upper()
                known_at = _parse_known_at(str(raw["known_at"]))
                observation_date = _parse_observation_date(str(raw["observation_date"]))
                value = float(str(raw["value"]).strip())
                rows.append(MacroLedgerRow(series_id, known_at, observation_date, value))
            except Exception as exc:  # pragma: no cover - contextualizes malformed external input
                raise ValueError(f"invalid macro ledger row at line {line_number}: {exc}") from exc
    return tuple(rows)


def ledger_summary(rows: Sequence[MacroLedgerRow]) -> Mapping[str, int]:
    counts = {series: 0 for series in MACRO_SERIES}
    for row in rows:
        counts[row.series_id] += 1
    return counts
