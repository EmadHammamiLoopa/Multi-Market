from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from .models import MarketBar

_REQUIRED = {"timestamp", "open", "high", "low", "close"}


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def load_ohlc_csv(path: str | Path) -> list[MarketBar]:
    bars: list[MarketBar] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = _REQUIRED - fieldnames
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            try:
                bars.append(
                    MarketBar(
                        timestamp=_parse_timestamp(row["timestamp"]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid row {row_number}: {exc}") from exc

    if not bars:
        raise ValueError("CSV contains no bars")

    bars.sort(key=lambda bar: bar.timestamp)
    timestamps = [bar.timestamp for bar in bars]
    if len(timestamps) != len(set(timestamps)):
        raise ValueError("Timestamps must be unique")
    return bars
