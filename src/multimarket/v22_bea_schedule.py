from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")

# Frozen official BEA Personal Income and Outlays release timestamps for the
# preregistered V2.2 evidence interval 2024-01-01..2026-08-22.
#
# Official sources include BEA archived PIO releases and post-shutdown schedule
# updates. Most releases are 08:30 America/New_York. Explicit 10:00 releases
# are retained below. The December 23, 2025 PIO data update is included because
# it revised July-September data and was published through BEA's API/tables.
#
# Sources:
#   https://www.bea.gov/news/archive?field_related_product_target_id=476
#   https://www.bea.gov/news/blog/2026-01-07/economic-release-schedule-updates-gdp-personal-income-and-outlays
#   https://www.bea.gov/news/blog/2026-01-15/economic-release-schedule-updates-gdp-personal-income-and-outlays
#   individual archived Personal Income and Outlays releases.

_RELEASES: tuple[tuple[str, str], ...] = (
    # 2024
    ("2024-01-26", "08:30"),
    ("2024-02-29", "08:30"),
    ("2024-03-29", "08:30"),
    ("2024-04-26", "08:30"),
    ("2024-05-31", "08:30"),
    ("2024-06-28", "08:30"),
    ("2024-07-26", "08:30"),
    ("2024-08-30", "08:30"),
    ("2024-09-27", "08:30"),
    ("2024-10-31", "08:30"),
    ("2024-11-27", "08:30"),
    ("2024-12-20", "08:30"),
    # 2025
    ("2025-01-31", "08:30"),
    ("2025-02-28", "08:30"),
    ("2025-03-28", "08:30"),
    ("2025-04-30", "10:00"),
    ("2025-05-30", "08:30"),
    ("2025-06-27", "08:30"),
    ("2025-07-31", "08:30"),
    ("2025-08-29", "08:30"),
    ("2025-09-26", "08:30"),
    ("2025-12-05", "10:00"),
    ("2025-12-23", "08:30"),
    # 2026 through frozen realtime end
    ("2026-01-22", "10:00"),
    ("2026-02-20", "08:30"),
    ("2026-03-13", "08:30"),
    ("2026-04-09", "08:30"),
    ("2026-04-30", "08:30"),
    ("2026-05-28", "08:30"),
    ("2026-06-25", "08:30"),
    ("2026-07-30", "08:30"),
)


def _known_at(raw_day: str, raw_clock: str) -> datetime:
    day = date.fromisoformat(raw_day)
    hour, minute = (int(part) for part in raw_clock.split(":"))
    local = datetime(day.year, day.month, day.day, hour, minute, tzinfo=EASTERN)
    return local.astimezone(timezone.utc)


def frozen_bea_pce_schedule(start_year: int, end_year: int) -> dict[date, datetime]:
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")
    result: dict[date, datetime] = {}
    for raw_day, raw_clock in _RELEASES:
        known_at = _known_at(raw_day, raw_clock)
        if start_year <= known_at.year <= end_year:
            result[known_at.date()] = known_at
    return result
