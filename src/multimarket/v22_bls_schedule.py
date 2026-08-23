from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")

# Frozen official BLS release dates for the V2.2 macro-ledger evidence window.
# Scope: 2024-01-01 through 2026-08-22 only.
# Sources:
#   https://www.bls.gov/schedule/2024/home.htm
#   https://www.bls.gov/schedule/2025/home.htm
#   https://www.bls.gov/schedule/2026/home.htm
#   https://www.bls.gov/bls/2025-lapse-revised-release-dates.htm
# All CPI and Employment Situation releases below are 08:30 America/New_York.
# Canceled releases during the 2025 lapse are intentionally absent.

_CPI_RELEASE_DATES = (
    # 2024
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10",
    "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
    "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
    # 2025
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-15", "2025-08-12",
    "2025-09-11", "2025-10-24", "2025-12-18",
    # 2026 through frozen realtime end
    "2026-01-13", "2026-02-13", "2026-03-11", "2026-04-10",
    "2026-05-12", "2026-06-10", "2026-07-14", "2026-08-12",
)

_EMPLOYMENT_RELEASE_DATES = (
    # 2024
    "2024-01-05", "2024-02-02", "2024-03-08", "2024-04-05",
    "2024-05-03", "2024-06-07", "2024-07-05", "2024-08-02",
    "2024-09-06", "2024-10-04", "2024-11-01", "2024-12-06",
    # 2025
    "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04",
    "2025-05-02", "2025-06-06", "2025-07-03", "2025-08-01",
    "2025-09-05", "2025-11-20", "2025-12-16",
    # 2026 through frozen realtime end
    "2026-01-09", "2026-02-11", "2026-03-06", "2026-04-03",
    "2026-05-08", "2026-06-05", "2026-07-02", "2026-08-07",
)


def _known_at(raw: str) -> datetime:
    day = date.fromisoformat(raw)
    local = datetime(day.year, day.month, day.day, 8, 30, tzinfo=EASTERN)
    return local.astimezone(timezone.utc)


def frozen_bls_schedule(start_year: int, end_year: int) -> dict[str, dict[date, datetime]]:
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")
    cpi = {
        ts.date(): ts
        for ts in map(_known_at, _CPI_RELEASE_DATES)
        if start_year <= ts.year <= end_year
    }
    employment = {
        ts.date(): ts
        for ts in map(_known_at, _EMPLOYMENT_RELEASE_DATES)
        if start_year <= ts.year <= end_year
    }
    return {"CPI": cpi, "EMPLOYMENT": employment}
