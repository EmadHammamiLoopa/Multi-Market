from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from .v22_macro import MACRO_SERIES, MacroLedgerRow


FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
BLS_YEAR_URL = "https://www.bls.gov/schedule/{year}/home.htm"
BEA_YEAR_URL = "https://www.bea.gov/news/schedule/full-{year}"
BEA_CURRENT_URL = "https://www.bea.gov/news/schedule/full"
USER_AGENT = "Multi-Market-Research/0.2.8 (+point-in-time macro audit)"
EASTERN = ZoneInfo("America/New_York")

_SERIES_FAMILY = {
    "CPIAUCSL": "CPI",
    "CPILFESL": "CPI",
    "PAYEMS": "EMPLOYMENT",
    "UNRATE": "EMPLOYMENT",
    "PCEPI": "PCE",
    "PCEPILFE": "PCE",
}

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        cleaned = " ".join(data.split())
        if cleaned:
            self.parts.append(cleaned)

    def text(self) -> str:
        return " ".join(self.parts)


def _fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def _fetch_json(url: str) -> dict[str, object]:
    return json.loads(_fetch_text(url))


def _html_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.text()


def _known_at_utc(year: int, month_name: str, day: int, clock: str) -> datetime:
    local = datetime.strptime(
        f"{year} {month_name} {day} {clock}", "%Y %B %d %I:%M %p"
    ).replace(tzinfo=EASTERN)
    return local.astimezone(timezone.utc)


def parse_bls_schedule(html: str, year: int) -> dict[str, dict[date, datetime]]:
    """Parse official BLS yearly calendar for CPI and Employment Situation releases."""
    text = _html_text(html)
    result: dict[str, dict[date, datetime]] = {"CPI": {}, "EMPLOYMENT": {}}
    pattern = re.compile(
        rf"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+"
        rf"(?P<month>{_MONTHS})\s+(?P<day>\d{{1,2}}),\s+(?P<year>\d{{4}})\s+"
        rf"(?P<time>\d{{1,2}}:\d{{2}}\s+[AP]M)\s+"
        rf"(?P<release>Consumer Price Index|Employment Situation)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        parsed_year = int(match.group("year"))
        if parsed_year != year:
            continue
        month_name = match.group("month")
        day = int(match.group("day"))
        release = match.group("release").lower()
        family = "CPI" if release.startswith("consumer") else "EMPLOYMENT"
        known_at = _known_at_utc(parsed_year, month_name, day, match.group("time").upper())
        result[family][known_at.date()] = known_at
    return result


def parse_bea_schedule(html: str, year: int) -> dict[date, datetime]:
    """Parse official BEA full-year schedule for Personal Income and Outlays."""
    text = _html_text(html)
    result: dict[date, datetime] = {}
    pattern = re.compile(
        rf"(?P<month>{_MONTHS})\s+(?P<day>\d{{1,2}})\s+"
        rf"(?P<time>\d{{1,2}}:\d{{2}}\s+[AP]M).*?"
        rf"Personal Income and Outlays",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        month_name = match.group("month")
        day = int(match.group("day"))
        known_at = _known_at_utc(year, month_name, day, match.group("time").upper())
        result[known_at.date()] = known_at
    return result


def load_official_release_schedule(start_year: int, end_year: int) -> dict[str, dict[date, datetime]]:
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")
    schedule: dict[str, dict[date, datetime]] = {
        "CPI": {},
        "EMPLOYMENT": {},
        "PCE": {},
    }
    for year in range(start_year, end_year + 1):
        bls = parse_bls_schedule(_fetch_text(BLS_YEAR_URL.format(year=year)), year)
        schedule["CPI"].update(bls["CPI"])
        schedule["EMPLOYMENT"].update(bls["EMPLOYMENT"])

        bea_url = BEA_CURRENT_URL if year == datetime.now(timezone.utc).year else BEA_YEAR_URL.format(year=year)
        try:
            bea_html = _fetch_text(bea_url)
        except Exception:
            # The current-year endpoint naming can vary; try the archived form too.
            bea_html = _fetch_text(BEA_YEAR_URL.format(year=year))
        schedule["PCE"].update(parse_bea_schedule(bea_html, year))
    return schedule


def fetch_alfred_changes(
    series_id: str,
    *,
    api_key: str,
    realtime_start: date,
    realtime_end: date,
    observation_start: date,
) -> list[dict[str, str]]:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "realtime_start": realtime_start.isoformat(),
        "realtime_end": realtime_end.isoformat(),
        "observation_start": observation_start.isoformat(),
        "output_type": "3",
        "limit": "100000",
    }
    payload = _fetch_json(f"{FRED_OBSERVATIONS_URL}?{urlencode(params)}")
    raw = payload.get("observations")
    if not isinstance(raw, list):
        raise ValueError(f"unexpected FRED response for {series_id}: no observations list")
    result: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        value = str(item.get("value", "")).strip()
        if value in {"", "."}:
            continue
        result.append({
            "realtime_start": str(item.get("realtime_start", "")),
            "date": str(item.get("date", "")),
            "value": value,
        })
    return result


def normalize_ledger_rows(
    raw_by_series: dict[str, list[dict[str, str]]],
    schedule: dict[str, dict[date, datetime]],
) -> tuple[list[MacroLedgerRow], dict[str, int]]:
    rows: list[MacroLedgerRow] = []
    excluded = {series: 0 for series in MACRO_SERIES}
    seen: set[tuple[str, datetime, date, float]] = set()

    for series_id in MACRO_SERIES:
        family = _SERIES_FAMILY[series_id]
        official = schedule[family]
        for raw in raw_by_series.get(series_id, []):
            realtime_day = date.fromisoformat(raw["realtime_start"])
            known_at = official.get(realtime_day)
            if known_at is None:
                # Preregistration rule: an unaudited date-only vintage is excluded,
                # never promoted to a guessed midnight/08:30 timestamp.
                excluded[series_id] += 1
                continue
            observation_date = date.fromisoformat(raw["date"])
            value = float(raw["value"])
            key = (series_id, known_at, observation_date, value)
            if key in seen:
                continue
            seen.add(key)
            rows.append(MacroLedgerRow(series_id, known_at, observation_date, value))

    rows.sort(key=lambda row: (row.known_at, row.series_id, row.observation_date, row.value))
    return rows, excluded


def write_ledger_csv(path: Path, rows: list[MacroLedgerRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("series_id", "known_at", "observation_date", "value"))
        for row in rows:
            writer.writerow((
                row.series_id,
                row.known_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                row.observation_date.isoformat(),
                format(row.value, ".15g"),
            ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the audited V2.2 point-in-time macro ledger from ALFRED/FRED and official BLS/BEA release schedules"
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--realtime-start", default="2024-01-01")
    parser.add_argument("--realtime-end", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--observation-start", default="2023-01-01")
    parser.add_argument("--api-key-env", default="FRED_API_KEY")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(
            f"Missing {args.api_key_env}. Set the FRED API key in your local environment; do not place it in source control."
        )

    realtime_start = date.fromisoformat(args.realtime_start)
    realtime_end = date.fromisoformat(args.realtime_end)
    observation_start = date.fromisoformat(args.observation_start)
    schedule = load_official_release_schedule(realtime_start.year, realtime_end.year)

    print("Official release timestamps")
    for family in ("CPI", "EMPLOYMENT", "PCE"):
        print(f"{family:10s}: {len(schedule[family])}")

    raw_by_series: dict[str, list[dict[str, str]]] = {}
    for series_id in MACRO_SERIES:
        rows = fetch_alfred_changes(
            series_id,
            api_key=api_key,
            realtime_start=realtime_start,
            realtime_end=realtime_end,
            observation_start=observation_start,
        )
        raw_by_series[series_id] = rows
        print(f"{series_id:10s}: ALFRED new/revised rows={len(rows)}")

    ledger, excluded = normalize_ledger_rows(raw_by_series, schedule)
    write_ledger_csv(Path(args.output_csv), ledger)

    print()
    print(f"Ledger rows written: {len(ledger)}")
    for series_id in MACRO_SERIES:
        kept = sum(row.series_id == series_id for row in ledger)
        print(f"{series_id:10s}: kept={kept:4d} excluded_unverified_timestamp={excluded[series_id]:4d}")
    print(f"Output: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
