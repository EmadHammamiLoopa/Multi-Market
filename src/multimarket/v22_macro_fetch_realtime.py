from __future__ import annotations

from datetime import date
from urllib.parse import urlencode

from . import v22_macro_fetch as _base
from . import v22_macro_fetch_static as _static


def fetch_alfred_changes(
    series_id: str,
    *,
    api_key: str,
    realtime_start: date,
    realtime_end: date,
    observation_start: date,
) -> list[dict[str, str]]:
    """Return point-in-time initial/revised values in a standard row schema.

    FRED output_type=3 is a vintage-date cross-tab, not the normal
    date/value/realtime_start row representation. The V2.2 ledger needs the
    latter because ``realtime_start`` is the date on which a value first became
    the active public revision. FRED output_type=1 (Observations by Real-Time
    Period) supplies exactly that schema.

    Rows whose real-time period began before the frozen requested interval are
    carry-in state, not new/revised observations inside the interval, so they
    are excluded here. Normalization later maps each retained date to an
    independently verified official release timestamp and drops unverified
    dates rather than guessing an intraday time.
    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "realtime_start": realtime_start.isoformat(),
        "realtime_end": realtime_end.isoformat(),
        "observation_start": observation_start.isoformat(),
        "output_type": "1",
        "limit": "100000",
    }
    payload = _base._fetch_json(f"{_base.FRED_OBSERVATIONS_URL}?{urlencode(params)}")
    raw = payload.get("observations")
    if not isinstance(raw, list):
        raise ValueError(f"unexpected FRED response for {series_id}: no observations list")

    result: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        realtime_raw = str(item.get("realtime_start", "")).strip()
        observation_raw = str(item.get("date", "")).strip()
        value = str(item.get("value", "")).strip()
        if not realtime_raw or not observation_raw or value in {"", "."}:
            continue
        try:
            revision_day = date.fromisoformat(realtime_raw)
        except ValueError:
            continue
        if revision_day < realtime_start or revision_day > realtime_end:
            continue
        key = (realtime_raw, observation_raw, value)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "realtime_start": realtime_raw,
                "date": observation_raw,
                "value": value,
            }
        )
    return result


def main(argv: list[str] | None = None) -> int:
    # Patch only the FRED response representation. The frozen BLS schedule,
    # official BEA schedule, normalization, masking, model policy and holdouts
    # remain unchanged.
    _base.fetch_alfred_changes = fetch_alfred_changes
    return _static.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
