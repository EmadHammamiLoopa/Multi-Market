from __future__ import annotations

from . import v22_macro_fetch as _base
from . import v22_macro_fetch_http as _http
from . import v22_macro_fetch_realtime as _realtime
from .v22_bea_schedule import frozen_bea_pce_schedule
from .v22_bls_schedule import frozen_bls_schedule


def load_official_release_schedule(start_year: int, end_year: int):
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")
    bls = frozen_bls_schedule(start_year, end_year)
    return {
        "CPI": dict(bls["CPI"]),
        "EMPLOYMENT": dict(bls["EMPLOYMENT"]),
        "PCE": dict(frozen_bea_pce_schedule(start_year, end_year)),
    }


def main(argv: list[str] | None = None) -> int:
    # Freeze both release calendars. FRED/ALFRED remains the only live data
    # source; schedule acquisition is fully reproducible and independent of
    # HTML parsing or later website edits.
    _base._fetch_text = _http._fetch_text
    _base._fetch_json = _http._fetch_json
    _base.fetch_alfred_changes = _realtime.fetch_alfred_changes
    _base.load_official_release_schedule = load_official_release_schedule
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
