from __future__ import annotations

from datetime import datetime, timezone

from . import v22_macro_fetch as _base
from . import v22_macro_fetch_http as _http
from .v22_bls_schedule import frozen_bls_schedule


def load_official_release_schedule(start_year: int, end_year: int):
    """Load frozen official BLS dates plus live official BEA PCE schedule.

    BLS is intentionally not contacted at runtime. Its release dates for the
    preregistered 2024-01-01..2026-08-22 evidence window are checked into the
    repository for reproducibility and to avoid BLS automated-access blocking.
    BEA remains fetched from its official schedule pages because those endpoints
    are accessible from the research runtime.
    """
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")

    bls = frozen_bls_schedule(start_year, end_year)
    schedule = {
        "CPI": dict(bls["CPI"]),
        "EMPLOYMENT": dict(bls["EMPLOYMENT"]),
        "PCE": {},
    }

    for year in range(start_year, end_year + 1):
        bea_url = (
            _base.BEA_CURRENT_URL
            if year == datetime.now(timezone.utc).year
            else _base.BEA_YEAR_URL.format(year=year)
        )
        try:
            bea_html = _http._fetch_text(bea_url)
        except Exception:
            bea_html = _http._fetch_text(_base.BEA_YEAR_URL.format(year=year))
        schedule["PCE"].update(_base.parse_bea_schedule(bea_html, year))

    return schedule


def main(argv: list[str] | None = None) -> int:
    # Patch only transport and schedule acquisition. Parsing, ALFRED vintage
    # normalization, masking and all V2.2 preregistered model semantics remain
    # unchanged.
    _base._fetch_text = _http._fetch_text
    _base._fetch_json = _http._fetch_json
    _base.load_official_release_schedule = load_official_release_schedule
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
