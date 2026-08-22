from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from . import v22_macro_fetch as _base


PRIMARY_USER_AGENT = (
    "Multi-Market-Research/0.2.9 "
    "(+https://github.com/EmadHammamiLoopa/Multi-Market)"
)
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/151.0.0.0 Safari/537.36 "
    "Multi-Market-Research/0.2.9 "
    "(+https://github.com/EmadHammamiLoopa/Multi-Market)"
)


def _request_headers(user_agent: str) -> dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Accept": "text/html,application/json,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }


def _fetch_text(url: str) -> str:
    """Fetch an official source with one explicit 403-compatible retry.

    BLS documents automated-access controls and can reject non-browser-looking
    clients. The first request identifies this research client and includes a
    contact URL. If the server returns HTTP 403, retry exactly once with a
    browser-compatible User-Agent while preserving the same project contact URL.
    No alternate/non-official data source is used.
    """
    last_error: HTTPError | None = None
    for user_agent in (PRIMARY_USER_AGENT, BROWSER_USER_AGENT):
        request = Request(url, headers=_request_headers(user_agent))
        try:
            with urlopen(request, timeout=60) as response:
                return response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            last_error = exc
            if exc.code != 403 or user_agent == BROWSER_USER_AGENT:
                raise

    assert last_error is not None
    raise last_error


def _fetch_json(url: str) -> dict[str, object]:
    return json.loads(_fetch_text(url))


def main(argv: list[str] | None = None) -> int:
    # Patch only transport. All schedule parsing, ALFRED normalization,
    # revision handling and V2.2 preregistered semantics remain unchanged.
    _base._fetch_text = _fetch_text
    _base._fetch_json = _fetch_json
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
