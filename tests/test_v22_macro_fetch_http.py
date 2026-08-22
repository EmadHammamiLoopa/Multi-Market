from __future__ import annotations

import io
import unittest
from email.message import Message
from unittest.mock import patch
from urllib.error import HTTPError

from multimarket.v22_macro_fetch_http import _fetch_text


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class V22MacroFetchHttpTests(unittest.TestCase):
    def test_403_retries_once_with_browser_compatible_headers(self) -> None:
        error = HTTPError(
            "https://www.bls.gov/schedule/2026/home.htm",
            403,
            "Forbidden",
            Message(),
            io.BytesIO(b"forbidden"),
        )
        with patch(
            "multimarket.v22_macro_fetch_http.urlopen",
            side_effect=[error, _Response(b"<html>ok</html>")],
        ) as mocked:
            text = _fetch_text("https://www.bls.gov/schedule/2026/home.htm")

        self.assertEqual(text, "<html>ok</html>")
        self.assertEqual(mocked.call_count, 2)
        first_request = mocked.call_args_list[0].args[0]
        second_request = mocked.call_args_list[1].args[0]
        self.assertIn("Multi-Market-Research", first_request.get_header("User-agent"))
        self.assertIn("Mozilla/5.0", second_request.get_header("User-agent"))

    def test_non_403_http_error_is_not_retried(self) -> None:
        error = HTTPError(
            "https://example.invalid",
            404,
            "Not Found",
            Message(),
            io.BytesIO(b"missing"),
        )
        with patch("multimarket.v22_macro_fetch_http.urlopen", side_effect=error) as mocked:
            with self.assertRaises(HTTPError):
                _fetch_text("https://example.invalid")
        self.assertEqual(mocked.call_count, 1)


if __name__ == "__main__":
    unittest.main()
