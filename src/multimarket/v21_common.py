from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .data import load_ohlc_csv
from .data_quality import audit_bars
from .v21_features import PeerMarket


def parse_peer(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError("--peer must use SYMBOL=CSV")
    symbol, path = value.split("=", 1)
    symbol = symbol.strip().upper()
    path = path.strip()
    if not symbol or not path:
        raise ValueError("--peer must use non-empty SYMBOL=CSV")
    return symbol, path


def hard_eligible_indices(bars, symbol: str) -> set[int]:
    return {
        row.index
        for row in audit_bars(list(bars), symbol=symbol)
        if row.session_eligible and not row.zero_range and not row.repeated_ohlc
    }


def load_peer_markets(values: Iterable[str], *, target_symbol: str) -> dict[str, PeerMarket]:
    parsed = [parse_peer(value) for value in values]
    symbols = [symbol for symbol, _ in parsed]
    if len(symbols) != len(set(symbols)):
        raise ValueError("duplicate --peer symbol")
    if target_symbol.upper() in symbols:
        raise ValueError("target symbol must not also be supplied as a peer")

    peers: dict[str, PeerMarket] = {}
    for symbol, path in parsed:
        bars = load_ohlc_csv(Path(path))
        peers[symbol] = PeerMarket.build(
            bars,
            eligible_indices=hard_eligible_indices(bars, symbol),
        )
    if not peers:
        raise ValueError("at least one --peer SYMBOL=CSV is required")
    return peers
