from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import websockets

from .v23_phase0d_book import DepthSequenceError, LocalOrderBook


DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT")
PUBLIC_WS_BASE = "wss://fstream.binance.com/public/stream?streams="
MARKET_WS_BASE = "wss://fstream.binance.com/market/stream?streams="
REST_DEPTH = "https://fapi.binance.com/fapi/v1/depth"
RAW_GZIP_COMPRESSLEVEL = 3
NORMALIZED_FIELDS = (
    "timestamp_utc", "symbol", "best_bid", "best_ask", "bid_qty_l1", "ask_qty_l1",
    "mid", "spread_bps", "microprice", "microprice_minus_mid_bps",
    "bid_depth_l5", "ask_depth_l5", "bid_depth_l10", "ask_depth_l10",
    "obi_l1", "obi_l5", "obi_l10", "agg_buy_qty_1s", "agg_sell_qty_1s",
    "agg_buy_count_1s", "agg_sell_count_1s", "trade_flow_imbalance_1s",
    "trade_count_imbalance_1s", "depth_sequence_valid", "last_depth_update_id",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utc_now().isoformat(timespec="microseconds")


def _build_stream_urls(symbols: tuple[str, ...]) -> tuple[str, str]:
    public_streams: list[str] = []
    market_streams: list[str] = []
    for symbol in symbols:
        s = symbol.lower()
        public_streams.extend([f"{s}@depth@100ms", f"{s}@bookTicker"])
        market_streams.append(f"{s}@aggTrade")
    return (
        PUBLIC_WS_BASE + "/".join(public_streams),
        MARKET_WS_BASE + "/".join(market_streams),
    )


def _classify_ws_event(stream: str, payload: dict[str, Any]) -> str:
    """Classify by exchange event type first, stream-name fallback second."""
    event_type = str(payload.get("e", ""))
    stream_lower = stream.lower()
    if event_type == "depthUpdate" or "@depth" in stream_lower:
        return "depth"
    if event_type.lower() == "aggtrade" or "@aggtrade" in stream_lower:
        return "agg_trade"
    if event_type == "bookTicker" or "@bookticker" in stream_lower:
        return "book_ticker"
    return "other"


class DailyJsonlWriter:
    """Append-only raw JSONL stored with lossless gzip compression."""

    def __init__(self, root: Path, symbol: str):
        self.root = root
        self.symbol = symbol
        self.day = None
        self.handle = None

    def write(self, record: dict[str, Any]) -> None:
        day = _utc_now().date().isoformat()
        if day != self.day:
            if self.handle:
                self.handle.close()
            path = self.root / "raw" / self.symbol
            path.mkdir(parents=True, exist_ok=True)
            file = path / f"{day}.jsonl.gz"
            self.handle = gzip.open(
                file,
                mode="at",
                encoding="utf-8",
                compresslevel=RAW_GZIP_COMPRESSLEVEL,
            )
            self.day = day
        assert self.handle is not None
        self.handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")

    def close(self) -> None:
        if self.handle:
            self.handle.close()


class DailyCsvWriter:
    def __init__(self, root: Path, symbol: str):
        self.root = root
        self.symbol = symbol
        self.day = None
        self.handle = None
        self.writer = None

    def write(self, record: dict[str, Any]) -> None:
        day = _utc_now().date().isoformat()
        if day != self.day:
            if self.handle:
                self.handle.close()
            path = self.root / "normalized" / self.symbol
            path.mkdir(parents=True, exist_ok=True)
            file = path / f"{day}.csv"
            existed = file.exists() and file.stat().st_size > 0
            self.handle = file.open("a", newline="", encoding="utf-8", buffering=1)
            self.writer = csv.DictWriter(self.handle, fieldnames=NORMALIZED_FIELDS)
            if not existed:
                self.writer.writeheader()
            self.day = day
        assert self.writer is not None
        self.writer.writerow({key: record.get(key) for key in NORMALIZED_FIELDS})

    def close(self) -> None:
        if self.handle:
            self.handle.close()


@dataclass(slots=True)
class TradeBucket:
    buy_qty: float = 0.0
    sell_qty: float = 0.0
    buy_count: int = 0
    sell_count: int = 0

    def add_agg_trade(self, payload: dict[str, Any]) -> None:
        qty = float(payload["q"])
        buyer_is_maker = bool(payload["m"])
        if buyer_is_maker:
            self.sell_qty += qty
            self.sell_count += 1
        else:
            self.buy_qty += qty
            self.buy_count += 1

    def clear(self) -> None:
        self.buy_qty = self.sell_qty = 0.0
        self.buy_count = self.sell_count = 0

    def consume(self) -> dict[str, float | int]:
        qty_total = self.buy_qty + self.sell_qty
        count_total = self.buy_count + self.sell_count
        result = {
            "agg_buy_qty_1s": self.buy_qty,
            "agg_sell_qty_1s": self.sell_qty,
            "agg_buy_count_1s": self.buy_count,
            "agg_sell_count_1s": self.sell_count,
            "trade_flow_imbalance_1s": (
                (self.buy_qty - self.sell_qty) / qty_total if qty_total > 0 else 0.0
            ),
            "trade_count_imbalance_1s": (
                (self.buy_count - self.sell_count) / count_total if count_total > 0 else 0.0
            ),
        }
        self.clear()
        return result


@dataclass(slots=True)
class SymbolState:
    symbol: str
    book: LocalOrderBook = field(default_factory=LocalOrderBook)
    buffered_depth: list[dict[str, Any]] = field(default_factory=list)
    snapshot_pending: bool = False
    trades: TradeBucket = field(default_factory=TradeBucket)


class Collector:
    def __init__(self, symbols: tuple[str, ...], output_dir: Path, snapshot_limit: int = 1000):
        self.symbols = symbols
        self.output_dir = output_dir
        self.snapshot_limit = snapshot_limit
        self.states = {symbol: SymbolState(symbol) for symbol in symbols}
        self.raw = {symbol: DailyJsonlWriter(output_dir, symbol) for symbol in symbols}
        self.normalized = {symbol: DailyCsvWriter(output_dir, symbol) for symbol in symbols}
        self.queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=200_000)
        self.http = httpx.AsyncClient(timeout=10.0)

    async def fetch_snapshot(self, symbol: str) -> None:
        try:
            response = await self.http.get(
                REST_DEPTH, params={"symbol": symbol, "limit": self.snapshot_limit}
            )
            response.raise_for_status()
            await self.queue.put(("snapshot", (symbol, response.json())))
        except Exception as exc:
            await self.queue.put(("snapshot_error", (symbol, repr(exc))))

    async def request_snapshot(self, state: SymbolState) -> None:
        if state.snapshot_pending:
            return
        state.snapshot_pending = True
        asyncio.create_task(self.fetch_snapshot(state.symbol))

    def _raw_record(self, symbol: str, stream: str, payload: dict[str, Any]) -> None:
        self.raw[symbol].write({
            "receive_time_utc": _iso_now(),
            "receive_time_ns": time.time_ns(),
            "stream": stream,
            "symbol": symbol,
            "exchange_event_time_ms": payload.get("E"),
            "payload": payload,
        })

    async def websocket_reader(self, channel: str, url: str) -> None:
        while True:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=60,
                    max_queue=100_000,
                    open_timeout=20,
                ) as ws:
                    async for raw in ws:
                        await self.queue.put(("ws", (channel, json.loads(raw))))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self.queue.put(("transport_error", (channel, repr(exc))))
                await asyncio.sleep(2.0)

    async def _handle_depth(self, state: SymbolState, payload: dict[str, Any]) -> None:
        if not state.book.valid:
            state.buffered_depth.append(payload)
            if len(state.buffered_depth) > 50_000:
                state.buffered_depth = state.buffered_depth[-10_000:]
            await self.request_snapshot(state)
            return
        try:
            state.book.apply_diff(payload)
        except DepthSequenceError as exc:
            self._raw_record(state.symbol, "integrity", {
                "E": payload.get("E"), "type": "DEPTH_SEQUENCE_GAP", "detail": str(exc)
            })
            state.book.reset()
            state.buffered_depth = [payload]
            state.trades.clear()
            await self.request_snapshot(state)

    async def _handle_snapshot(self, symbol: str, snapshot: dict[str, Any]) -> None:
        state = self.states[symbol]
        state.snapshot_pending = False
        state.book.load_snapshot(snapshot)
        last = int(snapshot["lastUpdateId"])
        buffered = [event for event in state.buffered_depth if int(event["u"]) >= last]
        state.buffered_depth = []
        bridged = False
        for event in buffered:
            if not bridged:
                if state.book.bridge(event):
                    bridged = True
                continue
            try:
                state.book.apply_diff(event)
            except DepthSequenceError:
                state.book.reset()
                state.buffered_depth = [event]
                state.trades.clear()
                await self.request_snapshot(state)
                return
        if not bridged:
            state.book.reset()
            state.trades.clear()
            await self.request_snapshot(state)

    async def processor(self) -> None:
        for state in self.states.values():
            await self.request_snapshot(state)
        while True:
            kind, item = await self.queue.get()
            if kind == "ws":
                channel, envelope = item
                stream = str(envelope.get("stream", ""))
                payload = envelope.get("data", envelope)
                symbol = str(payload.get("s", "")).upper()
                if symbol not in self.states:
                    continue
                # After the UM/CM stream migration, st=1 denotes USD-M and st=2 Coin-M.
                stream_type = payload.get("st")
                if stream_type is not None and int(stream_type) != 1:
                    continue
                self._raw_record(symbol, stream, payload)
                event_kind = _classify_ws_event(stream, payload)
                if event_kind == "depth":
                    await self._handle_depth(self.states[symbol], payload)
                elif event_kind == "agg_trade":
                    self.states[symbol].trades.add_agg_trade(payload)
                # bookTicker is intentionally preserved raw; reconstructed depth is authoritative.
            elif kind == "snapshot":
                symbol, snapshot = item
                await self._handle_snapshot(symbol, snapshot)
            elif kind == "snapshot_error":
                symbol, detail = item
                state = self.states[symbol]
                state.snapshot_pending = False
                state.trades.clear()
                self._raw_record(symbol, "integrity", {"type": "SNAPSHOT_ERROR", "detail": detail})
                await asyncio.sleep(1.0)
                await self.request_snapshot(state)
            elif kind == "transport_error":
                channel, detail = item
                for state in self.states.values():
                    self._raw_record(state.symbol, "integrity", {
                        "type": "WEBSOCKET_RECONNECT",
                        "channel": channel,
                        "detail": detail,
                    })
                    state.trades.clear()
                    if channel == "public":
                        state.book.reset()
                        state.buffered_depth.clear()
                        state.snapshot_pending = False
                        await self.request_snapshot(state)

    async def sampler(self) -> None:
        while True:
            now = time.time()
            await asyncio.sleep(max(0.0, 1.0 - (now % 1.0)))
            stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            for symbol, state in self.states.items():
                if not state.book.valid:
                    state.trades.clear()
                    continue
                trades = state.trades.consume()
                row = {"timestamp_utc": stamp, "symbol": symbol}
                row.update(state.book.snapshot_metrics())
                row.update(trades)
                self.normalized[symbol].write(row)

    async def run(self) -> None:
        public_url, market_url = _build_stream_urls(self.symbols)
        tasks = [
            asyncio.create_task(self.websocket_reader("public", public_url)),
            asyncio.create_task(self.websocket_reader("market", market_url)),
            asyncio.create_task(self.processor()),
            asyncio.create_task(self.sampler()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await self.http.aclose()
            for writer in self.raw.values():
                writer.close()
            for writer in self.normalized.values():
                writer.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect V2.3 Phase 0D Binance USD-M microstructure data")
    parser.add_argument("--symbol", action="append", dest="symbols", default=None)
    parser.add_argument("--output-dir", default="data/v23_phase0d_microstructure")
    parser.add_argument("--snapshot-limit", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    symbols = tuple(dict.fromkeys((args.symbols or DEFAULT_SYMBOLS)))
    symbols = tuple(symbol.upper() for symbol in symbols)
    unknown = sorted(set(symbols) - set(DEFAULT_SYMBOLS))
    if unknown:
        raise SystemExit(f"Phase 0D frozen targets only: {DEFAULT_SYMBOLS}; got {unknown}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    collector = Collector(symbols, output, snapshot_limit=args.snapshot_limit)
    try:
        asyncio.run(collector.run())
    except KeyboardInterrupt:
        print("collector_stopped=USER_INTERRUPT", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
