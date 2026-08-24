from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


class DepthSequenceError(RuntimeError):
    pass


@dataclass(slots=True)
class LocalOrderBook:
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    last_update_id: int | None = None
    valid: bool = False

    def reset(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.last_update_id = None
        self.valid = False

    def load_snapshot(self, payload: dict) -> None:
        self.bids = {
            float(price): float(qty)
            for price, qty in payload.get("bids", [])
            if float(qty) > 0.0
        }
        self.asks = {
            float(price): float(qty)
            for price, qty in payload.get("asks", [])
            if float(qty) > 0.0
        }
        self.last_update_id = int(payload["lastUpdateId"])
        self.valid = False

    @staticmethod
    def _apply_side(side: dict[float, float], updates: Iterable[list[str]]) -> None:
        for price_raw, qty_raw in updates:
            price = float(price_raw)
            qty = float(qty_raw)
            if qty == 0.0:
                side.pop(price, None)
            else:
                side[price] = qty

    def bridge(self, event: dict) -> bool:
        if self.last_update_id is None:
            raise DepthSequenceError("snapshot required before bridge")
        first = int(event["U"])
        final = int(event["u"])
        if final < self.last_update_id:
            return False
        if not (first <= self.last_update_id <= final):
            return False
        self._apply_side(self.bids, event.get("b", []))
        self._apply_side(self.asks, event.get("a", []))
        self.last_update_id = final
        self.valid = True
        self._validate_cross()
        return True

    def apply_diff(self, event: dict) -> None:
        if not self.valid or self.last_update_id is None:
            raise DepthSequenceError("book is not bridged")
        previous_final = int(event.get("pu", -1))
        if previous_final != self.last_update_id:
            self.valid = False
            raise DepthSequenceError(
                f"depth sequence gap: expected pu={self.last_update_id}, got pu={previous_final}"
            )
        self._apply_side(self.bids, event.get("b", []))
        self._apply_side(self.asks, event.get("a", []))
        self.last_update_id = int(event["u"])
        self._validate_cross()

    def _validate_cross(self) -> None:
        if not self.bids or not self.asks:
            self.valid = False
            raise DepthSequenceError("book side empty")
        if max(self.bids) >= min(self.asks):
            self.valid = False
            raise DepthSequenceError("crossed/locked local book")

    @property
    def best_bid(self) -> tuple[float, float]:
        price = max(self.bids)
        return price, self.bids[price]

    @property
    def best_ask(self) -> tuple[float, float]:
        price = min(self.asks)
        return price, self.asks[price]

    def top_bids(self, levels: int) -> list[tuple[float, float]]:
        return [(price, self.bids[price]) for price in sorted(self.bids, reverse=True)[:levels]]

    def top_asks(self, levels: int) -> list[tuple[float, float]]:
        return [(price, self.asks[price]) for price in sorted(self.asks)[:levels]]

    @staticmethod
    def _imbalance(bid_qty: float, ask_qty: float) -> float:
        total = bid_qty + ask_qty
        return (bid_qty - ask_qty) / total if total > 0.0 else 0.0

    def snapshot_metrics(self) -> dict[str, float | int | bool]:
        bid, bid_q = self.best_bid
        ask, ask_q = self.best_ask
        mid = (bid + ask) / 2.0
        spread_bps = (ask - bid) / mid * 10_000.0
        denom = bid_q + ask_q
        microprice = (ask * bid_q + bid * ask_q) / denom if denom > 0.0 else mid
        l5_bid = sum(q for _, q in self.top_bids(5))
        l5_ask = sum(q for _, q in self.top_asks(5))
        l10_bid = sum(q for _, q in self.top_bids(10))
        l10_ask = sum(q for _, q in self.top_asks(10))
        return {
            "best_bid": bid,
            "best_ask": ask,
            "bid_qty_l1": bid_q,
            "ask_qty_l1": ask_q,
            "mid": mid,
            "spread_bps": spread_bps,
            "microprice": microprice,
            "microprice_minus_mid_bps": (microprice - mid) / mid * 10_000.0,
            "bid_depth_l5": l5_bid,
            "ask_depth_l5": l5_ask,
            "bid_depth_l10": l10_bid,
            "ask_depth_l10": l10_ask,
            "obi_l1": self._imbalance(bid_q, ask_q),
            "obi_l5": self._imbalance(l5_bid, l5_ask),
            "obi_l10": self._imbalance(l10_bid, l10_ask),
            "depth_sequence_valid": self.valid,
            "last_depth_update_id": int(self.last_update_id or 0),
        }
