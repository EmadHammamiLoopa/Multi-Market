from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo


_CRYPTO = {"BTCUSD", "ETHUSD", "BTC-USD", "ETH-USD"}
_FX_METALS = {"EURUSD", "GBPUSD", "USDJPY", "XAUUSD"}
_US_RTH = {"QQQ", "NDX", "SPX", "GSPC", "^GSPC", "^NDX"}
_NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True, slots=True)
class TradabilityDecision:
    eligible: bool
    reason: str


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("/", "")


def tradability_at(symbol: str, timestamp: datetime) -> TradabilityDecision:
    """Return a conservative structural session-eligibility decision.

    This is intentionally a coarse research/data-quality gate, not a broker-specific
    trading-hours guarantee. Crypto is treated as 24/7. FX/metals reject UTC weekend
    bars. US equity/index proxies use DST-aware New York regular trading hours.
    Unknown instruments are left eligible but explicitly marked as unknown-policy.
    """
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    normalized = normalize_symbol(symbol)

    if normalized in _CRYPTO:
        return TradabilityDecision(True, "crypto_24_7")

    if normalized in _FX_METALS:
        if timestamp.weekday() >= 5:
            return TradabilityDecision(False, "weekend_closed")
        return TradabilityDecision(True, "weekday_fx_metal")

    if normalized in _US_RTH:
        local = timestamp.astimezone(_NEW_YORK)
        if local.weekday() >= 5:
            return TradabilityDecision(False, "us_equity_weekend")
        local_time = local.timetz().replace(tzinfo=None)
        if time(9, 30) <= local_time < time(16, 0):
            return TradabilityDecision(True, "us_regular_hours")
        return TradabilityDecision(False, "outside_us_regular_hours")

    return TradabilityDecision(True, "unknown_policy_assumed_eligible")
