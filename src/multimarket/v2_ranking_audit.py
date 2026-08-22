from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Iterable

from .data import load_ohlc_csv
from .data_quality import audit_bars
from .models import Direction
from .v2_labels import build_economic_event
from .v2_model import V2Config, V2PriceOnlyPredictor
from .v2_volatility import volatility_scaled_barriers_bps


@dataclass(frozen=True, slots=True)
class RankRow:
    decision_timestamp: str
    p_long: float
    p_short: float
    ev_long_bps: float
    ev_short_bps: float
    realized_long_bps: float
    realized_short_bps: float


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_baseline_rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    if not rows:
        raise ValueError("baseline audit JSON contains no rows")
    return rows


def _average_ranks(values: Iterable[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(indexed)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for pos in range(start, end):
            result[indexed[pos][0]] = rank
        start = end
    return result


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    dx = [value - mx for value in x]
    dy = [value - my for value in y]
    denom = sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    if denom == 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denom


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    return _pearson(_average_ranks(x), _average_ranks(y))


def _top_fraction_mean(scores: list[float], outcomes: list[float], fraction: float) -> float | None:
    if not scores:
        return None
    count = max(1, int(round(len(scores) * fraction)))
    chosen = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:count]
    return sum(outcomes[i] for i in chosen) / len(chosen)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ranking-only diagnostic for V2 probability/EV scores on frozen timestamps"
    )
    p.add_argument("csv")
    p.add_argument("--symbol", required=True)
    p.add_argument("--baseline-json", required=True)
    p.add_argument("--horizon-bars", type=int, default=6)
    p.add_argument("--volatility-span", type=int, default=48)
    p.add_argument("--volatility-min-observations", type=int, default=24)
    p.add_argument("--take-profit-vol-multiplier", type=float, default=1.0)
    p.add_argument("--stop-loss-vol-multiplier", type=float, default=1.0)
    p.add_argument("--round-trip-cost-bps", type=float, default=2.0)
    p.add_argument("--min-train-rows", type=int, default=5000)
    p.add_argument("--retrain-every", type=int, default=500)
    p.add_argument("--validation-fraction", type=float, default=0.20)
    p.add_argument("--embargo-bars", type=int, default=6)
    p.add_argument("--output-json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bars = load_ohlc_csv(args.csv)
    quality = audit_bars(bars, symbol=args.symbol)
    hard_eligible = {
        row.index
        for row in quality
        if row.session_eligible and not row.zero_range and not row.repeated_ohlc
    }
    frozen_rows = _load_baseline_rows(Path(args.baseline_json))
    index_by_timestamp = {bar.timestamp.astimezone(timezone.utc): i for i, bar in enumerate(bars)}

    config = V2Config(
        horizon_bars=args.horizon_bars,
        volatility_span=args.volatility_span,
        volatility_min_observations=args.volatility_min_observations,
        take_profit_vol_multiplier=args.take_profit_vol_multiplier,
        stop_loss_vol_multiplier=args.stop_loss_vol_multiplier,
        round_trip_cost_bps=args.round_trip_cost_bps,
        min_train_rows=args.min_train_rows,
        retrain_every=args.retrain_every,
        validation_fraction=args.validation_fraction,
        min_positive_probability=0.50,
        min_predicted_ev_bps=-1.0e12,
        embargo_bars=args.embargo_bars,
    )
    predictor = V2PriceOnlyPredictor(bars, config, eligible_indices=hard_eligible)

    rows: list[RankRow] = []
    blocked_feature = blocked_event = unavailable_data = unavailable_training = 0

    for frozen in frozen_rows:
        ts = _parse_timestamp(str(frozen["decision_timestamp"]))
        idx = index_by_timestamp.get(ts)
        if idx is None:
            unavailable_data += 1
            continue
        if idx not in predictor.clean_feature_indices:
            blocked_feature += 1
            continue
        vol = predictor.volatility[idx]
        if vol is None or vol <= 0:
            blocked_feature += 1
            continue
        tp_bps, sl_bps = volatility_scaled_barriers_bps(
            vol,
            horizon_bars=args.horizon_bars,
            take_profit_vol_multiplier=args.take_profit_vol_multiplier,
            stop_loss_vol_multiplier=args.stop_loss_vol_multiplier,
        )
        event = build_economic_event(
            bars,
            idx,
            horizon_bars=args.horizon_bars,
            take_profit_bps=tp_bps,
            stop_loss_bps=sl_bps,
            round_trip_cost_bps=args.round_trip_cost_bps,
            eligible_indices=hard_eligible,
        )
        if event is None:
            blocked_event += 1
            continue
        try:
            decision = predictor.predict_at_index(idx)
        except ValueError as exc:
            if "labeled historical rows" in str(exc):
                unavailable_training += 1
            else:
                unavailable_data += 1
            continue
        rows.append(
            RankRow(
                decision_timestamp=ts.isoformat().replace("+00:00", "Z"),
                p_long=decision.probability_long_positive,
                p_short=decision.probability_short_positive,
                ev_long_bps=decision.predicted_long_ev_bps,
                ev_short_bps=decision.predicted_short_ev_bps,
                realized_long_bps=event.long.net_return_bps,
                realized_short_bps=event.short.net_return_bps,
            )
        )

    p_long = [row.p_long for row in rows]
    p_short = [row.p_short for row in rows]
    ev_long = [row.ev_long_bps for row in rows]
    ev_short = [row.ev_short_bps for row in rows]
    real_long = [row.realized_long_bps for row in rows]
    real_short = [row.realized_short_bps for row in rows]

    metrics = {
        "spearman_p_long_vs_realized_long": _spearman(p_long, real_long),
        "spearman_p_short_vs_realized_short": _spearman(p_short, real_short),
        "spearman_ev_long_vs_realized_long": _spearman(ev_long, real_long),
        "spearman_ev_short_vs_realized_short": _spearman(ev_short, real_short),
        "top_20pct_p_long_mean_realized_bps": _top_fraction_mean(p_long, real_long, 0.20),
        "top_20pct_p_short_mean_realized_bps": _top_fraction_mean(p_short, real_short, 0.20),
        "top_20pct_ev_long_mean_realized_bps": _top_fraction_mean(ev_long, real_long, 0.20),
        "top_20pct_ev_short_mean_realized_bps": _top_fraction_mean(ev_short, real_short, 0.20),
    }

    print(f"Multi-Market V2 ranking diagnostic | {args.symbol}")
    print("=" * 78)
    print(f"Frozen samples           : {len(frozen_rows)}")
    print(f"Scored eligible rows     : {len(rows)}")
    print(f"Blocked feature history  : {blocked_feature}")
    print(f"Blocked future event     : {blocked_event}")
    print(f"Unavailable training     : {unavailable_training}")
    print(f"Unavailable data/other   : {unavailable_data}")
    for key, value in metrics.items():
        if value is None:
            print(f"{key:36s}: n/a")
        else:
            print(f"{key:36s}: {value:+.4f}")

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "symbol": args.symbol,
            "version": "V2-RANKING-DIAGNOSTIC",
            "baseline_json": args.baseline_json,
            "frozen_samples": len(frozen_rows),
            "scored_eligible_rows": len(rows),
            "blocked_feature_history": blocked_feature,
            "blocked_future_event": blocked_event,
            "unavailable_training": unavailable_training,
            "unavailable_data_or_other": unavailable_data,
            "metrics": metrics,
            "rows": [asdict(row) for row in rows],
            "note": (
                "Diagnostic only. This audit does not define or change a trading threshold. "
                "It measures whether existing V2 scores rank realized economic outcomes."
            ),
        }
        output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Audit JSON               : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
