from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .data import load_ohlc_csv
from .v2_labels import build_economic_event
from .v2_model import V2Config
from .v2_ranking_audit import _spearman, _top_fraction_mean
from .v2_volatility import volatility_scaled_barriers_bps
from .v21_common import hard_eligible_indices, load_peer_markets
from .v21_model import V21Config, V21CrossMarketRegimePredictor


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


def _load_rows(path: Path) -> list[dict[str, object]]:
    rows = json.loads(path.read_text(encoding="utf-8")).get("rows") or []
    if not rows:
        raise ValueError("baseline audit JSON contains no rows")
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ranking diagnostic for frozen V2.1 cross-market+regime scores")
    p.add_argument("csv")
    p.add_argument("--symbol", required=True)
    p.add_argument("--baseline-json", required=True)
    p.add_argument("--peer", action="append", default=[], metavar="SYMBOL=CSV")
    p.add_argument("--max-peer-staleness-minutes", type=int, default=15)
    p.add_argument("--output-json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bars = load_ohlc_csv(args.csv)
    hard_eligible = hard_eligible_indices(bars, args.symbol)
    peers = load_peer_markets(args.peer, target_symbol=args.symbol)
    frozen_rows = _load_rows(Path(args.baseline_json))
    index_by_timestamp = {bar.timestamp.astimezone(timezone.utc): i for i, bar in enumerate(bars)}

    # Ranking-only mode preserves all V2 training/model settings but removes the
    # decision threshold so every otherwise eligible timestamp can be scored.
    base = V2Config(min_positive_probability=0.50, min_predicted_ev_bps=-1.0e12)
    predictor = V21CrossMarketRegimePredictor(
        bars,
        V21Config(base=base, max_peer_staleness_minutes=args.max_peer_staleness_minutes),
        eligible_indices=hard_eligible,
        peers=peers,
    )

    rows: list[RankRow] = []
    blocked_feature = blocked_event = unavailable_data = unavailable_training = 0
    for frozen in frozen_rows:
        ts = _parse_timestamp(str(frozen["decision_timestamp"]))
        idx = index_by_timestamp.get(ts)
        if idx is None:
            unavailable_data += 1
            continue
        if idx not in predictor.feature_by_index:
            blocked_feature += 1
            continue
        vol = predictor.volatility[idx]
        if vol is None or vol <= 0:
            blocked_feature += 1
            continue
        tp_bps, sl_bps = volatility_scaled_barriers_bps(
            vol,
            horizon_bars=base.horizon_bars,
            take_profit_vol_multiplier=base.take_profit_vol_multiplier,
            stop_loss_vol_multiplier=base.stop_loss_vol_multiplier,
        )
        event = build_economic_event(
            bars,
            idx,
            horizon_bars=base.horizon_bars,
            take_profit_bps=tp_bps,
            stop_loss_bps=sl_bps,
            round_trip_cost_bps=base.round_trip_cost_bps,
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

    print(f"Multi-Market V2.1 ranking diagnostic | {args.symbol}")
    print("=" * 78)
    print(f"Frozen samples           : {len(frozen_rows)}")
    print(f"Scored eligible rows     : {len(rows)}")
    print(f"Blocked feature history  : {blocked_feature}")
    print(f"Blocked future event     : {blocked_event}")
    print(f"Unavailable training     : {unavailable_training}")
    print(f"Unavailable data/other   : {unavailable_data}")
    for key, value in metrics.items():
        print(f"{key:36s}: {value:+.4f}" if value is not None else f"{key:36s}: n/a")

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "symbol": args.symbol,
            "version": "V2.1-CROSS-MARKET-REGIME-RANKING",
            "baseline_json": args.baseline_json,
            "peers": sorted(peers),
            "feature_count": len(predictor.feature_names),
            "max_peer_staleness_minutes": args.max_peer_staleness_minutes,
            "frozen_samples": len(frozen_rows),
            "scored_eligible_rows": len(rows),
            "blocked_feature_history": blocked_feature,
            "blocked_future_event": blocked_event,
            "unavailable_training": unavailable_training,
            "unavailable_data_or_other": unavailable_data,
            "metrics": metrics,
            "rows": [asdict(row) for row in rows],
            "note": "Diagnostic only; no V2.1 threshold is defined or changed by this ranking audit.",
        }
        output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Audit JSON               : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
