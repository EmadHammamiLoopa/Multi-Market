from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .data import load_ohlc_csv
from .models import Direction
from .v2_labels import build_economic_event
from .v2_model import V2Config
from .v2_volatility import volatility_scaled_barriers_bps
from .v21_common import hard_eligible_indices, load_peer_markets
from .v21_model import V21Config, V21CrossMarketRegimePredictor


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
    p = argparse.ArgumentParser(description="Evaluate V2.1 cross-market+regime model on frozen timestamps")
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

    base = V2Config()
    predictor = V21CrossMarketRegimePredictor(
        bars,
        V21Config(base=base, max_peer_staleness_minutes=args.max_peer_staleness_minutes),
        eligible_indices=hard_eligible,
        peers=peers,
    )

    print(f"Multi-Market V2.1 cross-market+regime paired audit | {args.symbol}")
    print("=" * 116)
    print(
        f"Policy frozen: H={base.horizon_bars}, EWMA={base.volatility_span}, "
        f"TP/SL={base.take_profit_vol_multiplier:.2f}x/{base.stop_loss_vol_multiplier:.2f}x, "
        f"cost={base.round_trip_cost_bps:.2f}bp, p>={base.min_positive_probability:.2f}, "
        f"EV>{base.min_predicted_ev_bps:.2f}bp, embargo={base.embargo_bars}, "
        f"peer_stale<={args.max_peer_staleness_minutes}m"
    )
    print(f"Features: {len(predictor.feature_names)} ({', '.join(sorted(peers))} peers)")

    output_rows: list[dict[str, object]] = []
    gated = unavailable = no_trade = actionable = winners = losers = 0
    realized_nets: list[float] = []

    for frozen in frozen_rows:
        decision_ts = _parse_timestamp(str(frozen["decision_timestamp"]))
        idx = index_by_timestamp.get(decision_ts)
        if idx is None:
            unavailable += 1
            continue
        if idx not in predictor.feature_by_index:
            gated += 1
            no_trade += 1
            output_rows.append({"decision_timestamp": decision_ts.isoformat().replace("+00:00", "Z"), "prediction_direction": Direction.NO_TRADE.value, "gate": "BLOCK_FEATURE_HISTORY"})
            continue
        vol = predictor.volatility[idx]
        if vol is None or vol <= 0:
            gated += 1
            no_trade += 1
            output_rows.append({"decision_timestamp": decision_ts.isoformat().replace("+00:00", "Z"), "prediction_direction": Direction.NO_TRADE.value, "gate": "BLOCK_VOLATILITY"})
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
            gated += 1
            no_trade += 1
            output_rows.append({"decision_timestamp": decision_ts.isoformat().replace("+00:00", "Z"), "prediction_direction": Direction.NO_TRADE.value, "gate": "BLOCK_FUTURE_EVENT"})
            continue
        try:
            decision = predictor.predict_at_index(idx)
        except ValueError:
            unavailable += 1
            continue

        direction = decision.prediction.direction
        realized_net = None
        verdict = "NO_TRADE"
        exit_reason = None
        if direction is Direction.NO_TRADE:
            no_trade += 1
        else:
            actionable += 1
            outcome = event.long if direction is Direction.LONG else event.short
            realized_net = outcome.net_return_bps
            exit_reason = outcome.exit_reason
            realized_nets.append(realized_net)
            if realized_net > 0.0:
                winners += 1
                verdict = "NET_POSITIVE"
            else:
                losers += 1
                verdict = "NET_NONPOSITIVE"

        net_text = "-" if realized_net is None else f"{realized_net:+7.2f}bp"
        print(
            f"{decision_ts.strftime('%Y-%m-%d %H:%MZ'):19s} {direction.value:9s} "
            f"pL={decision.probability_long_positive:5.1%} pS={decision.probability_short_positive:5.1%} "
            f"evL={decision.predicted_long_ev_bps:+6.2f} evS={decision.predicted_short_ev_bps:+6.2f} "
            f"real={net_text:>9s} {verdict}"
        )
        output_rows.append({
            "decision_timestamp": decision_ts.isoformat().replace("+00:00", "Z"),
            "prediction_direction": direction.value,
            "probability_long_positive": decision.probability_long_positive,
            "probability_short_positive": decision.probability_short_positive,
            "predicted_long_ev_bps": decision.predicted_long_ev_bps,
            "predicted_short_ev_bps": decision.predicted_short_ev_bps,
            "realized_net_bps": realized_net,
            "exit_reason": exit_reason,
            "verdict": verdict,
            "gate": "PASS",
            "tp_bps": tp_bps,
            "sl_bps": sl_bps,
        })

    win_rate = winners / actionable if actionable else None
    expectancy = sum(realized_nets) / len(realized_nets) if realized_nets else None
    gains = sum(v for v in realized_nets if v > 0.0)
    losses_abs = -sum(v for v in realized_nets if v < 0.0)
    if losses_abs > 0.0:
        profit_factor = gains / losses_abs
        profit_factor_json = profit_factor
    elif gains > 0.0:
        profit_factor = float("inf")
        profit_factor_json = None
    else:
        profit_factor = None
        profit_factor_json = None

    print("\nAudit summary")
    print("=" * 64)
    print(f"Frozen samples         : {len(frozen_rows)}")
    print(f"Gate-blocked           : {gated}")
    print(f"Unavailable            : {unavailable}")
    print(f"Actionable             : {actionable}")
    print(f"Net-positive           : {winners}")
    print(f"Net-nonpositive        : {losers}")
    print(f"NO_TRADE               : {no_trade}")
    print(f"Selected win rate      : {win_rate:.2%}" if win_rate is not None else "Selected win rate      : n/a")
    print(f"Mean net expectancy    : {expectancy:+.3f} bp/trade" if expectancy is not None else "Mean net expectancy    : n/a")
    if profit_factor == float("inf"):
        print("Profit factor          : inf")
    else:
        print(f"Profit factor          : {profit_factor:.3f}" if profit_factor is not None else "Profit factor          : n/a")

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "symbol": args.symbol,
            "version": "V2.1-CROSS-MARKET-REGIME",
            "baseline_json": args.baseline_json,
            "peers": sorted(peers),
            "feature_count": len(predictor.feature_names),
            "max_peer_staleness_minutes": args.max_peer_staleness_minutes,
            "frozen_v2_config": vars(base),
            "frozen_samples": len(frozen_rows),
            "gate_blocked": gated,
            "unavailable": unavailable,
            "actionable": actionable,
            "net_positive": winners,
            "net_nonpositive": losers,
            "no_trade": no_trade,
            "selected_win_rate": win_rate,
            "mean_net_expectancy_bps": expectancy,
            "profit_factor": profit_factor_json,
            "profit_factor_is_infinite": profit_factor == float("inf"),
            "rows": output_rows,
        }
        output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Audit JSON             : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
