from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .data import load_ohlc_csv
from .data_quality import audit_bars
from .models import Direction
from .v2_labels import build_economic_event
from .v2_model import V2Config, V2PriceOnlyPredictor
from .v2_volatility import volatility_scaled_barriers_bps


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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate V2 price-only predictions on frozen historical decision timestamps")
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
    p.add_argument("--min-positive-probability", type=float, default=0.55)
    p.add_argument("--min-predicted-ev-bps", type=float, default=0.0)
    p.add_argument("--embargo-bars", type=int, default=6)
    p.add_argument("--output-json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bars = load_ohlc_csv(args.csv)
    quality = audit_bars(bars, symbol=args.symbol)
    hard_eligible = {
        row.index for row in quality
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
        min_positive_probability=args.min_positive_probability,
        min_predicted_ev_bps=args.min_predicted_ev_bps,
        embargo_bars=args.embargo_bars,
    )
    predictor = V2PriceOnlyPredictor(bars, config, eligible_indices=hard_eligible)

    print(f"Multi-Market V2 price-only paired audit | {args.symbol}")
    print("=" * 116)
    print(
        f"Policy: H={args.horizon_bars}, EWMA={args.volatility_span}, "
        f"TP/SL={args.take_profit_vol_multiplier:.2f}x/{args.stop_loss_vol_multiplier:.2f}x, "
        f"cost={args.round_trip_cost_bps:.2f}bp, p>={args.min_positive_probability:.2f}, "
        f"EV>{args.min_predicted_ev_bps:.2f}bp, embargo={args.embargo_bars}"
    )

    output_rows: list[dict[str, object]] = []
    gated = unavailable = no_trade = actionable = winners = losers = 0
    realized_nets: list[float] = []

    for frozen in frozen_rows:
        decision_ts = _parse_timestamp(str(frozen["decision_timestamp"]))
        idx = index_by_timestamp.get(decision_ts)
        if idx is None:
            unavailable += 1
            continue
        if idx not in predictor.clean_feature_indices:
            gated += 1
            no_trade += 1
            print(f"{decision_ts.strftime('%Y-%m-%d %H:%MZ'):19s} NO_TRADE  gate=BLOCK_FEATURE_HISTORY")
            output_rows.append({
                "decision_timestamp": decision_ts.isoformat().replace("+00:00", "Z"),
                "prediction_direction": Direction.NO_TRADE.value,
                "gate": "BLOCK_FEATURE_HISTORY",
            })
            continue
        vol = predictor.volatility[idx]
        if vol is None or vol <= 0:
            gated += 1
            no_trade += 1
            print(f"{decision_ts.strftime('%Y-%m-%d %H:%MZ'):19s} NO_TRADE  gate=BLOCK_VOLATILITY")
            output_rows.append({
                "decision_timestamp": decision_ts.isoformat().replace("+00:00", "Z"),
                "prediction_direction": Direction.NO_TRADE.value,
                "gate": "BLOCK_VOLATILITY",
            })
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
            gated += 1
            no_trade += 1
            print(f"{decision_ts.strftime('%Y-%m-%d %H:%MZ'):19s} NO_TRADE  gate=BLOCK_FUTURE_EVENT")
            output_rows.append({
                "decision_timestamp": decision_ts.isoformat().replace("+00:00", "Z"),
                "prediction_direction": Direction.NO_TRADE.value,
                "gate": "BLOCK_FUTURE_EVENT",
            })
            continue

        try:
            decision = predictor.predict_at_index(idx)
        except ValueError:
            unavailable += 1
            continue
        direction = decision.prediction.direction
        realized_net: float | None = None
        verdict = "NO_TRADE"
        exit_reason: str | None = None
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
    gains = sum(value for value in realized_nets if value > 0.0)
    losses_abs = -sum(value for value in realized_nets if value < 0.0)
    profit_factor = gains / losses_abs if losses_abs > 0.0 else (float("inf") if gains > 0.0 else None)

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
    print(f"Profit factor          : {profit_factor:.3f}" if profit_factor is not None else "Profit factor          : n/a")

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "symbol": args.symbol,
            "version": "V2-PRICE-ONLY-BASELINE",
            "baseline_json": args.baseline_json,
            "config": vars(args),
            "frozen_samples": len(frozen_rows),
            "gate_blocked": gated,
            "unavailable": unavailable,
            "actionable": actionable,
            "net_positive": winners,
            "net_nonpositive": losers,
            "no_trade": no_trade,
            "selected_win_rate": win_rate,
            "mean_net_expectancy_bps": expectancy,
            "profit_factor": profit_factor,
            "rows": output_rows,
        }
        output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Audit JSON             : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
