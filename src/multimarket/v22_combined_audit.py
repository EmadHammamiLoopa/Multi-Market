from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .data import load_ohlc_csv
from .models import Direction
from .v2_labels import build_economic_event
from .v2_model import V2Config
from .v2_ranking_audit import _spearman, _top_fraction_mean
from .v2_volatility import volatility_scaled_barriers_bps
from .v21_common import hard_eligible_indices, load_peer_markets
from .v22_macro import MacroLedgerIndex, ledger_summary, load_macro_ledger_csv
from .v22_model import V22Config, V22MacroContextPredictor


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
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    if not rows:
        raise ValueError("V2.2 timestamp manifest contains no rows")
    for row in rows:
        if set(row) != {"decision_timestamp"}:
            raise ValueError("V2.2 manifest rows must contain only decision_timestamp")
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Single-pass economic + ranking audit for frozen V2.2 macro-context predictions"
    )
    p.add_argument("csv")
    p.add_argument("--symbol", required=True)
    p.add_argument("--manifest-json", required=True)
    p.add_argument("--macro-ledger", required=True)
    p.add_argument("--peer", action="append", default=[], metavar="SYMBOL=CSV")
    p.add_argument("--max-peer-staleness-minutes", type=int, default=15)
    p.add_argument("--economic-output-json")
    p.add_argument("--ranking-output-json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bars = load_ohlc_csv(args.csv)
    hard_eligible = hard_eligible_indices(bars, args.symbol)
    peers = load_peer_markets(args.peer, target_symbol=args.symbol)
    frozen_rows = _load_rows(Path(args.manifest_json))
    macro_rows = load_macro_ledger_csv(args.macro_ledger)
    macro_ledger = MacroLedgerIndex(macro_rows)
    index_by_timestamp = {bar.timestamp.astimezone(timezone.utc): i for i, bar in enumerate(bars)}

    base = V2Config()
    print(f"Building V2.2 predictor | {args.symbol} | peers={','.join(sorted(peers))}", flush=True)
    print(f"Macro ledger rows={len(macro_rows)} summary={dict(ledger_summary(macro_rows))}", flush=True)
    predictor = V22MacroContextPredictor(
        bars,
        V22Config(base=base, max_peer_staleness_minutes=args.max_peer_staleness_minutes),
        eligible_indices=hard_eligible,
        peers=peers,
        macro_ledger=macro_ledger,
    )
    print(
        f"Predictor ready | features={len(predictor.feature_names)} "
        f"feature_rows={len(predictor.feature_points)} training_rows={len(predictor.training_points)}",
        flush=True,
    )

    economic_rows: list[dict[str, object]] = []
    rank_rows: list[RankRow] = []
    gated = unavailable = no_trade = actionable = winners = losers = 0
    blocked_feature = blocked_event = unavailable_data = unavailable_training = 0
    realized_nets: list[float] = []

    for number, frozen in enumerate(frozen_rows, start=1):
        ts = _parse_timestamp(str(frozen["decision_timestamp"]))
        print(f"[{number:02d}/{len(frozen_rows):02d}] {ts.strftime('%Y-%m-%d %H:%MZ')}", flush=True)
        idx = index_by_timestamp.get(ts)
        if idx is None:
            unavailable += 1
            unavailable_data += 1
            continue
        if idx not in predictor.feature_by_index:
            gated += 1
            no_trade += 1
            blocked_feature += 1
            economic_rows.append({
                "decision_timestamp": ts.isoformat().replace("+00:00", "Z"),
                "prediction_direction": Direction.NO_TRADE.value,
                "gate": "BLOCK_FEATURE_HISTORY",
            })
            continue
        vol = predictor.volatility[idx]
        if vol is None or vol <= 0:
            gated += 1
            no_trade += 1
            blocked_feature += 1
            economic_rows.append({
                "decision_timestamp": ts.isoformat().replace("+00:00", "Z"),
                "prediction_direction": Direction.NO_TRADE.value,
                "gate": "BLOCK_VOLATILITY",
            })
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
            blocked_event += 1
            economic_rows.append({
                "decision_timestamp": ts.isoformat().replace("+00:00", "Z"),
                "prediction_direction": Direction.NO_TRADE.value,
                "gate": "BLOCK_FUTURE_EVENT",
            })
            continue

        try:
            decision = predictor.predict_at_index(idx)
        except ValueError as exc:
            unavailable += 1
            if "labeled historical rows" in str(exc):
                unavailable_training += 1
            else:
                unavailable_data += 1
            continue

        rank_rows.append(
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

        economic_rows.append({
            "decision_timestamp": ts.isoformat().replace("+00:00", "Z"),
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
        profit_factor_is_infinite = False
    elif gains > 0.0:
        profit_factor = float("inf")
        profit_factor_json = None
        profit_factor_is_infinite = True
    else:
        profit_factor = None
        profit_factor_json = None
        profit_factor_is_infinite = False

    p_long = [row.p_long for row in rank_rows]
    p_short = [row.p_short for row in rank_rows]
    ev_long = [row.ev_long_bps for row in rank_rows]
    ev_short = [row.ev_short_bps for row in rank_rows]
    real_long = [row.realized_long_bps for row in rank_rows]
    real_short = [row.realized_short_bps for row in rank_rows]
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

    print("\nEconomic summary", flush=True)
    print(f"Actionable={actionable} positive={winners} nonpositive={losers} no_trade={no_trade}", flush=True)
    print(f"Win rate={win_rate:.2%}" if win_rate is not None else "Win rate=n/a", flush=True)
    print(f"Expectancy={expectancy:+.3f} bp" if expectancy is not None else "Expectancy=n/a", flush=True)
    if profit_factor == float("inf"):
        print("Profit factor=inf", flush=True)
    else:
        print(f"Profit factor={profit_factor:.3f}" if profit_factor is not None else "Profit factor=n/a", flush=True)

    print("\nRanking summary", flush=True)
    print(
        f"Scored={len(rank_rows)} blocked_feature={blocked_feature} blocked_event={blocked_event} "
        f"unavailable_training={unavailable_training} unavailable_other={unavailable_data}",
        flush=True,
    )
    for key, value in metrics.items():
        print(f"{key}: {value:+.4f}" if value is not None else f"{key}: n/a", flush=True)

    common = {
        "symbol": args.symbol,
        "manifest_json": args.manifest_json,
        "macro_ledger": args.macro_ledger,
        "macro_ledger_rows": len(macro_rows),
        "macro_ledger_summary": dict(ledger_summary(macro_rows)),
        "peers": sorted(peers),
        "feature_count": len(predictor.feature_names),
        "max_peer_staleness_minutes": args.max_peer_staleness_minutes,
        "frozen_v2_config": asdict(base),
        "frozen_samples": len(frozen_rows),
    }

    if args.economic_output_json:
        output = Path(args.economic_output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **common,
            "version": "V2.2-MACRO-CONTEXT",
            "gate_blocked": gated,
            "unavailable": unavailable,
            "actionable": actionable,
            "net_positive": winners,
            "net_nonpositive": losers,
            "no_trade": no_trade,
            "selected_win_rate": win_rate,
            "mean_net_expectancy_bps": expectancy,
            "profit_factor": profit_factor_json,
            "profit_factor_is_infinite": profit_factor_is_infinite,
            "rows": economic_rows,
        }
        output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Economic JSON: {output}", flush=True)

    if args.ranking_output_json:
        output = Path(args.ranking_output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **common,
            "version": "V2.2-MACRO-CONTEXT-RANKING",
            "scored_eligible_rows": len(rank_rows),
            "blocked_feature_history": blocked_feature,
            "blocked_future_event": blocked_event,
            "unavailable_training": unavailable_training,
            "unavailable_data_or_other": unavailable_data,
            "metrics": metrics,
            "rows": [asdict(row) for row in rank_rows],
            "note": "Single-pass diagnostic using the same frozen V2.2 predictions as the economic audit.",
        }
        output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Ranking JSON: {output}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
