from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import v23_phase0 as core
from .data import load_ohlc_csv
from .v21_common import load_peer_markets


ELASTIC_MAX_ITER = 50_000


def _make_model(name: str):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=core.RIDGE_ALPHA))
    if name == "elasticnet":
        return make_pipeline(
            StandardScaler(),
            ElasticNet(
                alpha=core.ELASTIC_ALPHA,
                l1_ratio=core.ELASTIC_L1_RATIO,
                max_iter=ELASTIC_MAX_ITER,
                random_state=0,
            ),
        )
    raise ValueError(name)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.8f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "V2.3 Phase 0 causal cross-sectional information audit; "
            "robust reporting only, no trading policy or PnL"
        )
    )
    parser.add_argument("csv")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--peer", action="append", default=[], metavar="SYMBOL=CSV")
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Numerical robustness amendment only: keep the frozen objective and
    # regularisation, but allow more coordinate-descent iterations.
    core._make_model = _make_model

    bars = load_ohlc_csv(args.csv)
    peers = load_peer_markets(args.peer, target_symbol=args.symbol)

    print(
        f"Building V2.3 Phase 0 rows | {args.symbol.upper()} | "
        f"peers={','.join(sorted(peers))}",
        flush=True,
    )
    rows = core.build_phase0_rows(bars, symbol=args.symbol, peers=peers)
    print(f"eligible_development_rows={len(rows)}", flush=True)
    payload = core.evaluate_rows(rows, symbol=args.symbol.upper())
    payload["robustness_amendment"] = {
        "elasticnet_max_iter": ELASTIC_MAX_ITER,
        "elasticnet_alpha": core.ELASTIC_ALPHA,
        "elasticnet_l1_ratio": core.ELASTIC_L1_RATIO,
        "min_train_rows_unchanged": core.MIN_TRAIN_ROWS,
        "unavailable_folds_are_not_relaxed": True,
    }

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nPooled six-bar incremental R2", flush=True)
    any_scored = False
    for model_name in ("ridge", "elasticnet"):
        block = payload.get("pooled", {}).get(model_name, {}).get("6", {})
        value = block.get("incremental_r2")
        non_jump = block.get("incremental_non_jump_r2")
        any_scored = any_scored or value is not None
        print(
            f"{model_name}: incremental_r2={_fmt(value)} "
            f"non_jump_incremental_r2={_fmt(non_jump)}",
            flush=True,
        )

    if not any_scored:
        print(
            "target_status=UNAVAILABLE_MIN_TRAIN_OR_NO_SCORED_FOLDS",
            flush=True,
        )
    print(f"Output: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
