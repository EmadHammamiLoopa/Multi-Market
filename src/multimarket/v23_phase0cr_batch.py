from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .v23_phase0_batch import DEFAULT_SYMBOLS, _child_env, resolve_parallel_plan
from .v23_phase0c import _sha256_file, load_phase0c_manifest
from .v23_phase0c_batch import _market_path, _parse_cost, _phase0b_has_scored_folds


def _write_unavailable(output: Path, *, symbol: str, phase0b_json: Path) -> None:
    payload = {
        "version": "V2.3-PHASE0CR-ASYNC-SENSOR-FEASIBILITY-REPAIR",
        "symbol": symbol,
        "evaluation_status": "UNAVAILABLE_PHASE0B_NO_SCORED_FOLDS",
        "reason": "Inherited Phase 0B boundaries contain no scored folds under frozen MIN_TRAIN_ROWS=5000",
        "phase0b_boundary_source": {
            "path": str(phase0b_json),
            "sha256": _sha256_file(phase0b_json),
        },
        "promotion_pass": False,
        "signal_candidate": None,
        "promoted_candidate": None,
    }
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _target_command(
    *, symbol: str, data_dir: Path, sensor_dir: Path, phase0b_dir: Path,
    manifest: Path, output_dir: Path, costs: dict[str, float],
) -> list[str]:
    _, linked = load_phase0c_manifest(manifest, symbol=symbol)
    command = [
        sys.executable, "-m", "multimarket.v23_phase0cr",
        str(_market_path(symbol, data_dir=data_dir, sensor_dir=sensor_dir)),
        "--symbol", symbol,
        "--phase0b-json", str(phase0b_dir / f"{symbol}_PHASE0B.json"),
        "--manifest", str(manifest),
    ]
    for peer in linked:
        command.extend([
            "--peer",
            f"{peer}={_market_path(peer, data_dir=data_dir, sensor_dir=sensor_dir)}",
        ])
    if symbol in costs:
        command.extend(["--round-trip-cost-bps", str(costs[symbol])])
    command.extend(["--output-json", str(output_dir / f"{symbol}_PHASE0CR.json")])
    return command


def _run_target(**kwargs) -> tuple[str, int]:
    symbol = kwargs["symbol"]
    threads_per_worker = kwargs.pop("threads_per_worker")
    command = _target_command(**kwargs)
    log_path = kwargs["output_dir"] / f"{symbol}_PHASE0CR.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=_child_env(threads_per_worker),
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(f"[{symbol}] {line}", end="", flush=True)
        return symbol, process.wait()


def _run_summary(*, symbols: tuple[str, ...], output_dir: Path) -> int:
    command = [sys.executable, "-m", "multimarket.v23_phase0c_summary"]
    command.extend(str(output_dir / f"{symbol}_PHASE0CR.json") for symbol in symbols)
    command.extend(["--output-json", str(output_dir / "V23_PHASE0CR_SUMMARY.json")])
    with (output_dir / "V23_PHASE0CR_SUMMARY.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return process.wait()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen V2.3 Phase 0C-R asynchronous-sensor repair")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sensor-dir", default="data/v23_phase0b_canonical")
    parser.add_argument("--phase0b-dir", default="evidence/v23/phase0b_scoring")
    parser.add_argument("--manifest", default="configs/v23_phase0c.json")
    parser.add_argument("--output-dir", default="evidence/v23/phase0cr_scoring")
    parser.add_argument("--cpu-budget", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--threads-per-worker", type=int, default=None)
    parser.add_argument("--symbol", action="append", dest="symbols", default=None)
    parser.add_argument("--cost", action="append", default=[], metavar="SYMBOL=BPS")
    parser.add_argument("--no-summary", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    symbols = tuple(dict.fromkeys(
        symbol.strip().upper() for symbol in (args.symbols or DEFAULT_SYMBOLS) if symbol.strip()
    ))
    if not symbols:
        raise SystemExit("at least one symbol is required")

    costs: dict[str, float] = {}
    for raw in args.cost:
        symbol, value = _parse_cost(raw)
        if symbol in costs:
            raise SystemExit(f"duplicate --cost for {symbol}")
        costs[symbol] = value
    unknown_costs = sorted(set(costs) - set(symbols))
    if unknown_costs:
        raise SystemExit(f"cost supplied for unselected target(s): {unknown_costs}")

    data_dir = Path(args.data_dir)
    sensor_dir = Path(args.sensor_dir)
    phase0b_dir = Path(args.phase0b_dir)
    manifest = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not manifest.is_file():
        raise SystemExit(f"missing frozen Phase 0C manifest: {manifest}")

    unavailable: set[str] = set()
    runnable: list[str] = []
    missing: list[str] = []
    for symbol in symbols:
        phase0b_json = phase0b_dir / f"{symbol}_PHASE0B.json"
        if not phase0b_json.is_file():
            missing.append(str(phase0b_json))
            continue
        if not _phase0b_has_scored_folds(phase0b_json):
            _write_unavailable(output_dir / f"{symbol}_PHASE0CR.json", symbol=symbol, phase0b_json=phase0b_json)
            (output_dir / f"{symbol}_PHASE0CR.log").write_text(
                "evaluation_status=UNAVAILABLE_PHASE0B_NO_SCORED_FOLDS\n", encoding="utf-8"
            )
            unavailable.add(symbol)
            print(f"[{symbol}] UNAVAILABLE: no Phase 0B scored folds", flush=True)
            continue

        target_path = _market_path(symbol, data_dir=data_dir, sensor_dir=sensor_dir)
        if not target_path.is_file():
            missing.append(str(target_path))
        _, linked = load_phase0c_manifest(manifest, symbol=symbol)
        for peer in linked:
            peer_path = _market_path(peer, data_dir=data_dir, sensor_dir=sensor_dir)
            if not peer_path.is_file():
                missing.append(str(peer_path))
        runnable.append(symbol)

    if missing:
        raise SystemExit(f"missing Phase 0C-R inputs: {sorted(set(missing))}")

    failures: list[str] = []
    if runnable:
        plan = resolve_parallel_plan(
            symbols=tuple(runnable), cpu_budget=args.cpu_budget,
            workers=args.workers, threads_per_worker=args.threads_per_worker,
        )
        with ThreadPoolExecutor(max_workers=plan.workers) as executor:
            futures = {
                executor.submit(
                    _run_target,
                    symbol=symbol,
                    data_dir=data_dir,
                    sensor_dir=sensor_dir,
                    phase0b_dir=phase0b_dir,
                    manifest=manifest,
                    output_dir=output_dir,
                    costs=costs,
                    threads_per_worker=plan.threads_per_worker,
                ): symbol
                for symbol in runnable
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    _, code = future.result()
                except Exception as exc:
                    failures.append(symbol)
                    print(f"[{symbol}] FAILED: {exc}", flush=True)
                    continue
                if code != 0:
                    failures.append(symbol)
                    print(f"[{symbol}] FAILED exit={code}", flush=True)
                else:
                    print(f"[{symbol}] COMPLETE", flush=True)

    if failures:
        print(f"batch_status=FAIL targets={','.join(sorted(failures))}", flush=True)
        return 1

    if not args.no_summary:
        code = _run_summary(symbols=symbols, output_dir=output_dir)
        if code != 0:
            print(f"summary_status=FAIL exit={code}", flush=True)
            return code
        print("summary_status=COMPLETE", flush=True)

    print("batch_status=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
