from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .v23_phase0_batch import _child_env, resolve_parallel_plan
from .v23_phase0c import EXPECTED_SENSORS


TARGETS = tuple(EXPECTED_SENSORS)
CORE_SYMBOLS = frozenset(TARGETS)


def _peer_path(symbol: str, *, data_dir: Path, sensor_dir: Path) -> Path:
    if symbol in CORE_SYMBOLS:
        return data_dir / f"{symbol}_5m.csv"
    return sensor_dir / f"{symbol}_5m.csv"


def target_command(
    *,
    symbol: str,
    data_dir: Path,
    sensor_dir: Path,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "multimarket.v23_phase0c",
        str(data_dir / f"{symbol}_5m.csv"),
        "--symbol",
        symbol,
    ]
    for peer in EXPECTED_SENSORS[symbol]:
        command.extend(
            [
                "--peer",
                f"{peer}={_peer_path(peer, data_dir=data_dir, sensor_dir=sensor_dir)}",
            ]
        )
    command.extend(
        [
            "--output-json",
            str(output_dir / f"{symbol}_PHASE0C.json"),
        ]
    )
    return command


def _required_files(
    symbols: tuple[str, ...],
    *,
    data_dir: Path,
    sensor_dir: Path,
) -> list[Path]:
    required: set[Path] = set()
    for symbol in symbols:
        required.add(data_dir / f"{symbol}_5m.csv")
        for peer in EXPECTED_SENSORS[symbol]:
            required.add(_peer_path(peer, data_dir=data_dir, sensor_dir=sensor_dir))
    return sorted(required)


def _run_target(
    *,
    symbol: str,
    data_dir: Path,
    sensor_dir: Path,
    output_dir: Path,
    threads_per_worker: int,
) -> tuple[str, int]:
    command = target_command(
        symbol=symbol,
        data_dir=data_dir,
        sensor_dir=sensor_dir,
        output_dir=output_dir,
    )
    log_path = output_dir / f"{symbol}_PHASE0C.log"
    env = _child_env(threads_per_worker)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(f"[{symbol}] {line}", end="", flush=True)
        return symbol, process.wait()


def _run_summary(*, symbols: tuple[str, ...], output_dir: Path) -> int:
    command = [sys.executable, "-m", "multimarket.v23_phase0c_summary"]
    command.extend(str(output_dir / f"{symbol}_PHASE0C.json") for symbol in symbols)
    command.extend(["--output-json", str(output_dir / "V23_PHASE0C_SUMMARY.json")])
    log_path = output_dir / "V23_PHASE0C_SUMMARY.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return process.wait()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen V2.3 Phase 0C target-specific signal audit. "
            "This runner does not score G/H/I and does not compute PnL."
        )
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sensor-dir", default="data/v23_phase0b_canonical")
    parser.add_argument("--output-dir", default="evidence/v23/phase0c")
    parser.add_argument("--cpu-budget", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--threads-per-worker", type=int, default=None)
    parser.add_argument(
        "--symbol",
        action="append",
        dest="symbols",
        choices=TARGETS,
        default=None,
        help="Target symbol; repeat to run a subset. Default is all five frozen targets.",
    )
    parser.add_argument("--no-summary", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    symbols = tuple(dict.fromkeys(args.symbols or TARGETS))
    data_dir = Path(args.data_dir)
    sensor_dir = Path(args.sensor_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = [str(path) for path in _required_files(symbols, data_dir=data_dir, sensor_dir=sensor_dir) if not path.is_file()]
    if missing:
        raise SystemExit(f"missing frozen Phase 0C input CSVs: {missing}")

    plan = resolve_parallel_plan(
        symbols=symbols,
        cpu_budget=args.cpu_budget,
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
    )
    (output_dir / "PHASE0C_EXECUTION_PLAN.json").write_text(
        json.dumps(
            {
                "logical_cpus": plan.logical_cpus,
                "cpu_budget": plan.cpu_budget,
                "workers": plan.workers,
                "threads_per_worker": plan.threads_per_worker,
                "nominal_thread_slots": plan.nominal_thread_slots,
                "symbols": list(symbols),
                "data_dir": str(data_dir),
                "sensor_dir": str(sensor_dir),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("===== V2.3 PHASE 0C EXECUTION PLAN =====", flush=True)
    print(f"symbols={','.join(symbols)}", flush=True)
    print(f"cpu_budget={plan.cpu_budget}", flush=True)
    print(f"workers={plan.workers}", flush=True)
    print(f"threads_per_worker={plan.threads_per_worker}", flush=True)
    print("G/H/I remain excluded by scorer construction", flush=True)
    print("economic/PnL outputs remain forbidden", flush=True)

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=plan.workers) as executor:
        futures = {
            executor.submit(
                _run_target,
                symbol=symbol,
                data_dir=data_dir,
                sensor_dir=sensor_dir,
                output_dir=output_dir,
                threads_per_worker=plan.threads_per_worker,
            ): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            symbol, return_code = future.result()
            if return_code != 0:
                failures.append(symbol)

    if failures:
        print(f"PHASE0C_BATCH=FAIL targets={','.join(sorted(failures))}", flush=True)
        return 1

    if args.no_summary:
        print("PHASE0C_BATCH=PASS summary=SKIPPED", flush=True)
        return 0

    summary_code = _run_summary(symbols=symbols, output_dir=output_dir)
    if summary_code != 0:
        print("PHASE0C_BATCH=FAIL summary_failed=true", flush=True)
        return summary_code

    print("PHASE0C_BATCH=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
