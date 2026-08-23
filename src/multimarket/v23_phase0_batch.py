from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_SYMBOLS = ("EURUSD", "XAUUSD", "BTCUSD", "ETHUSD", "QQQ")
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


@dataclass(frozen=True, slots=True)
class ParallelPlan:
    logical_cpus: int
    cpu_budget: int
    workers: int
    threads_per_worker: int
    nominal_thread_slots: int
    symbols: tuple[str, ...]


def resolve_parallel_plan(
    *,
    symbols: tuple[str, ...],
    cpu_budget: int | None,
    workers: int | None,
    threads_per_worker: int | None,
) -> ParallelPlan:
    logical = max(1, os.cpu_count() or 1)
    budget = logical if cpu_budget is None else int(cpu_budget)
    if budget <= 0:
        raise ValueError("cpu_budget must be positive")
    budget = min(budget, logical)

    resolved_workers = min(len(symbols), budget) if workers is None else int(workers)
    if resolved_workers <= 0:
        raise ValueError("workers must be positive")
    resolved_workers = min(resolved_workers, len(symbols), budget)

    resolved_threads = (
        max(1, budget // resolved_workers)
        if threads_per_worker is None
        else int(threads_per_worker)
    )
    if resolved_threads <= 0:
        raise ValueError("threads_per_worker must be positive")

    # Avoid hidden oversubscription by default. Explicit user overrides are still
    # bounded to the declared CPU budget to preserve predictable runtime behavior.
    resolved_threads = min(resolved_threads, max(1, budget // resolved_workers))

    return ParallelPlan(
        logical_cpus=logical,
        cpu_budget=budget,
        workers=resolved_workers,
        threads_per_worker=resolved_threads,
        nominal_thread_slots=resolved_workers * resolved_threads,
        symbols=symbols,
    )


def _child_env(threads_per_worker: int) -> dict[str, str]:
    env = dict(os.environ)
    value = str(threads_per_worker)
    for key in THREAD_ENV_KEYS:
        env[key] = value
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _target_command(
    *,
    symbol: str,
    symbols: tuple[str, ...],
    data_dir: Path,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "multimarket.v23_phase0_robust",
        str(data_dir / f"{symbol}_5m.csv"),
        "--symbol",
        symbol,
    ]
    for peer in symbols:
        if peer != symbol:
            command.extend(["--peer", f"{peer}={data_dir / f'{peer}_5m.csv'}"])
    command.extend(
        [
            "--output-json",
            str(output_dir / f"{symbol}_PHASE0.json"),
        ]
    )
    return command


def _run_target(
    *,
    symbol: str,
    symbols: tuple[str, ...],
    data_dir: Path,
    output_dir: Path,
    threads_per_worker: int,
) -> tuple[str, int]:
    command = _target_command(
        symbol=symbol,
        symbols=symbols,
        data_dir=data_dir,
        output_dir=output_dir,
    )
    log_path = output_dir / f"{symbol}_PHASE0.log"
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
        return_code = process.wait()
    return symbol, return_code


def _run_summary(*, symbols: tuple[str, ...], output_dir: Path) -> int:
    command = [
        sys.executable,
        "-m",
        "multimarket.v23_phase0_summary_robust",
    ]
    command.extend(str(output_dir / f"{symbol}_PHASE0.json") for symbol in symbols)
    command.extend(
        [
            "--output-json",
            str(output_dir / "V23_PHASE0_SUMMARY.json"),
        ]
    )
    log_path = output_dir / "V23_PHASE0_SUMMARY.log"
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
            "Run V2.3 Phase 0 targets concurrently with an explicit CPU/thread "
            "budget. This changes execution capacity only, not model semantics."
        )
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="evidence/v23/phase0")
    parser.add_argument("--cpu-budget", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--threads-per-worker", type=int, default=None)
    parser.add_argument(
        "--symbol",
        action="append",
        dest="symbols",
        default=None,
        help="Target symbol; repeat to override the default five-market universe",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not run the frozen Phase 0 promotion summary after target runs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    symbols = tuple(
        dict.fromkeys(
            symbol.strip().upper()
            for symbol in (args.symbols or DEFAULT_SYMBOLS)
            if symbol.strip()
        )
    )
    if not symbols:
        raise SystemExit("at least one symbol is required")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        str(data_dir / f"{symbol}_5m.csv")
        for symbol in symbols
        if not (data_dir / f"{symbol}_5m.csv").is_file()
    ]
    if missing:
        raise SystemExit(f"missing input CSVs: {missing}")

    plan = resolve_parallel_plan(
        symbols=symbols,
        cpu_budget=args.cpu_budget,
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
    )
    (output_dir / "FULL_CAPACITY_PLAN.json").write_text(
        json.dumps(asdict(plan), indent=2) + "\n",
        encoding="utf-8",
    )

    print("===== V2.3 FULL-CAPACITY EXECUTION PLAN =====", flush=True)
    print(f"logical_cpus={plan.logical_cpus}", flush=True)
    print(f"cpu_budget={plan.cpu_budget}", flush=True)
    print(f"market_workers={plan.workers}", flush=True)
    print(f"threads_per_worker={plan.threads_per_worker}", flush=True)
    print(f"nominal_thread_slots={plan.nominal_thread_slots}", flush=True)
    print(f"symbols={','.join(plan.symbols)}", flush=True)
    print("gpu_used=NO (Phase 0 Ridge/ElasticNet CPU implementation is frozen)", flush=True)
    print("research_semantics_changed=NO", flush=True)

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=plan.workers) as executor:
        futures = {
            executor.submit(
                _run_target,
                symbol=symbol,
                symbols=symbols,
                data_dir=data_dir,
                output_dir=output_dir,
                threads_per_worker=plan.threads_per_worker,
            ): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                _, return_code = future.result()
            except Exception as exc:  # pragma: no cover - defensive subprocess path
                failures.append(symbol)
                print(f"[{symbol}] FAILED: {exc}", flush=True)
                continue
            if return_code != 0:
                failures.append(symbol)
                print(f"[{symbol}] FAILED exit={return_code}", flush=True)
            else:
                print(f"[{symbol}] COMPLETE", flush=True)

    if failures:
        print(f"batch_status=FAIL targets={','.join(sorted(failures))}", flush=True)
        return 1

    if not args.no_summary:
        if len(symbols) != 5:
            print(
                "summary_status=SKIP (frozen summary requires exactly five targets)",
                flush=True,
            )
        else:
            summary_code = _run_summary(symbols=symbols, output_dir=output_dir)
            if summary_code != 0:
                print(f"summary_status=FAIL exit={summary_code}", flush=True)
                return summary_code
            print("summary_status=COMPLETE", flush=True)

    print("batch_status=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
