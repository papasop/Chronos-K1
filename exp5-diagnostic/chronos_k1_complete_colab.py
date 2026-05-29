"""Colab-friendly launcher for Chronos-K1 Experiment 5.

This file is intentionally a thin wrapper around the maintained benchmark:

    k1-manifold-core/benchmarks/experiment_5_full_sanity_reproduction.py

It exists so users looking for the historical "complete Colab" entry point can
find a single script, while the actual experiment logic remains maintained in
one canonical location.

Colab usage:

    !git clone https://github.com/papasop/Chronos-K1.git
    %cd Chronos-K1
    !python exp5-diagnostic/chronos_k1_complete_colab.py --smoke
    !CHRONOS_DEVICE=cuda python exp5-diagnostic/chronos_k1_complete_colab.py --full

The result should be interpreted as a primary world-model phenomenon benchmark:
substantial effect size with OOD persistence, close to significance, but not a
final statistical significance claim.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "k1-manifold-core"
BENCHMARK = CORE / "benchmarks" / "experiment_5_full_sanity_reproduction.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Chronos-K1 Experiment 5 Colab launcher.")
    parser.add_argument("--full", action="store_true", help="Run the N=10 full sanity reproduction.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny smoke check.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CORE / "results",
        help="Output directory passed through to the maintained benchmark.",
    )
    args = parser.parse_args()
    if args.full and args.smoke:
        parser.error("--full and --smoke cannot be used together")
    return args


def main() -> int:
    args = parse_args()

    command = [sys.executable, str(BENCHMARK)]
    if args.full:
        command.append("--full")
    else:
        command.append("--smoke")

    command.extend(["--output-dir", str(args.output_dir)])

    print("Chronos-K1 Experiment 5 launcher")
    print(f"Repository root: {ROOT}")
    print(f"Benchmark: {BENCHMARK}")
    print(f"Command: {' '.join(command)}")

    return subprocess.call(command, cwd=str(CORE))


if __name__ == "__main__":
    raise SystemExit(main())
