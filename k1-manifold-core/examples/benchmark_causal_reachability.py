"""Run the v0.4 causal reachability benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from k1_manifold_core.benchmarks.benchmark_causal_reachability import (
    plot_reachability,
    simulate_reachability,
    summarize_reachability,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = simulate_reachability()
    summary = summarize_reachability(result)
    output_json = root / "results" / "benchmark_causal_reachability.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    figure_paths = plot_reachability(result, root / "examples" / "outputs")

    print("===== Causal Reachability Benchmark =====")
    print(json.dumps(summary["metrics"], indent=2))
    print(f"saved results = {output_json}")
    for path in figure_paths:
        print(f"saved figure = {path}")


if __name__ == "__main__":
    main()
