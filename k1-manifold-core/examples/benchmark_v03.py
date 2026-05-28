"""Run the v0.3 constrained rollout benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from k1_manifold_core.benchmarks.v0_3 import write_benchmark_v03


def main() -> None:
    output_path = Path(__file__).resolve().parents[1] / "results" / "benchmark_v03.json"
    result = write_benchmark_v03(output_path)
    print(json.dumps(result["metrics"], indent=2))
    print(f"saved results = {output_path}")


if __name__ == "__main__":
    main()
