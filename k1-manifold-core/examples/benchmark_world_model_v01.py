"""Run the minimal Chronos-K1 latent world-model benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from k1_manifold_core.benchmarks.world_model_v01 import write_world_model_v01


def main() -> None:
    output_path = Path(__file__).resolve().parents[1] / "results" / "world_model_v01.json"
    result = write_world_model_v01(output_path)
    print("===== Chronos-K1 World-Model v0.1 Benchmark =====")
    print(json.dumps(result["metrics"], indent=2))
    print(f"saved results = {output_path}")


if __name__ == "__main__":
    main()
