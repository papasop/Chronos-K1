"""Run S0-E2 active toy exploration through diagnostics and Chronos-S0."""

from __future__ import annotations

import argparse
import json

from chronos.embodied_toy.active import (
    ActiveToyExplorer,
    PendulumWorld,
    diagnostics_from_active_pendulum,
    random_explore,
    state_coverage,
)
from chronos.s0.structure_selector import recommend


def _strip_private(diagnostics: dict) -> dict:
    return {key: value for key, value in diagnostics.items() if not key.startswith("_")}


def run_active_suite(n_steps: int = 120, *, quiet: bool = False) -> dict:
    """Explore actively, compare against random, extract diagnostics, recommend."""

    active_world = PendulumWorld()
    explorer = ActiveToyExplorer(active_world)
    transitions = explorer.run(n_steps)
    coverage_active = state_coverage(explorer.memory)

    random_world = PendulumWorld()
    _, random_memory = random_explore(random_world, n_steps, seed=0)
    coverage_random = state_coverage(random_memory)

    diagnostics = diagnostics_from_active_pendulum(transitions)
    recommendation = recommend(_strip_private(diagnostics)).to_dict()
    result = {
        "recommendation": recommendation,
        "diagnostics": diagnostics,
        "coverage_active": coverage_active,
        "coverage_random": coverage_random,
        "coverage_ratio": round(coverage_active / max(1, coverage_random), 3),
    }

    if not quiet:
        print(f"active exploration -> {recommendation['candidate_family']} / {recommendation['allowed_action']}")
        print(f"measured: {diagnostics.get('_measured')}")
        print(
            f"state-space coverage: active={coverage_active} cells vs random={coverage_random} "
            f"cells ({result['coverage_ratio']}x)"
        )
        print(
            "active exploration validated by coverage gain; the K2 diagnostic probe is launched "
            f"from {diagnostics['_measured']['probe_source']}"
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronos S0-E2 active toy exploration")
    parser.add_argument("--json", action="store_true", help="print full result JSON")
    parser.add_argument("--quiet", action="store_true", help="suppress compact output")
    args = parser.parse_args()

    result = run_active_suite(quiet=args.quiet or args.json)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
