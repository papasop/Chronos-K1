"""Run S0-E1 toy simulations through extractors and Chronos-S0."""

from __future__ import annotations

import argparse
import json

from chronos.embodied_toy.extractors import (
    diagnostics_from_contact_sim,
    diagnostics_from_object_persistence_sim,
    diagnostics_from_pendulum_sim,
)
from chronos.embodied_toy.simulations import (
    simulate_contact,
    simulate_object_persistence_fail,
    simulate_pendulum,
)
from chronos.s0.structure_selector import recommend


def _strip_private(diagnostics: dict) -> dict:
    return {key: value for key, value in diagnostics.items() if not key.startswith("_")}


def run_sim_suite(*, quiet: bool = False) -> dict[str, dict]:
    """Run simulate -> extract -> recommend for each toy world."""

    diagnostics_by_case = {
        "pendulum": diagnostics_from_pendulum_sim(simulate_pendulum()),
        "contact": diagnostics_from_contact_sim(simulate_contact()),
        "object_persistence": diagnostics_from_object_persistence_sim(simulate_object_persistence_fail()),
    }
    results = {}
    for name, diagnostics in diagnostics_by_case.items():
        recommendation = recommend(_strip_private(diagnostics)).to_dict()
        results[name] = {
            "diagnostics": diagnostics,
            "recommendation": recommendation,
        }
        if not quiet:
            measured = diagnostics.get("_measured", {})
            print(
                f"{name:18s} -> {recommendation['candidate_family']} / "
                f"{recommendation['allowed_action']}   measured={measured}"
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronos S0-E1 toy simulation diagnostics")
    parser.add_argument("--json", action="store_true", help="print full recommendation JSON")
    parser.add_argument("--quiet", action="store_true", help="suppress compact output")
    args = parser.parse_args()

    results = run_sim_suite(quiet=args.quiet or args.json)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
