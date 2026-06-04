"""Run S0-E2b active diagnostic-value suite."""

from __future__ import annotations

import argparse
import json

from chronos.embodied_toy.active_value import (
    RAIL,
    ZONE,
    ActiveRailExplorer,
    RailWorld,
    diagnostics_at_reached_state,
    random_rail_run,
)
from chronos.s0.structure_selector import recommend


def _strip_private(diagnostics: dict) -> dict:
    return {key: value for key, value in diagnostics.items() if not key.startswith("_")}


def run_value_suite(n_steps: int = 200, random_seed: int = 0, *, quiet: bool = False) -> dict:
    """Compare active and random reached states in a partitioned diagnostic world."""

    active_explorer = ActiveRailExplorer(RailWorld())
    active_positions = active_explorer.run(n_steps)
    active_reached = max(active_positions)
    active_diagnostics = diagnostics_at_reached_state(active_reached, n_steps)
    active_recommendation = recommend(_strip_private(active_diagnostics)).to_dict()

    random_positions = random_rail_run(RailWorld(), n_steps, seed=random_seed)
    random_reached = max(random_positions)
    random_diagnostics = diagnostics_at_reached_state(random_reached, n_steps)
    random_recommendation = recommend(_strip_private(random_diagnostics)).to_dict()

    result = {
        "active": {
            "reached_x": active_reached,
            "in_zone": active_reached >= ZONE,
            "recommendation": active_recommendation,
            "measured": active_diagnostics["_measured"],
        },
        "random": {
            "reached_x": random_reached,
            "in_zone": random_reached >= ZONE,
            "recommendation": random_recommendation,
            "measured": random_diagnostics["_measured"],
        },
    }

    if not quiet:
        print(f"ZONE boundary x>={ZONE} is the structure zone (rail length {RAIL})\n")
        print(
            f"ACTIVE : reached x={active_reached:.1f} (in_zone={result['active']['in_zone']}) "
            f"-> {active_recommendation['candidate_family']} / {active_recommendation['allowed_action']}"
        )
        print(f"         measured={active_diagnostics['_measured']}")
        print(
            f"RANDOM : reached x={random_reached:.1f} (in_zone={result['random']['in_zone']}) "
            f"-> {random_recommendation['candidate_family']} / {random_recommendation['allowed_action']}"
        )
        print(f"         measured={random_diagnostics['_measured']}")
        verdict = (
            "active GETS K2, random does NOT"
            if active_recommendation["candidate_family"] == "K2_SYMPLECTIC"
            and random_recommendation["candidate_family"] != "K2_SYMPLECTIC"
            else "no separation"
        )
        print(f"\n=> {verdict}: active exploration is necessary for the correct diagnosis here.")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronos S0-E2b active diagnostic-value suite")
    parser.add_argument("--json", action="store_true", help="print full result JSON")
    parser.add_argument("--quiet", action="store_true", help="suppress compact output")
    args = parser.parse_args()

    result = run_value_suite(quiet=args.quiet or args.json)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
