"""Run S0-E0 embodied toy diagnostics through Chronos-S0."""

from __future__ import annotations

import argparse
import json

from chronos.embodied_toy.toy_worlds import TOY_WORLDS
from chronos.s0.structure_selector import recommend


def run_toy_suite(*, quiet: bool = False) -> dict[str, dict]:
    """Run all toy diagnostic packets through S0."""

    results = {}
    for name, make_diagnostics in TOY_WORLDS.items():
        recommendation = recommend(make_diagnostics()).to_dict()
        results[name] = recommendation
        if not quiet:
            print(f"{name:16s} -> {recommendation['candidate_family']} / {recommendation['allowed_action']}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronos S0-E0 embodied toy suite")
    parser.add_argument("--json", action="store_true", help="print full recommendation JSON")
    parser.add_argument("--quiet", action="store_true", help="suppress compact output")
    args = parser.parse_args()

    results = run_toy_suite(quiet=args.quiet or args.json)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
