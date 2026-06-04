"""Run K3-E2b active topology regime search."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass

from chronos.k3.active_topology_search import (
    SCORE_MARGIN,
    START,
    active_search,
    diagnostics_from_vortex_search,
    has_active_advantage,
    random_search,
)
from chronos.s0.structure_selector import recommend


def _jsonable(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def run_search_suite(start=START, n_trials: int = 30, random_seed: int = 0, *, quiet: bool = False) -> dict:
    active = active_search(start, n_trials)
    random_result = random_search(start, n_trials, seed=random_seed)
    active_rec = recommend(diagnostics_from_vortex_search(active["best_metrics"])).to_dict()
    random_rec = recommend(diagnostics_from_vortex_search(random_result["best_metrics"])).to_dict()

    random_best_scores = sorted(random_search(start, n_trials, seed=seed)["best_score"] for seed in range(20))
    random_median_best = random_best_scores[len(random_best_scores) // 2]
    random_success_count = sum(1 for seed in range(20) if random_search(start, n_trials, seed=seed)["found_transport_ok"])

    verdict = "ACTIVE_DIAGNOSTIC_VALUE_PASSED" if has_active_advantage(active, random_result) else "ACTIVE_NO_ADVANTAGE"
    result = {
        "active": active,
        "random": random_result,
        "active_rec": active_rec,
        "random_rec": random_rec,
        "verdict": verdict,
        "random_median_best_score": random_median_best,
        "random_success_count_20": random_success_count,
        "score_margin": SCORE_MARGIN,
    }

    if not quiet:
        print("K3-E2b Active Topology Regime Search\n")
        for label, search_result, recommendation in (("ACTIVE", active, active_rec), ("RANDOM", random_result, random_rec)):
            regime = search_result["best_regime"]
            print(f"{label}:")
            print(f"  best_score = {search_result['best_score']:.3f}")
            print(f"  found_transport_ok = {search_result['found_transport_ok']}")
            print(f"  trials_to_success = {search_result['trials_to_success']}")
            print(
                f"  best_regime = sep={regime.pair_sep:.1f} core={regime.core_size:.1f} "
                f"dt={regime.dt_phys:.3f} horizon={regime.horizon}"
            )
            print(f"  S0 recommendation = {recommendation['candidate_family']} / {recommendation['allowed_action']}\n")
        print("RANDOM (statistics over 20 seeds):")
        print(f"  random_median_best_score = {random_median_best:.3f}")
        print(f"  random_reached_transport_ok = {random_success_count}/20\n")
        print("VERDICT:")
        print(f"  {verdict}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="K3-E2b active topology regime search")
    parser.add_argument("--json", action="store_true", help="print compact JSON verdict")
    parser.add_argument("--quiet", action="store_true", help="suppress compact output")
    args = parser.parse_args()

    result = run_search_suite(quiet=args.quiet or args.json)

    if args.json:
        print(
            json.dumps(
                {
                    "verdict": result["verdict"],
                    "active_rec": result["active_rec"],
                    "random_rec": result["random_rec"],
                    "active": _jsonable(result["active"]),
                    "random": _jsonable(result["random"]),
                    "random_median_best_score": result["random_median_best_score"],
                    "random_success_count_20": result["random_success_count_20"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
