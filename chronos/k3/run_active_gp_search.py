"""Run K3-E2d discriminating GP truth-only active topology search."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass

from chronos.k3.active_gp_search import (
    N_RANDOM_SEEDS,
    SCORE_MARGIN,
    START,
    active_search,
    diagnostics_from_vortex_search,
    passed_admission,
    random_search,
    write_memory_event,
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


def run_search_suite(start=START, n_trials: int = 16, random_seed: int = 0, *, quiet: bool = False) -> dict:
    active = active_search(start, n_trials)
    random_result = random_search(start, n_trials, seed=random_seed)
    active_rec = recommend(diagnostics_from_vortex_search(active["best_metrics"])).to_dict()
    random_rec = recommend(diagnostics_from_vortex_search(random_result["best_metrics"])).to_dict()
    random_best_scores = sorted(random_search(start, n_trials, seed=seed)["best_score"] for seed in range(N_RANDOM_SEEDS))
    random_median_best = random_best_scores[len(random_best_scores) // 2]
    random_success_count = sum(
        1 for seed in range(N_RANDOM_SEEDS) if random_search(start, n_trials, seed=seed)["found_transport_ok"]
    )
    verdict = (
        "GP_ACTIVE_DIAGNOSTIC_VALUE_PASSED"
        if passed_admission(active, random_median_best, random_success_count)
        else "GP_ACTIVE_NO_ADVANTAGE"
    )
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
        print("K3-E2d Discriminating GP Truth-Only Active Topology Regime Search\n")
        for label, search_result, recommendation in (("ACTIVE", active, active_rec), ("RANDOM", random_result, random_rec)):
            regime = search_result["best_regime"]
            metrics = search_result["best_metrics"]
            print(f"{label}:")
            print(f"  best_score = {search_result['best_score']:.3f}")
            print(f"  pair_intact = {metrics['pair_intact']}")
            print(f"  mean_pos_err = {metrics['mean_pos_err']:.2f} ({metrics['mean_pos_err_units']})")
            print(f"  trials_to_success = {search_result['trials_to_success']}")
            print(f"  best_regime = sep={regime.pair_sep:.1f} core={regime.core_size:.1f} push={regime.push:.1f}")
            print(f"  S0 = {recommendation['candidate_family']} / {recommendation['allowed_action']}\n")
        print(f"RANDOM (statistics over {N_RANDOM_SEEDS} seeds):")
        print(f"  random_median_best_score = {random_median_best:.3f}")
        print(f"  random_reached_transport_ok = {random_success_count}/{N_RANDOM_SEEDS}\n")
        print("VERDICT:")
        print(f"  {verdict}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="K3-E2d discriminating GP truth-only active search")
    parser.add_argument("--json", action="store_true", help="print compact JSON verdict")
    parser.add_argument("--quiet", action="store_true", help="suppress compact output")
    parser.add_argument("--memory", default=None, help="optional JSONL path for S0-M0 memory logging")
    args = parser.parse_args()

    result = run_search_suite(quiet=args.quiet or args.json)
    if args.memory:
        event = write_memory_event(result, args.memory)
        if not args.quiet:
            print(f"[M0] wrote memory event to {args.memory} (run_id={event.run_id})")
    if args.json:
        print(
            json.dumps(
                {"verdict": result["verdict"], "active_rec": result["active_rec"], "random_rec": result["random_rec"]},
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
