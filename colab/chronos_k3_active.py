"""Portable mirror for Colab.

Canonical implementation lives in chronos/k3/active_topology_search.py and
chronos/k3/run_active_topology_search.py.

K3-E2b tests guided active topology regime search on an interpretable toy
landscape. It is not K3 prior validation, not GP truth, not robotics, and not
RL/CNN training.
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import asdict, dataclass, replace

try:
    from chronos.k3.run_active_topology_search import run_search_suite

    SOURCE = "chronos.k3 (repo)"
except Exception:
    CTX_TOPOLOGY = "K3_TOPOLOGY_REGIME"
    POS_ERR_CEIL = 8.0
    TRANSPORT_OK_THRESHOLD = 0.6
    SCORE_MARGIN = 0.1

    @dataclass(frozen=True)
    class VortexRegime:
        pair_sep: float
        core_size: float
        dt_phys: float
        horizon: int
        angle: float = 0.0

    START = VortexRegime(pair_sep=5.0, core_size=1.0, dt_phys=0.06, horizon=120)
    BOUNDS = {"pair_sep": (4.0, 24.0), "core_size": (0.8, 6.0), "dt_phys": (0.005, 0.08), "horizon": (20, 160)}
    ACTIONS = [
        "increase_sep",
        "decrease_sep",
        "increase_core",
        "decrease_core",
        "increase_dt",
        "decrease_dt",
        "increase_horizon",
        "decrease_horizon",
        "rotate_angle",
    ]
    ACTION_STEPS = {
        "increase_sep": ("pair_sep", +2.0),
        "decrease_sep": ("pair_sep", -2.0),
        "increase_core": ("core_size", +0.5),
        "decrease_core": ("core_size", -0.5),
        "increase_dt": ("dt_phys", +0.01),
        "decrease_dt": ("dt_phys", -0.01),
        "increase_horizon": ("horizon", +20),
        "decrease_horizon": ("horizon", -20),
    }

    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _triangular_score(x: float, low: float, mid: float, high: float) -> float:
        if x <= low or x >= high:
            return 0.0
        return (x - low) / (mid - low) if x <= mid else (high - x) / (high - mid)

    def evaluate_vortex_regime(regime: VortexRegime) -> dict:
        sep = _triangular_score(regime.pair_sep, 6.0, 13.0, 20.0)
        core = _triangular_score(regime.core_size, 1.0, 3.0, 5.5)
        dt_score = _clamp((0.07 - regime.dt_phys) / 0.05, 0.0, 1.0)
        horizon_score = _clamp((140 - regime.horizon) / 100.0, 0.0, 1.0)
        eps = 0.05
        topology_score = ((sep + eps) * (core + eps) * (dt_score + eps) * (horizon_score + eps)) ** 0.25
        topology_score = float(min(1.0, max(0.0, topology_score - eps)))
        transport_ok = topology_score >= TRANSPORT_OK_THRESHOLD
        return {
            "pair_intact": bool(transport_ok),
            "vortex_lifetime_fraction": topology_score,
            "net_charge_error": 0.0 if transport_ok else 1.0,
            "pos_err": POS_ERR_CEIL * (1.0 - topology_score),
            "topology_score": topology_score,
            "transport_ok": bool(transport_ok),
        }

    def apply_action(regime: VortexRegime, action: str) -> VortexRegime:
        if action == "rotate_angle":
            return replace(regime, angle=(regime.angle + math.pi / 4) % (2 * math.pi))
        field, delta = ACTION_STEPS[action]
        low, high = BOUNDS[field]
        value = _clamp(getattr(regime, field) + delta, low, high)
        if field == "horizon":
            value = int(value)
        return replace(regime, **{field: value})

    def regime_distance(a: VortexRegime, b: VortexRegime) -> float:
        return math.sqrt(
            ((a.pair_sep - b.pair_sep) / 20.0) ** 2
            + ((a.core_size - b.core_size) / 5.2) ** 2
            + ((a.dt_phys - b.dt_phys) / 0.075) ** 2
            + ((a.horizon - b.horizon) / 140.0) ** 2
        )

    def novelty(candidate: VortexRegime, visited: list[VortexRegime]) -> float:
        return 1.0 if not visited else min(regime_distance(candidate, old) for old in visited)

    def active_search(start_regime: VortexRegime = START, n_trials: int = 30) -> dict:
        visited = []
        history = []
        current = start_regime
        trials_to_success = None
        best_score = 0.0
        for trial in range(n_trials):
            metrics = evaluate_vortex_regime(current)
            history.append((current, metrics))
            visited.append(current)
            if metrics["transport_ok"] and trials_to_success is None:
                trials_to_success = trial
            best_score = max(best_score, metrics["topology_score"])
            candidates = [apply_action(current, action) for action in ACTIONS]
            current = max(candidates, key=lambda candidate: (evaluate_vortex_regime(candidate)["topology_score"], novelty(candidate, visited)))
        best_regime, best_metrics = max(history, key=lambda item: item[1]["topology_score"])
        return {"best_regime": best_regime, "best_metrics": best_metrics, "best_score": best_score, "trials_to_success": trials_to_success, "found_transport_ok": best_metrics["transport_ok"]}

    def random_search(start_regime: VortexRegime = START, n_trials: int = 30, seed: int = 0) -> dict:
        rng = random.Random(seed)
        history = []
        current = start_regime
        trials_to_success = None
        best_score = 0.0
        for trial in range(n_trials):
            metrics = evaluate_vortex_regime(current)
            history.append((current, metrics))
            if metrics["transport_ok"] and trials_to_success is None:
                trials_to_success = trial
            best_score = max(best_score, metrics["topology_score"])
            current = apply_action(current, rng.choice(ACTIONS))
        best_regime, best_metrics = max(history, key=lambda item: item[1]["topology_score"])
        return {"best_regime": best_regime, "best_metrics": best_metrics, "best_score": best_score, "trials_to_success": trials_to_success, "found_transport_ok": best_metrics["transport_ok"]}

    def recommend(diagnostics: dict) -> dict:
        score = diagnostics["topological_transport_score"]
        if score >= TRANSPORT_OK_THRESHOLD:
            return {"candidate_family": "K3_TOPOLOGICAL", "confidence": "medium", "reason": "Topological object is transported.", "next_vpsl_gate": "constraint", "allowed_action": "continue"}
        return {"candidate_family": "K3_TOPOLOGICAL", "confidence": "low", "reason": "Field learnable but topological object not transported.", "next_vpsl_gate": "regime", "allowed_action": "do_not_promote"}

    def diagnostics_from_vortex_search(metrics: dict) -> dict:
        return {"diagnostic_context": CTX_TOPOLOGY, "field_learnable": True, "object_tracking_valid": bool(metrics["transport_ok"]), "topological_transport_score": float(metrics["topology_score"])}

    def has_active_advantage(active: dict, random_result: dict) -> bool:
        return (
            (active["found_transport_ok"] and not random_result["found_transport_ok"])
            or (active["best_score"] > random_result["best_score"] + SCORE_MARGIN)
            or (active["trials_to_success"] is not None and (random_result["trials_to_success"] is None or active["trials_to_success"] < random_result["trials_to_success"]))
        )

    def pure_novelty_search(start_regime: VortexRegime = START, n_trials: int = 30) -> bool:
        visited = [start_regime]
        current = start_regime
        for _ in range(n_trials):
            metrics = evaluate_vortex_regime(current)
            if metrics["transport_ok"]:
                return True
            current = max((apply_action(current, action) for action in ACTIONS), key=lambda candidate: novelty(candidate, visited))
            visited.append(current)
        return False

    def _jsonable(value):
        if isinstance(value, VortexRegime):
            return asdict(value)
        if isinstance(value, dict):
            return {key: _jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_jsonable(item) for item in value]
        if isinstance(value, tuple):
            return [_jsonable(item) for item in value]
        return value

    def run_search_suite(start=START, n_trials: int = 30, random_seed: int = 0, *, quiet: bool = False) -> dict:
        active = active_search(start, n_trials)
        random_result = random_search(start, n_trials, seed=random_seed)
        active_rec = recommend(diagnostics_from_vortex_search(active["best_metrics"]))
        random_rec = recommend(diagnostics_from_vortex_search(random_result["best_metrics"]))
        random_best_scores = sorted(random_search(start, n_trials, seed=seed)["best_score"] for seed in range(20))
        random_median_best = random_best_scores[len(random_best_scores) // 2]
        random_success_count = sum(1 for seed in range(20) if random_search(start, n_trials, seed=seed)["found_transport_ok"])
        verdict = "ACTIVE_DIAGNOSTIC_VALUE_PASSED" if has_active_advantage(active, random_result) else "ACTIVE_NO_ADVANTAGE"
        result = {"active": active, "random": random_result, "active_rec": active_rec, "random_rec": random_rec, "verdict": verdict, "random_median_best_score": random_median_best, "random_success_count_20": random_success_count}
        if not quiet:
            print("K3-E2b Active Topology Regime Search\n")
            for label, search_result, rec in (("ACTIVE", active, active_rec), ("RANDOM", random_result, random_rec)):
                regime = search_result["best_regime"]
                print(f"{label}:")
                print(f"  best_score = {search_result['best_score']:.3f}")
                print(f"  found_transport_ok = {search_result['found_transport_ok']}")
                print(f"  trials_to_success = {search_result['trials_to_success']}")
                print(f"  best_regime = sep={regime.pair_sep:.1f} core={regime.core_size:.1f} dt={regime.dt_phys:.3f} horizon={regime.horizon}")
                print(f"  S0 recommendation = {rec['candidate_family']} / {rec['allowed_action']}\n")
            print("RANDOM (statistics over 20 seeds):")
            print(f"  random_median_best_score = {random_median_best:.3f}")
            print(f"  random_reached_transport_ok = {random_success_count}/20\n")
            print("VERDICT:")
            print(f"  {verdict}")
        return result

    SOURCE = "embedded fallback"


def _tests() -> int:
    count = 0
    metrics = evaluate_vortex_regime(START)
    for key in ("pair_intact", "vortex_lifetime_fraction", "net_charge_error", "pos_err", "topology_score", "transport_ok"):
        assert key in metrics
        count += 1
    assert 0.0 <= metrics["topology_score"] <= 1.0
    count += 1
    stable = VortexRegime(pair_sep=13.0, core_size=3.0, dt_phys=0.01, horizon=40)
    bad = VortexRegime(pair_sep=4.0, core_size=0.8, dt_phys=0.08, horizon=160)
    assert evaluate_vortex_regime(bad)["topology_score"] < evaluate_vortex_regime(stable)["topology_score"]
    count += 1
    assert evaluate_vortex_regime(stable)["transport_ok"] is True
    count += 1
    active = active_search(START)
    random_result = random_search(START, seed=0)
    assert active["best_score"] > evaluate_vortex_regime(START)["topology_score"]
    count += 1
    assert has_active_advantage(active, random_result)
    count += 1
    random_median = sorted(random_search(START, seed=seed)["best_score"] for seed in range(20))[10]
    assert active["best_score"] > random_median
    count += 1
    active_rec = recommend(diagnostics_from_vortex_search(active["best_metrics"]))
    assert active_rec["candidate_family"] == "K3_TOPOLOGICAL"
    count += 1
    assert active_rec["allowed_action"] == "continue"
    count += 1
    bad_rec = recommend(diagnostics_from_vortex_search(evaluate_vortex_regime(bad)))
    assert bad_rec["candidate_family"] == "K3_TOPOLOGICAL" and bad_rec["allowed_action"] == "do_not_promote"
    count += 1
    assert active_rec["allowed_action"] in {"continue", "archive", "do_not_promote"}
    count += 1
    assert bad_rec["allowed_action"] in {"continue", "archive", "do_not_promote"}
    count += 1
    assert active_search(START)["best_score"] == active["best_score"]
    count += 1
    assert pure_novelty_search(START) is False
    count += 1
    assert run_search_suite(quiet=True)["verdict"] == "ACTIVE_DIAGNOSTIC_VALUE_PASSED"
    count += 1
    return count


if __name__ == "__main__":
    quiet = "--quiet" in sys.argv
    as_json = "--json" in sys.argv
    if not quiet:
        print(f"(source: {SOURCE})\n")
        suite_result = run_search_suite()
        if as_json:
            print("\n=== json ===")
            print(json.dumps(_jsonable({"verdict": suite_result["verdict"], "active_rec": suite_result["active_rec"], "random_rec": suite_result["random_rec"]}), ensure_ascii=False, indent=2))
    print("\n=== tests ===")
    print(f"ok all {_tests()} assertions passed")
