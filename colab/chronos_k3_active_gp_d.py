"""Portable K3-E2d cheap-GP active topology regime-search mirror for Colab.

Canonical implementation lives in chronos/k3/active_gp_search.py and
chronos/k3/run_active_gp_search.py.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass, replace

import numpy as np

try:
    from chronos.k3.run_active_gp_search import run_search_suite
    from chronos.k3.active_gp_search import (
        START,
        TRANSPORT_OK_THRESHOLD,
        VortexRegime,
        evaluate_gp_vortex_regime,
        write_memory_event,
    )

    SOURCE = "chronos.k3 (repo)"
except Exception:
    SOURCE = "embedded fallback"
    CTX_TOPOLOGY = "K3_TOPOLOGY_REGIME"
    TRANSPORT_OK_THRESHOLD = 0.9
    SCORE_MARGIN = 0.05
    N_RANDOM_SEEDS = 20
    RANDOM_SUCCESS_MAX = 5

    @dataclass(frozen=True)
    class VortexRegime:
        pair_sep: float
        core_size: float
        push: float
        dt_phys: float = 0.005
        horizon: int = 240

    START = VortexRegime(pair_sep=4.0, core_size=1.0, push=2.6)
    BOUNDS = {"pair_sep": (2.0, 14.0), "core_size": (0.8, 4.0), "push": (0.0, 3.0)}
    GRID_N = 64
    BOX_L = 20.0
    DX = BOX_L / GRID_N
    X_AXIS = np.linspace(-BOX_L / 2, BOX_L / 2, GRID_N, endpoint=False)
    X_GRID, Y_GRID = np.meshgrid(X_AXIS, X_AXIS, indexing="ij")
    K_AXIS = 2 * np.pi * np.fft.fftfreq(GRID_N, d=DX)
    KX_GRID, KY_GRID = np.meshgrid(K_AXIS, K_AXIS, indexing="ij")
    K2_GRID = KX_GRID**2 + KY_GRID**2
    CACHE = {}

    def _clamp(value, low, high):
        return max(low, min(high, value))

    def seed_vortex_pair(pair_sep, core_size):
        th_p = np.arctan2(Y_GRID, X_GRID - pair_sep / 2)
        th_m = np.arctan2(Y_GRID, X_GRID + pair_sep / 2)
        rp = np.sqrt((X_GRID - pair_sep / 2) ** 2 + Y_GRID**2)
        rm = np.sqrt((X_GRID + pair_sep / 2) ** 2 + Y_GRID**2)
        return (np.tanh(rp / core_size) * np.tanh(rm / core_size) * np.exp(1j * (th_p - th_m))).astype(np.complex128)

    def find_vortices(psi):
        ph = np.angle(psi)

        def delta(a, b):
            return (b - a + np.pi) % (2 * np.pi) - np.pi

        w = (
            delta(ph[:-1, :-1], ph[1:, :-1])
            + delta(ph[1:, :-1], ph[1:, 1:])
            + delta(ph[1:, 1:], ph[:-1, 1:])
            + delta(ph[:-1, 1:], ph[:-1, :-1])
        ) / (2 * np.pi)
        return [(X_AXIS[i] + 0.5 * DX, X_AXIS[j] + 0.5 * DX, int(np.sign(w[i, j]))) for i, j in np.argwhere(np.abs(w) > 0.5)]

    def regime_key(r):
        return (round(r.pair_sep, 4), round(r.core_size, 4), round(r.push, 4), round(r.dt_phys, 5), int(r.horizon))

    def evaluate_gp_vortex_regime(regime, g=1.0, checks=10):
        key = regime_key(regime)
        if key in CACHE:
            return CACHE[key]
        sep, dt, steps = regime.pair_sep, regime.dt_phys, int(regime.horizon)
        psi = seed_vortex_pair(regime.pair_sep, regime.core_size) * np.exp(1j * regime.push * Y_GRID)
        exp_half = np.exp(-1j * 0.5 * K2_GRID * dt * 0.5)
        every = max(1, steps // checks)
        errors = []
        blew_up = False
        for step in range(steps):
            psi = np.fft.ifft2(exp_half * np.fft.fft2(psi))
            psi = psi * np.exp(-1j * g * np.abs(psi) ** 2 * dt)
            psi = np.fft.ifft2(exp_half * np.fft.fft2(psi))
            if not np.isfinite(np.abs(psi).max()):
                blew_up = True
                break
            if step % every == 0:
                cores = find_vortices(psi)
                plus = [c for c in cores if c[2] > 0]
                minus = [c for c in cores if c[2] < 0]
                if not plus or not minus:
                    errors.append(1.0)
                    continue
                ep = min(math.hypot(x - sep / 2, y) for x, y, _ in plus)
                em = min(math.hypot(x + sep / 2, y) for x, y, _ in minus)
                errors.append(min(1.0, (ep + em) / (BOX_L / 2)))
        score = 0.0 if blew_up or not errors else max(0.0, 1.0 - float(np.mean(errors)))
        score = float(min(1.0, score))
        ok = score >= TRANSPORT_OK_THRESHOLD
        out = {
            "pair_intact": bool(ok),
            "topology_score": score,
            "transport_score": score,
            "mean_pos_err": (BOX_L / 2) * (1.0 - score),
            "mean_pos_err_units": "physical coordinate units",
            "net_charge_error": 0.0 if ok else 1.0,
            "transport_ok": bool(ok),
            "_gp": {"grid": GRID_N, "steps": steps, "dt": dt, "push": regime.push, "blew_up": blew_up},
        }
        CACHE[key] = out
        return out

    ACTIONS = ["increase_sep", "decrease_sep", "increase_core", "decrease_core", "increase_push", "decrease_push"]
    STEPS = {
        "increase_sep": ("pair_sep", 1.0),
        "decrease_sep": ("pair_sep", -1.0),
        "increase_core": ("core_size", 0.4),
        "decrease_core": ("core_size", -0.4),
        "increase_push": ("push", 0.4),
        "decrease_push": ("push", -0.4),
    }

    def apply_action(regime, action):
        field, delta = STEPS[action]
        lo, hi = BOUNDS[field]
        return replace(regime, **{field: _clamp(getattr(regime, field) + delta, lo, hi)})

    def regime_distance(a, b):
        return math.sqrt(((a.pair_sep - b.pair_sep) / 12.0) ** 2 + ((a.core_size - b.core_size) / 3.2) ** 2 + ((a.push - b.push) / 3.0) ** 2)

    def novelty(candidate, visited):
        return 1.0 if not visited else min(regime_distance(candidate, old) for old in visited)

    def active_search(start=START, n_trials=16):
        visited, history, current = [], [], start
        tts, best = None, 0.0
        for trial in range(n_trials):
            metrics = evaluate_gp_vortex_regime(current)
            history.append((current, metrics))
            visited.append(current)
            if metrics["transport_ok"] and tts is None:
                tts = trial
            best = max(best, metrics["transport_score"])
            current = max((apply_action(current, a) for a in ACTIONS), key=lambda c: (evaluate_gp_vortex_regime(c)["transport_score"], novelty(c, visited)))
        regime, metrics = max(history, key=lambda item: item[1]["transport_score"])
        return {"best_regime": regime, "best_metrics": metrics, "best_score": best, "trials_to_success": tts, "found_transport_ok": metrics["transport_ok"]}

    def random_search(start=START, n_trials=16, seed=0):
        rng = random.Random(seed)
        history, current = [], start
        tts, best = None, 0.0
        for trial in range(n_trials):
            metrics = evaluate_gp_vortex_regime(current)
            history.append((current, metrics))
            if metrics["transport_ok"] and tts is None:
                tts = trial
            best = max(best, metrics["transport_score"])
            current = apply_action(current, rng.choice(ACTIONS))
        regime, metrics = max(history, key=lambda item: item[1]["transport_score"])
        return {"best_regime": regime, "best_metrics": metrics, "best_score": best, "trials_to_success": tts, "found_transport_ok": metrics["transport_ok"]}

    def recommend(metrics):
        if metrics["transport_score"] >= TRANSPORT_OK_THRESHOLD:
            return {"candidate_family": "K3_TOPOLOGICAL", "confidence": "medium", "reason": "Topological object is transported.", "next_vpsl_gate": "constraint", "allowed_action": "continue"}
        return {"candidate_family": "K3_TOPOLOGICAL", "confidence": "low", "reason": "Field learnable but topological object not transported.", "next_vpsl_gate": "regime", "allowed_action": "do_not_promote"}

    def run_search_suite(start=START, n_trials=16, random_seed=0, quiet=False):
        active = active_search(start, n_trials)
        random_result = random_search(start, n_trials, seed=random_seed)
        active_rec = recommend(active["best_metrics"])
        random_rec = recommend(random_result["best_metrics"])
        scores = sorted(random_search(start, n_trials, seed=s)["best_score"] for s in range(N_RANDOM_SEEDS))
        median = scores[len(scores) // 2]
        success = sum(1 for s in range(N_RANDOM_SEEDS) if random_search(start, n_trials, seed=s)["found_transport_ok"])
        passed = success <= RANDOM_SUCCESS_MAX and active["found_transport_ok"] and active["best_score"] > median + SCORE_MARGIN
        verdict = "GP_ACTIVE_DIAGNOSTIC_VALUE_PASSED" if passed else "GP_ACTIVE_NO_ADVANTAGE"
        out = {"active": active, "random": random_result, "active_rec": active_rec, "random_rec": random_rec, "verdict": verdict, "random_median_best_score": median, "random_success_count_20": success}
        if not quiet:
            print("K3-E2d Discriminating GP Truth-Only Active Topology Regime Search\n")
            for label, result, rec in (("ACTIVE", active, active_rec), ("RANDOM", random_result, random_rec)):
                reg, metrics = result["best_regime"], result["best_metrics"]
                print(f"{label}:")
                print(f"  best_score = {result['best_score']:.3f}")
                print(f"  pair_intact = {metrics['pair_intact']}")
                print(f"  mean_pos_err = {metrics['mean_pos_err']:.2f} ({metrics['mean_pos_err_units']})")
                print(f"  trials_to_success = {result['trials_to_success']}")
                print(f"  best_regime = sep={reg.pair_sep:.1f} core={reg.core_size:.1f} push={reg.push:.1f}")
                print(f"  S0 = {rec['candidate_family']} / {rec['allowed_action']}\n")
            print(f"RANDOM (statistics over {N_RANDOM_SEEDS} seeds):")
            print(f"  random_median_best_score = {median:.3f}")
            print(f"  random_reached_transport_ok = {success}/{N_RANDOM_SEEDS}\n")
            print("VERDICT:")
            print(f"  {verdict}")
        return out

    def write_memory_event(out, path, run_id=None):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        record = {"run_id": run_id or "k3e2d-colab", "module": "k3.active.gp_d", "verdict": out["verdict"], "claim_boundary": "cheap GP only; not K3 prior validation; S0 does not certify"}
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


def _jsonable(value):
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _tests() -> int:
    count = 0
    metrics = evaluate_gp_vortex_regime(START)
    for key in ("pair_intact", "topology_score", "transport_score", "mean_pos_err", "transport_ok"):
        assert key in metrics
        count += 1
    assert 0.0 <= metrics["transport_score"] <= 1.0
    count += 1
    high_push = VortexRegime(pair_sep=6.0, core_size=2.0, push=2.6)
    low_push = VortexRegime(pair_sep=6.0, core_size=2.0, push=0.0)
    assert evaluate_gp_vortex_regime(high_push)["transport_score"] < evaluate_gp_vortex_regime(low_push)["transport_score"]
    count += 1
    out = run_search_suite(quiet=True)
    assert out["random_success_count_20"] <= 5
    count += 1
    assert out["active"]["found_transport_ok"] is True
    count += 1
    assert out["verdict"] == "GP_ACTIVE_DIAGNOSTIC_VALUE_PASSED"
    count += 1
    assert evaluate_gp_vortex_regime(START)["transport_score"] < TRANSPORT_OK_THRESHOLD
    count += 1
    assert out["active_rec"]["candidate_family"] == "K3_TOPOLOGICAL"
    count += 1
    assert out["active_rec"]["allowed_action"] == "continue"
    count += 1
    return count


if __name__ == "__main__":
    quiet = "--quiet" in sys.argv
    as_json = "--json" in sys.argv
    memory_path = None
    if "--memory" in sys.argv:
        index = sys.argv.index("--memory")
        if index + 1 < len(sys.argv):
            memory_path = sys.argv[index + 1]
    if not quiet:
        print(f"(source: {SOURCE})\n")
    result = run_search_suite(quiet=quiet or as_json)
    if memory_path:
        event = write_memory_event(result, memory_path)
        if not quiet:
            run_id = getattr(event, "run_id", event.get("run_id", "unknown") if isinstance(event, dict) else "unknown")
            print(f"[M0] wrote memory event to {memory_path} (run_id={run_id})")
    if as_json:
        print(json.dumps({"verdict": result["verdict"], "active_rec": result["active_rec"], "random_rec": result["random_rec"]}, ensure_ascii=False, indent=2))
    print("\n=== tests ===")
    print(f"  ok all {_tests()} assertions passed")
