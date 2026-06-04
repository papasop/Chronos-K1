"""K3-E2b active topology regime search on an interpretable toy landscape.

This is active regime search only. It is not K3 prior validation, not a proof
that topological priors work, not robotics, not RL training, not CNN training,
and not a Gross-Pitaevskii run.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

from chronos.s0.diagnostics_schema import CTX_TOPOLOGY

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


BOUNDS = {
    "pair_sep": (4.0, 24.0),
    "core_size": (0.8, 6.0),
    "dt_phys": (0.005, 0.08),
    "horizon": (20, 160),
}

START = VortexRegime(pair_sep=5.0, core_size=1.0, dt_phys=0.06, horizon=120)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _triangular_score(x: float, low: float, mid: float, high: float) -> float:
    """Triangular score with peak 1 at mid and 0 outside [low, high]."""

    if x <= low or x >= high:
        return 0.0
    if x <= mid:
        return (x - low) / (mid - low)
    return (high - x) / (high - mid)


def evaluate_vortex_regime(regime: VortexRegime) -> dict:
    """Evaluate a transparent toy survival landscape.

    This is not GP truth. A high score requires all four factor scores to be
    decent, creating a product-shaped stable region for active search.
    """

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
        "_factors": {
            "sep": round(sep, 3),
            "core": round(core, 3),
            "dt": round(dt_score, 3),
            "hor": round(horizon_score, 3),
        },
    }


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
    if not visited:
        return 1.0
    return min(regime_distance(candidate, old) for old in visited)


def active_search(start_regime: VortexRegime = START, n_trials: int = 30) -> dict:
    """Guided active search using observed topology score; novelty breaks ties."""

    visited_regimes = []
    history = []
    current = start_regime
    trials_to_success = None
    best_score = 0.0
    for trial in range(n_trials):
        metrics = evaluate_vortex_regime(current)
        history.append((current, metrics))
        visited_regimes.append(current)
        if metrics["transport_ok"] and trials_to_success is None:
            trials_to_success = trial
        best_score = max(best_score, metrics["topology_score"])
        candidates = [apply_action(current, action) for action in ACTIONS]
        current = max(
            candidates,
            key=lambda candidate: (evaluate_vortex_regime(candidate)["topology_score"], novelty(candidate, visited_regimes)),
        )
    best_regime, best_metrics = max(history, key=lambda item: item[1]["topology_score"])
    return {
        "history": history,
        "best_regime": best_regime,
        "best_metrics": best_metrics,
        "best_score": best_score,
        "trials_to_success": trials_to_success,
        "found_transport_ok": best_metrics["transport_ok"],
    }


def random_search(start_regime: VortexRegime = START, n_trials: int = 30, seed: int = 0) -> dict:
    """Random-action control with the same start and budget."""

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
    return {
        "history": history,
        "best_regime": best_regime,
        "best_metrics": best_metrics,
        "best_score": best_score,
        "trials_to_success": trials_to_success,
        "found_transport_ok": best_metrics["transport_ok"],
    }


def diagnostics_from_vortex_search(metrics: dict) -> dict:
    return {
        "diagnostic_context": CTX_TOPOLOGY,
        "field_learnable": True,
        "object_tracking_valid": bool(metrics["transport_ok"]),
        "topological_transport_score": float(metrics["topology_score"]),
    }


def has_active_advantage(active_result: dict, random_result: dict) -> bool:
    return (
        (active_result["found_transport_ok"] and not random_result["found_transport_ok"])
        or (active_result["best_score"] > random_result["best_score"] + SCORE_MARGIN)
        or (
            active_result["trials_to_success"] is not None
            and (
                random_result["trials_to_success"] is None
                or active_result["trials_to_success"] < random_result["trials_to_success"]
            )
        )
    )


def pure_novelty_search(start_regime: VortexRegime = START, n_trials: int = 30) -> bool:
    """Negative capability check: pure novelty alone does not solve this landscape."""

    visited = [start_regime]
    current = start_regime
    for _ in range(n_trials):
        metrics = evaluate_vortex_regime(current)
        if metrics["transport_ok"]:
            return True
        current = max((apply_action(current, action) for action in ACTIONS), key=lambda candidate: novelty(candidate, visited))
        visited.append(current)
    return False
