"""K3-E2d discriminating GP truth-only active topology search.

This is active regime search with a cheap real GP evaluator and a continuous,
position-based vortex transport metric. It is not K3 prior validation, not a
proof that topological priors work, not robotics, and not CNN/RL training.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

import numpy as np

from chronos.memory import MemoryEvent, append_event, new_timestamp
from chronos.s0.diagnostics_schema import CTX_TOPOLOGY

TRANSPORT_OK_THRESHOLD = 0.9
SCORE_MARGIN = 0.05
RANDOM_SUCCESS_MAX = 5
N_RANDOM_SEEDS = 20


@dataclass(frozen=True)
class VortexRegime:
    pair_sep: float
    core_size: float
    push: float
    dt_phys: float = 0.005
    horizon: int = 240


BOUNDS = {"pair_sep": (2.0, 14.0), "core_size": (0.8, 4.0), "push": (0.0, 3.0)}
START = VortexRegime(pair_sep=4.0, core_size=1.0, push=2.6)

GRID_N = 64
BOX_L = 20.0
DX = BOX_L / GRID_N
X_AXIS = np.linspace(-BOX_L / 2, BOX_L / 2, GRID_N, endpoint=False)
X_GRID, Y_GRID = np.meshgrid(X_AXIS, X_AXIS, indexing="ij")
K_AXIS = 2 * np.pi * np.fft.fftfreq(GRID_N, d=DX)
KX_GRID, KY_GRID = np.meshgrid(K_AXIS, K_AXIS, indexing="ij")
K2_GRID = KX_GRID**2 + KY_GRID**2

EVALUATION_CACHE: dict[tuple[float, float, float, float, int], dict] = {}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def seed_vortex_pair(pair_sep: float, core_size: float) -> np.ndarray:
    theta_plus = np.arctan2(Y_GRID, X_GRID - pair_sep / 2)
    theta_minus = np.arctan2(Y_GRID, X_GRID + pair_sep / 2)
    radius_plus = np.sqrt((X_GRID - pair_sep / 2) ** 2 + Y_GRID**2)
    radius_minus = np.sqrt((X_GRID + pair_sep / 2) ** 2 + Y_GRID**2)
    psi = np.tanh(radius_plus / core_size) * np.tanh(radius_minus / core_size)
    return (psi * np.exp(1j * (theta_plus - theta_minus))).astype(np.complex128)


def find_vortices(psi: np.ndarray) -> list[tuple[float, float, int]]:
    """Return plaquette-center coordinates for phase-winding singularities."""

    phase = np.angle(psi)

    def delta(a, b):
        return (b - a + np.pi) % (2 * np.pi) - np.pi

    winding = (
        delta(phase[:-1, :-1], phase[1:, :-1])
        + delta(phase[1:, :-1], phase[1:, 1:])
        + delta(phase[1:, 1:], phase[:-1, 1:])
        + delta(phase[:-1, 1:], phase[:-1, :-1])
    ) / (2 * np.pi)
    cores = []
    for i, j in np.argwhere(np.abs(winding) > 0.5):
        cores.append((X_AXIS[i] + 0.5 * DX, X_AXIS[j] + 0.5 * DX, int(np.sign(winding[i, j]))))
    return cores


def regime_key(regime: VortexRegime) -> tuple[float, float, float, float, int]:
    return (
        round(regime.pair_sep, 4),
        round(regime.core_size, 4),
        round(regime.push, 4),
        round(regime.dt_phys, 5),
        int(regime.horizon),
    )


def evaluate_gp_vortex_regime(regime: VortexRegime, g: float = 1.0, checks: int = 10) -> dict:
    """Score a regime by tracked vortex-position transport under cheap GP truth.

    The transport score is continuous: 1 minus the mean normalized position
    error of the detected +1/-1 pair from the intended static positions.
    ``mean_pos_err`` is reported in physical coordinate units, not grid-index
    units.
    """

    key = regime_key(regime)
    if key in EVALUATION_CACHE:
        return EVALUATION_CACHE[key]

    sep = regime.pair_sep
    dt = regime.dt_phys
    steps = int(regime.horizon)
    psi = seed_vortex_pair(regime.pair_sep, regime.core_size) * np.exp(1j * regime.push * Y_GRID)
    exp_kinetic_half = np.exp(-1j * 0.5 * K2_GRID * dt * 0.5)
    every = max(1, steps // checks)
    errors = []
    blew_up = False

    for step in range(steps):
        psi = np.fft.ifft2(exp_kinetic_half * np.fft.fft2(psi))
        psi = psi * np.exp(-1j * g * np.abs(psi) ** 2 * dt)
        psi = np.fft.ifft2(exp_kinetic_half * np.fft.fft2(psi))
        if not np.isfinite(np.abs(psi).max()):
            blew_up = True
            break
        if step % every == 0:
            cores = find_vortices(psi)
            plus = [core for core in cores if core[2] > 0]
            minus = [core for core in cores if core[2] < 0]
            if not plus or not minus:
                errors.append(1.0)
                continue
            plus_error = min(math.hypot(x - sep / 2, y) for x, y, _charge in plus)
            minus_error = min(math.hypot(x + sep / 2, y) for x, y, _charge in minus)
            errors.append(min(1.0, (plus_error + minus_error) / (BOX_L / 2)))

    transport_score = 0.0 if blew_up or not errors else max(0.0, 1.0 - float(np.mean(errors)))
    transport_score = float(min(1.0, transport_score))
    transport_ok = transport_score >= TRANSPORT_OK_THRESHOLD
    mean_pos_err = (BOX_L / 2) * (1.0 - transport_score)
    metrics = {
        "pair_intact": bool(transport_ok),
        "pair_intact_definition": "mean tracked-position transport_score >= TRANSPORT_OK_THRESHOLD",
        "topology_score": transport_score,
        "transport_score": transport_score,
        "mean_pos_err": mean_pos_err,
        "mean_pos_err_units": "physical coordinate units",
        "net_charge_error": 0.0 if transport_ok else 1.0,
        "transport_ok": bool(transport_ok),
        "_gp": {"grid": GRID_N, "steps": steps, "dt": dt, "push": regime.push, "blew_up": blew_up},
    }
    EVALUATION_CACHE[key] = metrics
    return metrics


ACTIONS = ["increase_sep", "decrease_sep", "increase_core", "decrease_core", "increase_push", "decrease_push"]
ACTION_STEPS = {
    "increase_sep": ("pair_sep", +1.0),
    "decrease_sep": ("pair_sep", -1.0),
    "increase_core": ("core_size", +0.4),
    "decrease_core": ("core_size", -0.4),
    "increase_push": ("push", +0.4),
    "decrease_push": ("push", -0.4),
}


def apply_action(regime: VortexRegime, action: str) -> VortexRegime:
    field, delta = ACTION_STEPS[action]
    low, high = BOUNDS[field]
    return replace(regime, **{field: _clamp(getattr(regime, field) + delta, low, high)})


def regime_distance(a: VortexRegime, b: VortexRegime) -> float:
    return math.sqrt(
        ((a.pair_sep - b.pair_sep) / 12.0) ** 2
        + ((a.core_size - b.core_size) / 3.2) ** 2
        + ((a.push - b.push) / 3.0) ** 2
    )


def novelty(candidate: VortexRegime, visited: list[VortexRegime]) -> float:
    if not visited:
        return 1.0
    return min(regime_distance(candidate, old) for old in visited)


def active_search(start: VortexRegime = START, n_trials: int = 16) -> dict:
    visited = []
    history = []
    current = start
    trials_to_success = None
    best_score = 0.0
    for trial in range(n_trials):
        metrics = evaluate_gp_vortex_regime(current)
        history.append((current, metrics))
        visited.append(current)
        if metrics["transport_ok"] and trials_to_success is None:
            trials_to_success = trial
        best_score = max(best_score, metrics["transport_score"])
        candidates = [apply_action(current, action) for action in ACTIONS]
        current = max(
            candidates,
            key=lambda candidate: (evaluate_gp_vortex_regime(candidate)["transport_score"], novelty(candidate, visited)),
        )
    best_regime, best_metrics = max(history, key=lambda item: item[1]["transport_score"])
    return {
        "best_regime": best_regime,
        "best_metrics": best_metrics,
        "best_score": best_score,
        "trials_to_success": trials_to_success,
        "found_transport_ok": best_metrics["transport_ok"],
    }


def random_search(start: VortexRegime = START, n_trials: int = 16, seed: int = 0) -> dict:
    rng = random.Random(seed)
    history = []
    current = start
    trials_to_success = None
    best_score = 0.0
    for trial in range(n_trials):
        metrics = evaluate_gp_vortex_regime(current)
        history.append((current, metrics))
        if metrics["transport_ok"] and trials_to_success is None:
            trials_to_success = trial
        best_score = max(best_score, metrics["transport_score"])
        current = apply_action(current, rng.choice(ACTIONS))
    best_regime, best_metrics = max(history, key=lambda item: item[1]["transport_score"])
    return {
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
        "topological_transport_score": float(metrics["transport_score"]),
    }


def passed_admission(active_result: dict, random_median_best: float, random_success_count: int) -> bool:
    return (
        random_success_count <= RANDOM_SUCCESS_MAX
        and active_result["found_transport_ok"]
        and active_result["best_score"] > random_median_best + SCORE_MARGIN
    )


def write_memory_event(out: dict, path: str, run_id: str | None = None) -> MemoryEvent:
    recommendation = out["active_rec"]
    resolved_run_id = run_id or f"k3e2d-{new_timestamp()}"
    event = MemoryEvent(
        timestamp=new_timestamp(),
        run_id=resolved_run_id,
        module="k3.active.gp_d",
        experiment_kind="k3_e2d_discriminating_gp_active_topology_search",
        verdict=out["verdict"],
        candidate_family=recommendation["candidate_family"],
        allowed_action=recommendation["allowed_action"],
        score=out["active"]["best_score"],
        payload={
            "active_best_score": out["active"]["best_score"],
            "active_found_transport_ok": out["active"]["found_transport_ok"],
            "active_trials_to_success": out["active"]["trials_to_success"],
            "random_median_best_score": out.get("random_median_best_score"),
            "random_success_count_20": out.get("random_success_count_20"),
        },
        claim_boundary=(
            "cheap real-GP active search (64x64, CPU) with continuous position-tracking transport; "
            "landscape made discriminating via a push velocity-kick failure dimension. NOT "
            "full-resolution GP, NOT K3 prior validation, NOT a proof topological priors work. "
            "Active is guided search. S0 does not certify."
        ),
        code_version="k3_e2d_continuous_transport_v1",
    )
    append_event(event, path)
    return event
