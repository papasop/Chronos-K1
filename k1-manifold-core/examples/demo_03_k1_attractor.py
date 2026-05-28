"""Plot K(t) and V(t) for Law II/III dynamics."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from k1_manifold_core.dynamics.critical_damping import critical_damping_matrix
from k1_manifold_core.dynamics.ode_solvers import integrate
from k1_manifold_core.dynamics.symplectic_dissipative import k1_potential_gradient, vector_field


def potential_and_k(x: np.ndarray, G: np.ndarray) -> tuple[float, float]:
    K = float(x.T @ G @ x)
    return 0.5 * (K - 1.0) ** 2, K


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "demo_03_k1_attractor.png"

    G = np.diag([1.0, -1.0])
    D = critical_damping_matrix(G)
    x0 = np.array([0.0, 0.8])
    dt = 1e-3
    steps = 10000

    def rhs(x: np.ndarray, _t: float) -> np.ndarray:
        return vector_field(x, G, D, lambda y: k1_potential_gradient(y, G))

    trajectory = integrate(rhs, x0, dt=dt, steps=steps)
    times = np.arange(steps + 1) * dt
    values = np.array([potential_and_k(x, G)[0] for x in trajectory])
    k_values = np.array([potential_and_k(x, G)[1] for x in trajectory])

    fig, (ax_k, ax_v) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax_k.plot(times, k_values, color="#1f77b4", linewidth=2)
    ax_k.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax_k.set_ylabel("K(t)")
    ax_k.set_title("K=1 attractor check")
    ax_k.grid(True, alpha=0.3)

    ax_v.plot(times, values, color="#d55e00", linewidth=2)
    ax_v.set_xlabel("t")
    ax_v.set_ylabel("V(t)")
    ax_v.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(f"x0 = {x0}")
    print(f"K(0) = {k_values[0]:.12f}")
    print(f"K(final) = {k_values[-1]:.12f}")
    print(f"V(0) = {values[0]:.12f}")
    print(f"V(final) = {values[-1]:.12e}")
    print(f"V monotone nonincreasing = {bool(np.all(np.diff(values) <= 1e-12))}")
    print(f"saved figure = {output_path}")


if __name__ == "__main__":
    main()
