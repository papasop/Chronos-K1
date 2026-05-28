"""Demo: K=1 attractor check under Law II/III dynamics."""

from __future__ import annotations

import numpy as np

from k1_manifold_core.dynamics.critical_damping import critical_damping_matrix
from k1_manifold_core.dynamics.ode_solvers import integrate
from k1_manifold_core.dynamics.symplectic_dissipative import k1_potential_gradient, vector_field


def potential_and_k(x: np.ndarray, G: np.ndarray) -> tuple[float, float]:
    K = float(x.T @ G @ x)
    return 0.5 * (K - 1.0) ** 2, K


def main() -> None:
    G = np.diag([1.0, -1.0])
    D = critical_damping_matrix(G)
    initial_states = [
        np.array([1.2, 0.0]),
        np.array([0.0, 0.8]),
        np.array([1.1, 0.1]),
        np.array([1.4, 0.2]),
    ]

    def rhs(x: np.ndarray, _t: float) -> np.ndarray:
        return vector_field(x, G, D, lambda y: k1_potential_gradient(y, G))

    for x0 in initial_states:
        trajectory = integrate(rhs, x0, dt=1e-3, steps=10000)
        values = np.array([potential_and_k(x, G)[0] for x in trajectory])
        k_values = np.array([potential_and_k(x, G)[1] for x in trajectory])

        print(f"x0 = {x0}")
        print(f"  K(0) = {k_values[0]:.12f}")
        print(f"  V(0) = {values[0]:.12f}")
        print(f"  x_final = {trajectory[-1]}")
        print(f"  K(final) = {k_values[-1]:.12f}")
        print(f"  V(final) = {values[-1]:.12e}")
        print(f"  V monotone nonincreasing = {bool(np.all(np.diff(values) <= 1e-12))}")


if __name__ == "__main__":
    main()
