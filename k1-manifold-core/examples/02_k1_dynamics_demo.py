"""Demo: Law II dynamics with Law III critical damping."""

from __future__ import annotations

import numpy as np

from k1_manifold_core.dynamics.critical_damping import critical_damping_matrix
from k1_manifold_core.dynamics.ode_solvers import integrate
from k1_manifold_core.dynamics.symplectic_dissipative import k1_potential_gradient, vector_field
from k1_manifold_core.geometry.lorentzian import QuadraticForm2D


def potential(x: np.ndarray, G: np.ndarray) -> float:
    K = float(x.T @ G @ x)
    return 0.5 * (K - 1.0) ** 2


def main() -> None:
    G = np.diag([1.0, -1.0])
    D = critical_damping_matrix(G)
    form = QuadraticForm2D(G)
    x0 = np.array([0.0, 0.8])

    def rhs(x: np.ndarray, _t: float) -> np.ndarray:
        return vector_field(x, G, D, lambda y: k1_potential_gradient(y, G))

    trajectory = integrate(rhs, x0, dt=1e-3, steps=400)
    values = np.array([potential(x, G) for x in trajectory])

    print("G = diag(1, -1)")
    print(f"D =\n{D}")
    print(f"x0 = {x0}")
    print(f"K(x0) = {form.K(x0):.6f}")
    print(f"V(x0) = {values[0]:.6f}")
    print(f"x_final = {trajectory[-1]}")
    print(f"K(x_final) = {form.K(trajectory[-1]):.6f}")
    print(f"V(x_final) = {values[-1]:.6f}")
    print(f"V decreased = {values[-1] < values[0]}")


if __name__ == "__main__":
    main()
