import numpy as np

from k1_manifold_core.dynamics.critical_damping import critical_damping_matrix
from k1_manifold_core.dynamics.ode_solvers import integrate
from k1_manifold_core.dynamics.symplectic_dissipative import k1_potential_gradient, vector_field


def potential_and_k(x, G):
    K = float(x.T @ G @ x)
    return 0.5 * (K - 1.0) ** 2, K


def test_k1_critical_damping_makes_v_decrease_and_k_approach_one_for_multiple_initial_states():
    G = np.diag([1.0, -1.0])
    D = critical_damping_matrix(G)
    initial_states = [
        np.array([1.2, 0.0]),
        np.array([0.0, 0.8]),
        np.array([1.1, 0.1]),
        np.array([1.4, 0.2]),
    ]

    def rhs(x, _t):
        return vector_field(x, G, D, lambda y: k1_potential_gradient(y, G))

    for x0 in initial_states:
        trajectory = integrate(rhs, x0, dt=1e-3, steps=10000)
        values = np.array([potential_and_k(x, G)[0] for x in trajectory])
        k_values = np.array([potential_and_k(x, G)[1] for x in trajectory])

        assert np.all(np.diff(values) <= 1e-12)
        assert values[-1] < values[0]
        assert abs(k_values[-1] - 1.0) < 1e-8
