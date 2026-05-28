import numpy as np

from k1_manifold_core.dynamics.critical_damping import critical_damping_matrix, local_restoring_rate
from k1_manifold_core.dynamics.symplectic_dissipative import k_dot


def test_k_dot_restores_near_normalized_base_point():
    G = np.diag([1.0, -1.0])
    D = critical_damping_matrix(G)
    x = np.array([0.0, 0.9])

    assert k_dot(x, G, D) > 0.0
    assert local_restoring_rate(G, np.array([1.0, 0.0])) == 4.0
