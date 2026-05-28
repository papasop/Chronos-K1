import numpy as np

from k1_manifold_core.axioms.validation import inertia, is_lorentzian, validate_det_negative
from k1_manifold_core.dynamics.critical_damping import critical_damping_matrix
from k1_manifold_core.dynamics.ode_solvers import integrate
from k1_manifold_core.dynamics.symplectic_dissipative import (
    k1_potential_gradient,
    law_ii_matrix,
    vector_field,
)
from k1_manifold_core.geometry.lorentzian import QuadraticForm2D
from k1_manifold_core.geometry.symplectic import induced_generator


def potential(x, G):
    K = float(x.T @ G @ x)
    return 0.5 * (K - 1.0) ** 2


def test_signature_is_1_1_for_lorentzian_form():
    G = np.diag([3.0, -2.0])

    assert inertia(G) == (1, 1, 0)
    assert is_lorentzian(G)


def test_det_negative_equivalent_to_2d_lorentzian_for_symmetric_nondegenerate_samples():
    samples = [
        np.array([[1.0, 0.0], [0.0, -1.0]]),
        np.array([[2.0, 0.4], [0.4, -1.0]]),
        np.array([[3.0, 1.0], [1.0, 0.5]]),
        np.array([[-2.0, 0.3], [0.3, -1.0]]),
    ]

    for G in samples:
        assert is_lorentzian(G) == validate_det_negative(G)


def test_quadratic_form_K_matches_matrix_formula():
    G = np.array([[2.0, 0.25], [0.25, -1.0]])
    x = np.array([1.5, -0.75])

    assert QuadraticForm2D(G).K(x) == float(x.T @ G @ x)


def test_law_ii_vector_field_matches_matrix_formula():
    G = np.diag([1.0, -1.0])
    D = 0.5 * np.eye(2)
    x = np.array([0.2, 0.9])
    grad = k1_potential_gradient(x, G)
    expected = (induced_generator(G) - D) @ grad

    assert np.allclose(law_ii_matrix(G, D), induced_generator(G) - D)
    assert np.allclose(vector_field(x, G, D, lambda _x: grad), expected)


def test_critical_damping_makes_potential_decrease_for_normal_fluctuation():
    G = np.diag([1.0, -1.0])
    D = critical_damping_matrix(G)
    x0 = np.array([0.0, 0.8])

    def rhs(x, t):
        return vector_field(x, G, D, lambda y: k1_potential_gradient(y, G))

    trajectory = integrate(rhs, x0, dt=1e-4, steps=200)
    values = np.array([potential(x, G) for x in trajectory])

    assert np.all(np.diff(values) <= 1e-10)
    assert values[-1] < values[0]
