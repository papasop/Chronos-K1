import numpy as np

from k1_manifold_core.experiments.null_flow import (
    ideal_leaf_probability,
    integrate_null_flow,
    leaf_box_probability,
    leaf_coordinate,
    null_flow_matrix,
    recovery_scaling_slope,
)


def test_null_flow_matrix_has_rank_one():
    A_c = null_flow_matrix()

    assert np.linalg.matrix_rank(A_c) == 1


def test_leaf_coordinate_c_is_conserved_to_roundoff():
    x0 = np.array([0.75, -0.25])
    trajectory = integrate_null_flow(x0, dt=1e-3, steps=5000)
    c_values = np.array([leaf_coordinate(x) for x in trajectory])

    assert np.max(np.abs(c_values - c_values[0])) < 1e-12


def test_leaf_probability_matches_four_c_squared():
    radius = 0.1
    estimated = leaf_box_probability(radius, samples=200_000, seed=2026)
    expected = ideal_leaf_probability(radius)

    assert abs(estimated - expected) < 3e-3


def test_recovery_time_loglog_slope_is_inverse_square():
    epsilons = np.array([0.05, 0.075, 0.1, 0.15, 0.2])
    slope = recovery_scaling_slope(epsilons)

    assert -2.1 <= slope <= -1.9
