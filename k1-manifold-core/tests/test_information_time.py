import pytest
import numpy as np

from k1_manifold_core.axioms.information_time import information_time


def test_information_time_is_dphi_over_h_for_scalars():
    assert information_time(6.0, 3.0) == 2.0
    assert information_time(1.25, 0.5) == 2.5


def test_information_time_is_dphi_over_h_for_arrays():
    d_phi = np.array([2.0, 6.0, 12.0])
    H = np.array([1.0, 3.0, 4.0])

    assert np.allclose(information_time(d_phi, H), np.array([2.0, 2.0, 3.0]))


def test_information_time_requires_positive_h():
    with pytest.raises(ValueError):
        information_time(1.0, 0.0)

    with pytest.raises(ValueError):
        information_time(np.array([1.0, 2.0]), np.array([1.0, -1.0]))
