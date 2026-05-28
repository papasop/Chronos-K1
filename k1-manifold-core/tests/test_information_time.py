import pytest

from k1_manifold_core.axioms.information_time import information_time


def test_information_time_is_dphi_over_h():
    assert information_time(6.0, 3.0) == 2.0
    assert information_time(1.25, 0.5) == 2.5


def test_information_time_requires_positive_h():
    with pytest.raises(ValueError):
        information_time(1.0, 0.0)
