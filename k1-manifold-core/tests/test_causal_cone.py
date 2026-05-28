import numpy as np

from k1_manifold_core.geometry.lorentzian import QuadraticForm2D, causal_classification


def test_causal_cone_classifies_timelike_lightlike_spacelike_vectors():
    G = np.diag([1.0, -1.0])
    form = QuadraticForm2D(G)

    timelike = np.array([2.0, 0.5])
    right_lightlike = np.array([1.0, 1.0])
    left_lightlike = np.array([1.0, -1.0])
    spacelike = np.array([0.5, 2.0])

    assert form.K(timelike) > 0.0
    assert form.K(right_lightlike) == 0.0
    assert form.K(left_lightlike) == 0.0
    assert form.K(spacelike) < 0.0

    assert causal_classification(G, timelike) == "timelike"
    assert causal_classification(G, right_lightlike) == "lightlike"
    assert causal_classification(G, left_lightlike) == "lightlike"
    assert causal_classification(G, spacelike) == "spacelike"
