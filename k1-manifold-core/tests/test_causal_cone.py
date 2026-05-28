import numpy as np

from k1_manifold_core.geometry.lorentzian import QuadraticForm2D, causal_classification


def test_canonical_causal_cone_classifies_timelike_lightlike_spacelike_vectors():
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


def test_nondiagonal_causal_cone_classifies_by_quadratic_form_sign():
    G = np.array([[2.0, 0.4], [0.4, -1.0]])
    form = QuadraticForm2D(G)

    # With x=(1, s), K=2+0.8s-s^2, so these slopes are exactly null.
    root = np.sqrt(0.8**2 + 8.0)
    null_right = np.array([1.0, (0.8 + root) / 2.0])
    null_left = np.array([1.0, (0.8 - root) / 2.0])
    timelike = np.array([1.0, 0.0])
    spacelike = np.array([0.0, 1.0])

    assert form.K(timelike) > 0.0
    assert abs(form.K(null_right)) < 1e-12
    assert abs(form.K(null_left)) < 1e-12
    assert form.K(spacelike) < 0.0

    assert causal_classification(G, timelike) == "timelike"
    assert causal_classification(G, null_right) == "lightlike"
    assert causal_classification(G, null_left) == "lightlike"
    assert causal_classification(G, spacelike) == "spacelike"
