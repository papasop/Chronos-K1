import numpy as np

from k1_manifold_core.axioms.validation import validate_det_negative
from k1_manifold_core.geometry.symplectic import generator_eigenvalues, spectral_threshold


def test_det_negative_and_real_spectrum():
    G = np.diag([4.0, -1.0])

    assert validate_det_negative(G)
    assert spectral_threshold(G) == 0.5
    eigenvalues = sorted(np.real(generator_eigenvalues(G)))
    assert np.allclose(eigenvalues, [-0.5, 0.5])
