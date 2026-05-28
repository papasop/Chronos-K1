import numpy as np

from k1_manifold_core.axioms.realizability import axiom_t_holds, find_zero_threshold_witness
from k1_manifold_core.axioms.validation import inertia, is_lorentzian


def test_realizability_selects_lorentzian_form():
    G = np.diag([1.0, -1.0])

    assert axiom_t_holds(G)
    assert find_zero_threshold_witness(G, tolerance=1e-2) is not None
    assert inertia(G) == (1, 1, 0)
    assert is_lorentzian(G)
