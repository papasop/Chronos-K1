import sympy as sp

from k1_manifold_core.spacetime.spherical import (
    K1,
    K2,
    reissner_nordstrom_f,
    ricci_components,
    schwarzschild_de_sitter_f,
    schwarzschild_f,
)


def differentiated(expr, r):
    return expr, sp.diff(expr, r), sp.diff(expr, r, 2)


def assert_zero(expr):
    assert sp.simplify(expr) == 0


def test_schwarzschild_satisfies_vacuum_k_conditions_and_ricci_flatness():
    r, M = sp.symbols("r M", positive=True, nonzero=True)
    f, fp, fpp = differentiated(schwarzschild_f(M, r), r)
    ricci = ricci_components(f, fp, fpp, r)

    assert_zero(K1(f, fp, fpp, r) - 1)
    assert_zero(K2(f, fp, r) - 1)
    assert_zero(ricci["R_tt"])
    assert_zero(ricci["R_rr"])
    assert_zero(ricci["R_theta_theta"])


def test_reissner_nordstrom_shows_rigidity_asymmetry():
    r, M, Q = sp.symbols("r M Q", positive=True, nonzero=True)
    f, fp, fpp = differentiated(reissner_nordstrom_f(M, Q, r), r)

    assert_zero(K1(f, fp, fpp, r) - 1)
    assert_zero(K2(f, fp, r) - (1 - Q**2 / r**2))
    assert sp.simplify(K2(f, fp, r) - 1) != 0


def test_schwarzschild_de_sitter_k_conditions_and_einstein_lambda_components():
    r, M, Lambda = sp.symbols("r M Lambda", positive=True, nonzero=True)
    f, fp, fpp = differentiated(schwarzschild_de_sitter_f(M, Lambda, r), r)
    ricci = ricci_components(f, fp, fpp, r)

    assert_zero(K1(f, fp, fpp, r) - (1 - 2 * Lambda * r**2))
    assert_zero(K2(f, fp, r) - (1 - Lambda * r**2))
    assert_zero(ricci["R_tt"] - (-Lambda * f))
    assert_zero(ricci["R_rr"] - (Lambda / f))
    assert_zero(ricci["R_theta_theta"] - Lambda * r**2)
