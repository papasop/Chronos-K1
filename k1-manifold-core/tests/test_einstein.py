from k1_manifold_core.spacetime.spherical import K1, K2, schwarzschild_de_sitter_f, schwarzschild_f


def test_schwarzschild_satisfies_vacuum_k_conditions():
    f = schwarzschild_f(1.0)

    assert abs(K1(f, 5.0) - 1.0) < 1e-5
    assert abs(K2(f, 5.0) - 1.0) < 1e-8


def test_schwarzschild_de_sitter_k_conditions():
    Lambda = 0.01
    r = 5.0
    f = schwarzschild_de_sitter_f(1.0, Lambda)

    assert abs(K1(f, r) - (1.0 - 2.0 * Lambda * r * r)) < 1e-4
    assert abs(K2(f, r) - (1.0 - Lambda * r * r)) < 1e-7
