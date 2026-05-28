from k1_manifold_core.spacetime.rindler import expected_tolman, rindler_dc, tolman_identity_product

kappa = 0.25
ell = 2.0

print("d_c =", rindler_dc(kappa, ell))
print("d_c * T_H =", tolman_identity_product(kappa, ell))
print("T_tol =", expected_tolman(kappa, ell))
