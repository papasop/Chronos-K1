from k1_manifold_core.spacetime.spherical import K1, K2, schwarzschild_f

f = schwarzschild_f(mass=1.0)
r = 5.0

print("K1 =", K1(f, r))
print("K2 =", K2(f, r))
