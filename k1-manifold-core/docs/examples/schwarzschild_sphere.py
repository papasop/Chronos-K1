from k1_manifold_core.spacetime.spherical import K1, K2, schwarzschild_f

r = 5.0
mass = 1.0
f = schwarzschild_f(mass, r)
fp = 2 * mass / r**2
fpp = -4 * mass / r**3

print("K1 =", K1(f, fp, fpp, r))
print("K2 =", K2(f, fp, r))
