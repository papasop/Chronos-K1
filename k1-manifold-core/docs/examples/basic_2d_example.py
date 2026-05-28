import numpy as np

from k1_manifold_core.geometry.symplectic import induced_generator, spectral_threshold

G = np.diag([1.0, -1.0])

print("J_G =")
print(induced_generator(G))
print("d_c =", spectral_threshold(G))
