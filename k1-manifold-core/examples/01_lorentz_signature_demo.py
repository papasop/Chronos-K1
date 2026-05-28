"""Demo: 2D Lorentzian signature and induced spectral threshold."""

from __future__ import annotations

import numpy as np

from k1_manifold_core.axioms.validation import determinant, inertia, is_lorentzian, validate_det_negative
from k1_manifold_core.geometry.symplectic import generator_eigenvalues, spectral_threshold


def main() -> None:
    G = np.array([[2.0, 0.4], [0.4, -1.0]])
    eigenvalues = generator_eigenvalues(G)

    print("G =")
    print(G)
    print(f"det(G) = {determinant(G):.6f}")
    print(f"inertia(G) = {inertia(G)}")
    print(f"is_lorentzian(G) = {is_lorentzian(G)}")
    print(f"det(G) < 0 = {validate_det_negative(G)}")
    print(f"spec(J_G) = {np.sort(np.real_if_close(eigenvalues))}")
    print(f"d_c = {spectral_threshold(G):.6f}")


if __name__ == "__main__":
    main()
