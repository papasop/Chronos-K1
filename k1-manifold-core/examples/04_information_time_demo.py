"""Demo: information time dt_info = dPhi / H."""

from __future__ import annotations

import numpy as np

from k1_manifold_core.axioms.information_time import information_time


def main() -> None:
    d_phi = 6.0
    H = 3.0
    dt_info = information_time(d_phi, H)

    print(f"dPhi = {d_phi}")
    print(f"H = {H}")
    print(f"dt_info = dPhi / H = {dt_info}")

    d_phi_values = np.array([2.0, 6.0, 12.0])
    H_values = np.array([1.0, 3.0, 4.0])
    print(f"dPhi array = {d_phi_values}")
    print(f"H array = {H_values}")
    print(f"dt_info array = {information_time(d_phi_values, H_values)}")


if __name__ == "__main__":
    main()
