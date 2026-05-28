"""Demo: information time dt_info = dPhi / H."""

from __future__ import annotations

from k1_manifold_core.axioms.information_time import information_time


def main() -> None:
    d_phi = 6.0
    H = 3.0
    dt_info = information_time(d_phi, H)

    print(f"dPhi = {d_phi}")
    print(f"H = {H}")
    print(f"dt_info = dPhi / H = {dt_info}")


if __name__ == "__main__":
    main()
