"""Einstein-sector formula checks for static spherical metrics."""

from __future__ import annotations

from typing import Any

from .spherical import K1, K2, ricci_components, scalar_curvature


def vacuum_residual(f: Any, fp: Any, fpp: Any, r: Any) -> Any:
    """Return a compact residual for the spherical-vacuum equivalence."""

    ricci = ricci_components(f, fp, fpp, r)
    return (
        abs(K1(f, fp, fpp, r) - 1)
        + abs(K2(f, fp, r) - 1)
        + abs(ricci["R_tt"])
        + abs(ricci["R_rr"])
        + abs(ricci["R_theta_theta"])
        + abs(scalar_curvature(f, fp, fpp, r))
    )
