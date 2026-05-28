"""Information-time helper formulas."""

from __future__ import annotations


def information_time(d_phi: float, H: float) -> float:
    """Return ``dt_info = dPhi / H`` for positive scalar cost ``H``."""

    if H <= 0.0:
        raise ValueError("H must be positive")
    return d_phi / H
