"""Conditional Clausius bridge utilities."""

from __future__ import annotations

from dataclasses import dataclass

from .temperature import effective_temperature


@dataclass(frozen=True)
class ClausiusCheck:
    holds: bool
    T_eff: float
    T_tol: float
    mismatch: float


def check_local_clausius(noise_sigma: float, d_c: float, T_tol: float, *, atol: float = 1e-10) -> ClausiusCheck:
    """Check the external identification ``T_eff = T_tol``."""

    T_eff = effective_temperature(noise_sigma, d_c)
    mismatch = abs(T_eff - T_tol)
    return ClausiusCheck(holds=mismatch <= atol, T_eff=T_eff, T_tol=T_tol, mismatch=mismatch)
