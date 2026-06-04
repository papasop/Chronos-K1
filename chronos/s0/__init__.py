"""Chronos-S0 structure recognition layer."""

from .adapters import diagnostics_from_k2_summary, diagnostics_from_k32d_summary, diagnostics_from_summary
from .diagnostics_schema import CTX_SYMPLECTIC, CTX_TOPOLOGY
from .structure_selector import Recommendation, k32d_explain, k32d_verdict, recommend

__all__ = [
    "CTX_SYMPLECTIC",
    "CTX_TOPOLOGY",
    "Recommendation",
    "diagnostics_from_k2_summary",
    "diagnostics_from_k32d_summary",
    "diagnostics_from_summary",
    "k32d_explain",
    "k32d_verdict",
    "recommend",
]
