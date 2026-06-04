"""Adapters from experiment summaries to Chronos-S0 diagnostics."""

from __future__ import annotations

import csv
import json
import os
from typing import Any

from .diagnostics_schema import CTX_SYMPLECTIC, CTX_TOPOLOGY


def diagnostics_from_k32d_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Translate a K3.2D vortex-regime summary row into S0 diagnostics."""

    pipeline_ok = bool(summary.get("pipeline_ok", False))
    transport_ok = bool(summary.get("transport_ok", False))
    if transport_ok and ("pair_frac" not in summary or summary.get("pair_frac") is None):
        raise ValueError(
            "k3_2d summary claims transport_ok=True but pair_frac is missing "
            "(no transport evidence)"
        )

    pair_frac = float(summary.get("pair_frac", 0.0))
    pair_frac = min(1.0, max(0.0, pair_frac))
    return {
        "diagnostic_context": CTX_TOPOLOGY,
        "field_learnable": pipeline_ok,
        "baseline_divergence": float(summary.get("hard_frac", 1.0)),
        "object_tracking_valid": transport_ok,
        "topological_transport_score": pair_frac,
    }


def diagnostics_from_k2_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Translate a K2 symplectic summary row into S0 diagnostics."""

    return {
        "diagnostic_context": CTX_SYMPLECTIC,
        "symplectic_improves_vs_controls": bool(summary.get("beats_baseline_and_controls", False)),
        "symplectic_jacobian_error": float(summary.get("symp_err", 1.0)),
        "field_learnable": True,
        "baseline_divergence": float(summary.get("hard_frac", 0.0)),
    }


ADAPTERS = {
    "k2": diagnostics_from_k2_summary,
    "k3_2d": diagnostics_from_k32d_summary,
}


def diagnostics_from_summary(kind: str, summary: dict[str, Any]) -> dict[str, Any]:
    if kind not in ADAPTERS:
        raise ValueError(f"unknown experiment kind: {kind!r} (known: {sorted(ADAPTERS)})")
    return ADAPTERS[kind](summary)


def _cli_coerce(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except (TypeError, ValueError):
            pass
    return text


def load_summary(path: str) -> dict[str, Any]:
    """Load a single summary row from JSON object/list or CSV."""

    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path) as handle:
            obj = json.load(handle)
        if isinstance(obj, list):
            if not obj:
                raise ValueError("JSON summary list is empty")
            obj = obj[0]
        if not isinstance(obj, dict):
            raise ValueError("JSON summary must be an object or a list of objects")
        return obj

    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no data rows in CSV summary: {path}")
    return {key: _cli_coerce(value) for key, value in rows[0].items()}
