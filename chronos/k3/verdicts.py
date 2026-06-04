"""Shared K3 verdict helpers.

The K3.2D regime gate deliberately separates field-prediction pipeline health
from topological-object transport health. This avoids the failure mode where a
low field MSE is promoted as a topology result even when the vortex pair is not
transported.
"""

REF_CEIL = 0.05
HARD_DIV_MAX = 0.5
PAIR_INTACT_MIN = 0.6
POS_ERR_CEIL = 8.0


def k32d_verdict(mode: str, ref_med: float, hard_frac: float, pair_frac: float, pos_med: float) -> str:
    """Return the two-layer K3.2D regime verdict."""

    mode = str(mode).upper()
    if mode not in {"SMOKE", "FULL"}:
        raise ValueError(f"mode must be SMOKE or FULL, got {mode!r}")

    pipeline_ok = (ref_med < REF_CEIL) and (hard_frac <= HARD_DIV_MAX)
    transport_ok = (pair_frac >= PAIR_INTACT_MIN) and (pos_med < POS_ERR_CEIL)

    if mode == "SMOKE":
        if not pipeline_ok:
            return "SMOKE_PIPELINE_FAIL"
        if not transport_ok:
            return "SMOKE_PIPELINE_OK_TRANSPORT_FAIL"
        return "SMOKE_TRANSPORT_OK"

    return "FULL_REGIME_VALIDATED" if (pipeline_ok and transport_ok) else "REGIME_UNRESOLVED"


def k32d_explain(verdict: str) -> str:
    explanations = {
        "SMOKE_PIPELINE_FAIL": (
            "Baseline did not learn field prediction or hard-diverged. Fix training/regime before promotion."
        ),
        "SMOKE_PIPELINE_OK_TRANSPORT_FAIL": (
            "Baseline learns field prediction but fails vortex transport. Low field MSE does not imply topology."
        ),
        "SMOKE_TRANSPORT_OK": "Pipeline learns and the vortex pair is transported; promote to FULL for N=30.",
        "FULL_REGIME_VALIDATED": "Baseline trains, vortex pair transported, position error graceful, hard-div low.",
        "REGIME_UNRESOLVED": "Regime not validated. This is not a prior test and not a rejection of 2D topology.",
    }
    return explanations.get(verdict, "unknown verdict")
