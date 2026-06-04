"""Diagnostic keys and conservative thresholds for Chronos-S0."""

K1_LORENTZ = "K1_LORENTZ"
K2_SYMPLECTIC = "K2_SYMPLECTIC"
K3_TOPOLOGICAL = "K3_TOPOLOGICAL"
K4_GAUGE = "K4_GAUGE"
K5_HILBERT = "K5_HILBERT"
UNRESOLVED = "UNRESOLVED"

GATE_REGIME = "regime"
GATE_CONSTRAINT = "constraint"
GATE_MECHANISM = "mechanism"
GATE_TRANSFER = "transfer"

ACT_CONTINUE = "continue"
ACT_ARCHIVE = "archive"
ACT_DO_NOT_PROMOTE = "do_not_promote"

CONF_LOW = "low"
CONF_MED = "medium"
CONF_HIGH = "high"

KNOWN_DIAGNOSTICS = {
    "energy_drift",
    "symplectic_jacobian_error",
    "symplectic_improves_vs_controls",
    "causal_violation_rate",
    "topological_transport_score",
    "object_tracking_valid",
    "gauge_residual",
    "unitarity_error",
    "baseline_divergence",
    "field_learnable",
    "diagnostic_context",
}

CTX_TOPOLOGY = "K3_TOPOLOGY_REGIME"
CTX_SYMPLECTIC = "K2_SYMPLECTIC"

TOPO_TRANSPORT_OK = 0.6
SYMPLECTIC_ERR_OK = 0.2
CAUSAL_VIOLATION_SIGNIF = 0.05
GAUGE_RESIDUAL_SIGNIF = 0.1
UNITARITY_ERR_SIGNIF = 0.1
