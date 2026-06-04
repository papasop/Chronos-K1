# K2 Reconstruction Notes

## Current Status

Available Source Code:

- `k2_1_symplectic_prior.py`
- `k2_1b_repair_controls.py`
- `k2_2a_transfer_h200.py`

Available Logs:

- K2.0 Regime Validation
- K2.0-B Refine Graceful Window

Missing Source Code:

- K2.0
- K2.0-B

## K2.0

Purpose:

Find a graceful-failure regime for FPU-β.

Known Outcome:

- S=1 identified as candidate regime.
- Higher S values became divergent.

Status:

Historical logs preserved. Source script missing.

## K2.0-B

Purpose:

Separate rollout accumulation from training instability.

Known Outcome:

- H=160 selected as clean VPSL regime.
- Rollout accumulation identified as dominant failure mode.

Status:

Historical logs preserved. Source script missing.

## Reproducibility Statement

K2.1, K2.1-B, and K2.2-A have preserved repository archive entrypoints.

K2.2-B has the full audit-fixed Colab training source restored. The remaining
K2 archive scripts preserve archived summaries and verdict semantics, but full
training reproduction for K2.1, K2.1-B, and K2.2-A still requires restoration
of the original Colab experiment sources.

K2.0 and K2.0-B are currently supported by:

- archived logs
- archived summaries
- archived conclusions

Future work may reconstruct the original K2.0/K2.0-B source files and restore
the full K2.1/K2.1-B/K2.2-A Colab training sources.
