# Chronos K2

K2 is the first VPSL-certified physical-structure milestone.

## Current Status

```text
K2 reached FULL_TRANSFER_CONFIRMED at K2.2-A and extended it through H=240 at
K2.2-B.
```

K2.1 onward currently has repository archive entrypoints and verdict logic.
Full training reproduction requires restoring the original Colab experiment
sources. The regime-discovery stages K2.0 and K2.0-B are preserved as
historical logs and are being reconstructed.

## Available Archive Entrypoints

- K2.1: `experiments/k2_1_symplectic_prior.py`
- K2.1-B: `experiments/k2_1b_repair_controls.py`
- K2.2-A: `experiments/k2_2a_transfer_h200.py`
- K2.2-B: `experiments/k2_2b_transfer_h240.py`

These are archive entrypoints, not complete Colab training scripts.

See `experiments/README.md` for the directory-level boundary statement.

## Missing Source Code

- K2.0: source script not currently available
- K2.0-B: source script not currently available

## Available Evidence

- K2.0 / K2.0-B logs: `historical_logs/`
- K2.1 / K2.1-B archive entrypoints and archived summaries
- K2.2-A archive entrypoint, archived summary, and verdict wiring
- K2.2-B archive entrypoint, archived summary, and verdict wiring

## Claim Boundary

K2 can be cited as a complete result chain with partial source-code
availability:

```text
K2 = complete result chain, partial source missing
K2.0/K2.0-B = historical log archive
K2.1+ = archive entrypoints available; full training source pending restoration
K2.2-B = FULL_TRANSFER_CONFIRMED through H=240
```

The final K2 claim remains:

```text
FULL_TRANSFER_CONFIRMED through H=240
```

## Next Reconstruction Work

- Reconstruct `k2_0_reconstructed.py`
- Reconstruct `k2_0b_reconstructed.py`
- Compare reconstructed outputs against the historical logs
- Promote reconstructed scripts only after the outputs match the archived
  regime/window conclusions
