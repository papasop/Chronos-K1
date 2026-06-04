# Chronos K2

K2 is the first VPSL-certified physical-structure milestone.

## Current Status

```text
K2 reached FULL_TRANSFER_CONFIRMED at K2.2-A and extended it through H=240 at
K2.2-B.
```

K2.1, K2.1-B, and K2.2-A currently have repository archive entrypoints and
verdict logic. K2.2-B has the full audit-fixed Colab training source restored
under `experiments/k2_2b_transfer_h240.py`. The regime-discovery stages K2.0
and K2.0-B are preserved as historical logs and are being reconstructed.

## Available Entrypoints

- K2.1: `experiments/k2_1_symplectic_prior.py`
- K2.1-B: `experiments/k2_1b_repair_controls.py`
- K2.2-A: `experiments/k2_2a_transfer_h200.py`
- K2.2-B: `experiments/k2_2b_transfer_h240.py`
- K2.3: `experiments/k2_3_wrong_omega_reanalysis.py`

K2.2-B is now a full Colab-scale training script. Running it directly launches
the expensive two-phase H=240 experiment. Use `py_compile` for repository CI
checks unless intentionally reproducing the full run on a GPU runtime.

See `experiments/README.md` for the directory-level boundary statement.

## Missing Source Code

- K2.0: source script not currently available
- K2.0-B: source script not currently available

## Available Evidence

- K2.0 / K2.0-B logs: `historical_logs/`
- K2.1 / K2.1-B archive entrypoints and archived summaries
- K2.2-A archive entrypoint, archived summary, and verdict wiring
- K2.2-B full audit-fixed Colab source, archived summary, and verdict wiring
- K2.3 wrong-Ω specificity reanalysis summary and code

## Claim Boundary

K2 can be cited as a complete result chain with partial source-code
availability:

```text
K2 = complete result chain, partial source missing
K2.0/K2.0-B = historical log archive
K2.1/K2.1-B/K2.2-A = archive entrypoints available
K2.2-B = full audit-fixed Colab training source restored
K2.2-B = FULL_TRANSFER_CONFIRMED through H=240
K2.3 = OMEGA_SPECIFICITY_CONFIRMED_NONDEGEN_AWARE hardening
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
