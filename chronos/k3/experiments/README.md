# K3 Experiment Entrypoints

This directory contains K3 candidate-structure experiments.

K3 is not certified yet. These scripts are regime-validation or candidate-test
entrypoints for future VPSL structures.

## Current Entrypoints

- `k3_0d_sine_gordon_winding_density.py`
- `k3_1_winding_density_prior.py`
- `k3_2d_0_vortex_regime.py`

## Boundary

K3.0-D is baseline-only regime validation for a winding-density / local
topological-structure target on periodic Sine-Gordon. K3.1 tests a
winding-density continuity prior and is archived as `NO_EFFECT`.

K3.2D.0 is a SMOKE-first 2D Gross-Pitaevskii vortex-antivortex regime
validation. It is baseline-only and does not test a prior.

None of these files certifies integer topological charge.
