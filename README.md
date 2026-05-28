# Chronos-K1

Chronos-K1 is a Lorentzian structural dynamics framework in which time is modeled as the cost of structural change and causal structure is encoded directly into the geometry of a state space.

The current repository is a reproducible research prototype. It contains the numerical core, tests, and demos needed to inspect the implemented mathematics. It does not claim to derive physical reality or solve world modeling.

## Core Idea

Most AI world models learn temporal structure and causality from data.

Chronos-K1 explores a different structural starting point:

- Information time is represented by `dt_info = dPhi / H`.
- Causality is represented by a Lorentzian quadratic form `G`.
- Local structural consistency is represented by the `K=1` manifold, where `K(x) = x.T @ G @ x`.
- Dynamics are modeled by a symplectic-dissipative flow, `xdot = (J_G - D) grad V`.
- The critical-damping choice `D = d_c I` is tested numerically as a local attractor mechanism for `K=1`.

## What Has Been Implemented

### Lorentzian Signature Verification

The tests verify the two-dimensional signature checks used by the core package:

- `Sig(G) = (1, 1)`
- `det(G) < 0`
- real spectral threshold `d_c`

### Information Time

The tests verify:

```text
dt_info = dPhi / H
```

for scalar and vector inputs, with positive `H`.

### Causal Cone Classification

The tests verify classification by the sign of:

```text
K(x) = x.T @ G @ x
```

into:

- timelike
- lightlike
- spacelike

The classification is tested for both canonical and non-diagonal Lorentzian forms.

### K=1 Attractor Dynamics

The tests verify that:

```text
V = 1/2 (K - 1)^2
```

decreases under the implemented Law II / Law III dynamics and that `K(t)` approaches `1` for the tested local trajectories.

### Spherical-Sector Reformulation

The `spacetime` tests use symbolic differentiation to verify the spherical-sector identities for Schwarzschild, Reissner-Nordstrom, and Schwarzschild-de Sitter examples.

These are numerical and symbolic reformulation checks, not a general derivation of Einstein gravity.

## Repository Layout

```text
k1-manifold-core/
  src/k1_manifold_core/
    axioms/
    geometry/
    dynamics/
    thermodynamics/
    spacetime/
  tests/
  examples/
  docs/
  lean4/
```

## Quick Start

```bash
cd k1-manifold-core
python -m pip install -e ".[dev]"
pytest -v
```

If your system Python blocks editable installs because of site-package permissions, use a virtual environment:

```bash
cd k1-manifold-core
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
pytest -v
```

## Run The Demos

```bash
cd k1-manifold-core
python examples/demo_01_information_time.py
python examples/demo_02_causal_cone.py
python examples/demo_03_k1_attractor.py
```

The demos generate figures in:

```text
k1-manifold-core/examples/outputs/
```

## Current Status

Chronos-K1 is an experimental research framework.

Implemented components are computational and numerical. Physical interpretations beyond the implemented mathematics are treated as hypotheses under investigation.

## Theory Boundary

The repository distinguishes theorem-level code checks, assumptions, and numerical experiments:

- Theorem-level checks: 2D Lorentzian signature tests, `K(x)` evaluation, Law II matrix form, and spherical-sector symbolic identities.
- Assumptions: Axioms R/E/T, nondegeneracy, the added `K=1` consistency condition, Law II, Law III, and thermodynamic identifications such as `T_eff = T_tol`.
- Numerical experiments: local `K=1` attractor trajectories and visualization demos.

No claim is made that the present code derives the full physical spacetime metric, the matter sector, or general relativity from first principles.

## Roadmap

- Higher-dimensional Lorentzian state spaces.
- Curved, state-dependent metrics.
- Stronger numerical tests for global `K=1` attractivity.
- JEPA integration on a separate experimental branch.
- World-model benchmarks after a baseline implementation exists.
- Long-horizon prediction experiments.

See [k1-manifold-core/docs/v0_2_roadmap.md](k1-manifold-core/docs/v0_2_roadmap.md) for the current v0.2 implementation map.
