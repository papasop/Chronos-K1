# K=1 Chronogeometrodynamics

This repository contains two linked tracks:

- `k1-manifold-core`: the publication-oriented mathematical and numerical core for the K=1 framework.
- `k1-jepa-experiment`: an exploratory application track for K=1 constraints in JEPA-style representation learning.

The core package implements the point-level bridge from realizability axioms to Lorentzian signature, the induced symplectic generator, symplectic-dissipative dynamics, local OU thermodynamics, and the spherical-sector algebraic reformulation of vacuum Einstein equations.

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
  docs/
  lean4/

k1-jepa-experiment/
  notebooks/
  src/
  experiments/
```

## Quick Start

```bash
cd k1-manifold-core
python -m pip install -e ".[dev]"
pytest
```

## Scope

This code is a computational companion to the K=1 paper draft. It is intentionally conservative: the core package encodes formulas that are explicit in the manuscript and marks external thermodynamic identifications as assumptions rather than deriving them.
