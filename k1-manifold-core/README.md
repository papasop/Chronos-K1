# k1-manifold-core

Code companion for **K=1 Chronogeometrodynamics**.

The package is organized around five layers:

- `axioms`: realizability checks for Axioms R, E, T and cost-form construction.
- `geometry`: Lorentzian quadratic forms, induced symplectic generators, geodesic utilities, and manifold abstractions.
- `dynamics`: Law II / Law III symplectic-dissipative evolution and ODE solvers.
- `thermodynamics`: local Clausius relation helpers, OU reductions, and effective temperature formulas.
- `spacetime`: Rindler, spherical, and Einstein-sector formulas.

The implementation is deliberately small and inspectable. It is not a symbolic GR system; it provides direct numerical functions for the 2D and spherical-sector formulas stated in the paper.

## Install

```bash
python -m pip install -e ".[dev]"
pytest
```

## Minimal Example

```python
import numpy as np
from k1_manifold_core.axioms.validation import is_lorentzian
from k1_manifold_core.geometry.lorentzian import QuadraticForm2D

G = np.diag([1.0, -1.0])
form = QuadraticForm2D(G)

assert is_lorentzian(G)
assert form.K(np.array([0.0, 1.0])) == -1.0
```
