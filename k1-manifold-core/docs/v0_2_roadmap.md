# v0.2 Roadmap

This roadmap keeps the repository scoped as reproducible companion code for the K=1 paper. It is not a broader theory statement.

## Completed Modules

- `axioms`: numerical helpers for the two-dimensional realizability layer and signature validation.
- `geometry`: 2D quadratic forms, `K(x) = x.T @ G @ x`, the canonical `J`, and `J_G = alpha G^-1 J`.
- `dynamics`: Law II matrix/vector field helpers, Law III critical damping `D = d_c I`, and a small RK4 integrator.
- `spacetime`: explicit spherical-sector formulas for `K1`, `K2`, and Ricci components, plus Rindler helper formulas.
- `tests`: executable checks for signature, K-form evaluation, Law II, critical-damping decrease in a local trajectory, and spherical-sector symbolic identities.
- `examples`: directly runnable scripts that print key values for signature, K=1 dynamics, and light-cone classification.

## Unfinished Modules

- Global attractivity of `K=1` is not implemented or claimed.
- Higher-dimensional signature reconstruction is not implemented.
- Full geodesic and curvature infrastructure is intentionally minimal.
- Matter-sector derivation is not implemented.
- Thermodynamic identification from first principles is not implemented.
- Lean formalization is only represented by placeholders.

## Theorem-Level Claims Encoded By Tests

- In two dimensions, for symmetric nondegenerate forms, `det(G) < 0` is equivalent to Lorentzian inertia `(1, 1)`.
- The induced generator used by the code is `J_G = alpha G^-1 J`.
- Law II is implemented as `xdot = (J_G - D) grad V`.
- In the spherical sector, the code verifies the symbolic algebra behind the vacuum and Lambda reformulation identities.

## Assumptions

- Axioms R, E, T and nondegeneracy are inputs to the realizability layer; the code tests consequences, not a formal proof system.
- `K=1` is treated as an added self-consistency condition.
- Law II is treated as a structural axiom.
- Law III uses the isotropic critical-damping choice `D = d_c I`.
- Thermodynamic formulas require external identifications such as `T_eff = T_tol`; the code does not derive them.
- Einstein-sector helpers are numerical or symbolic reformulations of stated spherical-sector formulas, not an autonomous derivation of general relativity.

## Numerical Experiments

- `examples/01_lorentz_signature_demo.py` prints determinant, inertia, Lorentzian status, spectrum of `J_G`, and `d_c`.
- `examples/02_k1_dynamics_demo.py` integrates one local Law II/Law III trajectory and prints `K` and `V` before and after.
- `examples/03_light_cone_demo.py` evaluates representative timelike, null, and spacelike vectors under `K(x)`.
- `tests/test_v0_2_core.py` includes a local numerical check that `V(t)` decreases along a normal fluctuation under critical damping.
