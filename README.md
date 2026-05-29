# Chronos-K1

[![tests](https://github.com/papasop/Chronos-K1/actions/workflows/tests.yml/badge.svg)](https://github.com/papasop/Chronos-K1/actions/workflows/tests.yml)

Chronos-K1 is a Lorentzian structural dynamics framework in which time is
modeled as the cost of structural change and causal structure is encoded
directly into the geometry of a state space.

The current repository is a reproducible research prototype. It contains
deterministic theory checks, dynamical validation benchmarks, and early AI
benchmarks. It does **not** claim to derive physical reality, solve world
modeling, or derive general relativity from first principles.

> Companion paper: *K=1 Chronogeometrodynamics - Lorentzian Geometry from
> Information Time, with a Self-Contained Realizability Foundation*
> (see `Chronos-K1.txt` / arXiv / Zenodo). This repository implements and
> numerically checks constructions from that paper; it does not extend the
> theoretical claims beyond it.

## Project Structure

Chronos-K1 currently contains three active layers plus one archive layer:

1. **Theory**
   - Lorentzian Signature Theorem
   - Information time, `dt_info = dPhi / H`
   - `K=1` null flow
   - recovery-time scaling, `T ~ c^-2`

2. **Dynamical Benchmarks**
   - `benchmark_v03`: noisy `K=1` recovery under Euclidean and Chronos-K1
     dynamics

3. **AI Benchmarks**
   - OOD light-cone classification
   - world-model causality stress test, Experiment 5

4. **Archive**
   - `world_model_v01`: minimal affine latent-transition prototype retained for
     historical comparison

## Core Idea

Most AI world models learn temporal structure and causality from data.
Chronos-K1 explores a different structural starting point:

- Causality is represented by a Lorentzian quadratic form `G`.
- Local structural consistency is the `K=1` manifold,
  `K(x) = x.T @ G @ x`.
- Dynamics are a symplectic-dissipative flow,
  `xdot = (J_G - D) grad V`.
- The critical-damping choice `D = d_c I` is studied as the selection mechanism
  for `K=1`.

The central mathematical claim is point-level: under realizability axioms
`R, E, T` plus nondegeneracy, the leading cost form `G` is forced to be
Lorentzian, `Sig(G) = (1,1)`. Dynamics, thermodynamics, field-equation
reformulation, and AI experiments are separate layers built around that
structure.

## Part I - Core Theory

### Lorentzian Signature

Two-dimensional signature checks verify:

- `Sig(G) = (1, 1)`
- `det(G) < 0`
- real spectral threshold `d_c = alpha * sqrt(-1/det(G))`

These tests cover canonical and non-diagonal Lorentzian forms.

### Information Time

The information-time helper verifies:

```text
dt_info = dPhi / H
```

for scalar and vector inputs with positive `H`.

### Causal Cone Classification

The causal-cone checks classify vectors by the sign of:

```text
K(x) = x.T @ G @ x
```

using the cost-sign convention: positive is timelike, zero is lightlike, and
negative is spacelike.

### K=1 Null Flow And `T ~ c^-2`

For the canonical form `G = diag(1, -1)` at critical damping, the generator
`A_c = J_G - d_c I` is rank-one:

```text
A_c = [[-1, 1], [1, -1]]
```

The flow has a conserved leaf coordinate `c = x1 + x2`. On each non-degenerate
leaf, the `K`-dynamics satisfies:

```text
d/dt (K - 1) = -4 c^2 (K - 1)
```

so recovery time scales as:

```text
T_recover(c) = Theta(c^-2)
```

`examples/demo_04_recovery_scaling.py` reproduces the rank check,
first-integral conservation, ideal `4c^2` leaf scaling, and log-log recovery
slope.

![Null-flow recovery scaling](k1-manifold-core/examples/outputs/demo_04_recovery_scaling.png)

### Spherical-Sector Reformulation

The `spacetime` tests use symbolic differentiation to verify spherical-sector
identities for Schwarzschild, Reissner-Nordstrom, and Schwarzschild-de Sitter
examples.

These are symbolic reformulation checks of an algebraic equivalence, **not** a
general derivation of Einstein gravity. The field-level conditions `K_i = 1`
are an independent ansatz.

## Part II - Dynamical Validation

### `benchmark_v03`: Noisy K=1 Recovery

This benchmark compares:

- Euclidean gradient dynamics, `xdot = -grad V`
- Chronos-K1 Lorentzian dynamics, `xdot = (J_G - D) grad V`

under shared Gaussian perturbations.

It reports:

- tail `|K(t)-1|`,
- tail potential `V(t)`,
- recovery time,
- long-horizon rollout error.

Run:

```bash
cd k1-manifold-core
python examples/benchmark_v03.py
```

Result:

```text
k1-manifold-core/results/benchmark_v03.json
```

## Part III - AI Benchmarks

AI benchmarks are research benchmarks, not pytest unit tests. They may involve
training, randomness, optional ML dependencies, and statistical comparisons.

Install optional benchmark dependencies:

```bash
cd k1-manifold-core
python -m pip install -r requirements-benchmarks.txt
```

### AI Benchmark 1 - OOD Light-Cone Classification

This benchmark trains classifiers on synthetic event differences from `box=2`
and evaluates OOD extrapolation on larger boxes.

Models:

- explicit Lorentzian score,
- Euclidean Mahalanobis baseline,
- Euclidean MLP baseline.

Run:

```bash
cd k1-manifold-core
python benchmarks/ood_extrapolation.py
```

Current AUC summary:

| Test box | Lorentz | Euclid Mahalanobis | Euclid MLP | Lorentz - MLP gap |
| --- | ---: | ---: | ---: | ---: |
| 2 | 1.0000 | 0.7256 | 0.9997 | +0.0003 |
| 4 | 1.0000 | 0.7278 | 0.9997 | +0.0003 |
| 8 | 1.0000 | 0.7213 | 0.9996 | +0.0004 |
| 12 | 1.0000 | 0.7217 | 0.9995 | +0.0005 |

Artifacts:

```text
k1-manifold-core/results/ood_extrapolation.json
k1-manifold-core/results/ood_extrapolation_auc.png
```

### AI Benchmark 2 - World-Model Causality Stress Test

Experiment 5 studies long-horizon rollout prediction under distribution shift
on synthetic Lorentzian oscillator trajectories.

It compares:

- Euclidean JEPA,
- Chronos-JEPA with Lorentzian latent geometry,
- multiple Chronos causal-regularization strengths.

It evaluates:

- final rollout MSE,
- causal-violation rate,
- Lorentz-interval drift,
- latent `K` drift,
- OOD extrapolation from `box=2` to `box=32`.

Run:

```bash
cd k1-manifold-core
python benchmarks/experiment_5_causal_stress_test.py
```

Current quick result: this is a negative ablation. Chronos-JEPA keeps rollout
MSE roughly comparable but does **not** reduce decoded causal-violation rates in
the current configuration. Stronger causal regularization increases decoded
causal violations on this stress test. It is included as a reproducible failure
mode and design target for future Chronos world-model regularizers.

Artifacts:

```text
k1-manifold-core/results/experiment_5_ablation_stress_summary.csv
k1-manifold-core/results/experiment_5_ablation_stress_raw.json
k1-manifold-core/results/experiment_5_violation_vs_box.png
k1-manifold-core/results/experiment_5_mse_vs_box.png
k1-manifold-core/results/experiment_5_violation_by_step.png
k1-manifold-core/results/experiment_5_K_drift_by_step.png
```

## Archived Benchmarks

### `world_model_v01`

`world_model_v01` is a minimal affine latent-transition benchmark retained for
historical comparison. It tests a simple `K=1` projection regularizer on a toy
hyperbolic latent dataset. It is not treated as current headline evidence.

Run:

```bash
cd k1-manifold-core
python examples/benchmark_world_model_v01.py
```

Result:

```text
k1-manifold-core/results/world_model_v01.json
```

## Repository Organization

```text
Chronos-K1
├── Theory
│   ├── Lorentz Signature
│   ├── K=1 Null Flow
│   └── T ~ c^-2
│
├── Benchmarks
│   ├── benchmark_v03
│   ├── ood_extrapolation
│   └── experiment_5_causal_stress
│
├── Archive
│   └── world_model_v01
│
└── Docs
```

On disk:

```text
k1-manifold-core/
  src/k1_manifold_core/
    axioms/
    geometry/
    dynamics/
    thermodynamics/
    spacetime/
    world_model/
  tests/
  benchmarks/
  examples/
  docs/
  results/
  lean4/
```

## Quick Start

```bash
cd k1-manifold-core
python -m pip install -e ".[dev]"
pytest -v
```

If your system Python blocks editable installs because of site-package
permissions, use a virtual environment:

```bash
cd k1-manifold-core
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
pytest -v
```

Expected test status:

```text
29 passed
```

## Reproduce

For a one-page reproduction guide, see [REPRODUCE.md](REPRODUCE.md).

For the benchmark narrative and current result tables, see
`k1-manifold-core/docs/benchmark_report.md`.

## Theory Boundary

The repository distinguishes theorem-level checks, assumptions, and numerical
experiments:

- **Theorem-level checks:** 2D Lorentzian signature tests; `K(x)` evaluation;
  Law II matrix form; rank-one structure of `A_c`; exact-leaf decay rate
  `4c^2`; spherical-sector symbolic identities.
- **Assumptions:** Axioms `R/E/T`, nondegeneracy, the added `K=1` consistency
  condition, Law II, Law III, field-level `K_i = 1`, and thermodynamic
  identifications such as `T_eff = T_tol`.
- **Numerical experiments:** local `K=1` trajectories, noisy recovery,
  light-cone classification, and world-model stress tests.

No claim is made that the present code derives the full physical spacetime
metric, the matter sector, or general relativity from first principles. The
thermodynamic bridge in the paper is explicitly conditional on external inputs
(`T_eff = T_tol`, `S ~ A`) and is not presented here as an autonomous
derivation.

## Next AI Milestones

- Lorenz attractor benchmark.
- Pendulum benchmark.
- Double-pendulum benchmark.
- N-body benchmark.
- Chronos-JEPA regularizer redesign for Experiment 5.
- Chronos-JEPA scaling study after the stress-test failure mode is addressed.

## Citation

If you use this code, please cite the companion paper (Y. Y. N. Li,
*K=1 Chronogeometrodynamics*) and this repository. A versioned snapshot with a
DOI will be archived on Zenodo; cite that DOI for reproducibility rather than
the moving `main` branch.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
