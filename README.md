# Chronos-K1

[![tests](https://github.com/papasop/Chronos-K1/actions/workflows/tests.yml/badge.svg)](https://github.com/papasop/Chronos-K1/actions/workflows/tests.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/papasop/Chronos-K1/blob/main/colab/chronos_k1_quickstart.ipynb)

Chronos-K1
A Lorentzian Information Geometry Framework
for Physics-Aware World Models

Chronos-K1 is a Lorentzian information-geometry framework with early evidence
for physics-sensitive inductive bias in latent world models.

Current evidence: Lorentz normalization exhibits a statistically significant
Metric x Dataset interaction (`N=30`, Wilcoxon `p=0.040`), while Euclidean and
random normalization do not.

## 1. What Is Chronos-K1?

Chronos-K1 studies whether causal structure can be injected as an explicit
inductive bias in world-model dynamics.

- Time/causality are modeled with a Lorentz-sign quadratic structure.
- Dynamics use a symplectic-dissipative update family.
- AI benchmarks test whether this bias helps on Lorentz-structured data.

This repository is a reproducible research prototype, not a claim of solved
Physics AI, general intelligence, or first-principles derivation of all
physics.

Companion paper:
- `Chronos-K1.txt` (`K=1 Chronogeometrodynamics` manuscript)

## 2. Evidence Ladder

### Ladder Levels

- Level 1: Theory
  - Law I / Law II / Law III
- Level 2: Numerical validation
  - 29+ tests
- Level 3: World-model stress tests
  - Exp5 / Exp5b (historical)
- Level 4: Physics sensitivity
  - Exp6: Does Chronos react differently to timelike vs spacelike data?
- Level 5: Metric specificity
  - Exp7: Is this difference specific to Lorentz normalization?

### Core Evidence Table

| Layer | Evidence | Status |
| --- | --- | --- |
| Theory | Lorentz signature, `K=1` null flow | ✅ |
| Numerical checks | pytest suite | ✅ |
| Exp5 | Historical world-model benchmark | Mixed |
| Exp6 | Physics sensitivity | Positive |
| Exp7 | Metric-specific interaction | Lorentz only (`p=0.040`) |
| Real physics datasets | pendulum / N-body / Lorenz | pending |

## 3. Theory Core

Law I: Realizability -> Lorentz Signature
- Axioms `R, E, T` constrain the leading cost form to Lorentzian type in 2D.

Law II: Symplectic-Dissipative Dynamics
- Core flow: `xdot = (J_G - D) grad V`.

Law III: Critical Damping -> Null Flow -> Invariant Foliation
- Critical damping yields the `K=1` null-flow dynamics and foliation structure.

Data-driven Bias ───────────────────► Explicit Physics Knowledge

LLM → WM → NODE → CHRONOS → Equiv → HNN → PINN
                  ▲
               OUR SPOT

Validated Physics Structure Priors

(Regime Validation
 + Constraint Validation
 + Structure Priors)

Positioning diagram asset:
- `k1-manifold-core/docs/chronos_positioning.svg`

## 4. Physics-AI Benchmarks

- Experiment 5: Historical world-model benchmark
- Experiment 5b: Fixed ablation diagnostic
- Experiment 6: Physics sensitivity benchmark
- Experiment 7: Metric-controlled normalization benchmark

Exp6 / Exp7 division of labor:
- Exp6 tests sensitivity: timelike vs spacelike structure response.
- Exp7 tests mechanism: whether that sensitivity is specific to Lorentz normalization.

### Experiment 7 - Metric-Controlled Normalization

Question:
Does Lorentz normalization exhibit a dataset-specific advantage on timelike
trajectories that Euclidean and random normalization do not?

Setup:
- Timelike geodesics (`eta(v,v) > 0`)
- Spacelike geodesics (`eta(v,v) < 0`)
- `N=30` independent seeds
- one-sided Wilcoxon test on Metric x Dataset interaction

Result:

| Metric | Timelike Improvement | Spacelike Improvement | Interaction p-value |
| --- | ---: | ---: | ---: |
| Lorentz | +7.1% | -0.7% | 0.040 |
| Euclidean | +1.5% | -1.1% | 0.278 |
| Random | +0.4% | -1.8% | 0.452 |

Interpretation:
Lorentz normalization was the only metric producing a statistically
significant timelike-spacelike interaction. Euclidean and random normalization
did not show significant interaction effects. This is evidence for a
metric-sensitive inductive bias rather than a generic normalization effect.

Detailed results:
- `k1-manifold-core/docs/benchmark_report.md`
- `k1-manifold-core/docs/experiment_6_physics_sensitivity.md`
- `k1-manifold-core/docs/experiment_7_metric_controlled_normalization.md`
- `k1-manifold-core/docs/physics_ai_evidence.md`

## 5. Reproduce Results

Use the reproduction guide for commands and run modes:
- `REPRODUCE.md`

Benchmark entrypoints live under:
- `k1-manifold-core/benchmarks/`

## 6. Repository Layout

```text
Chronos-K1/
├── README.md
├── REPRODUCE.md
├── Chronos-K1.txt
├── k1-manifold-core/
│   ├── src/k1_manifold_core/
│   ├── tests/
│   ├── benchmarks/
│   │   ├── experiment_5_causal_stress_test.py
│   │   ├── experiment_5b_causal_mechanism_ablation.py
│   │   ├── experiment_6_physics_sensitivity.py
│   │   ├── ood_extrapolation.py  (implementation backend)
│   │   └── experiment_7_metric_controlled_normalization.py
│   ├── docs/
│   │   ├── benchmark_report.md
│   │   ├── experiment_5_reproduction_protocol.md
│   │   └── experiment_7_metric_controlled_normalization.md
│   └── results/
│       ├── experiment_5* artifacts
│       ├── experiment_5b* artifacts
│       ├── experiment_6_physics_sensitivity/
│       ├── ood_extrapolation* artifacts (Exp6 backend outputs)
│       └── experiment_7_metric_controlled_normalization/
├── exp5-diagnostic/
└── archive/
```

## 7. Claim Boundary

What this repository supports:
- Evidence for a physics-sensitive inductive bias.
- Evidence for metric-sensitive behavior under controlled normalization tests.

What this repository does not claim:
- Chronos-K1 proves Physics AI.
- Lorentz structure is always best in all tasks.
- General intelligence or scientific discovery autonomy.

## 8. Roadmap

Near-term:
- Expand Exp6/Exp7 with stronger seed counts and stress settings.
- Add real-physics datasets: pendulum, N-body, Lorenz.
- Improve metric controls and confidence-interval reporting.

Mid-term:
- Scale to stronger latent world-model baselines.
- Validate robustness across architectures and dataset regimes.

Long-term:
- Test whether physics-sensitive inductive bias survives realistic data
  pipelines and model scale.
