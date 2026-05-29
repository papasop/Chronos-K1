# v0.3 Roadmap

This roadmap records the current Chronos-K1 research prototype after the
v0.3 task benchmark and reproducibility pass. It keeps the scope narrow:
implemented checks, runnable demos, and explicitly marked assumptions.

## Completed Modules

- `axioms`: information-time helpers and numerical checks around the
  realizability-to-signature layer.
- `geometry`: 2D Lorentzian quadratic forms, causal-cone classification,
  `K(x) = x.T @ G @ x`, and `J_G = alpha G^-1 J`.
- `dynamics`: Law II vector fields, Law III critical damping, and small ODE
  solvers used by the local `K=1` trajectory tests.
- `spacetime`: spherical-sector formulas for `K1`, `K2`, Ricci components,
  Rindler helpers, and symbolic reformulation checks.
- `benchmarks`: the v0.3 noisy `K=1` recovery comparison between a Euclidean
  gradient baseline and Chronos-K1 Lorentzian dynamics, plus the OOD
  light-cone classification research benchmark.
- `experiments`: causal projection and critical-damping null-flow experiments,
  including the recovery-scaling demo.
- `tests`: CI-backed pytest coverage for information time, causal cones,
  `K=1` dynamics, null-flow structure, spherical-sector formulas, and the v0.3
  benchmark metrics.
- `examples`: directly runnable demos that print key values and write figures
  into `examples/outputs/`.

## Reproducibility Status

- GitHub Actions runs `python -m pip install -e ".[dev]"` and `pytest -v` on
  every push and pull request.
- The repository root contains `requirements.txt` with lower-bound versions
  for `numpy`, `scipy`, `matplotlib`, `sympy`, and `pytest`.
- `REPRODUCE.md` maps each core numerical result to the command or test that
  reproduces it.
- The README includes a CI badge and a rendered demo figure.

## Theorem-Level Checks Encoded By Tests

- In two dimensions, for symmetric nondegenerate forms, `det(G) < 0` is
  equivalent to Lorentzian inertia `(1, 1)`.
- `K(x)` is evaluated as `x.T @ G @ x`.
- Law II is implemented as `xdot = (J_G - D) grad V`.
- In the canonical critical-damping null-flow setting, `A_c` is rank-one, the
  leaf coordinate is conserved to roundoff, and recovery time scales as
  `Theta(c^-2)`.
- In the spherical sector, symbolic checks verify the Schwarzschild,
  Reissner-Nordstrom asymmetry, and Schwarzschild-de Sitter reformulation
  identities.

## Assumptions

- Axioms R, E, T and nondegeneracy are inputs to the realizability layer; the
  code tests consequences and examples, not a machine-checked proof.
- `K=1` is an added self-consistency condition.
- Law II and Law III are structural modeling assumptions.
- The field-level conditions `K_i = 1` are treated as an ansatz in the
  spherical-sector reformulation.
- Thermodynamic identifications such as `T_eff = T_tol` and `S ~ A` remain
  external inputs and are not derived by the code.

## Numerical Experiments

- `examples/demo_01_information_time.py`: plots `dt_info = dPhi / H`.
- `examples/demo_02_causal_cone.py`: plots timelike, lightlike, and spacelike
  regions for a 2D Lorentzian form.
- `examples/demo_03_k1_attractor.py`: plots `K(t)` and `V(t)` recovery.
- `examples/demo_04_recovery_scaling.py`: reproduces the null-flow rank check,
  first-integral conservation, ideal `4c^2` leaf scaling, and log-log recovery
  slope.
- `examples/benchmark_world_model_v01.py`: compares a Euclidean affine latent
  transition with the same transition wrapped by a `K=1` projection regularizer
  on a toy hyperbolic sequence dataset.
- `examples/benchmark_v03.py`: writes `results/benchmark_v03.json` for the
  narrow noisy `K=1` recovery benchmark.
- `benchmarks/ood_extrapolation.py`: trains Lorentzian and Euclidean
  classifiers on light-cone labels from `box=2`, evaluates OOD boxes, and
  writes `results/ood_extrapolation.json`.
- `benchmarks/experiment_5_causal_stress_test.py`: scans Chronos latent
  predictor causal regularizer strength on Lorentzian oscillator rollouts.
  These are JEPA-style latent predictors, not Meta/LeCun JEPA implementations.
  The full sanity reproduction (`n_seeds=10`) keeps rollout MSE approximately
  unchanged while reducing decoded causal-violation rates under OOD stress. At
  `lambda=0.1`, `box=2`, violation drops from `0.2866` to `0.1475`
  (`p=0.0840`), so the result is presented as causal-consistency benchmark
  evidence rather than a final significance claim.
- `benchmarks/experiment_5b_causal_mechanism_ablation.py`: decomposes
  Experiment 5 into Euclidean, Chronos geometry-only, Chronos causal-only,
  Chronos interval-only, and Chronos full latent-predictor variants to probe
  which constraints contribute to causal-consistency preservation.
- `experiments/causal_projection_demo.py`: exploratory causal projection
  experiment. It is not presented as a superiority benchmark.

## Next Work

- Generalize the critical-damping null-flow analysis beyond the canonical
  `G = diag(1, -1)` case.
- Tighten benchmark baselines and separate deterministic recovery from noisy
  rollout behavior.
- Extend the world-model benchmarks beyond synthetic oscillator trajectories
  before making stronger AI-world-model claims.
- Add generated result snapshots only when they are tied to a reproducible
  script and test.
- Keep JEPA-style/world-model integration out of the core theorem claim path;
  present it as benchmark evidence and mechanism probing only.
