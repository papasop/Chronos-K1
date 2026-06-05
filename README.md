# Chronos

Validated Physics Structure Learning (VPSL)

A framework for discovering, validating, and transferring physics structure
priors.

## North Star

Chronos is the validation stack for a future physics-native baby-talk robot AI.

The long-term goal is not to add physics as a loss after learning, but to let a
developing agent choose a physical representation language first: Lorentz /
causal, symplectic / Hamiltonian, topological / defect, gauge, or Hilbert /
quantum-state.

Current status:

- Today: reproducible VPSL research prototype.
- S0: pre-robot developmental structure-recognition layer.
- K1/K2/K3: controlled validation of candidate physical representation
  languages.
- Not yet: a deployed robot, an RL agent, or a complete baby-talk system.

[![tests](https://github.com/papasop/Chronos-K1/actions/workflows/tests.yml/badge.svg)](https://github.com/papasop/Chronos-K1/actions/workflows/tests.yml)
[![k2-syntax](https://github.com/papasop/Chronos-K1/actions/workflows/k2_syntax.yml/badge.svg)](https://github.com/papasop/Chronos-K1/actions/workflows/k2_syntax.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/papasop/Chronos-K1/blob/main/colab/chronos_k1_quickstart.ipynb)

The Colab badge opens the K1 quickstart. K2 archive scripts live under
`chronos/k2/experiments/`.

`colab/chronos_core.py` is the portable pure-stdlib core for S0
recommendations, memory logging, and ClaimRecord scientific denominators. It
contains no torch, no GP/CNN training, and no self-evolution.
`colab/chronos_claims.py` is a portable ClaimRecord mirror. Canonical
implementation lives in `chronos/claims/`.

Current K2 scripts preserve archived verdict logic and summaries. Full Colab
training sources are still being restored.

Chronos-K1 is the first stage of the Chronos VPSL program.

Chronos is the working repository for VPSL: Validated Physics Structure
Learning. It studies whether explicit physical structure can be validated as a
transferable inductive bias under controlled regimes, fair controls, mechanism
checks, and claim boundaries.

This repository is no longer positioned as only a Lorentz / metric-prior
project. The K1 line remains the bounded framework validation stage. The K2
line is the first fully validated physical structure: a symplectic prior on the
FPU-β regime.

![VPSL pipeline](chronos/vpsl/vpsl_pipeline.svg)

## 1. What Is Chronos?

Chronos is a Validated Physics Structure Learning (VPSL) framework.

It determines:

1. When a physical structure should become a learning constraint.
2. Whether its effect is predictive.
3. Whether its mechanism is genuine.
4. Whether its advantage transfers.

Chronos is organized as milestones, not as a single architecture:

- S0: structure recognition / developmental learning layer that recommends
  which physical language should enter VPSL validation.
- K1: framework validation and bounded positive result for spectral /
  Lorentz-structured priors.
- K2: first VPSL-certified structure, validating a symplectic prior on FPU-β.
- K3: topological / winding-density structure search, with negative and
  unresolved regimes recorded.
- K4: gauge / cross-family structure discovery.
- K5: Hilbert / quantum-state representation.

## VPSL Structure Map

Chronos tracks physical structures as validation targets:

| Structure Family | Program / System | Scope | Current Status |
| --- | --- | --- | --- |
| Structure recognition layer | S0 | Selects candidate physical language from diagnostics | `S0_V0_4_PASSED`; emits recommendations, never certifies |
| Memory logging layer | S0-M0 | Append-only JSONL audit trail for runs, verdicts, recommendations, and claim boundaries | Logging only; does not learn or influence S0 |
| Pseudo-Riemannian / Lorentz structure | K1 / Klein-Gordon | Geometry, causality, metric signature, light-cone behavior | Partially confirmed; short-horizon evidence, mechanism transfer bounded |
| Spectral / dispersion structure | K1 / Klein-Gordon | Frequencies, dispersion relation, mode dynamics | `BOUNDED_POSITIVE`; coupled with Lorentz-sensitive validation |
| Symplectic / Hamiltonian structure | K2 / FPU-β | Phase space, Hamiltonian flow, long-horizon dynamics | `FULL_TRANSFER_CONFIRMED` through H=240; mechanism transfer confirmed |
| Topological / defect structure | K3.1 / K3.2D / K3-E2b / K3-E2d | Winding density, vortex transport, active topology regime search | K3.1 `NO_EFFECT`; K3.2D regime unresolved; K3-E2b toy active search `PASSED`; K3-E2d cheap GP active search `PASSED` |
| Gauge structure | K4 candidate | Local symmetry and gauge invariants | Pending |
| Hilbert / quantum-state structure | K5 candidate | Unitarity, quantum-state geometry, Born-compatible diagnostics | Pending |

The repository remains a reproducible research prototype. It does not claim a
solved Physics AI system, a universal architecture, or a proof that every
physical prior improves every task.

Companion historical manuscript:
- `Chronos-K1.txt` (`K=1 Chronogeometrodynamics`)

## 2. Evidence Ladder

VPSL treats a positive result as valid only when the regime, controls,
performance, and mechanism all survive the relevant gates.

| Stage | Result | Status |
| --- | --- | --- |
| S0 | Structure Recognition Layer | `S0_V0_4_PASSED`; recommends K-family, emits `s0_recommendation.csv`, never certifies |
| K1 | VPSL Framework Validation | Done |
| K1 | Spectral Prior | BOUNDED_POSITIVE |
| K2 | Symplectic Prior | FULL_TRANSFER_CONFIRMED through H=240 |
| K3 | Topological / Winding-Density Search | No certified structure yet; K3.1 `NO_EFFECT`; K3-E2b active toy search passed; K3-E2d cheap GP active search passed |

K3-E2b: guided active topology regime search on a transparent toy landscape;
active reaches a `transport_ok` regime while random control does not. Status:
`PASSED`.

K3-E2d: guided active topology regime search with a cheap real GP evaluator and
continuous vortex-position transport metric. Active passes the registered
admission criteria against a random control. Status: `PASSED`.

K1 established a bounded framework result: metric-sensitive behavior appears
under controlled normalization tests, but it is not promoted to a universal
physics-prior claim.

K2 established the first full VPSL transfer result:

- `symplectic < baseline`
- `symplectic < fair energy`
- `symplectic < fair L2`
- mechanism transfer confirmed by full symplectic Jacobian error reduction
- confirmation holds on the graceful-baseline subset, not only pooled rescue
- K2.2-B extended the transfer result to H=240 on the graceful-baseline subset
- K2.3 hardens the result with wrong-Ω controls: the canonical symplectic form
  outperforms shuffled and random antisymmetric 2-form penalties on rollout.
  The wrong-Ω controls under-drive the dynamics, so their raw Jacobian errors
  are treated as degenerate rather than valid mechanism evidence.

## 3. Chronos-S0: Structure Recognition Layer

S0 is the developmental layer of Chronos.

It does not predict physics directly. It observes diagnostic failures and
successes, then recommends which physical representation family should enter
VPSL validation.

S0 asks:

```text
Given a system S, which physical language should the learner try first?
```

- K1: Lorentz / causal language
- K2: Symplectic / Hamiltonian language
- K3: Topological / defect language
- K4: Gauge / local symmetry language
- K5: Hilbert / quantum-state language

S0 never certifies a structure. Certification still requires the VPSL gates:
regime, constraint, mechanism, and transfer validation.

The current S0 guardrail is the K3.2D lesson: low field prediction error is not
enough. A learner may imitate `[Re psi, Im psi]` while failing to transport the
vortex-antivortex pair as a topological object.

S0 v0.4 closes the first structure-recognition loop:
experiment summary -> diagnostics -> recommendation -> `s0_recommendation.csv`.

Current S0 verdict:

```text
S0_V0_4_PASSED
```

### S0-E0: Embodied Toy Diagnostics

S0-E0 is the pre-robot toy layer. It tests whether S0 can choose a physical
language from simple hand-written diagnostic packets:

```text
toy physical situation -> diagnostics -> S0 recommendation
```

Current toy cases:

- pendulum-like -> K2 / symplectic language
- causal-contact-like -> K1 / Lorentz-causal language
- vortex-transport-fail -> K3 / topological unresolved
- unknown -> `UNRESOLVED`

This is not robotics yet, not a simulation, and not a learned classifier.

### S0-E1: Toy Simulation Diagnostics

S0-E1 is the next computer-side toy layer:

```text
toy simulation -> diagnostic extractor -> S0 recommendation
```

It replaces hand-written toy diagnostics with measured diagnostics from small
deterministic toy trajectories:

- pendulum simulation -> energy-drift symplectic proxy -> K2
- contact simulation -> toy contact/event-ordering proxy -> K1
- object-persistence failure -> object tracking diagnostic -> K3 / `do_not_promote`

Current S0-E1 verdict:

```text
S0_E1_TOY_SIM_PASSED
```

This is still not robotics, not model training, not learned classification, and
not physical certification.

### S0-E2: Active Toy Exploration

S0-E2 adds deterministic action choice by novelty:

```text
agent chooses action by novelty -> toy world responds -> diagnostic probe -> S0 recommendation
```

The active explorer covers more toy state-space cells than a random-action
control, then launches the K2 diagnostic probe from a state active exploration
actually reached.

Current S0-E2 verdict:

```text
S0_E2_ACTIVE_TOY_PASSED
```

This is still not robotics, not RL training, not a neural network, not online
learning, and not physical certification.

### S0-E2b: Active Diagnostic Value

S0-E2b uses a partitioned toy world where the K2 structure signal only appears
in a far zone of state space. Active exploration reaches that zone; a random
control usually does not.

Observed split:

```text
active -> K2_SYMPLECTIC / continue
random -> UNRESOLVED / do_not_promote
```

Current S0-E2b verdict:

```text
S0_E2B_ACTIVE_VALUE_PASSED
```

This closes the E2 gap: active exploration is not just wired into the diagnostic
loop, it is necessary for the correct diagnosis in this toy.

### S0-M0: Memory Logging Layer

S0-M0 is an append-only audit trail, not a learning layer:

```text
run/verdict/recommendation -> MemoryEvent JSONL record -> read-only summary
```

Every memory record must carry a non-empty `claim_boundary`, so successes are
logged with what they do and do not establish. `allowed_action` mirrors S0's
never-certify boundary: `continue`, `archive`, or `do_not_promote`.

Current S0-M0 verdict:

```text
S0_M0_MEMORY_LOGGING_PASSED
```

S0-M0 does not change recommendations, rank future experiments, feed back into
S0, or self-evolve.

## 4. VPSL Gates

Historically, Chronos was positioned between learned world models and
explicit physics priors. The current repository foregrounds VPSL: a validation
framework for deciding when a physical structure is strong enough to become a
learning constraint.

The current VPSL discipline is:

- Regime validation is a necessary gate before comparing priors.
- Controls must be non-degenerate and fair at the tested horizon.
- Pooled improvements at stress horizons are context only.
- Transfer claims must hold on the graceful-baseline subset.
- Mechanism transfer is required for structure claims.

See:
- `chronos/vpsl/framework.md`
- `chronos/vpsl/gates.md`
- `chronos/vpsl/claim_taxonomy.md`
- `chronos/vpsl/verdicts.md`
- `chronos/vpsl/certified_structures.md`
- `chronos/vpsl/numbering.md`
- `chronos/archive/negative_results.md`

## 5. K1 Archive: Bounded Framework Result

K1 is now treated as the historical framework-validation archive. It studied
whether causal / metric structure can be injected as an explicit inductive bias
in world-model dynamics.

Historical K1 ingredients:

- Lorentz-sign quadratic structure for time / causality.
- Symplectic-dissipative update family.
- Exp6 / Exp7 tests for physics sensitivity and metric specificity.

The strongest K1 evidence remains bounded:

| Experiment | Question | Result |
| --- | --- | --- |
| Exp6 | Does Chronos react differently to timelike vs spacelike data? | Positive sensitivity |
| Exp7 | Is the effect specific to Lorentz normalization? | Lorentz-only interaction, Wilcoxon `p=0.040` |

Detailed K1 materials:
- `k1-manifold-core/`
- `chronos/k1/archive.md`
- `k1-manifold-core/docs/experiment_6_physics_sensitivity.md`
- `k1-manifold-core/docs/experiment_7_metric_controlled_normalization.md`

## 6. K2 Archive: Symplectic Prior

K2 Milestone: the symplectic prior became the first VPSL-certified physical
structure, achieving `FULL_TRANSFER_CONFIRMED` through H=240 on the FPU-β
benchmark.

See:
- `chronos/k2/archive.md`

## 7. Repository Layout

```text
Chronos/
├── README.md
├── REPRODUCE.md
├── MILESTONES.md
├── Chronos-K1.txt
├── archive/
├── chronos/
│   ├── ROADMAP.md
│   ├── archive/
│   │   └── negative_results.md
│   ├── embodied_toy/
│   │   ├── README.md
│   │   ├── run_toy_suite.py
│   │   ├── run_sim_suite.py
│   │   ├── run_active_suite.py
│   │   ├── run_active_value_suite.py
│   │   ├── simulations.py
│   │   ├── extractors.py
│   │   ├── active.py
│   │   ├── active_value.py
│   │   ├── toy_worlds.py
│   │   └── tests/
│   ├── s0/
│   │   ├── README.md
│   │   ├── S0_V0_3_PASSED.md
│   │   ├── S0_V0_4_PASSED.md
│   │   ├── adapters.py
│   │   ├── diagnostics_schema.py
│   │   ├── emitter.py
│   │   ├── run_selector.py
│   │   ├── structure_selector.py
│   │   └── tests/
│   ├── memory/
│   │   ├── README.md
│   │   ├── S0_M0_MEMORY_LOGGING_PASSED.md
│   │   ├── logging.py
│   │   ├── run_memory_demo.py
│   │   └── tests/
│   ├── k1/
│   │   └── archive.md
│   ├── k2/
│   │   ├── README.md
│   │   ├── archive.md
│   │   ├── experiments/
│   │   ├── historical_logs/
│   │   ├── reconstruction_notes.md
│   │   └── results/
│   ├── k3/
│   │   ├── K3_NEGATIVE_RESULTS_phi4_regime.md
│   │   ├── K3_E2B_ACTIVE_TOPOLOGY_SEARCH_PASSED.md
│   │   ├── K3_E2D_GP_ACTIVE_SEARCH_PASSED.md
│   │   ├── README.md
│   │   ├── active_gp_search.py
│   │   ├── active_topology_search.py
│   │   ├── run_active_gp_search.py
│   │   ├── run_active_topology_search.py
│   │   ├── verdicts.py
│   │   ├── archives/
│   │   ├── experiments/
│   │   ├── negative_results/
│   │   └── results/
│   └── vpsl/
│       ├── framework.md
│       ├── gates.md
│       ├── claim_taxonomy.md
│       ├── verdicts.md
│       ├── certified_structures.md
│       ├── numbering.md
│       ├── structure_claim_template.md
│       └── vpsl_pipeline.svg
├── k1-manifold-core/
└── exp5-diagnostic/
```

The repository name on GitHub may still be `Chronos-K1`; the scientific
positioning is now Chronos / VPSL, with K1 as the historical first milestone.

## 8. Claim Boundary

Supported:

- VPSL framework.
- Regime validation methodology.
- K1 bounded positive result.
- First fully validated physical structure: symplectic prior on FPU-β.

Not claimed:

- All physical priors help.
- All systems benefit from symplectic priors.
- General Physics-AI.
- Chronos is a single architecture.
- K2 generalizes beyond FPU-β yet.
- Pooled rescue at a stress horizon is enough for a structure claim.

## 9. Reproduce Results

Use the reproduction guide:
- `REPRODUCE.md`

K1 benchmark entrypoints live under:
- `k1-manifold-core/benchmarks/`

K2 entrypoints live under:
- `chronos/k2/experiments/`

## 10. Roadmap

See the program roadmap:
- `chronos/ROADMAP.md`

Current program:

- K1: Framework Validation - done.
- K2: First Certified Structure - done.
- K3: Topological attempts - Stage-2 winding-density prior test negative; K3-E2b active toy regime search passed; 2D vortex regime unresolved.
- S0: Structure Recognition Layer - `S0_V0_4_PASSED`.
- K4: Gauge / cross-family structure discovery - future.
- K5: Hilbert / quantum-state representation - future.
