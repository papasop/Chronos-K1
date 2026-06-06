# Chronos

**Validated representation stack for a future physics-native baby-talk AI.**

Chronos studies how a developing agent can learn to speak from verified
representations instead of free text generation.

The current VPSL milestone is not a chatbot and not a deployed robot. It is a
self-contained validation replay showing that physical-structure claims and
no-LLM grounded-language claims can share one scientific denominator:
`supports`, `does_not_support`, evidence level, next gate, and claim boundary.

[![tests](https://github.com/papasop/Chronos-K1/actions/workflows/tests.yml/badge.svg)](https://github.com/papasop/Chronos-K1/actions/workflows/tests.yml)
[![full-denominator](https://github.com/papasop/Chronos-K1/actions/workflows/full_denominator.yml/badge.svg)](https://github.com/papasop/Chronos-K1/actions/workflows/full_denominator.yml)
[![claims-tests](https://github.com/papasop/Chronos-K1/actions/workflows/claims_tests.yml/badge.svg)](https://github.com/papasop/Chronos-K1/actions/workflows/claims_tests.yml)
[![k2-syntax](https://github.com/papasop/Chronos-K1/actions/workflows/k2_syntax.yml/badge.svg)](https://github.com/papasop/Chronos-K1/actions/workflows/k2_syntax.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/papasop/Chronos-K1/blob/main/colab/chronos_k1_quickstart.ipynb)

## Current Canonical Replay

`chronos_full_denominator.py` is the current self-contained Chronos milestone.
It inlines the no-LLM grounded language stack and the Claim Denominator Layer,
then replays K2/K3 physical claims plus L-VPSL language claims under one
evidence-boundary system.

It includes happy-path tests and anti-cheat tests. It does not claim general
language understanding, open-domain conversation, LLM-level fluency, real robot
deployment, or universal physics AI.

`chronos_full_denominator.py` is a cleaned canonical replay distilled from
earlier exploratory Colab scripts. Historical raw drafts are not part of the
active API.

| Layer | Status |
| --- | --- |
| K2 symplectic claim | certified within tested FPU-beta scope |
| K3-E2c | archived negative result |
| K3-E2d | bounded positive evidence |
| L-VPSL no-LLM language | bounded positive evidence |
| Claim denominator | full replay passed |
| Anti-cheat tests | passed |
| Real robot grounding | not yet |
| Open-domain language | not supported |

### Reproduce the canonical replay

```bash
git clone https://github.com/papasop/Chronos-K1.git
cd Chronos-K1
python chronos_full_denominator.py
```

Expected summary:

```text
=== Self-tests (not just the happy ledger) ===
  language:10  claims:11  anti-cheat:6

=== FULL Chronos Denominator Replay (self-contained) ===
  by_claim_type      : {'certified_structure': 1, 'negative_result': 1, 'positive_evidence': 2}
  by_confidence_level: {'certified': 1, 'low': 1, 'medium': 2}
  count_total        : 4

  ok all full-denominator + self-test + anti-cheat assertions passed
```

## North Star: Baby-Talk AI Without Hallucinated Speech

The long-term goal is a physics-native baby-talk robot AI:

```text
world evidence -> physical representation -> semantic claim -> grounded utterance -> ClaimRecord
```

The agent should learn to say only what its verified representations support,
and say "I do not know" when the evidence does not support a claim.

Current status:

- Today: reproducible VPSL research prototype.
- Current milestone: `chronos_full_denominator.py`.
- Language stack: L1 negation, L2 causal boundary, L3 quantifier, L4 reference,
  L4A ambiguity hardening, L5 temporal ordering.
- Not yet: deployed robot, RL agent, open-domain conversation, or complete
  baby-talk system.

## What This Repository Is

Chronos is a VPSL and Claim Denominator validation stack for physical
representation and language representation claims. It records what each claim
supports, what it does not support, its evidence boundary, its evidence level,
its next gate, and its claim boundary.

## What This Repository Is Not

Chronos does not currently claim:

- a deployed robot
- a general chatbot
- open-domain language understanding
- LLM-level fluency
- autonomous AGI
- universal physics AI

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
- K2→K3: bridge validation target where symplectic Jacobian grammar is tested
  on nonlinear topological field systems such as Sine-Gordon.
- K4: gauge / cross-family structure discovery.
- K5: Hilbert / quantum-state representation.

## Architecture Grammar vs Physical Testbeds

Chronos separates **physical grammar** from **physical tasks**.

A physical grammar defines how a learner is allowed to represent or transform
states. A physical task provides the world where that grammar is tested.

| Layer | Object | Role | Chronos position |
| --- | --- | --- | --- |
| Grammar layer | `(Df)^T Omega (Df) = Omega` | Algebraic constraint on the learned update map | K2 symplectic / Hamiltonian grammar |
| Task layer | Sine-Gordon equation | Nonlinear field testbed with soliton and topological structure | K3 topology testbed and K2→K3 bridge |
| Validation layer | VPSL gates | Tests whether the grammar survives controls, diagnostics, mechanism checks, and transfer | ClaimRecord / evidence ladder |

In short:

```text
grammar = how the model is allowed to learn
task    = what physical world the grammar is tested on
VPSL    = whether the evidence is strong enough to promote the grammar
```

### K2: Symplectic Grammar

K2 asks whether a learned state update preserves Hamiltonian phase-space
structure.

For a learned update map `f`, the core mechanism condition is:

Mathematically:

```math
(Df)^\top \Omega (Df)=\Omega .
```

Machine-readable diagnostic:

```text
(Df)^T Omega (Df) = Omega
```

Here:

- `Df` is the Jacobian of the learned update map.
- `Omega` is the canonical symplectic form.
- The condition says that the learned deformation must preserve the symplectic
  structure.

This grammar is task-independent. It can apply to FPU chains, harmonic
oscillators, orbital systems, Sine-Gordon fields, and other Hamiltonian
systems.

```text
K2_SYMPLECTIC_JACOBIAN
role: architecture-level grammar
object: Jacobian of the learned update map
diagnostic: (Df)^T Omega (Df) = Omega
question: does the model preserve Hamiltonian phase-space structure?
controls: wrong-Omega, shuffled-Omega, random antisymmetric Omega
```

K2 therefore does not merely ask whether prediction loss improves. It asks
whether the model's internal dynamics lives inside the correct symplectic
geometry.

### K3: Topological Grammar

K3 asks whether the learner preserves topological objects, not just field
values.

A low field-prediction error is not enough if a defect, vortex, kink,
anti-kink, winding number, or soliton identity disappears during rollout.

```text
K3_TOPOLOGICAL_OBJECT
role: topology-level representation grammar
object: defect / winding / soliton identity
diagnostic: topological charge, winding density, defect transport
question: does the model preserve topological object identity under evolution?
controls: topology-blind predictors, field-error-only baselines
```

K3 therefore tests whether the representation carries object persistence in a
field, rather than only pixel-level or field-level accuracy.

### Sine-Gordon as the K2→K3 Bridge

The Sine-Gordon equation is the planned bridge between K2 and K3:

```text
phi_tt - phi_xx + sin(phi) = 0
```

It is useful because it combines:

- Hamiltonian field dynamics -> K2
- nonlinear wave evolution -> field prediction
- kink / anti-kink / soliton structure -> K3
- winding and topological charge -> K3 mechanism check
- possible failure of pure symplectic grammar -> motivation for an additional
  topology grammar

In Chronos terms:

```text
K2K3_SINE_GORDON_BRIDGE
role: bridge benchmark between symplectic dynamics and topological object persistence
task: Sine-Gordon nonlinear field evolution
K2 question: does the learned update preserve symplectic phase-space grammar?
K3 question: does the learned representation preserve kink / soliton / winding identity?
status: planned or experimental; not yet certified unless supported by a ClaimRecord
```

The purpose of Sine-Gordon is not only to solve one equation. Its purpose is to
expose whether symplectic grammar alone is sufficient, or whether a separate
topology grammar is required.

Current repository status: no Sine-Gordon benchmark implementation is included
yet. This bridge is a roadmap target and must not be read as an existing
certified result.

### Grammar / Task / Validation Map

| Grammar | Testbed | VPSL question | Possible output |
| --- | --- | --- | --- |
| K1 Lorentz / causal grammar | Klein-Gordon / causal contact tasks | Does the learner respect causal or metric structure? | `K1_LORENTZ / bounded` |
| K2 symplectic grammar: `(Df)^T Omega (Df) = Omega` | FPU-β chain | Does the learner preserve Hamiltonian phase-space dynamics? | `K2_SYMPLECTIC / certified or continue` |
| K3 topological grammar: angle / winding / defect continuity | vortex / winding tasks; Sine-Gordon planned | Does the learner preserve defect, soliton, or winding identity? | `K3_TOPOLOGICAL / unresolved or continue only with ClaimRecord` |
| K2→K3 bridge | Sine-Gordon | Can one representation preserve both symplectic dynamics and topological object persistence? | `K2K3_SINE_GORDON_BRIDGE / planned or experimental` |

This distinction is central to Chronos:

```text
Physics is not only a loss term.
Physics is a representation grammar.
Tasks are where the grammar is tested.
VPSL decides whether the grammar deserves to be promoted.
```

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
| K2→K3 | Sine-Gordon Bridge | Planned / experimental; not certified unless backed by a ClaimRecord |
| K3 | Topological / Winding-Density Search | No certified structure yet; K3.1 `NO_EFFECT`; K3-E2b active toy search passed; K3-E2d cheap GP active search passed |

A roadmap item is not a promoted claim. Promotion requires a `ClaimRecord`
with diagnostics, controls, evidence level, failure modes, and explicit claim
boundaries.

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
