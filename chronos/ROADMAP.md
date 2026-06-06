# Chronos Program Roadmap

Chronos is a VPSL program, not a single architecture.

## K1 - Framework Validation

Status:

```text
DONE
```

Role:

- establish the validation discipline
- produce a bounded positive spectral / Lorentz-sensitive result
- define the failure mode where performance can outrun mechanism

## K2 - First Certified Structure

Status:

```text
DONE
```

Certified structure:

```text
Symplectic prior on FPU-β
```

Verdict:

```text
FULL_TRANSFER_CONFIRMED
```

Evidence:

- K2.2-B extends `FULL_TRANSFER_CONFIRMED` through H=240.
- K2.3 wrong-Ω specificity control confirms that the result is not explained by
  arbitrary antisymmetric Jacobian regularization; wrong-Ω controls under-drive
  the dynamics and are treated as degenerate for raw Jacobian-error comparison.

## S0 - Structure Recognition Layer

Status:

```text
S0_V0_4_PASSED
```

Role:

- add a developmental structure-acquisition layer above VPSL
- observe diagnostic successes and failures before selecting a K-family
- recommend the candidate physical language, confidence, next VPSL gate, and
  allowed action
- never certify a structure directly

Current guardrail:

- field prediction can be learnable while object transport fails
- K3.2D.0 separates `pipeline_ok` from `transport_ok`
- low `[Re psi, Im psi]` error is not enough to promote a topological regime
- summary adapters, the `run_selector` CLI, and `emit_recommendation` close
  the loop from experiment summaries to emitted S0 recommendations

## S0-E0 - Embodied Toy Diagnostics

Status:

```text
S0_E0_TOY_SUITE_PASSED
```

Role:

- pre-robot toy layer
- tests whether S0 can choose a physical language from toy diagnostic packets
- no robotics, no real simulation, no training

Current toy cases:

- `pendulum` -> K2
- `causal_contact` -> K1
- `vortex_fail` -> K3 / `do_not_promote`
- `unknown` -> `UNRESOLVED`

## S0-E1 - Toy Simulation Diagnostics

Status:

```text
S0_E1_TOY_SIM_PASSED
```

Role:

- computer-side step after S0-E0
- replaces hand-written toy diagnostics with diagnostics extracted from simple
  deterministic toy simulations
- still no robotics, no model training, no learned selector, and no physical
  certification

Target loop:

```text
toy simulation -> diagnostic extractor -> S0 recommendation
```

Current diagnostic extractors:

- pendulum simulation -> measured energy drift / symplectic proxy -> K2
- contact simulation -> toy contact/event-ordering proxy -> K1
- toy object persistence -> measured `object_tracking_valid` -> K3 / `do_not_promote`

## S0-E2 - Active Toy Exploration

Status:

```text
S0_E2_ACTIVE_TOY_PASSED
```

Role:

- computer-side active exploration step after S0-E1
- chooses toy pendulum actions by novelty, not by RL training
- validates active exploration by coverage gain over a random-action control
- launches the K2 diagnostic probe from an actively reached state
- still no robotics, no neural network, no online learning, and no physical
  certification

Target loop:

```text
agent chooses action by novelty -> toy world responds -> diagnostic probe -> S0 recommendation
```

## S0-E2b - Active Diagnostic Value

Status:

```text
S0_E2B_ACTIVE_VALUE_PASSED
```

Role:

- partitioned toy world where the K2 structure signal only exists in a far
  zone
- active exploration reaches the structure zone; random control usually does
  not
- diagnostic probe is launched from the reached state, so the recommendation
  depends on exploration
- still no robotics, no RL training, no neural network, and no physical
  certification

Observed split:

```text
active -> K2_SYMPLECTIC / continue
random -> UNRESOLVED / do_not_promote
```

## S0-M0 - Memory Logging Layer

Status:

```text
S0_M0_MEMORY_LOGGING_PASSED
```

Role:

- append-only JSONL audit trail for Chronos run events
- records module, experiment kind, verdict, S0 recommendation, score, payload,
  claim boundary, and code version
- requires a non-empty `claim_boundary` on every event
- mirrors S0's never-certify action boundary
- does not learn, rank, self-evolve, or feed back into S0

## Y30 / Y20 - Cognitive and Debate Boundary Layer

Status:

```text
Y30_CORE_MINIMAL_PACKAGE
Y20_CORE_V0_2_DEBATE_AND_PHYSICS_SELF_AUDIT_PASSED
```

Role:

- Y30 contextualizes appearances, dependent conditions, projections, traces,
  and unknown boundaries.
- Y20 audits claims through objection/response/required-gate grammar.
- Neither layer certifies metaphysical or physical truth.
- K-family physics verdicts remain unchanged unless their VPSL gates pass.

## K3 - Structure Discovery Program

Status:

```text
ACTIVE / BOUNDED_NEGATIVE
```

Candidate families:

- topological structure
- multi-scale structure

Current Stage-1 candidate:

- K3.0-D: periodic Sine-Gordon winding-density regime validation
- representation: `[sin(u), cos(u), u_t]`
- target: winding-density / local topological structure, not integer-charge certification

Current Stage-2 result:

- K3.1: winding-density continuity prior test
- verdict: `NO_EFFECT`
- next direction: redesign the topological-density prior or move to a different topological structure

Current 2D restart:

- K3.2D.0: Gross-Pitaevskii vortex-antivortex regime validation
- status: SMOKE-first, baseline-only, no prior tested
- current lesson: a baseline may learn field prediction while failing vortex
  transport, so the regime verdict distinguishes pipeline health from
  topological transport
- goal: find a graceful 2D vortex-position regime before K3.2D.1 prior tests

Current active regime-search toy:

- K3-E2b: active topology regime search on an interpretable toy landscape
- verdict: `ACTIVE_DIAGNOSTIC_VALUE_PASSED`
- summary: guided active topology regime search on a transparent toy
  landscape; active reaches a `transport_ok` regime while random control does
  not. Status: `PASSED`.
- claim: guided regime search beats blind random search
- boundary: not K3 prior validation, not GP truth, not robotics, not RL/CNN
  training
- future path: K3-E2b toy landscape -> K3-E2c cheap GP truth-only active
  search -> K3.2D regime validation -> K3 prior test

Current cheap-GP active regime search:

- K3-E2d: discriminating GP truth-only active topology search
- verdict: `GP_ACTIVE_DIAGNOSTIC_VALUE_PASSED`
- summary: cheap real-GP evaluator with continuous vortex-position transport;
  active passes registered admission criteria against random control
- boundary: not K3 prior validation, not proof topological priors work, not
  full-resolution GP, not robotics, not CNN/RL training
- future path: harden the cheap-GP regime search before any K3.2D prior test

Entry requirement:

- pre-register regime, controls, mechanism, transfer horizon, and claim boundary

## K4 - Gauge / Cross-Family Structure Discovery

Status:

```text
FUTURE
```

Goal:

- test local-symmetry and gauge-invariant structures under VPSL gates
- compare gauge candidates against off-target, smoothness, and generic
  regularization controls
- keep cross-system transfer as a later hardening step, not the first K4 claim

## K5 - Hilbert / Quantum-State Representation

Status:

```text
FUTURE
```

Goal:

- test Hilbert-space and unitary structure candidates
- define diagnostics for quantum-state geometry, unitarity, and
  Born-compatible prediction regimes
- keep certification downstream of S0 recommendation and VPSL gates
