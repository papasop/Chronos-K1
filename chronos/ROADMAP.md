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
