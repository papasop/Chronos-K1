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
S0_V0_3_PASSED
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
- summary adapters and the `run_selector` CLI close the loop from experiment
  summaries to S0 recommendations

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
