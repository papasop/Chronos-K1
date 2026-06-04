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

## K3 - Structure Discovery Program

Status:

```text
STAGE_2_PRIOR_TEST_NEGATIVE
```

Candidate families:

- gauge structure
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
- goal: find a graceful 2D vortex-position regime before K3.2D.1 prior tests

Entry requirement:

- pre-register regime, controls, mechanism, transfer horizon, and claim boundary

## K4 - Cross-System Transfer

Status:

```text
FUTURE
```

Goal:

- test whether a certified structure transfers across systems, not only across
  horizons within one system
