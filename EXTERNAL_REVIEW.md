# External Review Guide

## What This Repo Claims

Chronos is a no-LLM validated representation stack. It records what a claim
supports, what it does not support, its evidence level, next gate, and claim
boundary.

It does not claim AGI, open-domain language, a deployed robot, or universal
physics AI.

## Fast Path: 5-Minute Replay

```bash
python chronos_full_denominator.py
python colab/chronos_y30_core_single.py
python colab/chronos_y20_core_single.py
```

Expected:

- full denominator replay passes
- Y30-Core self-tests pass
- Y20-Core self-tests pass

## What To Look For

Y30:

- contextualizes claims
- does not change physics verdicts
- is not physics evidence

Y20:

- asks objection / required-gate questions
- does not resolve physics claims by debate alone

K2:

- first VPSL-certified physical structure
- certified only within the tested FPU-beta scope

K3:

- active / bounded negative
- no K3 structure is certified yet
- field prediction OK does not imply object transport OK

## What This Repo Does Not Claim

- Buddhist doctrine is proven
- external world nonexistence is proven
- scientific realism is refuted
- Y30/Y20 are physics evidence
- all physical priors help
- general Physics-AI is solved

## Useful Review Files

- Main replay: `chronos_full_denominator.py`
- Expected outputs: `docs/EXPECTED_OUTPUTS.md`
- Boundary map: `docs/CLAIM_BOUNDARY_MAP.md`
- K3 status: `chronos/k3/K3_EXTERNAL_STATUS.md`
