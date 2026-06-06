# L-VPSL-0.4 Milestone

L-VPSL-0.4 is the current cleaned canonical replay for the no-LLM grounded
language stack and the Claim Denominator Layer. The active executable entrypoint
is `chronos_full_denominator.py`.

## Supports

- L1 negation
- L2 causal boundary
- L3 quantifier
- L4 reference
- L4A ambiguity hardening
- L5 temporal ordering
- Full denominator replay across K2, K3-E2c, K3-E2d, and L-VPSL claims
- Failure-path anti-cheat checks

## Does Not Support

- General language understanding
- Open-domain conversation
- LLM-level fluency
- Real robot deployment
- Universal physics AI

## Next Gate

Trace replay:

```text
sensor/action trace -> SemanticClaim / ActionEvent -> grounded language -> ClaimRecord
```
