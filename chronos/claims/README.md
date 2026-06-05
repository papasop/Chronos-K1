# Chronos Claims Layer

The Claims layer is the scientific denominator for Chronos K-words.

A K-word such as `K2_SYMPLECTIC` or `K3_TOPOLOGICAL` is incomplete by itself.
A meaningful Chronos claim must carry its evidence context:

- `evidence_level`
- `gate`
- `diagnostics`
- `controls`
- `failure_mode`
- `next_gate`
- `claim_boundary`

This layer is not memory, not a new experiment, and not a translator. It does
not learn, certify, change S0 recommendations, or feed back into S0. It records
what a result supports, what it does not support, and the boundary under which
the claim is valid.

## Claim Semantics

A `ClaimRecord` is not meaningful because of its name alone. Its semantics come
from the combination of:

- mathematical structure definition
- measured diagnostics
- fair controls
- VPSL gate
- evidence level
- explicit claim boundary

For example, `K3_TOPOLOGICAL / continue` is not a complete claim. A complete
record says what kind of evidence exists, which gate it has reached, what it
supports, what it does not support, and what the next gate is.

## Required Boundaries

`ClaimRecord` requires non-empty:

- `claim_id`
- `structure_family`
- `verdict`
- `claim_boundary`
- `timestamp`
- `supports`
- `does_not_support`

`allowed_action` must be one of:

```text
continue
archive
do_not_promote
```

The following are banned only as `allowed_action` values:

```text
certified
promote
proved
validated
```

They are not banned from verdicts or descriptive text. For example,
`FULL_REGIME_VALIDATED` is a legal verdict.

## Certified Verdict Rule

If the `verdict` field contains `CERTIFIED`, then the record must have:

```text
gate = transfer
evidence_level = vpsl_certified_structure
```

This rule checks only the `verdict` field. It does not scan `claim_boundary` or
other text, so phrases like `not certified` do not trigger it.

## Claim Status

`claim_status` is one of:

- `active`
- `superseded`
- `archived`

Claims are append-only audit records. Later records can supersede or narrow
earlier claims without mutating the original evidence.

## Files

```text
chronos/claims/
├── README.md
├── __init__.py
├── schema.py
├── failure_taxonomy.py
├── builders.py
├── replay.py
└── tests/
    └── test_claims.py
```

## Failure Modes

Standard failure modes include:

| Failure mode | Meaning |
| --- | --- |
| `PIPELINE_OK_TRANSPORT_FAIL` | Field prediction is learnable, but the topological object is not transported. |
| `ACTIVE_NO_ADVANTAGE` | Active search has no measured advantage over random control. |
| `RANDOM_ALSO_SUCCEEDS` | Random search also succeeds at a similar rate. |
| `TRUTH_STABLE_NEURAL_UNLEARNABLE` | Truth dynamics are stable, but the neural baseline cannot learn transport. |
| `PENDING_GPU_VALIDATION` | A result is waiting for GPU validation. |
| `CONTROL_DEGENERATE` | A control prior degenerates or collapses. |
| `REGIME_INVALID` | The graceful regime window is invalid or unresolved. |

## Builders

Builders convert existing result dictionaries into `ClaimRecord` objects:

- `claim_from_k3_e2b(summary_dict)`
- `claim_from_k3_e2c(summary_dict)`
- `claim_from_k3_e2d(summary_dict)`
- `claim_from_k3_2d_0_summary(summary_dict)`
- `claim_from_k2_summary(summary_dict)`

## Replay

`replay.py` provides read-only summaries and JSONL persistence:

- `summarize_claims(claims)`
- `claims_requiring_next_gate(claims)`
- `supersede_claim(claims, claim_id, reason)`
- `append_claim(claim, path)`
- `write_claims(claims, path)`
- `load_claims(path)`

This is intentionally minimal: no database, no cloud service, no self-evolving
memory, no new K-family, and no new prior test.
