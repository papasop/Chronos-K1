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
- `source_module`
- `code_version`
- `claim_type`
- `confidence_level`
- `evidence_scope`
- `replication`
- `risk_flags`

This layer is not memory, not a new experiment, and not a translator. It does
not learn, certify, change S0 recommendations, or feed back into S0. It records
what a result supports, what it does not support, and the boundary under which
the claim is valid.

The denominator is shared across physical and language-representation claims.
For example, a physics claim such as `K2_SYMPLECTIC` and a grounded-language
claim such as `LANGUAGE_GROUNDING` use the same `supports`,
`does_not_support`, `next_gate`, and `allowed_action` fields. A language claim
therefore cannot silently overclaim general language understanding any more
than a physical claim can silently overclaim certification.

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
- `source_module`
- `supports`
- `does_not_support`

## V2 Audit Fields

Claims also carry second-layer audit fields:

- `claim_type`: one of `positive_evidence`, `negative_result`, `unresolved_result`,
  `handoff`, `certified_structure`, or `boundary_note`
- `confidence_level`: one of `low`, `medium`, `high`, or `certified`
- `evidence_scope`: structured scope such as system, regime, model, and compute
- `replication`: structured repeatability metadata such as seeds and control counts
- `risk_flags`: audit tags such as `toy_landscape`, `transport_fail`, or `no_prior_test`

`confidence_level` is evidence strength only. It never changes
`allowed_action`. A claim may have `confidence_level = high` and still have
`allowed_action = do_not_promote`. `confidence_level = certified` is reserved
for `evidence_level = vpsl_certified_structure`.

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
| `MECHANISM_DECAYS` | A mechanism signal weakens or fails at later mechanism or transfer checks. |
| `DIAGNOSTICS_INSUFFICIENT` | Diagnostics are too thin to support a bounded structure claim. |
| `PRIOR_NO_EFFECT` | A prior does not beat the baseline and controls. |

## Builders

Builders convert existing result dictionaries into `ClaimRecord` objects:

- `claim_from_k3_e2b(summary_dict)`
- `claim_from_k3_e2c(summary_dict)`
- `claim_from_k3_e2d(summary_dict)`
- `claim_from_k3_2d_0_summary(summary_dict)`
- `claim_from_k2_summary(summary_dict)`
- `claim_from_language_grounding_summary(summary_dict)`

Builders never fabricate a pass. Negative active-search results such as
`GP_ACTIVE_NO_ADVANTAGE` are archived rather than continued, even if an input
recommendation dictionary happens to say `continue`.

`controls` and `diagnostics` are structured dictionaries. This keeps the
scientific denominator machine-readable instead of storing controls as a flat
label list.

## Replay

`replay.py` provides read-only summaries and JSONL persistence:

- `summarize_claims(claims)`
- `claims_requiring_next_gate(claims)`
- `claims_with_risk_flag(claims, flag)`
- `human_readable_summary(claim)`
- `supersede_claim(claims, claim_id, reason)`
- `append_claim(claim, path)`
- `write_claims(claims, path)`
- `load_claims(path)`

This is intentionally minimal: no database, no cloud service, no self-evolving
memory, no new K-family, and no new prior test.
