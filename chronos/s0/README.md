# Chronos-S0: Structure Recognition Layer

Current verdict:

```text
S0_V0_3_PASSED
```

S0 is the developmental layer of Chronos.

It does not predict physics directly and it never certifies a structure. It
observes diagnostic failures and successes, then recommends which physical
representation family should enter VPSL validation next.

S0 asks:

```text
Given a system S, which physical language should the learner try first?
```

Candidate language families:

- K1: Lorentz / causal language
- K2: Symplectic / Hamiltonian language
- K3: Topological / defect language
- K4: Gauge / local symmetry language
- K5: Hilbert / quantum-state language

Certification still requires the downstream VPSL gates:

```text
regime -> constraint -> mechanism -> transfer
```

## Developmental Lesson

K3.2D.0 motivates S0's first guardrail:

```text
field prediction ok != topological object transport ok
```

A learner can imitate field values while failing to preserve vortex identity,
charge, or transport. S0 therefore separates `pipeline_ok` from `transport_ok`
for topological regimes.

For a pre-emptive topology transport-failure guard, S0 requires:

- `field_learnable=True`
- a topology-regime diagnostic context for the pre-emptive guard:
  `diagnostic_context="K3_TOPOLOGY_REGIME"`

Auxiliary fields such as low `baseline_divergence` can travel with the packet,
but non-divergence alone is not treated as evidence that the field map was
learned.

When the diagnostic context is absent, transport failure does not globally
shadow stronger K1/K2/K4/K5 signals. If no stronger signal matches, S0 still
surfaces field-learnable-but-transport-failed packets as low-confidence K3
`REGIME_UNRESOLVED` recommendations.

## Run Tests

From the repository root:

```bash
python -m unittest chronos.s0.tests.test_structure_selector
```

Detailed verdict:

- `S0_V0_3_PASSED.md`
- `results/s0_v0_3_summary.csv`

## Run Selector

S0 can also close the loop from an experiment summary file to a recommendation:

```bash
python -m chronos.s0.run_selector --kind k3_2d --summary chronos/k3/results/k3_2d_0_summary.csv
python -m chronos.s0.run_selector --kind k2 --summary chronos/k2/results/k2_3_reanalysis_summary.csv --out rec.json
```

Supported adapters:

- `k3_2d`: K3.2D vortex-regime summaries.
- `k2`: K2 symplectic summaries.

The CLI emits a recommendation object with:

- candidate K-family
- confidence
- reason
- next VPSL gate
- allowed action
