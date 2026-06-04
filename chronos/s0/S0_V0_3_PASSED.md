# Chronos-S0 v0.3 Verdict

Final verdict:

```text
S0_V0_3_PASSED
```

Scope:

```text
rule-based structure recognition layer
```

Claim:

Chronos-S0 can read experiment diagnostics and recommend which physical
language family should enter VPSL validation next.

It can currently distinguish:

- missing / insufficient diagnostics -> `UNRESOLVED`
- symplectic mechanism evidence -> `K2_SYMPLECTIC`
- topology transport failure -> `K3_TOPOLOGICAL` / `do_not_promote`
- transported topology -> `K3_TOPOLOGICAL` / continue to constraint gate
- causal, gauge, and Hilbert diagnostic signals as candidate families

## K3.2D Transport Guard

S0 v0.3 encodes the K3.2D lesson:

```text
field prediction ok != topological-object transport ok
```

The K3.2D verdict gate separates:

- `pipeline_ok`: the field map is learnable and bounded.
- `transport_ok`: the vortex-antivortex pair is transported as a topological
  object.

The real K3.2D smoke pattern:

```text
ref=0.0374, hard=0.0, pair=0.0, pos=8.0
```

maps to:

```text
SMOKE_PIPELINE_OK_TRANSPORT_FAIL
```

## Context Guard

Topology transport failure only pre-empts other families inside an explicit
topology-regime context:

```text
diagnostic_context = K3_TOPOLOGY_REGIME
```

Without that context, a strong K2 signal still wins. This prevents a topology
failure diagnostic from globally shadowing a valid symplectic reading.

## Closed Loop

S0 v0.3 includes summary adapters, a CLI, and an experiment emit helper:

```bash
python -m chronos.s0.run_selector --kind k3_2d --summary s.csv
python -m chronos.s0.run_selector --kind k2 --summary s.json --out rec.json
```

```python
from chronos.s0 import emit_recommendation

emit_recommendation("k3_2d", summary, results_dir)
```

`emit_recommendation` writes `s0_recommendation.csv` and degrades without
raising, so experiment scripts can call it after saving their own summaries.

The CLI returns a recommendation object with:

- candidate K-family
- confidence
- reason
- next VPSL gate
- allowed action

## Non-Claims

S0 v0.3 does not certify structures.

It is not a learned classifier and not a physics predictor. It is a
developmental structure-language selector. Certification still happens only
through downstream VPSL gates.
