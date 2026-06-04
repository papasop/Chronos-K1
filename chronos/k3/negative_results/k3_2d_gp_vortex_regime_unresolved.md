# K3.2D.0: 2D GP Vortex Regime Unresolved

Status:

```text
REGIME_UNRESOLVED / TRANSPORT_FAIL
```

System:

```text
2D Gross-Pitaevskii vortex-antivortex pair
```

Stage:

```text
K3.2D.0 regime validation
```

## Result Boundary

This is a baseline-only regime validation result.

It is not a prior test, and it is not a rejection of 2D topological structure.

## Developmental Lesson

```text
Field prediction OK does not imply topological-object transport OK.
```

The baseline can learn short-horizon field prediction while still failing the
topological transport task: preserving and moving the vortex-antivortex pair as
objects with charge and identity.

Chronos therefore separates two gates:

- `pipeline_ok`: the field map is learnable and bounded.
- `transport_ok`: the vortex pair remains intact and its position error stays
  graceful.

The smoke verdict must not return a bare success when only `pipeline_ok` holds.

## Verdict Vocabulary

K3.2D.0 uses the shared verdict helper in:

```text
chronos/k3/verdicts.py
```

Smoke verdicts:

- `SMOKE_PIPELINE_FAIL`
- `SMOKE_PIPELINE_OK_TRANSPORT_FAIL`
- `SMOKE_TRANSPORT_OK`

Full verdicts:

- `FULL_REGIME_VALIDATED`
- `REGIME_UNRESOLVED`

## S0 Connection

This result motivates the first Chronos-S0 guardrail:

```text
low field error != topology success
```

S0 should recommend K3/topological follow-up as an unresolved regime when
field prediction is explicitly marked learnable but object transport fails in a
topology-regime diagnostic context:

```text
diagnostic_context = K3_TOPOLOGY_REGIME
```

Outside that context, the same packet should not globally shadow stronger K1,
K2, K4, or K5 signals. It should fall back to low-confidence K3 unresolved only
when no stronger structure signal is present. S0 should not promote the topology
prior gate until a valid transport regime exists.

## Next Actions

- Retune the 2D regime before any K3.2D prior test.
- Track both field-level diagnostics and vortex-object diagnostics.
- Preserve the distinction between `pipeline_ok` and `transport_ok` in all K3.2D
  summaries.
