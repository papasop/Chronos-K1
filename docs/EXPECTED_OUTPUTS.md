# Expected Outputs

Use this file to check whether a local reproduction succeeded.

## Full Denominator Replay

Command:

```bash
python chronos_full_denominator.py
```

Expected final signal:

```text
ok all full-denominator + self-test + anti-cheat assertions passed
```

Expected summary:

```text
by_claim_type      : {'certified_structure': 1, 'negative_result': 1, 'positive_evidence': 4}
by_confidence_level: {'certified': 1, 'low': 1, 'medium': 4}
count_total        : 6
```

## Y30 Replay

Command:

```bash
python colab/chronos_y30_core_single.py
```

Expected final signal:

```text
ok Y30-Core self-tests passed
```

Expected claim signal:

```text
Y30_CORE_TOY_MVP_PASSED
```

The package-level Y30 milestone is:

```text
Y30_CORE_V0_3_K_FAMILY_CONTEXT_BRIDGE_PASSED
```

## Y20 Replay

Command:

```bash
python colab/chronos_y20_core_single.py
```

Expected final signal:

```text
ok Y20-Core self-tests passed: 19
```

Expected claim signal:

```text
Y20_DEBATE_STRUCTURE_OK
```

The package-level Y20 milestone is:

```text
Y20_CORE_V0_2_DEBATE_AND_PHYSICS_SELF_AUDIT_PASSED
```
