# Chronos Numbering Discipline

Chronos uses milestone numbering for VPSL-certified work.

## Rule

K-stage work should be archived as:

```text
K1
K2
K3
...
```

Within a stage, sub-experiments should use:

```text
K2.0
K2.0-B
K2.1
K2.1-B
K2.2-A
```

## Boundary

Historical `ExpXX` labels belong to the K1 archive.

Examples:

- Exp5
- Exp5b
- Exp6
- Exp7
- Exp8-B
- Exp10.3
- Exp11.2-E

These may remain in historical filenames and reproduction notes, but they
should not be used as the primary archive structure for K2 or later.

## K2 Canonical Labels

| Label | Meaning |
| --- | --- |
| K2.0 | FPU-beta regime validation |
| K2.0-B | Transfer-window refinement |
| K2.1 | Initial symplectic prior comparison |
| K2.1-B | Fair-control repair |
| K2.2-A | H=200 symplectic transfer test |

## K3 And Later

Future structures should receive a K-stage label only after their VPSL plan is
defined:

- regime validation
- constraint validation
- mechanism test
- transfer test
- claim boundary
