"""K2.1-B fair-control repair archive entrypoint.

This file reserves the canonical path for the K2.1-B control-repair script.
The full experiment source should be restored from the original Colab or run
log before this entrypoint is used as executable evidence.

VPSL role:
- repair non-degenerate controls
- select fair energy and fair L2 configs
- establish `CONTROLS_REPAIRED` before K2.2-A transfer testing
"""

EXPERIMENT = "k2_1b_repair_controls"
CLAIM_ROLE = "CONTROLS_REPAIRED"


def main():
    raise RuntimeError(
        "K2.1-B archive placeholder only. Restore the original control-repair "
        "script before running this experiment."
    )


if __name__ == "__main__":
    main()
