"""K2.0-B FPU-beta window refinement archive entrypoint.

This file reserves the canonical path for the K2.0-B refinement script.
The full experiment source should be restored from the original Colab or run
log before this entrypoint is used as executable evidence.

VPSL role:
- refine the validated rollout window
- locate the H=200 stress setting
- report baseline hard-divergence context used by K2.2-A
"""

EXPERIMENT = "k2_0b_refine_window"
CLAIM_ROLE = "TRANSFER_WINDOW_REFINEMENT"


def main():
    raise RuntimeError(
        "K2.0-B archive placeholder only. Restore the original window-refinement "
        "script before running this experiment."
    )


if __name__ == "__main__":
    main()
