"""K2.0 FPU-beta regime validation archive entrypoint.

This file reserves the canonical path for the K2.0 regime-validation script.
The full experiment source should be restored from the original Colab or run
log before this entrypoint is used as executable evidence.

VPSL role:
- validate the FPU-beta regime before prior comparison
- identify clean and stress horizons
- establish baseline behavior used by later K2 stages
"""

EXPERIMENT = "k2_0_fpu_regime"
CLAIM_ROLE = "REGIME_VALIDATION_GATE"


def main():
    raise RuntimeError(
        "K2.0 archive placeholder only. Restore the original regime-validation "
        "script before running this experiment."
    )


if __name__ == "__main__":
    main()
