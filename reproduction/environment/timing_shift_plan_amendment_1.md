# Timing-shift plan: amendment 1

Written before any timing-shift result was inspected. The original plan's hash is
`acd132a1…07f11`; the plan file itself is unchanged.

## What happened

The first full run stopped at split 5, as the plan requires. At zero delay, the
XGBoost C1 and C2 scores differed from the locked record by 6.7 × 10⁻⁴ and
6.4 × 10⁻⁴. C3 was identical.

These checks all passed:

- RF, MLP and XGBoost scores in splits 0–4 matched exactly;
- the feature selection matched the locked record in every split reached
  (all KS and MI checks passed);
- the matrix hash is unchanged.

## Cause (supported interpretation)

- This run uses a CPU with AVX-512. The locked and operator-sensitivity runs
  used Colab machines, where OpenBLAS reported Haswell kernels.
- The library versions are identical, and so is the thread count (2).
- MLP reproduced exactly only after forcing Haswell BLAS kernels
  (`OPENBLAS_CORETYPE=Haswell`). XGBoost has no equivalent setting.
- A near-tie split decision that depends on floating-point rounding is the
  likely cause. This is not tested.

## Amendment

1. The zero-delay integrity check is split in two:
   - the feature selection must match the locked record exactly, and the RF and
     MLP scores must match within 1e-12; otherwise the run stops;
   - XGBoost score differences are recorded, and the run stops only if any
     exceeds 0.005.
2. All comparisons use this run's own C1, C2 and C3 from the same fits, so a
   hardware difference cannot bias a paired difference within the run.
3. The report states the largest XGBoost difference from the locked record.

Nothing else changes: the delays, variants, conditions and statistics stay as
planned.
