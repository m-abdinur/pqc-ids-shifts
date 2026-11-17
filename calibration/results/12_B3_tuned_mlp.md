# 12 B3 tuned MLP (Colab run 2026-10-10/11, 15,834 s)

Code: my_c2_code/colab_B3_tuned_mlp.py. Rules: decisions_log.md, 2026-10-10 "B3" rows (set before the run).
Raw JSON + per-split files: Google Drive, MyDrive/pqc_oav_calibration_B3_tuned_mlp/.
Environment: Python 3.13.16, scikit-learn 1.6.1, 2 CPUs. Data: 119,571 rows; 68 features (identifiers removed); candidates 53 / 37.

## Sanity
- Untuned MLP reproduces Table 9 at 4 decimals: C1 0.8381, C3 identifier-free 0.8066, C3 combined 0.7855.

## Results (MLP, C3 - C1 weighted F1, 10 splits)
| Design | Untuned | Tuned | Tuned 90% CI (uncorr.) | Tuned 90% CI (corr.) | Reading |
|---|---|---|---|---|---|
| Identifier-free (22.0 selected) | -0.0315 | -0.0284 | [-0.0312, -0.0255] | [-0.0350, -0.0218] | failure persists after tuning |
| Combined (10.1 selected) | -0.0526 | -0.0475 | [-0.0497, -0.0454] | [-0.0525, -0.0426] | failure persists after tuning |

- Tuning shrank the deficits by 0.0031 and 0.0051. TOST p > 0.999 in all four tuned/untuned tests.
- C1 (68 features): untuned 0.8381, tuned 0.8378. C2 - C1: untuned -0.2416, tuned -0.2319.
- Chosen configurations varied across splits (C1 mostly (128,64); combined C3 mostly (256,128)).

## Pre-set reading
- Both designs: "the MLP failure persists after tuning". The untuned-MLP limitation in 5.4 is replaced by this result (v7).
