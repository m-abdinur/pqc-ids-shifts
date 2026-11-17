# 11 B2 mismatch test (Colab run 2026-10-10, 8,027 s)

Code: my_c2_code/colab_B2_mismatch.py. Rules: decisions_log.md, 2026-10-10 "B2" rows (set before the run).
Raw JSON + per-split files: Google Drive, MyDrive/pqc_oav_calibration_B2_mismatch/.
Environment: Python 3.13.16, scikit-learn 1.6.1, xgboost 3.2.0, 2 CPUs. Data: 119,571 rows, 70 features, 55 candidates.
H = hybrid footprint (+1 packet, 40 B header, 1,460 max). B = bytes-only (no packet, no 1,460 max).

## Sanity checks
- C2 under H and B equal Table 7 for all models (exact).
- C3 H->H and A H->H equal the paper at 4 decimals for all models (C3 +0.0014 / -0.0013 / -0.0022; A +0.0004 / +0.0003 / +0.0008). The augmentation re-implementation reproduces the original.
- C3 H->B = C3 H->H: exact for XGBoost and MLP. RF differs by 0.000003 in the mean (same decisions). Likely cause: RF multi-threaded prediction summation order (Additional file 1, A1.1 records the same sensitivity). Not confirmed.

## Selection
- Prepared with H: 23 shifted (identical in all splits), mean 24.1 selected.
- Prepared with B: 14 shifted (identical in all splits; all 14 are in H's 23), mean 33.0 selected. The B screen does not flag the 9 packet-count, header-length and packet-rate features.

## X - C1 weighted F1, mean (share of the same-test C2 loss that remains), reading
| Cell (prepare -> test) | RF | XGBoost | MLP |
|---|---|---|---|
| C3 H->H (main) | +0.0014 survives | -0.0013 survives | -0.0022 survives |
| C3 B->B (matched) | -0.0007 survives | -0.0016 survives | -0.0032 survives |
| **C3 B->H** | +0.0013 survives | **-0.0195 (21%) not established** | -0.0050 uncorrected only |
| C3 H->B (check) | +0.0014 survives | -0.0013 survives | -0.0022 survives |
| A H->H (main) | +0.0004 survives | +0.0003 survives | +0.0008 survives |
| A B->B (matched) | -0.0001 survives | -0.0005 survives | +0.0004 survives |
| **A B->H** | -0.0037 uncorrected only | **-0.0678 (74%) not established** | **-0.1281 (51%) not established** |
| **A H->B** | -0.0004 survives | **-0.0098 (13%) not established** | -0.0063 uncorrected only |

90% intervals (uncorrected / corrected) for the failures:
- C3 B->H XGB [-0.0250, -0.0140] / [-0.0321, -0.0069]
- A B->H XGB [-0.0733, -0.0623] / [-0.0805, -0.0551]; MLP [-0.1602, -0.0959] / [-0.2020, -0.0541]
- A H->B XGB [-0.0146, -0.0050] / [-0.0209, +0.0012]
"Uncorrected only" cells: C3 B->H MLP corrected [-0.0098, -0.0002], Holm p 0.087; A B->H RF corrected [-0.0101, +0.0026], p 0.154; A H->B MLP corrected [-0.0113, -0.0013], p 0.213.

## Pre-set reading
- Matched simulations recover for every model and both defences.
- When the simulation omits the added packet (B->H): screening still recovers RF, fails for XGBoost and holds for MLP only without the overlap correction; augmentation fails for XGBoost and MLP and holds for RF only without the correction.
- When the simulation adds a packet that the test shift lacks (H->B): screening is unaffected by construction; augmentation fails for XGBoost and holds for MLP only without the correction.

## Interpretation (not part of the pre-set rules)
- Recovery depends on the simulation getting the packet structure right. Omitting the packet (B->H) costs more than adding one that is absent (H->B), and screening is less sensitive than augmentation.
