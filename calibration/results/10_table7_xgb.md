# 10 Table 7 XGBoost rerun (Colab run 2026-10-10, 1,205 s)

Code: my_c2_code/colab_table7_xgb.py. Rules: decisions_log.md, 2026-10-10 "Table 7 XGBoost rerun" rows (set before the run).
Raw JSON: Google Drive, MyDrive/pqc_oav_calibration_table7_xgb/table7_xgb_result.json.
Environment: Python 3.13.16, numpy 2.1.3, pandas 2.2.3, scipy 1.16.3, scikit-learn 1.6.1, xgboost 3.2.0, 2 CPUs. Data: 119,571 rows, 70 features.

## Checks (both pass)
- RF, all six cells: identical to results/07-08 (gap 0.0). The operators are unchanged.
- XGBoost hybrid all, pure all, hybrid 443: identical to results/09 (within 0.0001).

## XGBoost C2 - C1 under xgboost 3.2.0 (mean [95% CI], 10 splits)

| Footprint | Scope | New (3.2.0) | Old (results/07-08, unrecorded version) | Change |
|---|---|---|---|---|
| Hybrid | All bidirectional | -0.0912 [-0.0946, -0.0878] | -0.0897 | -0.0015 |
| Pure | All bidirectional | -0.0913 [-0.0943, -0.0884] | -0.0890 | -0.0024 |
| Bytes-only | All bidirectional | -0.0737 [-0.0776, -0.0698] | -0.0568 | -0.0169 |
| Hybrid | Port 443 | -0.0018 [-0.0025, -0.0010] | -0.0018 | +0.00001 |
| Pure | Port 443 | -0.0017 [-0.0024, -0.0011] | -0.0018 | +0.00007 |
| Bytes-only | Port 443 | -0.0014 [-0.0019, -0.0010] | -0.0010 | -0.0004 |

## Pre-set reading
- Bytes-only rule (2026-10-08), re-applied: bytes/pure = 0.807 -> "partly depends on the packet/max assumptions" (same category as before).
- Share of the XGBoost loss depending on the packet/max assumptions: 1 - bytes/pure = 19% (was 36% under the other version).
- Port-443 rule: all three XGBoost cells |C2 - C1| < 0.01 -> no material effect (unchanged).
- The bytes-only XGBoost cell is the most version-sensitive value seen (0.017).

## Run note
- The run shared the Colab runtime of the B1/B4 run (xgboost already imported). The loaded version was 3.2.0 (asserted in the script), and the XGBoost checks reproduced results/09 exactly, so the run is valid. The import guard in both scripts now allows an already-loaded xgboost 3.2.0.
- Clean-runtime replication (2026-10-10, fresh Colab runtime, xgboost 3.4.1 replaced by 3.2.0 before import, 1,202 s): every XGBoost mean, interval and per-split value, every RF sanity value, all checks and the pre-set readings are identical to the first run. Table 7 values in manuscript v5 stand.
