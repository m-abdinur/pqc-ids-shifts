# Port-443-only C2 run

Date: 2026-10-08 · Code: `my_c2_code/colab_port443_only.py` · Numbers: `results/08_port443_only.json` · Full output: Drive `MyDrive/pqc_oav_calibration_port443/`
Run: 10 splits, 9,711 s. Settings and reading rule fixed in `decisions_log.md` before the run.

## Bottom line

- **Applied to port-443 flows only, C2 has no material effect.** For all three models and all three footprints, |C2 − C1| is under 0.01 in weighted F1, which is the pre-set rule. The largest drop is MLP with the hybrid footprint: −0.0029.
- **That is 50–110 times smaller than the headline C2 effect:** RF −0.1147 → −0.0011, XGBoost −0.0897 → −0.0018, MLP −0.2514 → −0.0029.
- **The drops are small but mostly real:** the CIs exclude zero for every variant except RF bytes-only.
- **The biggest per-class change is Infiltration with MLP:** −0.0135 (hybrid443) and −0.0085 (bytes-only). Benign moves by 0.002 or less.
- **The sanity check passed:** 6,576 rows changed (2,791 Benign, 3,785 Infiltration), with 0 DoS and 0 Bot.

## C2 − C1, weighted F1 (mean of 10 paired splits, pipeline ci95)

| Model | hybrid768, all flows (reference) | hybrid768_443 | pure768_443 | bytes_only_443 |
|---|---|---|---|---|
| RF | −0.1147 [−0.1524, −0.0770] | −0.0011 [−0.0016, −0.0005] | −0.0010 [−0.0016, −0.0005] | −0.0002 [−0.0007, +0.0002] |
| XGBoost | −0.0897 [−0.0942, −0.0853] | −0.0018 [−0.0022, −0.0013] | −0.0018 [−0.0023, −0.0013] | −0.0010 [−0.0013, −0.0007] |
| MLP | −0.2514 [−0.2619, −0.2409] | −0.0029 [−0.0041, −0.0016] | −0.0028 [−0.0041, −0.0016] | −0.0017 [−0.0026, −0.0008] |

C3 − C1 is unchanged from the main run: RF +0.0014, XGBoost −0.0002, MLP −0.0022.

## Per-class F1 change (C2 − C1, mean over splits)

| Model | Class | all flows | hybrid768_443 | bytes_only_443 |
|---|---|---|---|---|
| RF | Benign | −0.0568 | −0.0012 | +0.0003 |
| RF | Infiltration | −0.2731 | −0.0030 | −0.0012 |
| XGBoost | Benign | −0.0153 | −0.0000 | −0.0001 |
| XGBoost | Infiltration | −0.3453 | −0.0071 | −0.0039 |
| MLP | Benign | −0.1148 | +0.0020 | +0.0018 |
| MLP | Infiltration | −0.4132 | −0.0135 | −0.0085 |

DoS and Bot change by 0.0002 or less in every port-443 variant. Their rows aren't modified, so any change there comes only through their precision.

## What it means

1. **The headline C2 effect comes from the scope, not just the size of ML-KEM.** It needs the footprint added to every bidirectional flow, including all DoS and Bot flows, none of which are on port 443. Limit it to port 443 and the effect almost disappears.
2. **C2 is a high-exposure stress test, not a deployment estimate.** (Not "worst case": the 2-packet footprint gave a larger RF loss, −0.1316 vs −0.1147, Table 19.) It can't be read as the expected effect of PQC migration.
3. **This weakens the case for C3.** Under the port-443 scope there is almost nothing to recover. C3 still works, but the problem it solves is mostly an artefact of the stress-test scope.
4. **This depends on the dataset.** In this class mix, no attack class except Infiltration has port-443 flows. A dataset with TLS-based attacks (e.g. web attacks or C2 channels over HTTPS) could show a larger port-443 effect. So these results don't show that the effect is negligible in general.

## Caveats

- **Weighted F1 dilutes the change.** Only 5.5% of rows change, so these numbers say little about how detection changes on the modified flows themselves. The loss on the modified rows alone is in `09_B1_B4.md` (B1).
- **Port 443 ≠ TLS 1.3 ≠ ML-KEM.** The run assumes every eligible 443 flow carries one ML-KEM-768 handshake. That makes it an upper bound for this scope.
- **The intervals are too narrow.** The 10 splits overlap, as in the earlier runs.
