# Bytes-only C2 variant and port-443 share

Date: 2026-10-08 · Code: `my_c2_code/colab_bytes_only_and_port443.py` · Numbers: `results/07_bytes_only_and_port443.json` · Full Colab output: Drive `MyDrive/pqc_oav_calibration_bytes_only/` (`bytes_only_result.json`, `partA_port443.json`, `bytes_only_run_summary.json`, `bytes_only_run_raw.json`)
Run: CODE_DIR `pqc_ids_dissertation/final_experiment_package_v6/01_timing_free/code`, 10 splits, 7,471 s. Settings and reading rule fixed in `decisions_log.md` before the run.

## Bottom line

- **The RF headline mostly rests on the untested assumptions.** With the same bytes but no added packet and no 1,460 max, RF's C2 loss falls from −0.1147 to −0.0314, so about 73% of it disappears. The pre-set rule says: "comes mainly from the packet/max assumptions".
- **XGBoost depends on them partly:** −0.0890 → −0.0568, about 36% of the loss gone.
- **MLP barely depends on them:** −0.2513 → −0.2393, about 5% gone. The rule still says "partly", because the gap (0.0120) is just over the 0.01 band.
- **What still holds:** C2 lowers weighted F1 for all three models even without the packet/max assumptions; every CI excludes 0. C3 recovery is unchanged: the C3 − C1 gap stays within 0.0022 of zero.
- **Most eligible flows aren't TLS-port flows:** only 8.9% (6,576 of 74,127) use destination port 443. None of the DoS or Bot flows do.
- **The paper's counts are confirmed:** 119,571 rows and 74,127 eligible flows.

## Part A: rows, eligibility, port 443

| | Eligible flows | On Dst Port 443 | Share |
|---|---|---|---|
| All | 74,127 (of 119,571 rows) | 6,576 | 8.87% |
| Benign | 25,322 | 2,791 | 11.02% |
| DoS | 8,361 | 0 | 0.00% |
| Bot | 15,105 | 0 | 0.00% |
| Infiltration | 25,339 | 3,785 | 14.94% |

## Part B: C2 − C1 weighted F1 (mean of 10 paired splits, pipeline ci95)

| Model | hybrid768 (same run) | pure768 (same run) | **bytes-only** | bytes-only / pure | Pre-set reading |
|---|---|---|---|---|---|
| RF | −0.1147 [−0.1524, −0.0770] | −0.1147 [−0.1518, −0.0776] | **−0.0314 [−0.0406, −0.0223]** | 0.27 | mainly packet/max |
| XGBoost | −0.0897 [−0.0942, −0.0853] | −0.0890 [−0.0936, −0.0843] | **−0.0568 [−0.0682, −0.0454]** | 0.64 | partly |
| MLP | −0.2514 [−0.2619, −0.2409] | −0.2513 [−0.2618, −0.2409] | **−0.2393 [−0.2470, −0.2316]** | 0.95 | partly (0.0120 from pure; the band is 0.01) |

C3 − C1 is identical for all three footprints, because C3's features aren't touched: RF +0.0014 [+0.0006, +0.0022], XGBoost −0.0002 [−0.0009, +0.0006], MLP −0.0022 [−0.0055, +0.0012].

**Check against the paper:**
- **RF and MLP** reproduce the paper's C2 − C1 to 4 decimals.
- **XGBoost is flagged:** this run gives hybrid −0.0897 vs −0.0912 in the paper (gap 0.0015), and pure −0.0890 vs −0.0913 (gap 0.0023). It was not corrected here. A likely cause is a different XGBoost version in the Colab runtime of that day (checked later in `09_B1_B4.md`). The bytes-only comparison uses the same-run values, so it isn't affected.

## Caveats on this run

1. **"Bytes-only" still moves the maxima.** To keep max ≥ mean, each maximum was raised to the new mean where needed. That happened in 73.6% (Fwd), 79.4% (Bwd) and 71.1% (overall) of eligible flows. So this isn't "maxima untouched". It removes the jump to 1,460 bytes, not every change to the maxima. Some of the remaining RF loss may come from this floor.
2. **This is a lower bracket, not the truth.** Real pure ML-KEM-768 traffic probably sits between "0 added packets" and "+1 packet per direction". The server side may well add a packet, because ~1,062 extra bytes go into a ~5.3 KB server flight. This run shows how much the result moves across that range. It doesn't show where reality is.
3. **The CIs treat the 10 splits as independent, but they overlap.** That's the same unresolved issue as the main results, so the intervals are too narrow.

## Implications

- **"C2 degrades all three detectors and C3 restores them" still holds.** It survives the bytes-only lower bracket.
- **The RF magnitude (−0.1147) is not the effect of ML-KEM-768 alone.** About three quarters of it depends on the +1 packet / 1,460 max assumptions, which can't be checked here and which PQC-OAV's Client Hello size argues against for pure ML-KEM-768. As a range, the RF loss runs from −0.031 (bytes-only) to −0.115 (+1 packet).
- **The 443 share supports the "stress test" framing the paper already uses** (eligibility is "a stress-test choice, not an adoption estimate"): 91% of the flows C2 changes are not on port 443, including every DoS and Bot flow.

A port-443-only C2 variant is in `08_port443_only.md`.
