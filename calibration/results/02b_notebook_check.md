# Does the code match what the paper says?

Date: 2026-10-08

**Verdict: yes, for the C2 operator and the main setup.** All 23 feature changes, the byte and packet increments, the four footprints, the eligibility rule, the splits and the seeds match the paper and `02_my_approximation.md`. No mismatches were found. Two items can't be confirmed from code alone (see "Not checked").

## Files checked

| File | Role |
|---|---|
| `pilot_option_b.py` (12,020 bytes, modified 2026-07-12) | **The C2 operator** (`make_approx_mlkem`) |
| `revised_pipeline_main_only.py` (16,604 bytes, 2026-07-12) | Main C1/C2/C3 run, footprints, KS + MI selection |
| `option_b_full_experiment.py` (12,191 bytes, 2026-07-12) | Classifier settings (`make_model`) |
| `pilot_pqc_stable_pipeline.py` (14,402 bytes, 2026-07-12) | Loader, `prepare_xy`, KS + Holm, evaluation |

Google Drive also holds older copies of `pilot_option_b.py`, dated June 2026. They're the same size (12,020 bytes), so they're probably identical, but only the July copy was checked. The check covers the Python modules that the Colab notebooks import, not the notebooks themselves.

## Line-by-line check

| # | Item | Paper says | Code does | Match? | Code (in `pilot_option_b.py` unless noted) |
|---|---|---|---|---|---|
| 1 | Byte increments | +1,184 fwd, +1,088 bwd | `"d_fwd_bytes": 1184.0`, `"d_bwd_bytes": 1088.0`, added to TotLen Fwd/Bwd Pkts and Subflow Fwd/Bwd Byts | **Yes** | `MLKEM = {...}`; step "1. byte totals" |
| 2a | Packets | +1 per direction | `"d_fwd_pkts": 1.0`, `"d_bwd_pkts": 1.0`, added to Tot Fwd/Bwd Pkts and Subflow Fwd/Bwd Pkts | **Yes** | step "2. packet counts" |
| 2b | Header length | +40 per added packet | `Fwd/Bwd Header Len + d_pkts * hdr_bytes`, `hdr_bytes = 40.0` | **Yes** | step "3. header lengths" |
| 3 | Maxima | max(M, 1,460) | `np.maximum(M, cfg["added_pkt_size"])` on Fwd/Bwd/overall Pkt Len Max, `added_pkt_size = 1460.0` | **Yes** | step "5. maxima" |
| 4 | Means | b'/p'; overall = summed bytes / summed packets | `fwd_mean = n_tlf / n_tfp` → Fwd Pkt Len Mean, Fwd Seg Size Avg (same for bwd); `(n_tlf + n_tlb) / (n_tfp + n_tbp)` → Pkt Len Mean, Pkt Size Avg | **Yes** | step "4. packet-length means" |
| 5 | Rates | × b'/b and × p'/p; zero-byte flows keep the old rate | `ratio = np.where(old > 0, new / clip(old, 1), 1.0)` for Flow Byts/s, Flow Pkts/s, Fwd Pkts/s, Bwd Pkts/s | **Yes**, including the known zero-byte behaviour | step "6. rates" |
| 6 | Eligibility | fwd AND bwd packets > 0 | `mask = (Tot Fwd Pkts > 0) & (Tot Bwd Pkts > 0)`; changes only where the mask is true | **Yes (rule)**. The 74,127 / 119,571 count needs the data. | top of `make_approx_mlkem` |
| 7 | 23 changed columns | 23, with duration and IAT untouched | 4 bytes + 4 packets + 2 headers + 6 means + 3 maxima + 4 rates = **23**. No duration or IAT column is touched. | **Yes** | whole function |
| 8a | Footprints | pure 1,152/1,056; hybrid-512 800/768; 2-packet variant | `VARIANTS`: hybrid768 1184/1088/1 pkt, pure768 1152/1056/1, hybrid768_p2 1184/1088/2, hybrid512 800/768/1; all use 40-byte headers and a 1,460 max | **Yes** | `revised_pipeline_main_only.py`, `VARIANTS` |
| 8b | Max-length references 1,200 / 1,330 | Separate sensitivity run | Not in these four files | **Can't tell** | Probably in `PQC_IDS_Operator_Sensitivity.ipynb` |
| 9a | Splits and seeds | 10 stratified splits, seed 42 | `N_SPLITS = 10`, `SPLIT_SEEDS = 0..9`, `MODEL_SEED = 42`, `TEST_SIZE = 0.30`, `stratify=y`, `PER_CLASS = 30000` | **Yes** | `revised_pipeline_main_only.py`, CONFIG |
| 9b | Classifiers | RF, XGBoost, MLP | RF 200 trees; XGB 300 trees, depth 6, learning rate 0.3, hist, subsample/colsample 0.9; MLP (64, 32), early stopping, max_iter 200; all random_state 42 | **Yes** (paper gives no hyperparameters to compare against) | `option_b_full_experiment.py`, `make_model` |
| 9c | 70 features | 70 | `prepare_xy` drops Label, Timestamp, Flow ID, Src/Dst IP, Src Port, NaN/inf rows and zero-variance columns. The count isn't hard-coded. | **Can't tell without data** | `pilot_pqc_stable_pipeline.py` |
| 9d | 55 candidates | 70 − 15 not reconstructable | `UNMODELLABLE` has 15 names (7 std/var + 8 active/idle) | **Yes** | `revised_pipeline_main_only.py` |
| 9e | Holm α, MI threshold | α = 0.05, τ = 0.05 | `HOLM_ALPHA = 0.05`, `MI_HEADLINE = 0.05`, sweep 0 / 0.01 / 0.05 / 0.10 / 0.20 | **Yes** | same file |

**Data types:** the loader casts all features to float32, and the operator casts its output back to float32. So columns the operator doesn't touch stay bit-for-bit identical. That backs up the paper's claim that the selected features were "exactly identical before and after transformation".

## One thing the PQC-OAV data says about the code's reasoning

The code justifies the extra packet like this: `+1184 B over the ~1460 B MSS -> +1 forward packet`. That only holds if the Client Hello actually crosses one TCP segment. In PQC-OAV, the pure ML-KEM-768 Client Hello record is 1,429 bytes, or about 1,434 with the 5-byte TLS record header. That's **under** a 1,460-byte segment, so in that lab it probably fit in one packet, with no extra forward packet. A hybrid share (+32 bytes) or a real browser Client Hello, which is usually bigger, would cross it.

This is an inference: PQC-OAV has no packet counts, so it can't be tested. But it means the "+1 forward packet" assumption depends on how large the classical Client Hello was to begin with. Analysis 04 returns to it.

## Not checked

1. **Eligibility count (74,127) and feature count (70).** These need the CSE-CIC-IDS2018 files. The output of `PQC_IDS_Main_Reproduction.ipynb` would show them.
2. **Max-length references (1,200 / 1,330) and the two corrected operators.** These are probably in `PQC_IDS_Operator_Sensitivity.ipynb`, which was not checked here.

## Effect on `02_my_approximation.md`

No changes needed. The table there matches the code. Its "written from the paper, not the code" warning is now resolved for the main operator.
