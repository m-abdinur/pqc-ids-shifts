# What the C2 approximation does

Date: 2026-10-08

**Source:** `thesis report/PQC_IDS_Journal_Revised.docx`, Section 3.1 and Tables 19, 23 and 24. It was not cross-checked against the thesis docx; the journal version is newer.

**Checked against the code on 2026-10-08 (see `02b_notebook_check.md`):** the operator in `pilot_option_b.py` (`make_approx_mlkem`) and the footprints in `revised_pipeline_main_only.py` match this table exactly. Not yet confirmed: the 74,127 eligible-record count, the 70-feature count, and the 1,200/1,330 max-length runs.

## The operator (main condition, "hybrid ML-KEM-768-like")

For each eligible flow and each direction (b = bytes, p = packets, H = header length):
`b' = b + Δ`, `p' = p + 1`, `H' = H + 40`. Means become `b'/p'`, maxima become `max(M, 1460)`, and rates are scaled by `b'/b` or `p'/p`.

- **Δ forward = +1,184 bytes.** This is the ML-KEM-768 encapsulation key, which is also the net growth of the hybrid client share over X25519 (1,216 − 32).
- **Δ backward = +1,088 bytes.** This is the ML-KEM-768 ciphertext, which is also the net growth of the hybrid server share (1,120 − 32).
- **Eligible flows:** forward and backward packet counts both above zero. That's 74,127 of 119,571 records (61.99%), across all protocols and ports.
- **23 of 70 features change.** Duration, IAT, minima, flags, Init Win, Down/Up Ratio, the 7 dispersion fields and the 8 active/idle fields are left as they were.

## Feature table

| Feature(s) | Shift applied | Direction | Flows affected | Assumption | Source |
|---|---|---|---|---|---|
| TotLen Fwd Pkts, Subflow Fwd Byts | +1,184 bytes | Fwd | Eligible (61.99%) | Client key material travels forward | FIPS 203 Table 3; RFC 10024 §4.1 (as cited in the paper) |
| TotLen Bwd Pkts, Subflow Bwd Byts | +1,088 bytes | Bwd | Eligible | Server key material travels backward | FIPS 203 Table 3; RFC 10024 §4.2 |
| Tot Fwd Pkts, Subflow Fwd Pkts | +1 packet | Fwd | Eligible | One extra packet per direction | ⚠ **No source.** The paper calls it "an analytical assumption, rather than a measured or mandated effect" |
| Tot Bwd Pkts, Subflow Bwd Pkts | +1 packet | Bwd | Eligible | Same | ⚠ **No source** |
| Fwd Header Len, Bwd Header Len | +40 bytes | Both | Eligible | IPv4 + TCP headers with no options, for the extra packet | RFC 791 / RFC 9293 (standard sizes, but they depend on the +1-packet assumption) |
| Fwd Pkt Len Max, Bwd Pkt Len Max, Pkt Len Max | max(M, 1,460) | Both | Eligible | A full-size segment at 1,500 MTU | ⚠ **Assumption.** MSS for 1,500 MTU, not measured. Sensitivity checks at 1,200 and 1,330. |
| Fwd Pkt Len Mean, Fwd Seg Size Avg | Recomputed b'/p' | Fwd | Eligible | Follows from the above | Derived |
| Bwd Pkt Len Mean, Bwd Seg Size Avg | Recomputed b'/p' | Bwd | Eligible | Follows from the above | Derived |
| Pkt Len Mean, Pkt Size Avg | (b'fwd + b'bwd)/(p'fwd + p'bwd) | Both | Eligible | Follows from the above | Derived |
| Flow Byts/s | × b'/b (left unchanged if b = 0) | Both | Eligible | Duration unchanged | Derived. ⚠ Known bug: 17,354 zero-byte flows keep their old rate. |
| Flow Pkts/s, Fwd Pkts/s, Bwd Pkts/s | × p'/p | Both | Eligible | Duration unchanged | Derived |
| Flow Duration, all IAT fields | **No change** | n/a | n/a | No timing cost in the main condition | ⚠ **Assumption.** Tested separately with 0.1–100 ms delays. |

**Known operator inconsistency:** after the shift, 58,242 of 74,127 eligible flows (78.57%) have a directional maximum larger than their byte total. Fixing it changed mean C2 loss by at most 0.017 (Table 24).

## Alternative footprints already run (Table 19)

| Footprint | Fwd / Bwd bytes | Packets per direction | C2 − C1: RF / XGB / MLP |
|---|---|---|---|
| Hybrid ML-KEM-512-like | +800 / +768 | 1 | −0.0810 / −0.0935 / −0.2458 |
| **Pure ML-KEM-768-like** | **+1,152 / +1,056** | 1 | **−0.1147 / −0.0913 / −0.2513** |
| Hybrid ML-KEM-768-like (main) | +1,184 / +1,088 | 1 | −0.1147 / −0.0912 / −0.2514 |
| Hybrid 768-like, 2 packets | +1,184 / +1,088 | 2 | −0.1316 / −0.0925 / −0.2411 |

The pure ML-KEM-768 footprint is the one PQC-OAV can check, because the dataset uses pure ML-KEM-768 (see `01_audit.md`). Its C2 loss is practically the same as the main hybrid footprint.

## Numbers without a measured source

1. **+1 packet per direction.** It drives 4 packet-count features, 2 header-length features, all 6 means and all 4 rates. PQC-OAV can't test it because it has no packet counts.
2. **The 1,460-byte maximum.** It drives 3 maximum features, which are the most-shifted features by KS distance (Fwd Pkt Len Max D = 0.62). PQC-OAV can't test it.
3. **Forward = client, backward = server.** The paper says this mapping "cannot be independently verified".
4. **The eligibility rule** (all bidirectional flows). It's a stress-test choice, not an adoption estimate.
5. **No timing change.** PQC-OAV can test this, but only on a lab network.
