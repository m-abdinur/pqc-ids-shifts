# PQC-OAV dataset audit

Date: 2026-10-08 · Code: `scripts/01_audit.py` · Full tables: `results/01_audit_tables.txt`

## Bottom line

- **The PQC sessions use pure ML-KEM-768, not the hybrid X25519MLKEM768.** Every `mlkem768` session has a 1,184-byte client key share. The hybrid group (code 0x11EC, 1,216 bytes) appears only in one adversarial scenario (500 sessions, none completed).
- **The data is almost deterministic.** In each normal scenario, the key-share size, Client Hello record length and Server record length each have exactly 1 value across 3,500 sessions. Total handshake bytes have 1 to 4 values. Only duration varies. Bootstrap CIs and TOST on the byte deltas will be zero-width and meaningless, so analyses 03 and 04 need adjusting (see the end of this note).
- **There's a clean like-for-like pair:** `normal_X25519_rsa_cert` vs `normal_mlkem768_rsa_cert`. Both use the same RSA certificate, so the only difference is the key exchange. That matches what C2 changes.
- **There's no per-packet data.** No packet counts, packet lengths, header bytes or inter-arrival times. The server key-share size is missing (N/A) for all 40,010 sessions.

## 1. Files and linkage

| File | Rows | Notes |
|---|---|---|
| `raw_sessions.jsonl` | 40,010 | One session summary per line (27 fields after flattening). Not packet-level. |
| `ml_features_and_labels.csv` | 40,010 | 36 columns: engineered features, one-hot context, `label`, `split`, `taxonomy`, `ID` |
| `scenario_manifest.csv` | 40,010 | `ID`, `scenario_id`, `taxonomy_class`, `subtype`, `operational_class`, `anomaly_label`, `run_idx` |

Linkage: `raw.scenario_id` is `<taxonomy>\<ID>`, the features `ID` has no extension, and the manifest `ID` ends in `.pcap`. All 40,010 rows link across the three files. There are no duplicate IDs.

## 2. Counts

**Key-exchange group (raw `e1_alg_suite`):** x25519 14,500 · mlkem768 14,000 · mlkem1024 10,500 · N/A 500 (TLS 1.2 scenario) · Unknown(0x11EC) 500 · Unknown(0x11EB) 10.

**Labels:** 35,000 Normal, 5,010 Anomaly.

**Normal scenarios (3,500 sessions each, all successful TLS 1.3):**

| Scenario | KEM | Certificate |
|---|---|---|
| normal_X25519_rsa_cert | x25519 | RSA |
| normal_X25519_rsa3072_cert | x25519 | RSA-3072 |
| normal_X25519_ecdsa_p256_cert | x25519 | ECDSA P-256 |
| normal_X25519_ecdsa_p384_cert | x25519 | ECDSA P-384 |
| normal_mlkem768_rsa_cert | mlkem768 | RSA |
| normal_mlkem768_mldsa65_cert | mlkem768 | ML-DSA-65 |
| normal_mlkem768_mldsa87_cert | mlkem768 | ML-DSA-87 |
| normal_mlkem1024_rsa_cert | mlkem1024 | RSA |
| normal_mlkem1024_mldsa65_cert | mlkem1024 | ML-DSA-65 |
| normal_mlkem1024_mldsa87_cert | mlkem1024 | ML-DSA-87 |

**Anomaly scenarios (500 each, 10 for malformed key share):** hybrid_pki, kem_mismatch, pqc_over_tls12, gray_rsa_mimicry, high_jitter_classic, high_latency_pqc, high_loss_pqc, alg_mismatch, downgrade_overt, data_exfil, malformed_keyshare.

## 3. Pure or hybrid?

| Group | Client key share (bytes) | Expected (FIPS 203 / RFC) |
|---|---|---|
| x25519 | 32 | 32 |
| mlkem768 | 1,184 | 1,184 pure ML-KEM-768 encapsulation key |
| mlkem1024 | 1,568 | 1,568 pure ML-KEM-1024 |
| Unknown(0x11EC) = X25519MLKEM768 | 1,216 | 1,216 hybrid (1,184 + 32) |

**Verdict: pure ML-KEM-768.** The hybrid only appears in `model_gray_rsa_mimicry`, which is labelled adversarial, and none of those sessions completed (493 incomplete, 7 failed). It can confirm the hybrid key-share size but can't serve as a normal baseline.

This matters for the paper. The main C2 footprint is the hybrid (+1,184 / +1,088 net). Table 19 already includes a "Pure ML-KEM-768-like" footprint (+1,152 / +1,056), and that is the one this dataset can check directly.

## 4. What detail exists

| Field | Available? | Notes |
|---|---|---|
| Client key-share size | Yes | Constant per scenario |
| Server key-share size | **No** | N/A in all 40,010 sessions |
| Client Hello record length | Yes | x25519 285, mlkem768 1,429 |
| Server record length | Partly | 5,316–5,333 for RSA, ECDSA and ML-DSA-65 certificates whatever the KEM (7,502 for ML-DSA-87), so it doesn't capture the ML-KEM ciphertext. It's probably a certificate-bearing record, not the Server Hello. |
| Total handshake bytes (`e2c_total_bytes`) | Yes | Both directions combined. The page doesn't say whether it counts IP/TCP headers. |
| Handshake duration | Yes | Lab Docker network (172.18.x.x), medians 3.5–7.2 ms |
| Packet counts, packet lengths, header bytes, IAT | **No** | Not in any file. No PCAPs. |

## 5. Which CICFlowMeter features can be checked

| C2 feature group | Can PQC-OAV check it? |
|---|---|
| TotLen Fwd Pkts / Subflow Fwd Byts (+1,184) | **Yes, roughly.** Client key-share delta, or Client Hello record delta |
| TotLen Bwd Pkts / Subflow Bwd Byts (+1,088) | **Only indirectly.** Total-bytes delta minus client delta. There's no direct server field. |
| Total bytes, both directions (+2,272) | **Yes.** `e2c_total_bytes` delta on the RSA pair |
| Tot Fwd/Bwd Pkts, Subflow Pkts (+1 each) | No. No packet counts. |
| Fwd/Bwd Header Len (+40 each) | No |
| Pkt Len Max (raised to 1,460) | No |
| Pkt Len Mean, Seg Size Avg | No. They need packet counts. |
| Flow Byts/s, Pkts/s | No. They need packet counts, and the lab durations aren't realistic. |
| Flow Duration (unchanged in main C2) | **Yes, as a lab-only check.** Duration delta on the RSA pair |
| IAT, Active/Idle, flags, Init Win | No |

So **4 of the 23 altered features** can be checked (the forward and backward byte totals and their subflow copies), and only at the handshake-total level. The other 19 can't be. The data can also check the "no timing change" assumption, but only on a lab network.

## 6. Data problems

- **Near-constant values.** Sizes don't vary within a scenario, so there are 1,405 exact duplicate feature rows. That's expected for a scripted testbed, but it means "n = 3,500" doesn't give 3,500 independent size measurements.
- **Server key-share size is missing everywhere.** `e2_struct_ke.key_length` is also missing everywhere. `e3_cert_info.sig_alg` is missing in 39,510 of 40,010 sessions.
- **Some anomaly labels can't be told apart by size.** `anomaly_hybrid_pki` has exactly the same sizes as `normal_mlkem768_rsa_cert` (10,412 total bytes). `net_high_loss_pqc` matches `normal_mlkem768_mldsa65_cert` (18,340). This doesn't affect this analysis, which uses only the normal scenarios.
- **Features and raw data agree.** `e2c_total_bytes` matches 100%. `e2_client_size` matches 98.75%; the other 1.25% are the 500 sessions where it's N/A.

## Consequences for analyses 03 and 04

1. **The matched pair** `normal_X25519_rsa_cert` vs `normal_mlkem768_rsa_cert` is the primary comparison, since the certificate is held constant. `normal_mlkem1024_rsa_cert` is a secondary pair. The ML-DSA scenarios are left out, because they also change the certificate, and C2 doesn't model that.
2. **Byte deltas are exact numbers, not estimates.** They are reported as fixed values, because bootstrap CIs and TOST would be zero-width and misleading. Only the duration delta gets a bootstrap CI.
3. **Both footprints are compared:** the main hybrid footprint (+1,184 / +1,088) and the pure footprint (+1,152 / +1,056) from Table 19.
4. **What can't be tested:** added packets, header bytes, the 1,460 maximum length, means and rates. Those are 19 of the 23 altered features.
