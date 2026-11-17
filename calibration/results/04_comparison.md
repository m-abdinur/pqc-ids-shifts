# C2 footprints vs real PQC-OAV differences

Date: 2026-10-08 · Code: `scripts/04_compare.py` · Numbers: `results/04_comparison_numbers.csv` · Inputs: `results/03_real_shifts.csv`, `results/02_my_approximation.md`, `data/raw/raw_sessions.jsonl`

No test, no TOST, no margin. The byte comparison is descriptive, because the primary total (+2,206) was seen in the audit. Percent = (C2 value − real value) / real value. A positive number means the C2 footprint is bigger.

## Bottom line

- **The byte sizes are right.** The pure ML-KEM-768 footprint is within 8 bytes of the real value in each direction, and 2 bytes (0.09%) on the total. The pure ML-KEM-1024 footprint is within 9 bytes, and 1 byte on the total.
- **The main hybrid footprint is 66 bytes (3%) above the real total.** 64 of those bytes are the two 32-byte X25519 shares that a hybrid carries and this pure-KEM dataset doesn't. That's a difference in what was measured, not a calibration error.
- **This check covers the safest part of C2.** The +1 packet assumption drives 15 of the 19 features this dataset can't check. The data gives no support for it (see section 1b). It can't refute it either.
- **Duration:** the "no timing change" assumption fits the lab data. The 0.1 ms delay is above the ML-KEM-768 CIs. This is a single-host Docker network, so it says little about real networks.

## ML-KEM sizes used (checked)

| | ML-KEM-768 | ML-KEM-1024 |
|---|---|---|
| Encapsulation key (client share) | 1,184 | 1,568 |
| Ciphertext (server share) | 1,088 | 1,568 |
| Pure footprint = size − 32 (X25519) | +1,152 / +1,056 | **+1,536 / +1,536** (confirmed) |

Source: **FIPS 203, Table 3 "Sizes (in bytes) of keys and ciphertexts of ML-KEM", p. 39**. The values come from `Module-Lattice-BasedKeyEncapsulation Mechanism Standard.pdf` (the August 2024 version with minor corrections). It gives ML-KEM-768 = 1184 / 2400 / 1088 / 32 and ML-KEM-1024 = 1568 / 3168 / 1568 / 32 (encapsulation key / decapsulation key / ciphertext / shared secret). draft-ietf-tls-mlkem-09 §3 and Go's `crypto/mlkem` give the same values. PQC-OAV's client shares (1,184 and 1,568) match the encapsulation-key sizes independently.

## 1. Mapping: how direct is each one?

| Real quantity | C2 feature | How direct | Why |
|---|---|---|---|
| Client Hello record diff (+1,144), or key-share diff (+1,152) | TotLen Fwd Pkts, Subflow Fwd Byts (+1,184 main / +1,152 pure) | **Fairly direct for one handshake** | Both are client-to-server payload bytes. The fixed 5-byte record header cancels out in the difference. Two gaps: (1) the paper's forward = client mapping "cannot be independently verified", and (2) the Client Hello grows 8 bytes less than the key share for reasons the data doesn't show. |
| Implied server side (+1,062), **derived** | TotLen Bwd Pkts, Subflow Bwd Byts (+1,088 main / +1,056 pure) | **Indirect** | It's a subtraction (total − Client Hello), not a measured field. The dataset has no server key-share size. `e2_server_record_len` is documented as the "Length of the Server Hello record", but it only moves +6 bytes, so it can't contain the 1,088-byte ciphertext. The documentation and the data disagree here. |
| Total diff (+2,206) | Fwd + Bwd total (+2,272 main / +2,208 pure) | **Most direct, if headers aren't counted** | It's measured, but it combines both directions. Whether it includes per-packet headers changes what it means (1b). |

**The bigger mapping gap applies to all three rows.** PQC-OAV measures one TLS 1.3 handshake. C2 adds the increment to every one of 74,127 bidirectional IDS2018 flows, whatever the protocol or port. This dataset confirms the size of the increment per handshake. It says nothing about whether applying it to all those flows is realistic.

### 1b. Does `e2c_total_bytes` include IP/TCP headers? Probably yes. This is an inference.

The dataset page only says "Total bytes transferred during the handshake". But the short, failed scenarios have just 1–2 TLS records, so their totals can be rebuilt. The model is: total = sum of (record length + 5-byte TLS header + 66). Here 66 = Ethernet 14 + IPv4 20 + TCP 20 + 12 bytes of TCP options, assumed to be timestamps.

| Scenario (outcome, n) | Client rec | Server rec | Total | Residual | 66 × records | Fits? |
|---|---|---|---|---|---|---|
| anomaly_kem_mismatch (Failure, 500) | 1,429 | 2 | 1,573 | 132 | 132 | yes |
| raf_alg_mismatch (Failure, 500) | 1,429 | 2 | 1,573 | 132 | 132 | yes |
| anomaly_pqc_over_tls12 (Failure, 500) | 162 | 2 | 306 | 132 | 132 | yes |
| threat_data_exfil (Incomplete, 493) | 1,429 | — | 1,500 | 66 | 66 | yes |
| model_gray_rsa_mimicry (Incomplete, 493) | 1,519 | — | 1,590 | 66 | 66 | yes |
| threat_data_exfil / gray_rsa_mimicry (Failure, 7 each) | 1,429 / 1,519 | — | 1,573 / 1,663 | 139 | 66 | only if one unrecorded 2-byte alert record is added (2 + 5 + 66 = 73) |
| raf_robustness_malformed_keyshare (Failure, 10) | 129 | 2 | 285 | 144 | 132 | **no** (12 bytes off) |

Five of eight configurations fit exactly, two fit with one extra alert record, and one doesn't fit. So the total very likely counts whole captured frames, headers included.

What that means for the normal pairs:

- **No extra-packet bytes show up.** Total diff − Client Hello diff − (ciphertext − 32) − server record diff leaves **exactly 0** for ML-KEM-768 (2,206 − 1,144 − 1,056 − 6) and for ML-KEM-1024 (3,073 − 1,528 − 1,536 − 9). If PQC had added a captured packet, about 66 more bytes would appear. They don't. So in this capture, PQC and classical handshakes have the same number of frames. Caveat: this assumes the server side is just the ciphertext change plus the server record change. That fits two pairs exactly, but it's still an assumption.
- **The capture probably doesn't show real wire segments.** Under the same model, the hybrid Client Hello is a single 1,590-byte frame, and the ML-KEM-1024 Client Hello (1,813 + 5) would be a single ~1,884-byte frame. Both are bigger than a standard 1,514-byte Ethernet frame. That points to segmentation offload in the capture, or a non-standard MTU. Either way, frame counts here aren't 1,500-MTU packet counts.
- **So the +40 header bytes per added packet can't be confirmed or refuted from this total.** If an extra packet had been captured, it would show as about +66 here, not +40. It doesn't show at all.

## 2. C2 value vs real value

| Pair | Footprint | Quantity | C2 | Real | Diff (bytes) | Diff (%) |
|---|---|---|---|---|---|---|
| 768 | Main hybrid | Fwd vs Client Hello diff | 1,184 | 1,144 | +40 | +3.50 |
| 768 | Main hybrid | Fwd vs key-share diff | 1,184 | 1,152 | +32 | +2.78 |
| 768 | Main hybrid | Bwd vs implied server (derived) | 1,088 | 1,062 | +26 | +2.45 |
| 768 | Main hybrid | Total | 2,272 | 2,206 | +66 | +2.99 |
| 768 | Pure | Fwd vs Client Hello diff | 1,152 | 1,144 | +8 | +0.70 |
| 768 | Pure | Fwd vs key-share diff | 1,152 | 1,152 | 0 | 0.00 |
| 768 | Pure | Bwd vs implied server (derived) | 1,056 | 1,062 | −6 | −0.56 |
| 768 | Pure | Total | 2,208 | 2,206 | +2 | +0.09 |
| 1024 | Pure 1024 | Fwd vs Client Hello diff | 1,536 | 1,528 | +8 | +0.52 |
| 1024 | Pure 1024 | Fwd vs key-share diff | 1,536 | 1,536 | 0 | 0.00 |
| 1024 | Pure 1024 | Bwd vs implied server (derived) | 1,536 | 1,545 | −9 | −0.58 |
| 1024 | Pure 1024 | Total | 3,072 | 3,073 | −1 | −0.03 |

The pure footprints are within 0.7% everywhere. The main hybrid is 2.5–3.5% high, which is almost all the 32-byte X25519 share in each direction. Table 19 already shows that the pure and hybrid footprints give practically the same C2 loss (−0.1147 / −0.0913 / −0.2513 vs −0.1147 / −0.0912 / −0.2514). So recalibrating to the measured +1,144 / +1,062 would move the bytes by 8 and 6. There's no reason to expect a visible change in F1.

## 3. Duration (lab Docker network)

| Pair | Statistic | Real diff (ms) | 95% CI | Fits the main assumption (0)? | Fits smallest tested delay (0.1 ms)? |
|---|---|---|---|---|---|
| ML-KEM-768 | median | +0.026 | [−0.006, +0.055] | yes | no, 0.1 is above the CI |
| ML-KEM-768 | mean | −0.022 | [−0.107, +0.061] | yes | no, 0.1 is above the CI |
| ML-KEM-1024 | median | +0.079 | [+0.048, +0.108] | no, the CI excludes 0 | yes |
| ML-KEM-1024 | mean | −0.014 | [−0.095, +0.066] | yes | no |

For ML-KEM-768, the lab data fits "no timing change" and is below the smallest tested delay. It's a single-host Docker network with handshakes of about 5 ms. In real networks, PQC timing costs come from extra round trips and packet loss on bigger flights, and this testbed doesn't reproduce those. Also, the measure is handshake duration, while CICFlowMeter's Flow Duration covers the whole flow. So the result is "consistent with no measurable cost in a lab setting". It doesn't validate the assumption for real traffic.

## 4. The 19 altered features this dataset cannot check

| Assumption they depend on | Features | Count |
|---|---|---|
| **+1 packet per direction** (direct) | Tot Fwd Pkts, Tot Bwd Pkts, Subflow Fwd Pkts, Subflow Bwd Pkts | 4 |
| **+40 header bytes** (also needs the +1 packet) | Fwd Header Len, Bwd Header Len | 2 |
| **1,460 maximum** | Fwd Pkt Len Max, Bwd Pkt Len Max, Pkt Len Max | 3 |
| **Derived means** (byte increment ÷ packets + 1) | Fwd Pkt Len Mean, Fwd Seg Size Avg, Bwd Pkt Len Mean, Bwd Seg Size Avg, Pkt Len Mean, Pkt Size Avg | 6 |
| **Derived rates** (unchanged duration; three also need the +1 packet) | Flow Byts/s (bytes only), Flow Pkts/s, Fwd Pkts/s, Bwd Pkts/s | 4 |
| | **Total** | **19** |

The +1 packet assumption feeds 15 of these 19 (4 + 2 + 6 + 3 rates). The 3 maxima are the most-shifted features (Fwd Pkt Len Max KS D = 0.62). Neither assumption is tested here.

What the data hints about them, all inference:
- **+1 forward packet:** the pure ML-KEM-768 Client Hello (1,429 + 5 = 1,434 bytes) fits within one 1,460-byte segment, even with 12 bytes of TCP options (1,448). In this lab, a pure ML-KEM-768 client probably sent no extra packet. The ML-KEM-1024 Client Hello (1,818) would cross one segment. So would a pure Client Hello plus a 32-byte X25519 share (1,466). The only hybrid Client Hello in the data (1,519 + 5) comes from the adversarial gray_rsa_mimicry scenario. It's 90 bytes bigger than the pure one, not 32, so it isn't a clean hybrid reference. So whether "+1 packet" holds depends on the KEM and on how big the classical Client Hello was to begin with.
- **+40 header bytes:** if the 66-byte frame model is right, this lab's TCP header carries 12 bytes of options. Then an added packet would add 52 bytes of IPv4 + TCP, not 40. That doesn't affect anything above.
- **1,460 maximum:** the capture shows frames larger than a 1,500 MTU allows (1b), so it can't show the real maximum segment size.

Testing the +1 packet and 1,460 assumptions needs PCAPs of real (ideally hybrid) PQC TLS traffic with packet counts. That's a separate analysis.

**Verdict: no.** The parts this dataset can check (the 4 forward/backward byte features, plus the no-timing-change assumption in a lab) don't need re-calibrating. The pure footprints match within 0.7%. The +1 packet assumption, which drives 15 of the other 19 features, is still untested, and this dataset gives no support for it.
