# Real classical vs ML-KEM handshake shifts

Date: 2026-10-08 · Code: `scripts/03_real_shifts.py` · Numbers: `results/03_real_shifts.csv` · Run log: `results/03_real_shifts_log.txt`

## Bottom line

- Every byte quantity is constant within each group (1 distinct value in 3,500 sessions). So the differences below are exact, with no spread and no test.
- Primary pair (ML-KEM-768 vs X25519, both RSA cert): total handshake bytes go up by **+2,206**. The Client Hello record goes up by **+1,144**. The implied server side is **+1,062** (derived, not measured).
- Secondary pair (ML-KEM-1024 vs X25519): total **+3,073**, Client Hello **+1,528**, implied server side **+1,545** (derived).
- Duration barely moves on this lab Docker network: all differences are under 0.1 ms. For ML-KEM-768 both CIs include zero.

## Groups

All three groups: 3,500 sessions each, all labelled Normal, all `Success`, all TLS 1.3. No sessions were dropped by the filter. No missing values.

| Quantity | X25519 (classical) | ML-KEM-768 (primary) | ML-KEM-1024 (secondary) |
|---|---|---|---|
| n | 3,500 | 3,500 | 3,500 |
| Client key-share size (bytes) | 32 | 1,184 | 1,568 |
| Client Hello record length | 285 | 1,429 | 1,813 |
| Server record length | 5,316 | 5,322 | 5,325 |
| Total handshake bytes (`e2c_total_bytes`) | 8,206 | 10,412 | 11,279 |
| Constant within group? | yes, all 4 | yes, all 4 | yes, all 4 |

## Differences (PQC − classical), exact

| Quantity | ML-KEM-768 − X25519 | ML-KEM-1024 − X25519 | Measured or derived |
|---|---|---|---|
| Client key-share size | +1,152 | +1,536 | measured |
| Client Hello record length | +1,144 | +1,528 | measured |
| Server record length | +6 | +9 | measured |
| Total handshake bytes | +2,206 | +3,073 | measured |
| **Implied server side** = total diff − Client Hello diff | **+1,062** | **+1,545** | **derived, not measured** |

Notes on these numbers:

- **The implied server side is a subtraction, not a reading.** It puts every byte of the total change that isn't in the Client Hello record on the server side. If any other client record changed size, it would land here too. The dataset has no server key-share field (N/A everywhere), so this can't be checked directly.
- **The server record length is not the ML-KEM ciphertext.** It only moves by 6 or 9 bytes. As the audit found, it looks like a certificate-bearing record, not the Server Hello.
- **The Client Hello grows 8 bytes less than the key share** in both pairs (+1,144 vs +1,152; +1,528 vs +1,536). These fields can't show why. Without packet data, it isn't possible to tell which other part of the Client Hello is smaller in the PQC runs.
- **It isn't known whether `e2c_total_bytes` counts IP/TCP headers.** The dataset page doesn't say. This matters for analysis 04.
- **The +2,206 primary total was already seen in the audit.** So this is a descriptive number, not a blind test (see decisions_log.md).

## Handshake duration (`e6b_flow_duration_ms`)

Lab Docker network (172.18.x.x). These timings say nothing about real-world latency.

| Group | Median (ms) | Mean (ms) |
|---|---|---|
| X25519 | 4.920 | 5.376 |
| ML-KEM-768 | 4.946 | 5.354 |
| ML-KEM-1024 | 4.999 | 5.362 |

| Pair | Difference in median (ms), 95% CI | Difference in mean (ms), 95% CI |
|---|---|---|
| ML-KEM-768 − X25519 | +0.026 [−0.006, +0.055] | −0.022 [−0.107, +0.061] |
| ML-KEM-1024 − X25519 | +0.079 [+0.048, +0.108] | −0.014 [−0.095, +0.066] |

Percentile bootstrap, 10,000 resamples, seed 42. Each group was resampled on its own with replacement. A fresh generator (seed 42) was used for each pair.

In plain terms: for ML-KEM-768, the data can't tell the duration apart from X25519. For ML-KEM-1024, the median is about 0.08 ms higher and the CI excludes zero, but the mean shows no difference. Either way, the change is tiny next to a ~5 ms handshake, and it comes from a lab network.
