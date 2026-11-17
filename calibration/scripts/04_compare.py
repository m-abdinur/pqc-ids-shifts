"""Compare the C2 byte footprints with the real PQC-OAV differences.

Run from anywhere:  python scripts/04_compare.py
Reads   results/03_real_shifts.csv, data/raw/raw_sessions.jsonl
Writes  results/04_comparison_numbers.csv  (every number used in results/04_comparison.md)
        prints all tables to the console

No test, no TOST, no margin (decisions_log.md, 2026-10-08). Byte comparison is descriptive.
Percent = (C2 value - real value) / real value * 100. Positive = the C2 footprint is larger.

ML-KEM sizes (bytes): encapsulation key 768: 1184, 1024: 1568; ciphertext 768: 1088, 1024: 1568.
Source: FIPS 203 (Aug 2024), Table 3 "Sizes (in bytes) of keys and ciphertexts of ML-KEM", p. 39,
read from the local PDF 'Module-Lattice-BasedKeyEncapsulation Mechanism Standard.pdf' on 2026-10-08.
Also matches draft-ietf-tls-mlkem-09 Section 3 and Go crypto/mlkem constants.
PQC-OAV client key shares (1184, 1568) independently match the encapsulation key sizes.
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "04_comparison_numbers.csv"
X25519 = 32
EK = {"768": 1184, "1024": 1568}
CT = {"768": 1088, "1024": 1568}

r3 = pd.read_csv(RES / "03_real_shifts.csv")
dif = r3[r3.section == "difference"].copy()
dif["kem"] = dif["pair"].str.extract(r"mlkem(\d+)_rsa")[0]

def real(kem, q, stat=None):
    s = dif[(dif.kem == kem) & (dif.quantity == q)]
    if stat:
        s = s[s.statistic == stat]
    assert len(s) == 1, (kem, q, stat)
    return s.iloc[0]

FOOT = {
    "main hybrid ML-KEM-768 (+1,184/+1,088)": ("768", 1184, 1088),
    "pure ML-KEM-768 (+1,152/+1,056)": ("768", EK["768"] - X25519, CT["768"] - X25519),
    "pure ML-KEM-1024 (+1,536/+1,536)": ("1024", EK["1024"] - X25519, CT["1024"] - X25519),
}
assert FOOT["pure ML-KEM-768 (+1,152/+1,056)"][1:] == (1152, 1056)
assert FOOT["pure ML-KEM-1024 (+1,536/+1,536)"][1:] == (1536, 1536)

rows = []
def add(block, **kw):
    rows.append({"block": block, **kw})

# ---------- 1/2. byte mapping ----------
for name, (kem, fwd, bwd) in FOOT.items():
    ch = real(kem, "client_hello_record_len").value
    ks = real(kem, "client_key_share_bytes").value
    srv = real(kem, "implied_server_side_bytes").value
    tot = real(kem, "total_handshake_bytes").value
    for feat, mine, rv, rsrc in [
        ("Fwd bytes (TotLen Fwd Pkts, Subflow Fwd Byts)", fwd, ch, "Client Hello record diff (measured)"),
        ("Fwd bytes (TotLen Fwd Pkts, Subflow Fwd Byts)", fwd, ks, "client key-share diff (measured)"),
        ("Bwd bytes (TotLen Bwd Pkts, Subflow Bwd Byts)", bwd, srv, "implied server side (DERIVED)"),
        ("Fwd + Bwd total", fwd + bwd, tot, "e2c_total_bytes diff (measured)"),
    ]:
        add("bytes", footprint=name, kem=kem, feature=feat, real_source=rsrc, mine=mine, real=rv,
            diff_bytes=mine - rv, diff_pct=round((mine - rv) / rv * 100, 2))

# ---------- 1. header check: does e2c_total_bytes include per-packet headers? ----------
# Short, failed/incomplete scenarios have only 1-2 TLS records, so the total can be rebuilt.
# Model tested: total = sum over records of (record length + 5-byte TLS record header + H),
# with H = 66 = Ethernet 14 + IPv4 20 + TCP 20 + 12 (timestamps option). This is an inference.
raw = pd.json_normalize([json.loads(l) for l in open(ROOT / "data/raw/raw_sessions.jsonl", encoding="utf-8")])
raw["scen"] = raw["scenario_id"].str.split("\\\\").str[1].str.replace(r"_run\d+$", "", regex=True)
for c in ["client_record_len", "server_record_len"]:
    raw[c] = pd.to_numeric(raw[f"e2_struct_ke.{c}"], errors="coerce")
SHORT = ["anomaly_kem_mismatch", "raf_alg_mismatch", "anomaly_pqc_over_tls12", "threat_data_exfil",
         "model_gray_rsa_mimicry", "raf_robustness_malformed_keyshare"]
H = 66
for s in SHORT:
    g = raw[raw.scen == s].groupby(["client_record_len", "server_record_len", "e2c_total_bytes", "e9_conn_outcome"],
                                   dropna=False).size().reset_index(name="n")
    for _, x in g.iterrows():
        tls_bytes = x.client_record_len + 5 + (0 if pd.isna(x.server_record_len) else x.server_record_len + 5)
        n_rec = 1 + (0 if pd.isna(x.server_record_len) else 1)
        resid = x.e2c_total_bytes - tls_bytes
        add("header_check", scenario=s, outcome=x.e9_conn_outcome, n=int(x.n),
            client_record_len=x.client_record_len, server_record_len=x.server_record_len,
            total=x.e2c_total_bytes, tls_bytes_incl_5B_headers=tls_bytes, records=n_rec,
            residual=resid, predicted_residual=n_rec * H, fits_66_per_record=resid == n_rec * H)

# Normal pairs: total diff - Client Hello diff - (ciphertext - 32) - server record diff
for kem in ["768", "1024"]:
    left = (real(kem, "total_handshake_bytes").value - real(kem, "client_hello_record_len").value
            - (CT[kem] - X25519) - real(kem, "server_record_len").value)
    add("closure", kem=kem, total_diff=real(kem, "total_handshake_bytes").value,
        client_hello_diff=real(kem, "client_hello_record_len").value, ciphertext_minus_x25519=CT[kem] - X25519,
        server_record_diff=real(kem, "server_record_len").value, leftover=left)

# ---------- 3. duration ----------
for kem in ["768", "1024"]:
    for stat in ["difference in median", "difference in mean"]:
        x = real(kem, "handshake_duration_ms", stat)
        add("duration", kem=kem, statistic=stat, real=x.value, ci_low=x.ci_low, ci_high=x.ci_high,
            zero_in_ci=x.ci_low <= 0 <= x.ci_high, point1ms_in_ci=x.ci_low <= 0.1 <= x.ci_high)

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
for b in out.block.unique():
    print(f"\n## {b}\n", out[out.block == b].dropna(axis=1, how="all").drop(columns="block").to_string(index=False))
print(f"\nWrote {OUT}")
