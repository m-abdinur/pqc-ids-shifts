"""Real classical vs ML-KEM handshake shifts in CIC PQC-OAV-dataset-2025.

Run from anywhere:  python scripts/03_real_shifts.py
Writes results/03_real_shifts.csv  (all numbers)
       results/03_real_shifts_log.txt  (filter counts, distinct-value checks, printed tables)

Decisions used (decisions_log.md, 2026-10-08):
- Groups: normal, successful sessions of normal_X25519_rsa_cert (classical),
  normal_mlkem768_rsa_cert (primary PQC), normal_mlkem1024_rsa_cert (secondary PQC).
- Byte quantities: exact values, no TOST, no CI.
- Duration only: 95% percentile bootstrap CI, 10,000 resamples, seed 42.
  Each group is resampled independently with replacement (same size as the group).
  A fresh numpy Generator(seed=42) is used for each pair, so each pair is reproducible on its own.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT_CSV = ROOT / "results" / "03_real_shifts.csv"
OUT_LOG = ROOT / "results" / "03_real_shifts_log.txt"

CLASSICAL = "normal_X25519_rsa_cert"
PQC = {"primary": "normal_mlkem768_rsa_cert", "secondary": "normal_mlkem1024_rsa_cert"}
GROUPS = [CLASSICAL, PQC["primary"], PQC["secondary"]]
B, SEED, CHUNK = 10_000, 42, 500

# quantity name -> source column
BYTE_Q = {
    "client_key_share_bytes": "client_size",
    "client_hello_record_len": "client_record_len",
    "server_record_len": "server_record_len",
    "total_handshake_bytes": "e2c_total_bytes",
}
DUR = "e6b_flow_duration_ms"

log = []
def say(s=""):
    print(s); log.append(str(s))

# ---------- load and link ----------
raw = pd.json_normalize([json.loads(l) for l in open(RAW / "raw_sessions.jsonl", encoding="utf-8")])
raw["ID"] = raw["scenario_id"].str.split("\\\\").str[1] + ".pcap"
man = pd.read_csv(RAW / "scenario_manifest.csv")
d = raw.merge(man, on="ID", suffixes=("", "_m"), validate="one_to_one")
for c in ["client_size", "client_record_len", "server_record_len"]:
    d[c] = pd.to_numeric(d[f"e2_struct_ke.{c}"], errors="coerce")
d["e2c_total_bytes"] = pd.to_numeric(d["e2c_total_bytes"], errors="coerce")
d[DUR] = pd.to_numeric(d[DUR], errors="coerce")

# ---------- filter ----------
say("## Filtering")
in_groups = d[d["scenario_id_m"].isin(GROUPS)]
say(f"rows in the 3 scenarios: {len(in_groups)}")
say(in_groups.groupby(["scenario_id_m", "anomaly_label", "e9_conn_outcome", "e1_alg_suite", "e8_proto_context"]).size().to_string())
sel = in_groups[(in_groups["anomaly_label"] == "Normal") & (in_groups["e9_conn_outcome"] == "Success")].copy()
say(f"rows kept (Normal & Success): {len(sel)}")
missing = sel[list(BYTE_Q.values()) + [DUR]].isna().sum()
say("missing values in kept rows:\n" + missing.to_string())
assert missing.sum() == 0, "unexpected missing values"

rows = []  # long-format output
def add(**kw):
    rows.append(kw)

# ---------- per-group values ----------
say("\n## Per-group values (constant check)")
group_val = {}
for g in GROUPS:
    s = sel[sel["scenario_id_m"] == g]
    n = len(s)
    add(section="group", group=g, pair="", quantity="n_sessions", statistic="count", value=n,
        ci_low="", ci_high="", n_distinct="", constant="", distinct_values="", note="")
    for q, col in BYTE_Q.items():
        vc = s[col].value_counts().sort_index()
        const = len(vc) == 1
        v = float(vc.index[0]) if const else float(s[col].median())
        group_val[(g, q)] = v
        dv = "; ".join(f"{int(k)} (n={c})" for k, c in vc.items())
        add(section="group", group=g, pair="", quantity=q,
            statistic="value" if const else "median", value=v, ci_low="", ci_high="",
            n_distinct=len(vc), constant=const, distinct_values=dv, note="")
        say(f"{g} | {q}: n={n}, distinct={len(vc)} -> {dv}")
    for stat, fn in [("median", np.median), ("mean", np.mean)]:
        v = float(fn(s[DUR].to_numpy()))
        group_val[(g, f"dur_{stat}")] = v
        add(section="group", group=g, pair="", quantity="handshake_duration_ms", statistic=stat, value=v,
            ci_low="", ci_high="", n_distinct=s[DUR].nunique(), constant=False, distinct_values="",
            note="lab Docker network (172.18.x.x)")
    say(f"{g} | duration ms: median={group_val[(g,'dur_median')]:.4f}, mean={group_val[(g,'dur_mean')]:.4f}, "
        f"distinct={s[DUR].nunique()}")

# ---------- differences ----------
def boot_diff(x_cls, x_pqc, rng):
    """percentile bootstrap of (PQC - classical) median and mean differences."""
    meds, means = np.empty(B), np.empty(B)
    for i in range(0, B, CHUNK):
        k = min(CHUNK, B - i)
        a = x_cls[rng.integers(0, len(x_cls), size=(k, len(x_cls)))]
        b = x_pqc[rng.integers(0, len(x_pqc), size=(k, len(x_pqc)))]
        meds[i:i + k] = np.median(b, axis=1) - np.median(a, axis=1)
        means[i:i + k] = b.mean(axis=1) - a.mean(axis=1)
    return meds, means

say("\n## PQC - classical differences")
for label, g in PQC.items():
    pair = f"{label}: {g} - {CLASSICAL}"
    diffs = {}
    for q in BYTE_Q:
        diffs[q] = group_val[(g, q)] - group_val[(CLASSICAL, q)]
        add(section="difference", group="", pair=pair, quantity=q, statistic="exact difference",
            value=diffs[q], ci_low="", ci_high="", n_distinct="", constant="", distinct_values="",
            note="measured (field difference)")
        say(f"{pair} | {q}: {diffs[q]:+.0f}")
    implied = diffs["total_handshake_bytes"] - diffs["client_hello_record_len"]
    add(section="difference", group="", pair=pair, quantity="implied_server_side_bytes",
        statistic="derived difference", value=implied, ci_low="", ci_high="", n_distinct="", constant="",
        distinct_values="", note="DERIVED, not measured: total-bytes diff minus Client Hello record diff")
    say(f"{pair} | implied server-side (DERIVED): {implied:+.0f}")

    x_cls = sel.loc[sel["scenario_id_m"] == CLASSICAL, DUR].to_numpy(dtype=float)
    x_pqc = sel.loc[sel["scenario_id_m"] == g, DUR].to_numpy(dtype=float)
    meds, means = boot_diff(x_cls, x_pqc, np.random.default_rng(SEED))
    for stat, point, bs in [("median", np.median(x_pqc) - np.median(x_cls), meds),
                            ("mean", x_pqc.mean() - x_cls.mean(), means)]:
        lo, hi = np.percentile(bs, [2.5, 97.5])
        add(section="difference", group="", pair=pair, quantity="handshake_duration_ms",
            statistic=f"difference in {stat}", value=float(point), ci_low=float(lo), ci_high=float(hi),
            n_distinct="", constant="", distinct_values="",
            note=f"95% percentile bootstrap, B={B}, seed={SEED}; lab Docker network")
        say(f"{pair} | duration diff in {stat}: {point:+.4f} ms, 95% CI [{lo:+.4f}, {hi:+.4f}]")

OUT_CSV.parent.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
OUT_LOG.write_text("\n".join(log), encoding="utf-8")
print(f"\nWrote {OUT_CSV}\nWrote {OUT_LOG}")
