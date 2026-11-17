"""Audit of CIC PQC-OAV-dataset-2025 (no shift analysis).

Run from the project folder:  python scripts/01_audit.py
Writes results/01_audit_tables.txt
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "01_audit_tables.txt"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)

feat = pd.read_csv(RAW / "ml_features_and_labels.csv")
man = pd.read_csv(RAW / "scenario_manifest.csv")
raw = pd.json_normalize([json.loads(l) for l in open(RAW / "raw_sessions.jsonl", encoding="utf-8")])
raw["ID"] = raw["scenario_id"].str.split("\\\\").str[1] + ".pcap"
feat["IDp"] = feat["ID"] + ".pcap"

lines = []
def out(title, obj):
    lines.append(f"\n## {title}\n{obj if isinstance(obj, str) else obj.to_string()}")

out("1. Shapes", f"features {feat.shape}, manifest {man.shape}, raw {raw.shape}")
out("1. raw_sessions columns", "\n".join(raw.columns))
out("1. feature columns", "\n".join(feat.columns))
out("1. ID linkage", "\n".join([
    f"raw IDs found in manifest: {raw.ID.isin(man.ID).sum()} / {len(raw)}",
    f"feature IDs found in manifest: {feat.IDp.isin(man.ID).sum()} / {len(feat)}",
    f"duplicate IDs: raw {raw.ID.duplicated().sum()}, features {feat.ID.duplicated().sum()}, manifest {man.ID.duplicated().sum()}",
]))

d = raw.merge(man, on="ID", suffixes=("", "_m"))
for c in ["client_size", "server_size", "client_record_len", "server_record_len"]:
    d[c] = pd.to_numeric(d[f"e2_struct_ke.{c}"], errors="coerce")

out("2. sessions by key-exchange group (raw e1_alg_suite)", d.e1_alg_suite.value_counts(dropna=False))
out("2. sessions by scenario x subtype x class x label x KEM",
    d.groupby(["scenario_id_m", "subtype", "operational_class", "anomaly_label", "e1_alg_suite"]).size())
out("2. outcome by scenario", d.groupby(["scenario_id_m", "e9_conn_outcome"]).size().unstack(fill_value=0))

out("3. client key-share size (bytes) by KEM group", d.groupby("e1_alg_suite")["client_size"].agg(["min", "median", "max", "nunique"]))
out("3. server key-share size: non-missing values", str(d["server_size"].notna().sum()))

cols = ["client_size", "client_record_len", "server_record_len", "e2c_total_bytes", "e6b_flow_duration_ms"]
g = d.groupby("scenario_id_m")[cols]
out("4. per-scenario medians", g.median().round(2))
out("4. per-scenario number of distinct values", g.nunique())
out("4. per-scenario min / max total bytes", d.groupby("scenario_id_m")["e2c_total_bytes"].agg(["min", "max"]))
out("4. per-scenario duration quantiles (ms)", d.groupby("scenario_id_m")["e6b_flow_duration_ms"].quantile([.05, .5, .95]).unstack().round(2))

out("6. missing values per raw column (as NaN or 'N/A')",
    pd.Series({c: int(d[c].isna().sum() + (d[c].astype(str) == "N/A").sum()) for c in raw.columns}))
fx = feat.merge(d[["ID", "e2c_total_bytes", "client_size"]], left_on="IDp", right_on="ID", suffixes=("", "_raw"))
out("6. features vs raw agreement", "\n".join([
    f"e2c_total_bytes equal: {(fx.e2c_total_bytes == fx.e2c_total_bytes_raw).mean():.4f}",
    f"e2_client_size equal: {(pd.to_numeric(fx.e2_client_size, errors='coerce') == fx.client_size).mean():.4f}",
]))
out("6. exact duplicate rows in feature matrix (excluding ID/split)",
    str(feat.drop(columns=["ID", "IDp", "split"]).duplicated().sum()))

OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT}")
