# =============================================================================
# Port-443-only C2 variants  (PQC-OAV calibration, follow-up to the bytes-only run)
# Paste this whole file into ONE Colab cell and run it. Allow about 2.5-3 hours.
#
# What it does
#   Check (fast, no fitting): on the full data, counts rows each port-443
#     variant changes, per class. Stops if any DoS/Bot row changes or the total
#     exceeds the 6,576 eligible port-443 flows.
#   Run (slow): the revised pipeline (A1 + B2) with four footprints:
#     hybrid768 (all flows; FPREP + same-run reference, unchanged) and
#     hybrid768_443, pure768_443, bytes_only_443 (operator applied only to
#     eligible flows with Dst Port = 443).
#   Reports mean paired C2 - C1 and C3 - C1 (weighted F1) and the mean
#     per-class F1 change, per model and variant.
#
# The four .py modules are imported unchanged. In-memory changes for this
# run only: VARIANTS, MI_THRESHOLDS=[0.05], OUT_PREFIX, and a wrapper around
# make_approx_mlkem that only acts when cfg has "only_dst_port" or
# "floor_max_at_mean". Settings fixed in decisions_log.md (2026-10-08).
# =============================================================================
import sys, json, time
from pathlib import Path
import numpy as np
from google.colab import drive

drive.mount("/content/drive")
MY = Path("/content/drive/MyDrive")

# Same folder the bytes-only run used. Change only if the folder was moved.
CODE_DIR = MY / "pqc_ids_dissertation/final_experiment_package_v6/01_timing_free/code"
for n in ("pilot_option_b.py", "revised_pipeline_main_only.py",
          "pilot_pqc_stable_pipeline.py", "option_b_full_experiment.py"):
    if not (CODE_DIR / n).is_file():
        raise SystemExit(f"Missing {n} in {CODE_DIR}. Set CODE_DIR by hand.")
sys.path.insert(0, str(CODE_DIR))
try:
    import xgboost  # noqa: F401
except ImportError:
    import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "-q", "xgboost"], check=True)

import pilot_pqc_stable_pipeline as base
import pilot_option_b as ob
import option_b_full_experiment as full
import revised_pipeline_main_only as rp

OUT = MY / "pqc_oav_calibration_port443"
OUT.mkdir(exist_ok=True)

H768 = dict(rp.VARIANTS["hybrid768"]); P768 = dict(rp.VARIANTS["pure768"])
BYTES_ONLY = {"d_fwd_bytes": 1152.0, "d_bwd_bytes": 1056.0, "d_fwd_pkts": 0.0, "d_bwd_pkts": 0.0,
              "hdr_bytes": 40.0, "added_pkt_size": 0.0, "floor_max_at_mean": True}
VARS_443 = {"hybrid768_443": {**H768, "only_dst_port": 443},
            "pure768_443": {**P768, "only_dst_port": 443},
            "bytes_only_443": {**BYTES_ONLY, "only_dst_port": 443}}
EXTRA = ("floor_max_at_mean", "only_dst_port")

_orig_make = ob.make_approx_mlkem
def make_approx_mlkem_wrapped(X, cfg=ob.MLKEM):
    """The operator unchanged; then (a) if only_dst_port is set, every row that is
    not on that Dst Port is put back to its original values; (b) if
    floor_max_at_mean, directional maxima are raised to the new directional mean
    where it exceeds them (eligible rows in scope only), overall max to the larger
    directional max."""
    Xa = _orig_make(X, {k: v for k, v in cfg.items() if k not in EXTRA})
    scope = ((X["Tot Fwd Pkts"] > 0) & (X["Tot Bwd Pkts"] > 0)).values
    port = cfg.get("only_dst_port")
    if port is not None:
        on_port = (X["Dst Port"].values == port)
        for c in Xa.columns:
            Xa[c] = np.where(on_port, Xa[c].values, X[c].values).astype(Xa[c].dtype)
        scope = scope & on_port
    if cfg.get("floor_max_at_mean", False):
        for mx, mean in (("Fwd Pkt Len Max", "Fwd Pkt Len Mean"), ("Bwd Pkt Len Max", "Bwd Pkt Len Mean")):
            a = Xa[mx].values.astype("float64"); b = Xa[mean].values.astype("float64")
            Xa[mx] = np.where(scope, np.maximum(a, b), a).astype("float32")
        a = Xa["Pkt Len Max"].values.astype("float64")
        dm = np.maximum(Xa["Fwd Pkt Len Max"].values, Xa["Bwd Pkt Len Max"].values).astype("float64")
        Xa["Pkt Len Max"] = np.where(scope, np.maximum(a, dm), a).astype("float32")
    return Xa
ob.make_approx_mlkem = make_approx_mlkem_wrapped

# ---- sanity check on the full data (no fitting) ----
data = base.load_and_clean(rp.FILES, base.LABEL_COL, base.LABEL_NORMALISE, base.CLASSES, rp.PER_CLASS, rp.MODEL_SEED)
X, y, feats, c2i = base.prepare_xy(data, base.NON_FEATURE_COLS, base.CLASSES)
i2c = {i: c for c, i in c2i.items()}; ya = np.asarray(y)
elig443 = int((((X["Tot Fwd Pkts"] > 0) & (X["Tot Bwd Pkts"] > 0)).values & (X["Dst Port"].values == 443)).sum())
check = {"eligible_443": elig443, "changed_rows": {}}
for v, cfg in VARS_443.items():
    Xv = make_approx_mlkem_wrapped(X, cfg)
    changed = np.zeros(len(X), bool)
    for c in X.columns:
        a, b = X[c].values, Xv[c].values
        changed |= ~((a == b) | (np.isnan(a) & np.isnan(b)))
    per = {i2c[k]: int((changed & (ya == k)).sum()) for k in sorted(set(ya.tolist()))}
    check["changed_rows"][v] = {"total": int(changed.sum()), "per_class": per}
    del Xv
del data
print(json.dumps(check, indent=2))
for v, r in check["changed_rows"].items():
    if r["per_class"].get("DoS", 0) or r["per_class"].get("Bot", 0) or r["total"] > elig443:
        raise SystemExit(f"Sanity check failed for {v}. Nothing was fitted.")
(OUT / "sanity_check.json").write_text(json.dumps(check, indent=2))
print("Sanity check passed.")

# ---- run the pipeline ----
rp.VARIANTS = {"hybrid768": H768, **VARS_443}
rp.MI_THRESHOLDS = [rp.MI_HEADLINE]
rp.OUT_PREFIX = str(OUT / "port443_run")
assert rp.FPREP == "hybrid768" and rp.N_SPLITS == 10 and rp.MODEL_SEED == 42
summary = rp.run_revised(run_transfer=True)

# ---- paired differences + per-class F1 change ----
raw = json.loads(Path(rp.OUT_PREFIX + "_raw.json").read_text())
seeds = sorted(raw, key=int)
res = {"sanity_check": check, "models": {}}
for kind, cl in summary["classifiers"].items():
    rw = cl["raw_wf1"]; c1 = np.array(rw["C1"]); row = {}
    for v in rp.VARIANTS:
        d2 = np.array(rw[v]["C2"]) - c1; d3 = np.array(rw[v]["C3"]) - c1
        m2 = float(np.mean(d2))
        pc = {}
        for cls in base.CLASSES:
            diffs = [raw[s]["results"][kind]["per_variant"][v]["C2"]["per_class_f1"][cls]
                     - raw[s]["results"][kind]["C1"]["per_class_f1"][cls] for s in seeds]
            pc[cls] = float(np.mean(diffs))
        row[v] = {"C2_minus_C1": [float(x) for x in full.ci95(d2)],
                  "C3_minus_C1": [float(x) for x in full.ci95(d3)],
                  "per_class_F1_C2_minus_C1": pc,
                  "reading": ("no material effect on weighted F1 (|C2-C1| < 0.01)" if abs(m2) < 0.01
                              else f"C2 - C1 = {m2:+.4f}")}
    res["models"][kind] = row
(OUT / "port443_result.json").write_text(json.dumps(res, indent=2))

print("\n" + "=" * 70 + "\nRESULTS\n" + "=" * 70)
print(json.dumps(res, indent=2))
print("Saved in", OUT)
