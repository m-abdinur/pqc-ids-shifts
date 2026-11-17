# =============================================================================
# Bytes-only C2 variant + port-443 share  (PQC-OAV calibration, Prompt 7 follow-up)
# Paste this whole file into ONE Colab cell and run it. Allow several hours.
#
# What it does
#   Part A (fast, no model fits): on the same rows the main run uses, counts
#     total rows, eligible rows (fwd AND bwd packets > 0) and the share of
#     eligible rows with Dst Port = 443, overall and per class.
#   Part B (slow): reruns the revised pipeline (A1 + B2) with three footprints:
#     hybrid768 and pure768 (same-run references, unchanged) and the new
#     pure768_bytes_only (0 added packets, no 1,460 max). Reports mean paired
#     C2 - C1 and C3 - C1 per model.
#
# What it does NOT change
#   The four .py modules are only imported, never edited. Seeds, splits,
#   per-class sampling, models, KS+Holm, MI selection and FPREP=hybrid768 are
#   the module defaults. The only edits are made in memory for this run:
#   VARIANTS (adds one footprint), MI_THRESHOLDS = [0.05] (skips the sweep;
#   the headline C3 uses MI 0.05 anyway), OUT_PREFIX, and a wrapper around
#   make_approx_mlkem that only acts when cfg has "floor_max_at_mean".
#
# Settings fixed before running: see decisions_log.md (2026-10-08 rows).
# =============================================================================
import sys, json, time, importlib
from pathlib import Path
import numpy as np
from google.colab import drive

drive.mount("/content/drive")
MY = Path("/content/drive/MyDrive")

# ---- 1. find the folder that holds all four modules (July 2026 versions) ----
CODE_DIR = None          # set to a Path to skip the search, e.g. MY / "some/folder"
NEEDED = {"pilot_option_b.py": 12020, "revised_pipeline_main_only.py": 16604,
          "pilot_pqc_stable_pipeline.py": 14402, "option_b_full_experiment.py": 12191}
if CODE_DIR is None:
    cands = []
    for p in MY.rglob("pilot_option_b.py"):
        d = p.parent
        if all((d / n).is_file() and (d / n).stat().st_size == s for n, s in NEEDED.items()):
            cands.append((max((d / n).stat().st_mtime for n in NEEDED), d))
    if not cands:
        raise SystemExit("No folder with all four modules found. Set CODE_DIR by hand.")
    cands.sort(reverse=True)
    print("Folders with all four modules (newest first):")
    for t, d in cands:
        print("  ", time.strftime("%Y-%m-%d %H:%M", time.gmtime(t)), d)
    CODE_DIR = cands[0][1]
print("Using CODE_DIR =", CODE_DIR)
sys.path.insert(0, str(CODE_DIR))

try:
    import xgboost  # noqa: F401
except ImportError:
    import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "-q", "xgboost"], check=True)

import pilot_pqc_stable_pipeline as base
import pilot_option_b as ob
import option_b_full_experiment as full
import revised_pipeline_main_only as rp

OUT = MY / "pqc_oav_calibration_bytes_only"
OUT.mkdir(exist_ok=True)

# ---- 2. bytes-only footprint (decisions_log.md, 2026-10-08) ----
BYTES_ONLY = {"d_fwd_bytes": 1152.0, "d_bwd_bytes": 1056.0,
              "d_fwd_pkts": 0.0, "d_bwd_pkts": 0.0,     # no added packets -> no header bytes
              "hdr_bytes": 40.0,                          # multiplied by 0 packets, so unused
              "added_pkt_size": 0.0,                      # max(M, 0) = M: no 1,460 max
              "floor_max_at_mean": True}

_orig_make = ob.make_approx_mlkem
def make_approx_mlkem_wrapped(X, cfg=ob.MLKEM):
    """Calls the operator unchanged. Only for cfg with floor_max_at_mean=True,
    raises each directional max to the new directional mean where the mean
    exceeds it (eligible rows only), and the overall max to the larger
    directional max. Keeps max >= mean once no packet is added."""
    floor = bool(cfg.get("floor_max_at_mean", False))
    Xa = _orig_make(X, {k: v for k, v in cfg.items() if k != "floor_max_at_mean"})
    if not floor:
        return Xa
    m = ((X["Tot Fwd Pkts"] > 0) & (X["Tot Bwd Pkts"] > 0)).values
    for mx, mean in (("Fwd Pkt Len Max", "Fwd Pkt Len Mean"), ("Bwd Pkt Len Max", "Bwd Pkt Len Mean")):
        if mx in Xa.columns and mean in Xa.columns:
            a = Xa[mx].values.astype("float64"); b = Xa[mean].values.astype("float64")
            Xa[mx] = np.where(m, np.maximum(a, b), a).astype("float32")
    if {"Pkt Len Max", "Fwd Pkt Len Max", "Bwd Pkt Len Max"} <= set(Xa.columns):
        a = Xa["Pkt Len Max"].values.astype("float64")
        dm = np.maximum(Xa["Fwd Pkt Len Max"].values, Xa["Bwd Pkt Len Max"].values).astype("float64")
        Xa["Pkt Len Max"] = np.where(m, np.maximum(a, dm), a).astype("float32")
    return Xa
ob.make_approx_mlkem = make_approx_mlkem_wrapped   # rp calls ob.make_approx_mlkem at run time

# ---- 3. Part A: port-443 share and eligibility check (no fitting) ----
data = base.load_and_clean(rp.FILES, base.LABEL_COL, base.LABEL_NORMALISE, base.CLASSES, rp.PER_CLASS, rp.MODEL_SEED)
X, y, feats, c2i = base.prepare_xy(data, base.NON_FEATURE_COLS, base.CLASSES)
i2c = {i: c for c, i in c2i.items()}
elig = ((X["Tot Fwd Pkts"] > 0) & (X["Tot Bwd Pkts"] > 0)).values
partA = {"rows": int(len(X)), "eligible": int(elig.sum()),
         "paper_rows": 119571, "paper_eligible": 74127, "dst_port_present": "Dst Port" in X.columns}
if partA["dst_port_present"]:
    p443 = (X["Dst Port"].values == 443)
    partA["eligible_443"] = int((elig & p443).sum())
    partA["eligible_443_share"] = float((elig & p443).sum() / elig.sum())
    partA["per_class"] = {}
    for k in sorted(set(np.asarray(y).tolist())):
        sel = elig & (np.asarray(y) == k)
        partA["per_class"][i2c[k]] = {"eligible": int(sel.sum()),
                                      "eligible_443": int((sel & p443).sum()),
                                      "share_443": float((sel & p443).sum() / max(sel.sum(), 1))}
Xb = make_approx_mlkem_wrapped(X, BYTES_ONLY)
partA["bytes_only_floor_raised_rows"] = {c: int(((Xb[c].values > X[c].values) & elig).sum())
                                        for c in ("Fwd Pkt Len Max", "Bwd Pkt Len Max", "Pkt Len Max") if c in X.columns}
partA["bytes_only_pkt_counts_unchanged"] = bool(np.array_equal(Xb["Tot Fwd Pkts"].values, X["Tot Fwd Pkts"].values)
                                                and np.array_equal(Xb["Tot Bwd Pkts"].values, X["Tot Bwd Pkts"].values))
del Xb, data
(OUT / "partA_port443.json").write_text(json.dumps(partA, indent=2))
print(json.dumps(partA, indent=2))

# ---- 4. Part B: rerun the revised pipeline with three footprints ----
rp.VARIANTS = {"hybrid768": rp.VARIANTS["hybrid768"],
               "pure768": rp.VARIANTS["pure768"],
               "pure768_bytes_only": BYTES_ONLY}
rp.MI_THRESHOLDS = [rp.MI_HEADLINE]
rp.OUT_PREFIX = str(OUT / "bytes_only_run")
assert rp.FPREP == "hybrid768" and rp.N_SPLITS == 10 and rp.MODEL_SEED == 42
summary = rp.run_revised(run_transfer=True)

# ---- 5. paired differences and the pre-set reading ----
PAPER = {"hybrid768": {"rf": -0.1147, "xgb": -0.0912, "mlp": -0.2514},
         "pure768":   {"rf": -0.1147, "xgb": -0.0913, "mlp": -0.2513}}
res = {"partA": partA, "models": {}}
for kind, cl in summary["classifiers"].items():
    raw = cl["raw_wf1"]; c1 = np.array(raw["C1"])
    row = {}
    for v in rp.VARIANTS:
        d2 = np.array(raw[v]["C2"]) - c1; d3 = np.array(raw[v]["C3"]) - c1
        row[v] = {"C2_minus_C1": [float(x) for x in full.ci95(d2)], "C3_minus_C1": [float(x) for x in full.ci95(d3)],
                  "per_split_C2_minus_C1": [float(x) for x in d2]}
    b, p = row["pure768_bytes_only"]["C2_minus_C1"][0], row["pure768"]["C2_minus_C1"][0]
    if abs(b - p) <= 0.01:
        reading = "does not depend on the packet/max assumptions"
    elif abs(b) < 0.5 * abs(p):
        reading = "comes mainly from the packet/max assumptions"
    else:
        reading = "partly depends on the packet/max assumptions"
    row["reading"] = reading
    row["paper_check"] = {v: {"paper": PAPER[v].get(kind), "this_run": row[v]["C2_minus_C1"][0],
                              "flag_gt_0.001": bool(PAPER[v].get(kind) is not None and
                                                     abs(row[v]["C2_minus_C1"][0] - PAPER[v][kind]) > 0.001)}
                          for v in ("hybrid768", "pure768")}
    res["models"][kind] = row
(OUT / "bytes_only_result.json").write_text(json.dumps(res, indent=2))

print("\n" + "=" * 70 + "\nRESULTS\n" + "=" * 70)
print(json.dumps(res, indent=2))
print("Saved in", OUT)
