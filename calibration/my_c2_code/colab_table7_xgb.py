# =============================================================================
# Table 7 rerun: XGBoost C2 - C1 under xgboost 3.2.0 for all six footprint x
# scope cells. RF is refitted as a sanity check of the operators only.
# Paste this whole file into ONE cell of a FRESH Colab runtime. About 30-45 min.
# Rules fixed before running: decisions_log.md, 2026-10-10 "Table 7 XGBoost rerun" rows.
#
# Why one fit per split is enough: the footprints change only the test data,
# and B4 showed separately fitted XGBoost models give identical predictions
# under a fixed version (10/10 splits). RF is deterministic as well.
# The operators are the same code as results/07 and 08 (the make_approx_mlkem
# plus the bytes-only and port-443 wrapper from colab_port443_only.py).
# The four .py modules are imported unchanged.
# =============================================================================
import sys, os, json, time, platform, subprocess
from pathlib import Path

XGB_VERSION = "3.2.0"           # Table S15
if "xgboost" in sys.modules:
    _loaded = getattr(sys.modules["xgboost"], "__version__", None)
    if _loaded != XGB_VERSION:
        raise SystemExit(f"xgboost {_loaded} is already imported. Use Runtime > Restart session, then run this cell again.")
    print(f"xgboost {_loaded} already imported in this session; continuing.")
try:
    from importlib.metadata import version as _v
    have = _v("xgboost")
except Exception:
    have = None
if have != XGB_VERSION:
    print(f"Installing xgboost=={XGB_VERSION} (found {have}) ...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"xgboost=={XGB_VERSION}"])
    if r.returncode != 0:
        raise SystemExit(f"Could not install xgboost=={XGB_VERSION}.")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from google.colab import drive

drive.mount("/content/drive")
MY = Path("/content/drive/MyDrive")

CODE_DIR = None          # set to a Path to skip the search
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
    CODE_DIR = cands[0][1]
print("Using CODE_DIR =", CODE_DIR)
sys.path.insert(0, str(CODE_DIR))

import xgboost, sklearn, scipy, pandas
import pilot_pqc_stable_pipeline as base
import pilot_option_b as ob
import option_b_full_experiment as full
import revised_pipeline_main_only as rp

OUT = MY / "pqc_oav_calibration_table7_xgb"
OUT.mkdir(exist_ok=True)
env = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pandas.__version__,
       "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
       "cpu_count": os.cpu_count(), "code_dir": str(CODE_DIR)}
print(json.dumps(env, indent=2))
assert xgboost.__version__ == XGB_VERSION and full.HAVE_XGB
assert rp.N_SPLITS == 10 and rp.MODEL_SEED == 42 and rp.TEST_SIZE == 0.30 and rp.PER_CLASS == 30000

# ---- operators: identical to colab_port443_only.py (results/08) ----
H768 = dict(rp.VARIANTS["hybrid768"]); P768 = dict(rp.VARIANTS["pure768"])
BYTES_ONLY = {"d_fwd_bytes": 1152.0, "d_bwd_bytes": 1056.0, "d_fwd_pkts": 0.0, "d_bwd_pkts": 0.0,
              "hdr_bytes": 40.0, "added_pkt_size": 0.0, "floor_max_at_mean": True}
VARIANTS = {"hybrid_all": H768, "pure_all": P768, "bytes_all": BYTES_ONLY,
            "hybrid_443": {**H768, "only_dst_port": 443}, "pure_443": {**P768, "only_dst_port": 443},
            "bytes_443": {**BYTES_ONLY, "only_dst_port": 443}}
EXTRA = ("floor_max_at_mean", "only_dst_port")
_orig_make = ob.make_approx_mlkem
def make_approx_mlkem_wrapped(X, cfg):
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

# ---- references (results/07, 08, 09) ----
RF_REF = {"hybrid_all": -0.11467699810827794, "pure_all": -0.11467955895038422, "bytes_all": -0.031434466019075256,
          "hybrid_443": -0.001053909042977097, "pure_443": -0.001047858409272906, "bytes_443": -0.0002288245967603908}
XGB_REF_09 = {"hybrid_all": -0.0911691433179719, "pure_all": -0.09132443565826528, "hybrid_443": -0.0017662508853624748}
XGB_OLD_07_08 = {"hybrid_all": -0.08971832843423763, "pure_all": -0.08897122960734032, "bytes_all": -0.05679829485225933,
                 "hybrid_443": -0.0017756862788117834, "pure_443": -0.0017737903186603842, "bytes_443": -0.0010004878589127907}

t0 = time.time()
data = base.load_and_clean(rp.FILES, base.LABEL_COL, base.LABEL_NORMALISE, base.CLASSES, rp.PER_CLASS, rp.MODEL_SEED)
X, y, feats, c2i = base.prepare_xy(data, base.NON_FEATURE_COLS, base.CLASSES)
del data
print(f"rows={len(X)} features={len(feats)}  ({time.time()-t0:.0f}s)")
assert len(X) == 119571 and len(feats) == 70, "data differ from the paper (119,571 rows, 70 features)"

def wf1(yt, yp): return float(f1_score(yt, yp, average="weighted"))   # same call as base.evaluate
acc = {k: {v: [] for v in VARIANTS} for k in ("xgb", "rf")}
for s in range(rp.N_SPLITS):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=rp.TEST_SIZE, random_state=s, stratify=y)
    tests = {v: make_approx_mlkem_wrapped(Xte, cfg)[feats].values for v, cfg in VARIANTS.items()}
    for kind in ("xgb", "rf"):
        m = full.make_model(kind).fit(Xtr[feats].values, ytr)          # module settings, n_jobs as in the module
        c1 = wf1(yte, m.predict(Xte[feats].values))
        for v, Z in tests.items():
            acc[kind][v].append(wf1(yte, m.predict(Z)) - c1)
        print(f"  split {s} {kind:>3}: " + "  ".join(f"{v} {acc[kind][v][-1]:+.4f}" for v in VARIANTS)
              + f"  ({time.time()-t0:.0f}s)")

def ci(a):
    m_, lo, hi = full.ci95(a); return [float(m_), float(lo), float(hi)]
res = {"env": env, "xgb": {}, "rf_sanity": {}, "checks": {}}
for v in VARIANTS:
    res["xgb"][v] = {"C2_minus_C1": ci(acc["xgb"][v]), "per_split": acc["xgb"][v],
                     "old_07_08": XGB_OLD_07_08[v], "change_vs_old": float(np.mean(acc["xgb"][v]) - XGB_OLD_07_08[v])}
    res["rf_sanity"][v] = {"C2_minus_C1": float(np.mean(acc["rf"][v])), "ref_07_08": RF_REF[v],
                           "gap": float(np.mean(acc["rf"][v]) - RF_REF[v])}
res["checks"]["rf_all_within_0.0001"] = bool(all(abs(r["gap"]) <= 1e-4 for r in res["rf_sanity"].values()))
res["checks"]["xgb_vs_09_within_0.0001"] = {v: bool(abs(np.mean(acc["xgb"][v]) - XGB_REF_09[v]) <= 1e-4) for v in XGB_REF_09}

b, p = res["xgb"]["bytes_all"]["C2_minus_C1"][0], res["xgb"]["pure_all"]["C2_minus_C1"][0]
res["xgb_reading"] = {
    "bytes_over_pure": float(b / p), "share_depending_on_packet_max": float(1 - b / p),
    "bytes_only_rule": ("does not depend on the packet/max assumptions" if abs(b - p) <= 0.01 else
                        "comes mainly from the packet/max assumptions" if abs(b) < 0.5 * abs(p) else
                        "partly depends on the packet/max assumptions"),
    "port443_rule": {v: ("no material effect (|C2-C1| < 0.01)" if abs(res["xgb"][v]["C2_minus_C1"][0]) < 0.01
                         else "C2 - C1 = %+.4f" % res["xgb"][v]["C2_minus_C1"][0]) for v in ("hybrid_443", "pure_443", "bytes_443")}}
(OUT / "table7_xgb_result.json").write_text(json.dumps(res, indent=2))
print("\n" + "=" * 70 + "\nRESULTS\n" + "=" * 70)
print(json.dumps(res, indent=2))
print(f"Saved in {OUT}  (total {time.time()-t0:.0f}s)")
