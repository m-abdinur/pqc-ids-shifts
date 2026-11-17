# =============================================================================
# B1 (loss on shifted flows only) + B4 (XGBoost reproducibility)
# PQC-OAV calibration, follow-up to the mock Q1 review of manuscript v3.
# Paste this whole file into ONE cell of a FRESH Colab runtime and run it.
# Allow about 1.5-2.5 hours. Rules for reading the output are fixed in
# decisions_log.md (2026-10-10 rows) before this is run.
#
# B4 (runs first, XGBoost only)
#   Pins xgboost==3.2.0 (Table S15), records versions and CPU count, then fits
#   XGBoost on the main hybrid and pure footprints with the paper's protocol
#   (separate C1 and C2 fits, n_jobs=-1), and again with n_jobs=1.
#   Reports mean C2 - C1 against the paper (-0.0912 hybrid, -0.0913 pure)
#   and whether separately fitted models give identical predictions.
#
# B1 (RF, XGBoost, MLP; hybrid footprint only)
#   One fit per split and model on original training data (the C1/C2 model;
#   Additional file 1, A1.1 shows one fit reproduces the stored scores).
#   Predicts the original test set (C1) and the hybrid-transformed test set
#   (C2). Because the operator works row by row, the port-443 shift is the
#   full shift restricted to port-443 rows, so both scopes come from the same
#   predictions. Reports weighted F1 C2 - C1 on these test subsets:
#     ALL            every test row (sanity check vs results/07 and 08)
#     ALL_443scope   every test row, port-443 scope (sanity check vs 08)
#     E_all          eligible (shifted) test rows
#     E_443          eligible test rows with Dst Port = 443
#     E_non443       eligible test rows not on port 443
#     BI_non443      eligible Benign + Infiltration test rows not on port 443
#   plus the share of rows in each subset whose predicted class changed.
#
# The four .py modules are imported unchanged. Nothing is written to them.
# =============================================================================
import sys, os, json, time, platform, subprocess
from pathlib import Path

XGB_VERSION = "3.2.0"           # Table S15
RUN_B4, RUN_B1 = True, True

# ---- 0. pin xgboost before anything imports it ----
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from google.colab import drive

drive.mount("/content/drive")
MY = Path("/content/drive/MyDrive")

# ---- 1. find the folder with the four July 2026 modules (same rule as before) ----
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

OUT = MY / "pqc_oav_calibration_B1_B4"
OUT.mkdir(exist_ok=True)

env = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pandas.__version__,
       "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
       "cpu_count": os.cpu_count(), "code_dir": str(CODE_DIR)}
print(json.dumps(env, indent=2))
assert xgboost.__version__ == XGB_VERSION, "xgboost version not pinned"
if sklearn.__version__ != "1.6.1":
    print(f"NOTE: scikit-learn {sklearn.__version__} differs from Table S15 (1.6.1). Not pinned; RF/MLP sanity checks below show whether it matters.")
assert full.HAVE_XGB, "option_b_full_experiment could not import XGBClassifier"
assert rp.N_SPLITS == 10 and rp.MODEL_SEED == 42 and rp.TEST_SIZE == 0.30 and rp.PER_CLASS == 30000

H768 = dict(rp.VARIANTS["hybrid768"]); P768 = dict(rp.VARIANTS["pure768"])
PAPER = {"hybrid768": -0.0912, "pure768": -0.0913}                    # Table 4 / Table S14, XGBoost
SAME_RUN_07_08 = {"rf": (-0.11467699810827794, -0.001053909042977097),  # results/07, 08 (hybrid all, hybrid 443)
                  "xgb": (-0.08971832843423763, -0.0017756862788117834),
                  "mlp": (-0.25139635711327807, -0.0028545998651784534)}

# ---- 2. data, exactly as the pipeline loads it ----
t0 = time.time()
data = base.load_and_clean(rp.FILES, base.LABEL_COL, base.LABEL_NORMALISE, base.CLASSES, rp.PER_CLASS, rp.MODEL_SEED)
X, y, feats, c2i = base.prepare_xy(data, base.NON_FEATURE_COLS, base.CLASSES)
del data
i2c = {i: c for c, i in c2i.items()}
print(f"rows={len(X)} features={len(feats)}  ({time.time()-t0:.0f}s)")
assert len(X) == 119571 and len(feats) == 70, "data differ from the paper (119,571 rows, 70 features)"

def ci(a):
    m, lo, hi = full.ci95(a); return [float(m), float(lo), float(hi)]

def fit(kind, Xtr, ytr, n_jobs=None):
    m = full.make_model(kind)
    if n_jobs is not None: m.set_params(n_jobs=n_jobs)
    if kind == "mlp":
        sc = StandardScaler().fit(Xtr[feats]); m.fit(sc.transform(Xtr[feats]), ytr)
        return m, (lambda Z: m.predict(sc.transform(Z[feats])))
    m.fit(Xtr[feats].values, ytr)
    return m, (lambda Z: m.predict(Z[feats].values))

def wf1(yt, yp): return float(f1_score(yt, yp, average="weighted", zero_division=0))

result = {"env": env}

# ============================== B4 ==========================================
if RUN_B4:
    print("\n" + "=" * 70 + "\nB4  XGBoost reproducibility\n" + "=" * 70)
    b4 = {}
    for nj in (-1, 1):
        d = {"hybrid768": [], "pure768": []}; same_pred = 0
        for s in range(rp.N_SPLITS):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=rp.TEST_SIZE, random_state=s, stratify=y)
            _, p1 = fit("xgb", Xtr, ytr, nj)                       # C1 fit
            yp1 = p1(Xte); c1 = wf1(yte, yp1)
            preds_orig = [yp1]
            for v, cfg in (("hybrid768", H768), ("pure768", P768)):
                _, p2 = fit("xgb", Xtr, ytr, nj)                   # separate C2 fit, as in the pipeline
                d[v].append(wf1(yte, p2(ob.make_approx_mlkem(Xte, cfg))) - c1)
                preds_orig.append(p2(Xte))
            same_pred += int(all(np.array_equal(preds_orig[0], q) for q in preds_orig[1:]))
            print(f"  n_jobs={nj:>2} split {s}: hybrid {d['hybrid768'][-1]:+.4f}  pure {d['pure768'][-1]:+.4f}  ({time.time()-t0:.0f}s)")
        b4[f"n_jobs={nj}"] = {v: {"C2_minus_C1": ci(d[v]), "per_split": d[v],
                                  "gap_vs_paper": float(np.mean(d[v]) - PAPER[v]),
                                  "matches_paper_4dp": bool(abs(np.mean(d[v]) - PAPER[v]) <= 0.0001)} for v in d}
        b4[f"n_jobs={nj}"]["splits_with_identical_predictions_across_fits"] = same_pred
    for v in ("hybrid768", "pure768"):
        b4[f"thread_effect_{v}"] = float(np.mean(b4["n_jobs=1"][v]["per_split"]) - np.mean(b4["n_jobs=-1"][v]["per_split"]))
    m1 = b4["n_jobs=-1"]
    b4["reading"] = {
        "E1": ("paper values reproduce with xgboost 3.2.0; the earlier gap came from the later runs' environment"
               if m1["hybrid768"]["matches_paper_4dp"] and m1["pure768"]["matches_paper_4dp"]
               else "paper values do not reproduce in this environment (see gap_vs_paper)"),
        "E2": {k: f"{b4[k]['splits_with_identical_predictions_across_fits']}/10 splits identical" for k in ("n_jobs=-1", "n_jobs=1")},
        "E3": {v: b4[f"thread_effect_{v}"] for v in ("hybrid768", "pure768")}}
    result["B4"] = b4
    (OUT / "B4_result.json").write_text(json.dumps(b4, indent=2))

# ============================== B1 ==========================================
if RUN_B1:
    print("\n" + "=" * 70 + "\nB1  loss on shifted flows only (hybrid footprint)\n" + "=" * 70)
    SUBS = ("ALL", "ALL_443scope", "E_all", "E_443", "E_non443", "BI_non443")
    acc = {k: {s: {"loss": [], "n": [], "changed": []} for s in SUBS} for k in ("rf", "xgb", "mlp")}
    BI = [c2i["Benign"], c2i["Infiltration"]]
    preds = {}
    for s in range(rp.N_SPLITS):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=rp.TEST_SIZE, random_state=s, stratify=y)
        Xte_h = ob.make_approx_mlkem(Xte, H768)
        changed = np.zeros(len(Xte), bool)
        for c in feats:
            a, b = Xte[c].values, Xte_h[c].values
            changed |= ~((a == b) | (np.isnan(a) & np.isnan(b)))
        elig = ((Xte["Tot Fwd Pkts"] > 0) & (Xte["Tot Bwd Pkts"] > 0)).values
        assert np.array_equal(changed, elig), "shifted rows differ from eligible rows"
        p443 = (Xte["Dst Port"].values == 443)
        Xte_443 = Xte_h.copy()
        for c in feats:                                   # port-443 scope = full shift on port-443 rows only
            Xte_443[c] = np.where(elig & p443, Xte_h[c].values, Xte[c].values).astype(Xte_h[c].dtype)
        masks = {"E_all": elig, "E_443": elig & p443, "E_non443": elig & ~p443,
                 "BI_non443": elig & ~p443 & np.isin(yte, BI)}
        for kind in ("rf", "xgb", "mlp"):
            _, pr = fit(kind, Xtr, ytr)                   # paper settings (n_jobs as in the module)
            y1, y2, y3 = pr(Xte), pr(Xte_h), pr(Xte_443)
            a = acc[kind]
            a["ALL"]["loss"].append(wf1(yte, y2) - wf1(yte, y1)); a["ALL"]["n"].append(len(yte))
            a["ALL"]["changed"].append(float(np.mean(y1 != y2)))
            a["ALL_443scope"]["loss"].append(wf1(yte, y3) - wf1(yte, y1)); a["ALL_443scope"]["n"].append(len(yte))
            a["ALL_443scope"]["changed"].append(float(np.mean(y1 != y3)))
            assert np.array_equal(y3[elig & p443], y2[elig & p443]) and np.array_equal(y3[~(elig & p443)], y1[~(elig & p443)])
            for nm, mk in masks.items():
                a[nm]["loss"].append(wf1(yte[mk], y2[mk]) - wf1(yte[mk], y1[mk]))
                a[nm]["n"].append(int(mk.sum())); a[nm]["changed"].append(float(np.mean(y1[mk] != y2[mk])))
            preds[f"s{s}_{kind}_C1"] = y1.astype(np.int8); preds[f"s{s}_{kind}_C2"] = y2.astype(np.int8)
            print(f"  split {s} {kind:>3}: ALL {a['ALL']['loss'][-1]:+.4f}  E_all {a['E_all']['loss'][-1]:+.4f}  "
                  f"E_443 {a['E_443']['loss'][-1]:+.4f}  BI_non443 {a['BI_non443']['loss'][-1]:+.4f}  ({time.time()-t0:.0f}s)")
        preds[f"s{s}_yte"] = yte.astype(np.int8); preds[f"s{s}_elig"] = elig; preds[f"s{s}_p443"] = p443
    np.savez_compressed(OUT / "B1_predictions.npz", **preds)

    b1 = {}
    for kind, a in acc.items():
        row = {nm: {"C2_minus_C1": ci(a[nm]["loss"]), "mean_rows": float(np.mean(a[nm]["n"])),
                    "share_pred_changed": float(np.mean(a[nm]["changed"]))} for nm in SUBS}
        L = {nm: row[nm]["C2_minus_C1"][0] for nm in SUBS}
        def ratio(num, den): return float(num / den) if den < 0 else None
        r = ratio(L["E_443"], L["E_all"]); r_bi = ratio(L["E_443"], L["BI_non443"])
        row["ratio_E443_over_Eall"] = r; row["ratio_E443_over_BInon443"] = r_bi
        row["reading"] = {
            "B1-b": ("n/a (no loss on E_all)" if r is None else
                     "port-443 result is mainly dilution" if r >= 0.5 else
                     "port-443 flows are less affected than eligible flows overall; not only dilution"),
            "B1-c": ("n/a (no loss on BI_non443)" if r_bi is None else
                     "within Benign+Infiltration, port-443 flows are about as affected; B1-b difference mainly class composition" if r_bi >= 0.5 else
                     "port-443 flows are less affected even within Benign+Infiltration")}
        ref_all, ref_443 = SAME_RUN_07_08[kind]
        row["sanity_vs_07_08"] = {"ALL_gap": float(L["ALL"] - ref_all), "ALL_443scope_gap": float(L["ALL_443scope"] - ref_443),
                                  "within_0.0001": bool(abs(L["ALL"] - ref_all) <= 1e-4 and abs(L["ALL_443scope"] - ref_443) <= 1e-4)}
        b1[kind] = row
    result["B1"] = b1
    (OUT / "B1_result.json").write_text(json.dumps(b1, indent=2))

(OUT / "B1_B4_result.json").write_text(json.dumps(result, indent=2))
print("\n" + "=" * 70 + "\nRESULTS\n" + "=" * 70)
print(json.dumps(result, indent=2))
print(f"Saved in {OUT}  (total {time.time()-t0:.0f}s)")
