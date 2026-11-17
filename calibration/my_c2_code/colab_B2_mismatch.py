# =============================================================================
# B2 mismatch test: do screening (C3) and augmentation (A) still recover when
# the simulation used to prepare them differs in packet structure from the
# shift met at test time?
# Paste this whole file into ONE cell of a Colab runtime and run it. About 3 h.
# If the runtime disconnects, run the cell again: finished splits are reloaded
# from Drive and skipped.
# Rules fixed before running: decisions_log.md, 2026-10-10 "B2 mismatch" rows.
#
# Footprints: H = hybrid768 (main), B = bytes-only (pure bytes, 0 added packets,
# no 1,460 max, maxima floored at the new mean; same wrapper as results/07-10).
# Cells (prepare -> test): H->H (main, sanity), B->B (matched), B->H and H->B
# (mismatched).
#   C3 prepared with P: KS + Holm screen on training data transformed with P,
#      then MI >= 0.05 (rp.ks_holm_stable and rp.mi_rank, unchanged).
#   A prepared with P: original training rows + their P-transformed copy,
#      labels duplicated, all 70 features (Additional file 1, A13).
# The four .py modules are imported unchanged.
# =============================================================================
import sys, os, json, time, platform, subprocess
from pathlib import Path

XGB_VERSION = "3.2.0"
if "xgboost" in sys.modules:
    _loaded = getattr(sys.modules["xgboost"], "__version__", None)
    if _loaded != XGB_VERSION:
        raise SystemExit(f"xgboost {_loaded} is already imported. Use Runtime > Restart session, then run this cell again.")
else:
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

import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from google.colab import drive

drive.mount("/content/drive")
MY = Path("/content/drive/MyDrive")
CODE_DIR = None
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

import xgboost, sklearn, scipy
import pilot_pqc_stable_pipeline as base
import pilot_option_b as ob
import option_b_full_experiment as full
import revised_pipeline_main_only as rp

OUT = MY / "pqc_oav_calibration_B2_mismatch"
OUT.mkdir(exist_ok=True)
env = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
       "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
       "cpu_count": os.cpu_count(), "code_dir": str(CODE_DIR)}
print(json.dumps(env, indent=2))
assert xgboost.__version__ == XGB_VERSION and full.HAVE_XGB
assert rp.N_SPLITS == 10 and rp.MODEL_SEED == 42 and rp.TEST_SIZE == 0.30 and rp.PER_CLASS == 30000
assert rp.MI_HEADLINE == 0.05 and rp.HOLM_ALPHA == 0.05 and rp.APPLY_DROP_UNMODELLABLE and not rp.DROP_IDENTIFIERS

# ---- footprints (same code as results/07-10) ----
H768 = dict(rp.VARIANTS["hybrid768"])
BYTES_ONLY = {"d_fwd_bytes": 1152.0, "d_bwd_bytes": 1056.0, "d_fwd_pkts": 0.0, "d_bwd_pkts": 0.0,
              "hdr_bytes": 40.0, "added_pkt_size": 0.0, "floor_max_at_mean": True}
_orig_make = ob.make_approx_mlkem
def T(X, cfg):
    Xa = _orig_make(X, {k: v for k, v in cfg.items() if k != "floor_max_at_mean"})
    if cfg.get("floor_max_at_mean", False):
        m = ((X["Tot Fwd Pkts"] > 0) & (X["Tot Bwd Pkts"] > 0)).values
        for mx, mean in (("Fwd Pkt Len Max", "Fwd Pkt Len Mean"), ("Bwd Pkt Len Max", "Bwd Pkt Len Mean")):
            a = Xa[mx].values.astype("float64"); b = Xa[mean].values.astype("float64")
            Xa[mx] = np.where(m, np.maximum(a, b), a).astype("float32")
        a = Xa["Pkt Len Max"].values.astype("float64")
        dm = np.maximum(Xa["Fwd Pkt Len Max"].values, Xa["Bwd Pkt Len Max"].values).astype("float64")
        Xa["Pkt Len Max"] = np.where(m, np.maximum(a, dm), a).astype("float32")
    return Xa
FP = {"H": H768, "B": BYTES_ONLY}
CELLS = [("H", "H"), ("B", "B"), ("B", "H"), ("H", "B")]

# ---- data ----
t0 = time.time()
data = base.load_and_clean(rp.FILES, base.LABEL_COL, base.LABEL_NORMALISE, base.CLASSES, rp.PER_CLASS, rp.MODEL_SEED)
X, y, feats, c2i = base.prepare_xy(data, base.NON_FEATURE_COLS, base.CLASSES)
del data
assert len(X) == 119571 and len(feats) == 70, "data differ from the paper"
cand = [c for c in feats if c not in rp.UNMODELLABLE]
assert len(cand) == 55
print(f"rows={len(X)} features={len(feats)} candidates={len(cand)}  ({time.time()-t0:.0f}s)")

def wf1(yt, yp): return float(f1_score(yt, yp, average="weighted"))
def select(Xtr, ytr, Xtr_p, seed):
    stable, shifted, _ = rp.ks_holm_stable(Xtr[cand], Xtr_p[cand], cand, rp.HOLM_ALPHA)
    mi = rp.mi_rank(Xtr, stable, ytr, seed)
    sub = [f for f in stable if mi.get(f, 0.0) >= rp.MI_HEADLINE]
    return (sub if sub else stable[:1]), sorted(shifted)
def fit_pred(kind, Xfit, yfit, cols, tests):
    """Fit once on Xfit[cols]; return predictions for each test frame."""
    m = full.make_model(kind)
    if kind == "mlp":
        sc = StandardScaler().fit(Xfit[cols]); m.fit(sc.transform(Xfit[cols]), yfit)
        return [m.predict(sc.transform(Z[cols])) for Z in tests]
    m.fit(Xfit[cols].values, yfit)
    return [m.predict(Z[cols].values) for Z in tests]

KINDS = ("rf", "xgb", "mlp")
for s in range(rp.N_SPLITS):
    fn = OUT / f"split_{s}.json"
    if fn.exists():
        print(f"  split {s}: already done, skipped"); continue
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=rp.TEST_SIZE, random_state=s, stratify=y)
    Xtr_p = {k: T(Xtr, c) for k, c in FP.items()}
    Xte_t = {k: T(Xte, c) for k, c in FP.items()}
    sel, shf = {}, {}
    for k in FP:
        sel[k], shf[k] = select(Xtr, ytr, Xtr_p[k], s)
    rec = {"selected": sel, "shifted": shf, "scores": {}}
    for kind in KINDS:
        sc = {}
        p1, p2H, p2B = fit_pred(kind, Xtr, ytr, feats, [Xte, Xte_t["H"], Xte_t["B"]])
        sc["C1"] = wf1(yte, p1); sc["C2_H"] = wf1(yte, p2H); sc["C2_B"] = wf1(yte, p2B)
        for P in FP:                                            # screening prepared with P
            qH, qB = fit_pred(kind, Xtr, ytr, sel[P], [Xte_t["H"], Xte_t["B"]])
            sc[f"C3_{P}_H"] = wf1(yte, qH); sc[f"C3_{P}_B"] = wf1(yte, qB)
        for P in FP:                                            # augmentation prepared with P
            Xa = pd.concat([Xtr, Xtr_p[P]], ignore_index=True); ya = np.concatenate([ytr, ytr])
            qH, qB = fit_pred(kind, Xa, ya, feats, [Xte_t["H"], Xte_t["B"]])
            sc[f"A_{P}_H"] = wf1(yte, qH); sc[f"A_{P}_B"] = wf1(yte, qB)
        rec["scores"][kind] = sc
        print(f"  split {s} {kind:>3}: C2_H {sc['C2_H']-sc['C1']:+.4f} C2_B {sc['C2_B']-sc['C1']:+.4f} | "
              + " ".join(f"C3 {P}->{Q} {sc[f'C3_{P}_{Q}']-sc['C1']:+.4f}" for P, Q in CELLS) + " | "
              + " ".join(f"A {P}->{Q} {sc[f'A_{P}_{Q}']-sc['C1']:+.4f}" for P, Q in CELLS)
              + f"  ({time.time()-t0:.0f}s)")
    fn.write_text(json.dumps(rec))

# ---- analysis ----
recs = [json.loads((OUT / f"split_{s}.json").read_text()) for s in range(rp.N_SPLITS)]
K = rp.N_SPLITS; NTE, NTR = 35872, 83699
CORR = np.sqrt(1.0 / K + NTE / NTR) / np.sqrt(1.0 / K)          # Nadeau-Bengio factor (about 2.3)
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj
def tost(d, margin=0.01, corrected=False):
    d = np.asarray(d, float); n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    if corrected: se *= CORR
    t_lo, t_hi = (m + margin) / se, (m - margin) / se
    p = max(1 - stats.t.cdf(t_lo, n - 1), stats.t.cdf(t_hi, n - 1))
    h90 = stats.t.ppf(0.95, n - 1) * se
    return m, p, (m - h90, m + h90)
res = {"env": env, "selection": {}, "cells": {}, "sanity": {}}
res["selection"] = {P: {"n_selected_mean": float(np.mean([len(r["selected"][P]) for r in recs])),
                        "n_shifted_mean": float(np.mean([len(r["shifted"][P]) for r in recs])),
                        "shifted_identical_across_splits": len({tuple(r["shifted"][P]) for r in recs}) == 1,
                        "shifted_split0": recs[0]["shifted"][P]} for P in FP}
for defence in ("C3", "A"):
    for P, Q in CELLS:
        cell = {}
        for corrected in (False, True):
            ms, ps, cis = [], [], []
            for kind in KINDS:
                d = [r["scores"][kind][f"{defence}_{P}_{Q}"] - r["scores"][kind]["C1"] for r in recs]
                m_, p_, ci_ = tost(d, 0.01, corrected); ms.append(m_); ps.append(p_); cis.append(ci_)
            adj = holm(ps)
            for i, kind in enumerate(KINDS):
                eq = bool(cis[i][0] > -0.01 and cis[i][1] < 0.01 and adj[i] < 0.05)
                c2 = np.mean([r["scores"][kind][f"C2_{Q}"] - r["scores"][kind]["C1"] for r in recs])
                e = cell.setdefault(kind, {"mean_minus_C1": float(ms[i]), "C2_minus_C1_same_test": float(c2),
                                           "remaining_share_of_C2_loss": float(ms[i] / c2) if c2 < 0 else None})
                tag = "corrected" if corrected else "uncorrected"
                e[tag] = {"ci90": [float(cis[i][0]), float(cis[i][1])], "holm_p": float(adj[i]), "equivalent_0.01": eq}
        for kind in KINDS:
            u, c = cell[kind]["uncorrected"]["equivalent_0.01"], cell[kind]["corrected"]["equivalent_0.01"]
            cell[kind]["reading"] = ("recovery survives" if u and c else
                                     "recovery survives only without the overlap correction" if u else
                                     "recovery not established")
        res["cells"][f"{defence} {P}->{Q}"] = cell
PAPER = {"C3": {"rf": 0.0014, "xgb": -0.0013, "mlp": -0.0022}, "A": {"rf": 0.0004, "xgb": 0.0003, "mlp": 0.0008}}
for kind in KINDS:
    res["sanity"][kind] = {
        "C2_H_minus_C1": float(np.mean([r["scores"][kind]["C2_H"] - r["scores"][kind]["C1"] for r in recs])),
        "C2_B_minus_C1": float(np.mean([r["scores"][kind]["C2_B"] - r["scores"][kind]["C1"] for r in recs])),
        "C3_HH_matches_paper_4dp": bool(round(res["cells"]["C3 H->H"][kind]["mean_minus_C1"], 4) == PAPER["C3"][kind]),
        "A_HH_matches_paper_4dp": bool(round(res["cells"]["A H->H"][kind]["mean_minus_C1"], 4) == PAPER["A"][kind]),
        "C3_HB_equals_C3_HH": bool(all(abs(r["scores"][kind]["C3_H_B"] - r["scores"][kind]["C3_H_H"]) < 1e-12 for r in recs))}
(OUT / "B2_result.json").write_text(json.dumps(res, indent=2))
print("\n" + "=" * 70 + "\nRESULTS\n" + "=" * 70)
print(json.dumps(res, indent=2))
print(f"Saved in {OUT}  (total {time.time()-t0:.0f}s)")
