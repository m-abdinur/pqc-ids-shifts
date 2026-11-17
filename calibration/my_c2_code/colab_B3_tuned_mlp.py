# =============================================================================
# B3: is the MLP recovery failure a tuning artefact?
# Tunes the MLP inside each training partition and repeats the two designs
# where MLP screening failed: identifier-free and combined (Table 9).
# Paste this whole file into ONE cell of a Colab runtime. About 3-5 h.
# If the runtime disconnects, run the cell again: finished splits are reloaded
# from Drive and skipped.
# Rules fixed before running: decisions_log.md, 2026-10-10 "B3" rows.
#
# Designs (as in Table 9; feature lists from v6_timing_free_pipeline.py):
#   identifier-free: Dst Port and Init Fwd Win Byts removed everywhere
#                    (68 features); C3 candidates = 68 minus the 15
#                    unmodellable fields (53).
#   combined:        same 68 features; C3 candidates also exclude the timing
#                    and rate fields (37).
#   Both share C1 and C2 (68 features). Hybrid footprint; KS + Holm, MI 0.05.
# Tuning: 6 configurations (hidden layers (64,32) / (128,64) / (256,128) x
#   alpha 1e-4 / 1e-2; everything else as the paper's MLP). Chosen by weighted
#   F1 on a stratified 20% validation part of the training partition (original
#   data), separately for the C1 model and for each C3 model, then refitted on
#   the full training partition. Test data are never used for tuning.
# The paper's untuned MLP is refitted in the same run as a sanity check.
# The four .py modules are imported unchanged.
# =============================================================================
import sys, os, json, time, platform, subprocess
from pathlib import Path
XGB_VERSION = "3.2.0"   # not used by B3, pinned so the environment matches the other runs
if "xgboost" not in sys.modules:
    try:
        from importlib.metadata import version as _v
        have = _v("xgboost")
    except Exception:
        have = None
    if have != XGB_VERSION:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"xgboost=={XGB_VERSION}"])

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
import sklearn, scipy
import pilot_pqc_stable_pipeline as base
import pilot_option_b as ob
import option_b_full_experiment as full
import revised_pipeline_main_only as rp

OUT = MY / "pqc_oav_calibration_B3_tuned_mlp"
OUT.mkdir(exist_ok=True)
env = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
       "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "cpu_count": os.cpu_count(),
       "code_dir": str(CODE_DIR)}
print(json.dumps(env, indent=2))
assert rp.N_SPLITS == 10 and rp.MODEL_SEED == 42 and rp.TEST_SIZE == 0.30 and rp.PER_CLASS == 30000
assert rp.MI_HEADLINE == 0.05 and rp.HOLM_ALPHA == 0.05

IDENT = ["Dst Port", "Init Fwd Win Byts"]
TIMING_AND_RATE = ["Flow Duration", "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Flow IAT Min", "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Tot",
    "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd Pkts/s", "Bwd Pkts/s", "Active Mean",
    "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"]   # v6_timing_free_pipeline.py
H768 = dict(rp.VARIANTS["hybrid768"])
GRID = [{"hidden_layer_sizes": h, "alpha": a} for h in [(64, 32), (128, 64), (256, 128)] for a in [1e-4, 1e-2]]
DEFAULT = {"hidden_layer_sizes": (64, 32), "alpha": 1e-4}

t0 = time.time()
data = base.load_and_clean(rp.FILES, base.LABEL_COL, base.LABEL_NORMALISE, base.CLASSES, rp.PER_CLASS, rp.MODEL_SEED)
X, y, feats70, c2i = base.prepare_xy(data, base.NON_FEATURE_COLS, base.CLASSES)
del data
assert len(X) == 119571 and len(feats70) == 70
feats = [c for c in feats70 if c not in IDENT]
cand = {"idfree": [c for c in feats if c not in rp.UNMODELLABLE],
        "combined": [c for c in feats if c not in set(rp.UNMODELLABLE) | set(TIMING_AND_RATE)]}
assert len(feats) == 68 and len(cand["idfree"]) == 53 and len(cand["combined"]) == 37, (len(feats), {k: len(v) for k, v in cand.items()})
print(f"rows={len(X)} features={len(feats)} candidates={ {k: len(v) for k, v in cand.items()} }  ({time.time()-t0:.0f}s)")

def wf1(yt, yp): return float(f1_score(yt, yp, average="weighted"))
def mlp(params):
    m = full.make_model("mlp"); m.set_params(**params); return m
def fit(params, Xf, yf, cols):
    sc = StandardScaler().fit(Xf[cols]); m = mlp(params).fit(sc.transform(Xf[cols]), yf)
    return lambda Z: m.predict(sc.transform(Z[cols]))
def tune(Xtr, ytr, cols, seed):
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.20, random_state=seed, stratify=ytr)
    scores = []
    for g in GRID:
        scores.append(wf1(yv, fit(g, Xa, ya, cols)(Xv)))
    best = int(np.argmax(scores))
    return GRID[best], scores
def select(Xtr, ytr, Xtr_h, cands, seed):
    stable, shifted, _ = rp.ks_holm_stable(Xtr[cands], Xtr_h[cands], cands, rp.HOLM_ALPHA)
    mi = rp.mi_rank(Xtr, stable, ytr, seed)
    sub = [f for f in stable if mi.get(f, 0.0) >= rp.MI_HEADLINE]
    return sub if sub else stable[:1]

for s in range(rp.N_SPLITS):
    fn = OUT / f"split_{s}.json"
    if fn.exists():
        print(f"  split {s}: already done, skipped"); continue
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=rp.TEST_SIZE, random_state=s, stratify=y)
    Xtr_h, Xte_h = ob.make_approx_mlkem(Xtr, H768), ob.make_approx_mlkem(Xte, H768)
    rec = {"selected": {}, "tuning": {}, "scores": {}}
    # C1 / C2 (68 features), default and tuned
    best, sc_ = tune(Xtr, ytr, feats, s); rec["tuning"]["C1"] = {"best": best, "val_scores": sc_}
    for tag, params in (("default", DEFAULT), ("tuned", best)):
        pr = fit(params, Xtr, ytr, feats)
        rec["scores"][f"C1_{tag}"] = wf1(yte, pr(Xte)); rec["scores"][f"C2_{tag}"] = wf1(yte, pr(Xte_h))
    for d in ("idfree", "combined"):
        sel = select(Xtr, ytr, Xtr_h, cand[d], s); rec["selected"][d] = sel
        best, sc_ = tune(Xtr, ytr, sel, s); rec["tuning"][f"C3_{d}"] = {"best": best, "val_scores": sc_}
        for tag, params in (("default", DEFAULT), ("tuned", best)):
            rec["scores"][f"C3_{d}_{tag}"] = wf1(yte, fit(params, Xtr, ytr, sel)(Xte_h))
    fn.write_text(json.dumps(rec, default=list))
    S = rec["scores"]
    print(f"  split {s}: C1 default {S['C1_default']:.4f} tuned {S['C1_tuned']:.4f} | "
          f"idfree C3-C1 default {S['C3_idfree_default']-S['C1_default']:+.4f} tuned {S['C3_idfree_tuned']-S['C1_tuned']:+.4f} | "
          f"combined C3-C1 default {S['C3_combined_default']-S['C1_default']:+.4f} tuned {S['C3_combined_tuned']-S['C1_tuned']:+.4f} | "
          f"best C1 {rec['tuning']['C1']['best']}  ({time.time()-t0:.0f}s)")

# ---- analysis ----
recs = [json.loads((OUT / f"split_{s}.json").read_text()) for s in range(rp.N_SPLITS)]
K, NTE, NTR = rp.N_SPLITS, 35872, 83699
CORR = np.sqrt(1.0 / K + NTE / NTR) / np.sqrt(1.0 / K)
def tost(d, margin=0.01, corrected=False):
    d = np.asarray(d, float); n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    if corrected: se *= CORR
    p = max(1 - stats.t.cdf((m + margin) / se, n - 1), stats.t.cdf((m - margin) / se, n - 1))
    h = stats.t.ppf(0.95, n - 1) * se
    return {"mean": float(m), "ci90": [float(m - h), float(m + h)], "tost_p": float(p),
            "equivalent_0.01": bool(m - h > -margin and m + h < margin and p < 0.05)}
def col(k): return np.array([r["scores"][k] for r in recs])
res = {"env": env, "grid": GRID, "designs": {}, "sanity": {}, "tuning_choices": {}}
for tag in ("default", "tuned"):
    res[f"C1_{tag}_mean"] = float(col(f"C1_{tag}").mean())
    res[f"C2_minus_C1_{tag}"] = float((col(f"C2_{tag}") - col(f"C1_{tag}")).mean())
for d in ("idfree", "combined"):
    out = {"n_selected_mean": float(np.mean([len(r["selected"][d]) for r in recs]))}
    for tag in ("default", "tuned"):
        diff = col(f"C3_{d}_{tag}") - col(f"C1_{tag}")
        u, c = tost(diff), tost(diff, corrected=True)
        out[tag] = {"C3_mean": float(col(f"C3_{d}_{tag}").mean()), "uncorrected": u, "corrected": c,
                    "reading": ("recovery established" if u["equivalent_0.01"] and c["equivalent_0.01"] else
                                "recovery established only without the overlap correction" if u["equivalent_0.01"] else
                                "recovery not established")}
    t, dflt = out["tuned"], out["default"]
    out["B3_reading"] = ("the MLP failure in this design was a tuning artefact" if t["reading"] == "recovery established"
                         else "the MLP failure persists after tuning")
    out["tuned_minus_default_deficit"] = float(t["uncorrected"]["mean"] - dflt["uncorrected"]["mean"])
    res["designs"][d] = out
    res["tuning_choices"][f"C3_{d}"] = [str(r["tuning"][f"C3_{d}"]["best"]) for r in recs]
res["tuning_choices"]["C1"] = [str(r["tuning"]["C1"]["best"]) for r in recs]
PAPER = {"C1": 0.8381, "C3_idfree": 0.8066, "C3_combined": 0.7855}       # Table 9, MLP
res["sanity"] = {"C1_default_4dp": round(res["C1_default_mean"], 4), "C3_idfree_default_4dp": round(res["designs"]["idfree"]["default"]["C3_mean"], 4),
                 "C3_combined_default_4dp": round(res["designs"]["combined"]["default"]["C3_mean"], 4)}
res["sanity"]["matches_table9"] = bool(res["sanity"]["C1_default_4dp"] == PAPER["C1"] and res["sanity"]["C3_idfree_default_4dp"] == PAPER["C3_idfree"]
                                       and res["sanity"]["C3_combined_default_4dp"] == PAPER["C3_combined"])
(OUT / "B3_result.json").write_text(json.dumps(res, indent=2))
print("\n" + "=" * 70 + "\nRESULTS\n" + "=" * 70)
print(json.dumps(res, indent=2))
print(f"Saved in {OUT}  (total {time.time()-t0:.0f}s)")
