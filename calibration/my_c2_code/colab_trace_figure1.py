# =============================================================================
# Figure 1 trace: which matrices each condition is trained and tested on.
# Paste into ONE Colab cell. About 5-10 minutes (data loading dominates).
#
# What it does
#   * Runs revised_pipeline_main_only.one_split() unchanged for the main
#     (hybrid768) footprint, but with a stand-in classifier (sklearn
#     DummyClassifier), so no real models are trained. It traces data flow only;
#     it does not reproduce any score.
#   * Wraps train_test_split, ks_holm_stable, mi_rank, fit_eval and
#     make_approx_mlkem to record the row index of every matrix they receive.
#   * Rebuilds condition A the way the augmentation runs did (original training
#     partition stacked with its transformed copy, all 70 features, tested on the
#     transformed test set).
#   * Asserts: train and test rows are disjoint; no test row enters the KS screen,
#     MI selection, scaler fitting or model fitting; C1 is scored on X, C2 and A on
#     T(X), C3 on T(X) restricted to the selected set F.
#   * Prints a shape table per condition and saves it to Drive.
# The four .py modules are imported unchanged.
# =============================================================================
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from google.colab import drive

drive.mount("/content/drive")
MY = Path("/content/drive/MyDrive")
N_TRACE = 10                       # splits to trace (all ten by default)

CODE_DIR = None                    # set to a Path to skip the search
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
    CODE_DIR = sorted(cands, reverse=True)[0][1]
print("Using CODE_DIR =", CODE_DIR)
sys.path.insert(0, str(CODE_DIR))

import pilot_pqc_stable_pipeline as base
import pilot_option_b as ob
import option_b_full_experiment as full
import revised_pipeline_main_only as rp

# ---------------------------------------------------------------- bookkeeping
CUR = {}          # current split: seed, train index, test index
TAG = {}          # id(DataFrame) -> "T(X_tr)" or "T(X)"
LOG = []          # one row per traced call

def idx(df):
    return set(df.index)

# stand-in classifier: records the shape it is fitted on
class Recorder(DummyClassifier):
    def fit(self, X, y, sample_weight=None):
        LOG.append({"split": CUR["seed"], "step": "model.fit", "rows": int(X.shape[0]), "cols": int(X.shape[1])})
        return super().fit(X, y, sample_weight)
full.make_model = lambda kind: Recorder(strategy="prior")

_tts = rp.train_test_split
def traced_split(*a, **k):
    Xtr, Xte, ytr, yte = _tts(*a, **k)
    assert idx(Xtr).isdisjoint(idx(Xte)), "train and test rows overlap"
    CUR.update(train=idx(Xtr), test=idx(Xte), n_tr=len(Xtr), n_te=len(Xte))
    TAG.clear()                    # object ids are only meaningful within one split
    return Xtr, Xte, ytr, yte
rp.train_test_split = traced_split

_T = ob.make_approx_mlkem
def traced_T(X, cfg=ob.MLKEM):
    out = _T(X, cfg)
    assert list(out.index) == list(X.index), "T changed the row index"
    if idx(X) <= CUR["train"]:
        TAG[id(out)] = "T(X_tr)"
    elif idx(X) <= CUR["test"]:
        TAG[id(out)] = "T(X)"
    else:
        raise AssertionError("T applied to a mix of train and test rows")
    return out
ob.make_approx_mlkem = traced_T

_ks = rp.ks_holm_stable
def traced_ks(Xo, Xa, cols, alpha):
    assert idx(Xo) <= CUR["train"] and idx(Xa) <= CUR["train"], "TEST ROWS ENTERED THE KS SCREEN"
    LOG.append({"split": CUR["seed"], "step": "KS+Holm (X_tr vs T(X_tr))", "rows": len(Xo), "cols": len(cols)})
    s, sh, D = _ks(Xo, Xa, cols, alpha)
    CUR["S"] = list(s)
    return s, sh, D
rp.ks_holm_stable = traced_ks

_mi = rp.mi_rank
def traced_mi(Xtr, stable, ytr, seed):
    assert idx(Xtr) <= CUR["train"], "TEST ROWS ENTERED MI SELECTION"
    assert TAG.get(id(Xtr)) is None, "MI was given transformed values"
    LOG.append({"split": CUR["seed"], "step": "MI (original X_tr)", "rows": len(Xtr), "cols": len(stable)})
    return _mi(Xtr, stable, ytr, seed)
rp.mi_rank = traced_mi

_fe = rp.fit_eval
def traced_fit_eval(kind, scale, Xtr_df, cols, ytr, Xev_df, yev, i2c):
    assert idx(Xtr_df) <= CUR["train"], "TEST ROWS ENTERED FITTING"
    assert TAG.get(id(Xtr_df)) is None, "C1/C2/C3 model trained on transformed values"
    assert idx(Xev_df) <= CUR["test"], "evaluation on non-test rows"
    test = TAG.get(id(Xev_df), "X")
    if len(cols) == len(CUR["full"]):
        cond = "C1" if test == "X" else "C2"
    else:
        assert test == "T(X)" and set(cols) <= set(CUR["S"]), "C3 features outside the stable set"
        cond = "C3"
    LOG.append({"split": CUR["seed"], "step": cond, "kind": kind, "train": f"X_tr ({len(Xtr_df)} x {len(cols)})",
                "test": f"{test} ({len(Xev_df)} x {len(cols)})", "scaler_fitted_on": "X_tr" if scale else "-"})
    return _fe(kind, scale, Xtr_df, cols, ytr, Xev_df, yev, i2c)
rp.fit_eval = traced_fit_eval

# ---------------------------------------------------------------- data
t0 = time.time()
data = base.load_and_clean(rp.FILES, base.LABEL_COL, base.LABEL_NORMALISE, base.CLASSES, rp.PER_CLASS, rp.MODEL_SEED)
X, y, full_features, c2i = base.prepare_xy(data, base.NON_FEATURE_COLS, base.CLASSES)
del data
i2c = {i: c for c, i in c2i.items()}
cand = [c for c in full_features if c not in rp.UNMODELLABLE]
print(f"rows={len(X)} features={len(full_features)} candidates={len(cand)}  ({time.time()-t0:.0f}s)")
CUR["full"] = full_features
variants = {rp.FPREP: rp.VARIANTS[rp.FPREP]}
kinds = [k for k, _, _ in full.CLASSIFIERS]

for seed in rp.SPLIT_SEEDS[:N_TRACE]:
    CUR["seed"] = seed
    res, stable_v, shifted_v, D_v, mi_prep = rp.one_split(seed, X, y, full_features, cand, i2c, kinds, variants)
    # condition A, built as in the augmentation runs: [X_tr ; T(X_tr)], all 70 features, tested on T(X)
    Xtr, Xte, ytr, yte = traced_split(X, y, test_size=rp.TEST_SIZE, random_state=seed, stratify=y)
    Xtr_t, Xte_t = traced_T(Xtr, rp.VARIANTS[rp.FPREP]), traced_T(Xte, rp.VARIANTS[rp.FPREP])
    Xa = pd.concat([Xtr, Xtr_t]); ya = np.concatenate([ytr, ytr])
    assert idx(Xa) <= CUR["train"] and len(Xa) == 2 * len(Xtr), "A used non-training rows"
    Recorder(strategy="prior").fit(Xa[full_features].values, ya)
    LOG.append({"split": seed, "step": "A", "train": f"X_tr + T(X_tr) ({len(Xa)} x {len(full_features)})",
                "test": f"T(X) ({len(Xte_t)} x {len(full_features)})"})
    print(f"  split {seed}: train={CUR['n_tr']} test={CUR['n_te']}  checks passed  ({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------- report
log = pd.DataFrame(LOG)
print("\nTrain/test matrices by condition (split 0; RF and MLP stand-ins; C3 rows include the MI-threshold sweep):")
cols = ["step", "train", "test", "scaler_fitted_on"]
print(log[(log.split == 0) & log.step.isin(["C1", "C2", "C3", "A"]) & (log.kind.isin(["rf", "mlp"]) | log.kind.isna())][cols]
      .drop_duplicates().to_string(index=False))
print("\nSelection steps (split 0):")
print(log[(log.split == 0) & log.step.str.startswith(("KS", "MI"))][["step", "rows", "cols"]].to_string(index=False))
print(f"\nALL ASSERTIONS PASSED for {N_TRACE} splits: test rows never entered the KS screen, MI selection, "
      "scaler fitting or model fitting.")
OUT = MY / "pqc_oav_calibration_figure1_trace"; OUT.mkdir(exist_ok=True)
log.to_csv(OUT / "figure1_trace_log.csv", index=False)
print("Saved", OUT / "figure1_trace_log.csv")
