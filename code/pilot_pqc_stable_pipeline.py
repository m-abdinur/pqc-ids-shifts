# ================================================================
# PILOT — PQC-aware stable-feature pipeline (vertical slice)
# Dissertation: approximated ML-KEM-768 traffic feature shift on ML-IDS
#
# Purpose of THIS script (not the full experiment):
#   * Run the full C1 -> C2 -> pipeline -> C3 path ONCE, end to end.
#   * One classifier (Random Forest), one stratified split.
#   * Prove the path is executable and leakage-safe BEFORE writing Ch3.
#   * Answer 4 questions in one run:
#       Q1  Is there a C1 -> C2 effect? (does the shift degrade detection)
#       Q2  Does C3 recover any of it? (C2 -> C3)
#       Q3  Which features does KS + Holm flag as shifted vs keep as stable?
#       Q4  Is the leakage discipline correct? (selection on TRAIN only)
#
# It is deliberately small. Scale to 3 classifiers x 10 splits only AFTER
# this runs clean and the numbers make sense.
# ================================================================

import json, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings("ignore")

# Holm-Bonferroni: prefer the standard implementation, fall back to manual.
try:
    from statsmodels.stats.multitest import multipletests
    _HAVE_SM = True
except Exception:
    _HAVE_SM = False


# ============================ CONFIG ============================
# Edit ONLY this block.

RANDOM_SEED       = 42
MULTIPLIER        = 1.505      # main multiplier (pilot uses the main only)
TEST_SIZE         = 0.30
HOLM_ALPHA        = 0.05
PER_CLASS_SAMPLE  = 30000      # subsample per class for speed; None = use all
RF_TREES          = 200
OUTPUT_PREFIX     = "pilot"    # output files: pilot_results.json / pilot_ks.csv

# Map each source CSV to the attack label it contributes (besides Benign).
# Benign rows are pooled from ALL files. >>> EDIT PATHS TO YOUR COLAB <<<
FILES = {
    "/content/data/Friday-16-02-2018_DoS.csv":            "DoS",
    "/content/data/Friday-02-03-2018_Bot.csv":            "Bot",
    "/content/data/Wednesday-28-02-2018_Infiltration.csv": "Infiltration",
    "/content/data/Thursday-01-03-2018_Infiltration.csv":  "Infiltration",
}

CLASSES   = ["Benign", "DoS", "Bot", "Infiltration"]   # the four-class target
LABEL_COL = "Label"

# Byte-volume features the ML-KEM-768 approximation perturbs.
# >>> VERIFY these match df.columns EXACTLY (name variants exist) <<<
BYTE_FEATURES  = ["TotLen Fwd Pkts", "TotLen Bwd Pkts",
                  "Subflow Fwd Byts", "Subflow Bwd Byts"]
# Flow Byts/s is DERIVED: both byte components scale by the same multiplier,
# so the rate scales by the same multiplier (no need for Flow Duration here).
FLOW_BYTES_CANDIDATES = ["Flow Byts/s", "Flow Bytes/s"]

# Columns that are NOT model features (dropped before training).
NON_FEATURE_COLS = ["Label", "Timestamp", "Flow ID", "Src IP", "Dst IP",
                    "Src Port", "_y"]

# Raw label -> canonical class. CSE-CIC-IDS2018 uses 'Infilteration' (sic).
LABEL_NORMALISE = {
    "Benign": "Benign", "BENIGN": "Benign",
    "DoS attacks-Hulk": "DoS", "DoS attacks-SlowHTTPTest": "DoS",
    "DoS attacks-GoldenEye": "DoS", "DoS attacks-Slowloris": "DoS",
    "DDoS attacks-LOIC-HTTP": "DoS",   # remove if you exclude DDoS from DoS
    "Bot": "Bot",
    "Infilteration": "Infiltration", "Infiltration": "Infiltration",
}
# ================================================================


def load_and_clean(files, label_col, label_map, classes, per_class, seed,
                   chunksize=150_000):
    """
    Memory-safe loader. The CSE-CIC-IDS2018 Friday files are ~1M rows and contain
    repeated header rows, which makes pandas read every column as text (huge in
    RAM). We therefore read in CHUNKS and convert features to float32 immediately,
    so the full pooled dataset stays small (~1 GB) instead of overflowing memory.
    Keeps Benign + that file's attack, pools all files, then subsamples per class.
    """
    pieces = []
    for path, attack in files.items():
        keep = {"Benign", attack}
        fbuf = []
        for chunk in pd.read_csv(path, low_memory=False, chunksize=chunksize):
            chunk.columns = [c.strip() for c in chunk.columns]       # strip header whitespace
            lab = chunk[label_col].astype(str).str.strip()
            mask = (lab != label_col)                                # drop embedded header rows
            if not mask.any():
                continue
            sub = chunk.loc[mask].copy()
            sub["_y"] = lab.loc[mask].map(label_map).values
            sub = sub[sub["_y"].isin(keep)]
            if len(sub) == 0:
                continue
            featcols = [c for c in sub.columns if c not in (label_col, "_y")]
            num = sub[featcols].apply(pd.to_numeric, errors="coerce").astype("float32")
            num["_y"] = sub["_y"].values                             # keep label as object col
            fbuf.append(num)
        fpiece = pd.concat(fbuf, ignore_index=True) if fbuf else pd.DataFrame()
        print(f"  loaded {path.split('/')[-1]:45s} kept={len(fpiece):>8,}  ({attack})")
        pieces.append(fpiece)
        del fbuf
    data = pd.concat(pieces, ignore_index=True)
    data = data[data["_y"].isin(classes)].copy()

    if per_class is not None:
        parts = []
        for c in classes:
            sub = data[data["_y"] == c]
            if len(sub) > per_class:
                sub = sub.sample(per_class, random_state=seed)
            parts.append(sub)
        data = pd.concat(parts, ignore_index=True)
    return data


def prepare_xy(data, non_feature_cols, classes):
    """Coerce features to numeric, drop inf/NaN, return X, y(int), feature list."""
    feat_cols = [c for c in data.columns if c not in non_feature_cols]
    X = data[feat_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # drop columns that are entirely NaN (e.g. a non-numeric meta col we missed)
    X = X.dropna(axis=1, how="all")
    # drop rows with any remaining NaN
    good = X.notna().all(axis=1)
    X = X[good]
    y_name = data.loc[good, "_y"]

    # drop zero-variance columns (no information, and they trivialise KS)
    nunique = X.nunique()
    X = X[nunique[nunique > 1].index]

    class_to_int = {c: i for i, c in enumerate(classes)}
    y = y_name.map(class_to_int).astype(int).values
    return X.reset_index(drop=True), y, list(X.columns), class_to_int


def resolve_flow_bytes(columns):
    for c in FLOW_BYTES_CANDIDATES:
        if c in columns:
            return c
    return None


def make_approx(X, byte_features, flow_bytes_col, m):
    """Apply the approximated ML-KEM-768 byte-volume shift."""
    Xa = X.copy()
    for c in byte_features:
        if c in Xa.columns:
            Xa[c] = Xa[c] * m
    if flow_bytes_col and flow_bytes_col in Xa.columns:
        Xa[flow_bytes_col] = Xa[flow_bytes_col] * m   # derived: scales by same m
    return Xa


def stable_features_via_ks(X_tr_orig, X_tr_approx, feature_cols, alpha):
    """
    KS two-sample test per feature (TRAIN ONLY), Holm-Bonferroni correction.
    A feature REJECTED by Holm = significantly shifted = removed.
    Stable subset = features NOT rejected.
    """
    pvals = []
    for c in feature_cols:
        a = X_tr_orig[c].values
        b = X_tr_approx[c].values
        if np.array_equal(a, b):
            p = 1.0                      # unperturbed -> identical -> no shift
        else:
            try:
                _, p = ks_2samp(a, b)
            except Exception:
                p = 1.0
        pvals.append(p)
    pvals = np.array(pvals)

    if _HAVE_SM:
        reject, _, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    else:
        # manual Holm step-down
        order = np.argsort(pvals)
        m = len(pvals)
        reject = np.zeros(m, dtype=bool)
        for rank, idx in enumerate(order):
            if pvals[idx] <= alpha / (m - rank):
                reject[idx] = True
            else:
                break

    shifted = [feature_cols[i] for i in range(len(feature_cols)) if reject[i]]
    stable  = [feature_cols[i] for i in range(len(feature_cols)) if not reject[i]]
    ks_table = pd.DataFrame({"feature": feature_cols, "ks_pvalue": pvals,
                             "flagged_shifted": reject}).sort_values("ks_pvalue")
    return stable, shifted, ks_table


def evaluate(model, X_eval, y_eval, int_to_class):
    yp    = model.predict(X_eval)
    proba = model.predict_proba(X_eval)
    wf1   = f1_score(y_eval, yp, average="weighted")
    try:
        auc = roc_auc_score(y_eval, proba, multi_class="ovr",
                            average="weighted", labels=model.classes_)
    except Exception:
        auc = float("nan")
    per = f1_score(y_eval, yp, average=None, labels=sorted(int_to_class))
    per_named = {int_to_class[i]: float(per[k])
                 for k, i in enumerate(sorted(int_to_class))}
    return {"weighted_f1": float(wf1), "roc_auc_ovr_weighted": float(auc),
            "per_class_f1": per_named}


def run_pilot(files=FILES, per_class=PER_CLASS_SAMPLE, m=MULTIPLIER,
              seed=RANDOM_SEED, test_size=TEST_SIZE, alpha=HOLM_ALPHA,
              rf_trees=RF_TREES, out_prefix=OUTPUT_PREFIX):
    t0 = time.time()
    print("=" * 64, "\n1) LOAD + CLEAN\n" + "-" * 64)
    data = load_and_clean(files, LABEL_COL, LABEL_NORMALISE, CLASSES, per_class, seed)
    X, y, features, class_to_int = prepare_xy(data, NON_FEATURE_COLS, CLASSES)
    int_to_class = {i: c for c, i in class_to_int.items()}
    flow_bytes_col = resolve_flow_bytes(features)

    dist = {int_to_class[i]: int((y == i).sum()) for i in sorted(int_to_class)}
    print(f"\n  final rows={len(X):,}  features={len(features)}  "
          f"flow_bytes_col={flow_bytes_col}")
    print(f"  class distribution: {dist}")
    missing = [c for c in BYTE_FEATURES if c not in features]
    if missing:
        print(f"  !! WARNING: byte features not found in data: {missing}")
        print(f"     check names against: {features[:12]} ...")

    print("\n2) SPLIT (stratified, one split)\n" + "-" * 64)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    print(f"  train={len(Xtr):,}  test={len(Xte):,}")

    # ---- C1: classical baseline (train original, eval original) ----
    print("\n3) C1  classical baseline\n" + "-" * 64)
    rf1 = RandomForestClassifier(n_estimators=rf_trees, random_state=seed, n_jobs=-1)
    rf1.fit(Xtr[features], ytr)
    C1 = evaluate(rf1, Xte[features], yte, int_to_class)
    print(f"  weighted F1={C1['weighted_f1']:.4f}  ROC-AUC={C1['roc_auc_ovr_weighted']:.4f}")

    # ---- C2: same C1 model, approximated test ----
    print("\n4) C2  approximated shift, NO pipeline (reuses C1 model)\n" + "-" * 64)
    Xte_approx = make_approx(Xte, BYTE_FEATURES, flow_bytes_col, m)
    C2 = evaluate(rf1, Xte_approx[features], yte, int_to_class)
    print(f"  weighted F1={C2['weighted_f1']:.4f}  ROC-AUC={C2['roc_auc_ovr_weighted']:.4f}")

    # ---- Pipeline: KS+Holm + MI, TRAIN PARTITION ONLY (leakage-safe) ----
    print("\n5) STABLE-FEATURE PIPELINE  (computed on TRAIN only)\n" + "-" * 64)
    Xtr_approx = make_approx(Xtr, BYTE_FEATURES, flow_bytes_col, m)
    stable, shifted, ks_table = stable_features_via_ks(
        Xtr[features], Xtr_approx[features], features, alpha)
    print(f"  flagged SHIFTED ({len(shifted)}): {shifted}")
    print(f"  kept STABLE: {len(stable)} features")
    mi = mutual_info_classif(Xtr[stable], ytr, random_state=seed)
    mi_rank = pd.Series(mi, index=stable).sort_values(ascending=False)
    print(f"  top stable features by MI: {list(mi_rank.head(8).index)}")

    # ---- C3: NEW model on stable features, eval approximated test ----
    print("\n6) C3  approximated shift, WITH pipeline\n" + "-" * 64)
    rf3 = RandomForestClassifier(n_estimators=rf_trees, random_state=seed, n_jobs=-1)
    rf3.fit(Xtr[stable], ytr)
    C3 = evaluate(rf3, Xte_approx[stable], yte, int_to_class)
    print(f"  weighted F1={C3['weighted_f1']:.4f}  ROC-AUC={C3['roc_auc_ovr_weighted']:.4f}")

    # ---- Report ----
    d_12 = C2["weighted_f1"] - C1["weighted_f1"]
    d_23 = C3["weighted_f1"] - C2["weighted_f1"]
    print("\n" + "=" * 64 + "\nSUMMARY (weighted F1)\n" + "-" * 64)
    print(f"  C1 baseline           {C1['weighted_f1']:.4f}")
    print(f"  C2 shift, no pipeline {C2['weighted_f1']:.4f}   (C1->C2 delta {d_12:+.4f})")
    print(f"  C3 shift, pipeline    {C3['weighted_f1']:.4f}   (C2->C3 delta {d_23:+.4f})")
    print("\nPER-CLASS F1")
    hdr = f"  {'class':14s}{'C1':>9s}{'C2':>9s}{'C3':>9s}"
    print(hdr)
    for c in CLASSES:
        print(f"  {c:14s}{C1['per_class_f1'].get(c, float('nan')):>9.4f}"
              f"{C2['per_class_f1'].get(c, float('nan')):>9.4f}"
              f"{C3['per_class_f1'].get(c, float('nan')):>9.4f}")

    print("\nREAD IT LIKE THIS")
    print(f"  C1->C2 {d_12:+.4f} : {'shift degrades detection (effect present)' if d_12 < -0.01 else 'little/no effect from the shift'}")
    print(f"  C2->C3 {d_23:+.4f} : {'pipeline recovers performance' if d_23 > 0.01 else ('pipeline neutral' if abs(d_23) <= 0.01 else 'pipeline hurts (dropped useful features)')}")

    # ---- Save for Chapter 3 ----
    results = {"config": {"multiplier": m, "seed": seed, "test_size": test_size,
                          "holm_alpha": alpha, "per_class_sample": per_class,
                          "rf_trees": rf_trees, "n_features": len(features),
                          "class_distribution": dist},
               "stable_features": stable, "shifted_features": shifted,
               "mi_ranking": {k: float(v) for k, v in mi_rank.items()},
               "C1": C1, "C2": C2, "C3": C3,
               "deltas": {"C1_to_C2_weighted_f1": d_12,
                          "C2_to_C3_weighted_f1": d_23}}
    with open(f"{out_prefix}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    ks_table.to_csv(f"{out_prefix}_ks.csv", index=False)
    print(f"\nsaved: {out_prefix}_results.json  and  {out_prefix}_ks.csv")
    print(f"elapsed: {time.time() - t0:.1f}s")
    return results


if __name__ == "__main__":
    run_pilot()
