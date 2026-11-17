# ================================================================
# OPTION B — FULL STATISTICAL EXPERIMENT
# Three classifiers (Random Forest, XGBoost, MLP) x conditions C1/C2/C3
# x 10 repeated stratified splits (seeds 0-9), full ML-KEM-768 footprint.
#
# Per classifier it reports, across the 10 splits:
#   - mean and 95% CI of weighted F1, ROC-AUC, and per-class F1 (C1/C2/C3)
#   - C1->C2 and C2->C3 deltas
#   - paired t-tests (C1 vs C2, C2 vs C3) with p-values
#   - Cohen's d (paired) for each comparison
#   - one-way ANOVA across C1/C2/C3 with eta-squared
#
# Leakage rule: KS+Holm (hard filter) and MI (ranking-only) are computed
# on EACH split's TRAINING partition only. MLP uses a StandardScaler frozen
# on the original training partition. Memory-safe chunked loader is reused.
#
# Requires: pilot_pqc_stable_pipeline.py and pilot_option_b.py in the folder.
# ================================================================

import json, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import mutual_info_classif

import pilot_pqc_stable_pipeline as base   # validated loader / evaluate / KS+Holm
import pilot_option_b as ob                # validated ML-KEM-768 footprint

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

# ============================ CONFIG ============================
N_SPLITS        = 10                 # repeated stratified splits
SPLIT_SEEDS     = list(range(N_SPLITS))   # 0..9 -> the only deliberate variation
MODEL_SEED      = 42                 # fixed, so splits are the only thing that changes
PER_CLASS       = 30000
TEST_SIZE       = 0.30
HOLM_ALPHA      = 0.05
RF_TREES        = 200
MI_SUBSAMPLE    = 20000              # MI is reporting-only; subsample for speed
DROP_IDENTIFIERS = False             # B3 toggle (needs supervisor sign-off)
OUT_PREFIX      = "option_b_full"

FILES = ob.FILES                     # same Drive paths as Option B
# ================================================================


def make_model(kind):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=RF_TREES, random_state=MODEL_SEED, n_jobs=-1)
    if kind == "xgb":
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.3,
                             tree_method="hist", subsample=0.9, colsample_bytree=0.9,
                             n_jobs=-1, random_state=MODEL_SEED,
                             eval_metric="mlogloss", verbosity=0)
    return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True,
                         n_iter_no_change=10, random_state=MODEL_SEED)


CLASSIFIERS = [("rf", "Random Forest", False),
               ("xgb", "XGBoost", False),
               ("mlp", "MLP (scaled)", True)]


def one_split(split_seed, X, y, features, int_to_class):
    """Run C1/C2/C3 for all classifiers on one stratified split."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE,
                                          random_state=split_seed, stratify=y)
    Xte_a = ob.make_approx_mlkem(Xte)     # perturbed test (full footprint)
    Xtr_a = ob.make_approx_mlkem(Xtr)     # perturbed train (for KS only)

    # pipeline on TRAIN only: KS+Holm hard filter, MI ranking-only
    stable, shifted, _ = base.stable_features_via_ks(Xtr[features], Xtr_a[features],
                                                     features, HOLM_ALPHA)
    if MI_SUBSAMPLE and len(Xtr) > MI_SUBSAMPLE:
        idx = np.random.RandomState(split_seed).choice(len(Xtr), MI_SUBSAMPLE, replace=False)
        mi = mutual_info_classif(Xtr[stable].values[idx], ytr[idx], random_state=MODEL_SEED)
    else:
        mi = mutual_info_classif(Xtr[stable], ytr, random_state=MODEL_SEED)
    mi_rank = {f: float(v) for f, v in zip(stable, mi)}

    res = {}
    for kind, _, scale in CLASSIFIERS:
        if kind == "xgb" and not HAVE_XGB:
            continue
        if scale:
            sc1 = StandardScaler().fit(Xtr[features])
            tr1, te1, te2 = sc1.transform(Xtr[features]), sc1.transform(Xte[features]), sc1.transform(Xte_a[features])
        else:
            tr1, te1, te2 = Xtr[features].values, Xte[features].values, Xte_a[features].values
        m1 = make_model(kind).fit(tr1, ytr)
        C1 = base.evaluate(m1, te1, yte, int_to_class)     # baseline
        C2 = base.evaluate(m1, te2, yte, int_to_class)     # shift, no pipeline (same model)

        if scale:
            sc3 = StandardScaler().fit(Xtr[stable])
            tr3, te3 = sc3.transform(Xtr[stable]), sc3.transform(Xte_a[stable])
        else:
            tr3, te3 = Xtr[stable].values, Xte_a[stable].values
        m3 = make_model(kind).fit(tr3, ytr)
        C3 = base.evaluate(m3, te3, yte, int_to_class)     # shift, with pipeline

        res[kind] = {"C1": C1, "C2": C2, "C3": C3}
    return res, stable, shifted, mi_rank


# ---------- statistics helpers ----------
def ci95(arr):
    a = np.asarray(arr, float); m = a.mean()
    if len(a) < 2:
        return m, m, m
    se = a.std(ddof=1) / np.sqrt(len(a))
    h = stats.t.ppf(0.975, len(a) - 1) * se
    return m, m - h, m + h

def cohen_d_paired(after, before):
    d = np.asarray(after, float) - np.asarray(before, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0

def eta_squared(groups):
    allv = np.concatenate([np.asarray(g, float) for g in groups]); gm = allv.mean()
    ss_tot = ((allv - gm) ** 2).sum()
    ss_bet = sum(len(g) * (np.mean(g) - gm) ** 2 for g in groups)
    return float(ss_bet / ss_tot) if ss_tot > 0 else 0.0


def run_full(files=FILES, n_splits=N_SPLITS, per_class=PER_CLASS,
             drop_identifiers=DROP_IDENTIFIERS, out_prefix=OUT_PREFIX):
    t0 = time.time()
    print("=" * 70)
    print(f"OPTION B FULL EXPERIMENT  |  {n_splits} splits  |  "
          f"classifiers: RF, {'XGBoost, ' if HAVE_XGB else '(xgboost missing) '}MLP")
    print(f"drop_identifiers={drop_identifiers}  footprint={ob.MLKEM}")
    print("=" * 70)
    if not HAVE_XGB:
        print("  !! xgboost not installed -> run  !pip install xgboost  and re-run for all three.")

    data = base.load_and_clean(files, base.LABEL_COL, base.LABEL_NORMALISE,
                               base.CLASSES, per_class, MODEL_SEED)
    X, y, features, class_to_int = base.prepare_xy(data, base.NON_FEATURE_COLS, base.CLASSES)
    int_to_class = {i: c for c, i in class_to_int.items()}
    if drop_identifiers:
        rm = [c for c in ob.IDENTIFIER_FEATURES if c in features]
        features = [c for c in features if c not in ob.IDENTIFIER_FEATURES]
        print(f"  B3: dropped {rm} -> {len(features)} features")
    print(f"  data: {len(X):,} rows, {len(features)} model features, classes={base.CLASSES}\n")

    kinds = [k for k, _, _ in CLASSIFIERS if not (k == "xgb" and not HAVE_XGB)]
    # accumulators: acc[kind][cond]['wf1'|'auc'|'per'][...]
    acc = {k: {c: {"wf1": [], "auc": [], "per": {cl: [] for cl in base.CLASSES}}
               for c in ("C1", "C2", "C3")} for k in kinds}
    mi_accum = {}
    shifted_sets = []

    for i, s in enumerate(SPLIT_SEEDS[:n_splits], 1):
        res, stable, shifted, mi_rank = one_split(s, X, y, features, int_to_class)
        shifted_sets.append(tuple(sorted(shifted)))
        for f, v in mi_rank.items():
            mi_accum.setdefault(f, []).append(v)
        for k in kinds:
            for c in ("C1", "C2", "C3"):
                acc[k][c]["wf1"].append(res[k][c]["weighted_f1"])
                acc[k][c]["auc"].append(res[k][c]["roc_auc_ovr_weighted"])
                for cl in base.CLASSES:
                    acc[k][c]["per"][cl].append(res[k][c]["per_class_f1"][cl])
        # incremental save so a disconnect never loses progress
        with open(f"{out_prefix}_raw.json", "w") as fh:
            json.dump({"acc": acc, "n_done": i}, fh)
        print(f"  split {i}/{n_splits} (seed {s}) done  |  elapsed {time.time()-t0:.0f}s")

    # ---------- summarise ----------
    summary = {"config": {"n_splits": n_splits, "per_class": per_class,
                          "drop_identifiers": drop_identifiers, "footprint": ob.MLKEM,
                          "n_features": len(features),
                          "shifted_features_consistent": len(set(shifted_sets)) == 1,
                          "n_shifted": len(shifted_sets[0])},
               "mi_ranking_mean": dict(sorted(((f, float(np.mean(v))) for f, v in mi_accum.items()),
                                              key=lambda kv: -kv[1])),
               "classifiers": {}}

    print("\n" + "=" * 70 + f"\nRESULTS  (mean [95% CI], n = {n_splits} splits)\n" + "=" * 70)
    for k, label, _ in CLASSIFIERS:
        if k not in kinds:
            continue
        wf1 = {c: acc[k][c]["wf1"] for c in ("C1", "C2", "C3")}
        auc = {c: acc[k][c]["auc"] for c in ("C1", "C2", "C3")}
        m1, l1, h1 = ci95(wf1["C1"]); m2, l2, h2 = ci95(wf1["C2"]); m3, l3, h3 = ci95(wf1["C3"])
        a1 = ci95(auc["C1"]); a2 = ci95(auc["C2"]); a3 = ci95(auc["C3"])
        d12 = m2 - m1; d23 = m3 - m2
        t12 = stats.ttest_rel(wf1["C1"], wf1["C2"]); t23 = stats.ttest_rel(wf1["C2"], wf1["C3"])
        cd12 = cohen_d_paired(wf1["C2"], wf1["C1"]); cd23 = cohen_d_paired(wf1["C3"], wf1["C2"])
        F, pA = stats.f_oneway(wf1["C1"], wf1["C2"], wf1["C3"])
        eta = eta_squared([wf1["C1"], wf1["C2"], wf1["C3"]])

        print(f"\n### {label}  (n={n_splits})")
        print(f"  weighted F1   C1 {m1:.4f} [{l1:.4f},{h1:.4f}]   "
              f"C2 {m2:.4f} [{l2:.4f},{h2:.4f}]   C3 {m3:.4f} [{l3:.4f},{h3:.4f}]")
        print(f"  ROC-AUC       C1 {a1[0]:.4f} [{a1[1]:.4f},{a1[2]:.4f}]   "
              f"C2 {a2[0]:.4f} [{a2[1]:.4f},{a2[2]:.4f}]   C3 {a3[0]:.4f} [{a3[1]:.4f},{a3[2]:.4f}]")
        print(f"  C1->C2 delta  {d12:+.4f}   paired t p={t12.pvalue:.2e}   Cohen d={cd12:+.2f}   "
              f"({'shift DEGRADES' if d12 < 0 and t12.pvalue < 0.05 else 'no sig. change'})")
        print(f"  C2->C3 delta  {d23:+.4f}   paired t p={t23.pvalue:.2e}   Cohen d={cd23:+.2f}   "
              f"({'pipeline RECOVERS' if d23 > 0 and t23.pvalue < 0.05 else 'no sig. change'})")
        print(f"  ANOVA C1/C2/C3  F={F:.2f}  p={pA:.2e}  eta^2={eta:.3f}")
        print(f"  per-class F1 (mean C1 / C2 / C3):")
        for cl in base.CLASSES:
            print(f"    {cl:14s}{np.mean(acc[k]['C1']['per'][cl]):.4f} / "
                  f"{np.mean(acc[k]['C2']['per'][cl]):.4f} / {np.mean(acc[k]['C3']['per'][cl]):.4f}")

        summary["classifiers"][k] = {
            "label": label,
            "weighted_f1": {c: {"mean": ci95(wf1[c])[0], "ci_low": ci95(wf1[c])[1], "ci_high": ci95(wf1[c])[2]} for c in ("C1","C2","C3")},
            "roc_auc": {c: {"mean": ci95(auc[c])[0], "ci_low": ci95(auc[c])[1], "ci_high": ci95(auc[c])[2]} for c in ("C1","C2","C3")},
            "per_class_f1": {cl: {c: {"mean": ci95(acc[k][c]["per"][cl])[0],
                                      "ci_low": ci95(acc[k][c]["per"][cl])[1],
                                      "ci_high": ci95(acc[k][c]["per"][cl])[2]} for c in ("C1","C2","C3")}
                            for cl in base.CLASSES},
            "delta_C1_C2": d12, "delta_C2_C3": d23,
            "ttest_C1_C2_p": float(t12.pvalue), "ttest_C2_C3_p": float(t23.pvalue),
            "cohen_d_C1_C2": cd12, "cohen_d_C2_C3": cd23,
            "anova_F": float(F), "anova_p": float(pA), "eta_squared": eta,
            "raw_weighted_f1": wf1}

    with open(f"{out_prefix}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nshifted-feature set identical across splits: {summary['config']['shifted_features_consistent']} "
          f"({summary['config']['n_shifted']} features)")
    print(f"saved: {out_prefix}_summary.json (+ {out_prefix}_raw.json)   total elapsed {time.time()-t0:.0f}s")
    return summary


if __name__ == "__main__":
    run_full()
