# ================================================================
# COMBINED A1 + B2 REVISED-PIPELINE RERUN
#
# Two pipeline changes vs the locked design, applied together:
#
#   A1  DROP-UNMODELLABLE  -- features the approximated footprint cannot
#       recompute (packet-length Std/Var, IAT Std, Active/Idle families)
#       are removed from the candidate pool BEFORE the KS filter. They
#       were previously certified "stable" only because the transform
#       never touched them (KS D = 0 by construction), which is a
#       construct-validity hole: the pipeline cannot claim a feature is
#       distributionally stable under ML-KEM if it never modelled that
#       feature's response. Verified against the package: the 15 dropped
#       features all have D = 0 and none were in the shifted set.
#
#   B2  MI-THRESHOLD SELECTION  -- after KS+Holm yields the stable set,
#       mutual information is used as an ACTUAL SELECTION step (not just
#       ranking): features with MI below a threshold are dropped, and C3
#       trains on the MI-selected stable subset. This makes the pipeline
#       satisfy RO2 literally ("stable AND useful for classification").
#       The threshold is swept so the choice is a reported sensitivity
#       curve, not a magic number; a headline threshold is also recorded.
#
# Resulting pipeline:  drop-unmodellable -> KS+Holm (modellable pool)
#                      -> MI threshold selection -> C3 on MI-stable subset
#
# Leakage rules unchanged: drop-list is fixed and label-independent;
# KS+Holm and MI computed on each split's TRAINING partition only;
# MLP scaler frozen on the relevant training partition; nothing fitted
# on any test set.
#
# Honesty note baked into the output: removing the unmodellable family
# (~13.5% of prior C3 importance for RF) and then MI-thresholding will
# REDUCE the stable set. Recovery may no longer reach within delta=0.01.
# That is a result to report, not a bug to suppress -- it is the honest
# statement that the MODELLABLE, LABEL-RELEVANT stable features alone
# recover performance to within the reported margin.
#
# Requires the usual modules on sys.path. Reuses ob.make_approx_mlkem,
# the transfer footprint variants, and option_b_full's helpers.
# ================================================================

import json, time, warnings
import numpy as np
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

import pilot_pqc_stable_pipeline as base
import pilot_option_b as ob
import option_b_full_experiment as full

warnings.filterwarnings("ignore")
try:
    from statsmodels.stats.multitest import multipletests
    HAVE_SM = True
except Exception:
    HAVE_SM = False

# ============================ CONFIG ============================
N_SPLITS    = 10
SPLIT_SEEDS = list(range(N_SPLITS))
MODEL_SEED  = 42
PER_CLASS   = 30000
TEST_SIZE   = 0.30
HOLM_ALPHA  = 0.05
DROP_IDENTIFIERS = False        # set True + change OUT_PREFIX for the B3 variant

# --- A1: the unmodellable drop-list (verified D=0, none shifted) ---
APPLY_DROP_UNMODELLABLE = True
UNMODELLABLE = ["Fwd IAT Std", "Flow IAT Std", "Fwd Pkt Len Std", "Bwd Pkt Len Std",
                "Bwd IAT Std", "Pkt Len Std", "Pkt Len Var",
                "Active Mean", "Active Std", "Active Max", "Active Min",
                "Idle Mean", "Idle Std", "Idle Max", "Idle Min"]

# --- B2: MI-threshold selection sweep ---
APPLY_MI_SELECTION = True
MI_THRESHOLDS   = [0.0, 0.01, 0.05, 0.10, 0.20]   # 0.0 == KS-only (no MI cut)
MI_HEADLINE     = 0.05                              # single number for the main tables
MI_SUBSAMPLE    = 20000

# --- transfer footprints (same as the transfer experiment) ---
FPREP    = "hybrid768"
VARIANTS = {
    "hybrid768":    {"d_fwd_bytes": 1184.0, "d_bwd_bytes": 1088.0, "d_fwd_pkts": 1.0, "d_bwd_pkts": 1.0, "hdr_bytes": 40.0, "added_pkt_size": 1460.0},
    "pure768":      {"d_fwd_bytes": 1152.0, "d_bwd_bytes": 1056.0, "d_fwd_pkts": 1.0, "d_bwd_pkts": 1.0, "hdr_bytes": 40.0, "added_pkt_size": 1460.0},
    "hybrid768_p2": {"d_fwd_bytes": 1184.0, "d_bwd_bytes": 1088.0, "d_fwd_pkts": 2.0, "d_bwd_pkts": 2.0, "hdr_bytes": 40.0, "added_pkt_size": 1460.0},
    "hybrid512":    {"d_fwd_bytes":  800.0, "d_bwd_bytes":  768.0, "d_fwd_pkts": 1.0, "d_bwd_pkts": 1.0, "hdr_bytes": 40.0, "added_pkt_size": 1460.0},
}

RUN_TRANSFER = False           # R_oracle phase: main matched footprint only
OUT_PREFIX   = "r_oracle_main"
FILES = ob.FILES
# ================================================================


def ks_holm_stable(Xtr_orig, Xtr_approx, cols, alpha):
    """KS two-sample per feature (TRAIN only) + Holm; returns (stable, shifted, D-dict)."""
    pvals, Ds = [], {}
    for c in cols:
        a, b = Xtr_orig[c].values, Xtr_approx[c].values
        if np.array_equal(a, b):
            D, p = 0.0, 1.0
        else:
            D, p = ks_2samp(a, b)
        pvals.append(p); Ds[c] = float(D)
    pvals = np.asarray(pvals)
    if HAVE_SM:
        reject = multipletests(pvals, alpha=alpha, method="holm")[0]
    else:
        order = np.argsort(pvals); m = len(pvals); reject = np.zeros(m, bool)
        for rank, idx in enumerate(order):
            if pvals[idx] <= alpha / (m - rank):
                reject[idx] = True
            else:
                break
    stable  = [cols[i] for i in range(len(cols)) if not reject[i]]
    shifted = [cols[i] for i in range(len(cols)) if reject[i]]
    return stable, shifted, Ds


def mi_rank(Xtr, stable, ytr, seed):
    """MI of each stable feature with the label (TRAIN only)."""
    if MI_SUBSAMPLE and len(Xtr) > MI_SUBSAMPLE:
        idx = np.random.RandomState(seed).choice(len(Xtr), MI_SUBSAMPLE, replace=False)
        mi = mutual_info_classif(Xtr[stable].values[idx], ytr[idx], random_state=MODEL_SEED)
    else:
        mi = mutual_info_classif(Xtr[stable].values, ytr, random_state=MODEL_SEED)
    return {f: float(v) for f, v in zip(stable, mi)}


def fit_eval(kind, scale, Xtr_df, cols, ytr, Xev_df, yev, i2c):
    if scale:
        sc = StandardScaler().fit(Xtr_df[cols])
        m = full.make_model(kind).fit(sc.transform(Xtr_df[cols]), ytr)
        ev = sc.transform(Xev_df[cols])
    else:
        m = full.make_model(kind).fit(Xtr_df[cols].values, ytr)
        ev = Xev_df[cols].values
    return base.evaluate(m, ev, yev, i2c)


def one_split(seed, X, y, full_features, cand_features, i2c, kinds, variants):
    """
    full_features : all 70 model features -> used for C1 and C2 (real-world conditions)
    cand_features : modellable subset (70 - unmodellable [- identifiers]) -> KS/MI/C3 only
    """
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE,
                                          random_state=seed, stratify=y)
    Xtr_v = {v: ob.make_approx_mlkem(Xtr, cfg) for v, cfg in variants.items()}
    Xte_v = {v: ob.make_approx_mlkem(Xte, cfg) for v, cfg in variants.items()}

    # KS+Holm on the CANDIDATE pool only (drops unmodellable from stability testing)
    stable_v, shifted_v, D_v = {}, {}, {}
    for v in variants:
        s, sh, D = ks_holm_stable(Xtr[cand_features], Xtr_v[v][cand_features],
                                  cand_features, HOLM_ALPHA)
        stable_v[v], shifted_v[v], D_v[v] = s, sh, D

    # MI ranking of the KS-stable set under the prep footprint (candidate pool)
    mi_prep = mi_rank(Xtr, stable_v[FPREP], ytr, seed)

    def mi_subset(thr):
        sub = [f for f in stable_v[FPREP] if mi_prep.get(f, 0.0) >= thr]
        return sub if sub else stable_v[FPREP][:1]

    res = {}
    for kind, label, scale in full.CLASSIFIERS:
        if kind == "xgb" and not full.HAVE_XGB:
            continue
        # C1: baseline on ALL 70 features (real-world; matches archived runs)
        C1 = fit_eval(kind, scale, Xtr, full_features, ytr, Xte, yte, i2c)
        out = {"C1": C1, "per_variant": {}, "mi_sweep": {}, "selected_counts": {}}

        # MI-threshold sweep: C3 at each threshold (matched/prep footprint)
        for thr in MI_THRESHOLDS:
            sub = mi_subset(thr)
            C3 = fit_eval(kind, scale, Xtr, sub, ytr, Xte_v[FPREP], yte, i2c)
            out["mi_sweep"][str(thr)] = C3["weighted_f1"]
            out["selected_counts"][str(thr)] = len(sub)

        head_sub = mi_subset(MI_HEADLINE) if APPLY_MI_SELECTION else stable_v[FPREP]
        for v in variants:
            # C2: ALL-70 baseline model evaluated on the perturbed test set (real-world
            # no-mitigation condition; identical to archived C2 under matched seeds)
            C2 = fit_eval(kind, scale, Xtr, full_features, ytr, Xte_v[v], yte, i2c)
            # C3: MI-selected modellable-stable subset
            C3 = fit_eval(kind, scale, Xtr, head_sub, ytr, Xte_v[v], yte, i2c)
            # R_oracle: train on the approximated shifted training set and test on
            # the approximated shifted test set, using all 70 features. This is an
            # oracle retraining baseline, not the proposed mitigation.
            R_oracle = fit_eval(kind, scale, Xtr_v[v], full_features, ytr, Xte_v[v], yte, i2c)
            out["per_variant"][v] = {"C2": C2, "C3": C3, "R_oracle": R_oracle}
        res[kind] = out
    return res, stable_v, shifted_v, D_v, mi_prep


def run_revised(run_transfer=RUN_TRANSFER):
    t0 = time.time()
    variants = VARIANTS if run_transfer else {FPREP: VARIANTS[FPREP]}
    kinds = [k for k, _, _ in full.CLASSIFIERS if not (k == "xgb" and not full.HAVE_XGB)]

    print("=" * 74)
    print(f"REVISED PIPELINE (A1 drop-unmodellable + B2 MI-selection) | "
          f"{N_SPLITS} splits | transfer={run_transfer}")
    print(f"drop_unmodellable={APPLY_DROP_UNMODELLABLE} ({len(UNMODELLABLE)} feats) | "
          f"MI selection={APPLY_MI_SELECTION} thresholds={MI_THRESHOLDS} headline={MI_HEADLINE}")
    print(f"drop_identifiers={DROP_IDENTIFIERS} | per_class={PER_CLASS} | seed={MODEL_SEED}")
    print("=" * 74)
    if not full.HAVE_XGB:
        print("  !! xgboost not installed -> !pip install xgboost")

    data = base.load_and_clean(FILES, base.LABEL_COL, base.LABEL_NORMALISE,
                               base.CLASSES, PER_CLASS, MODEL_SEED)
    X, y, full_features, c2i = base.prepare_xy(data, base.NON_FEATURE_COLS, base.CLASSES)
    i2c = {i: c for c, i in c2i.items()}

    # full_features (all 70) drive C1 and C2 -- the real-world conditions, unchanged.
    # cand_features is the restricted pool for KS / MI / C3 candidacy only.
    n_full = len(full_features)
    dropped = []
    cand_features = list(full_features)
    if APPLY_DROP_UNMODELLABLE:
        dropped = [c for c in UNMODELLABLE if c in cand_features]
        cand_features = [c for c in cand_features if c not in UNMODELLABLE]
    if DROP_IDENTIFIERS:   # B3 toggle: identifiers also leave candidacy (still in C1/C2)
        cand_features = [c for c in cand_features if c not in ob.IDENTIFIER_FEATURES]
    print(f"  C1/C2 features: {n_full} (unchanged, real-world)")
    print(f"  KS/MI/C3 candidate pool: {len(cand_features)} "
          f"(dropped {len(dropped)} unmodellable{', + identifiers' if DROP_IDENTIFIERS else ''})")
    print(f"  data: {len(X):,} rows\n")

    acc = {k: {"C1": [],
               **{v: {"C2": [], "C3": [], "R_oracle": []} for v in variants}} for k in kinds}
    mi_sweep_acc = {k: {str(t): [] for t in MI_THRESHOLDS} for k in kinds}
    sel_counts   = {k: {str(t): [] for t in MI_THRESHOLDS} for k in kinds}
    stable_sizes, shifted_sets = [], []
    raw = {}

    for i, seed in enumerate(SPLIT_SEEDS[:N_SPLITS], 1):
        res, stable_v, shifted_v, D_v, mi_prep = one_split(
            seed, X, y, full_features, cand_features, i2c, kinds, variants)
        raw[str(seed)] = {"results": {k: {"C1": res[k]["C1"],
                                          "per_variant": res[k]["per_variant"],
                                          "mi_sweep": res[k]["mi_sweep"],
                                          "selected_counts": res[k]["selected_counts"]} for k in kinds},
                          "stable_prep": stable_v[FPREP], "shifted_prep": sorted(shifted_v[FPREP]),
                          "mi_prep": mi_prep, "ks_D_prep": D_v[FPREP]}
        stable_sizes.append(len(stable_v[FPREP]))
        shifted_sets.append(tuple(sorted(shifted_v[FPREP])))
        for k in kinds:
            acc[k]["C1"].append(res[k]["C1"]["weighted_f1"])
            for v in variants:
                acc[k][v]["C2"].append(res[k]["per_variant"][v]["C2"]["weighted_f1"])
                acc[k][v]["C3"].append(res[k]["per_variant"][v]["C3"]["weighted_f1"])
                acc[k][v]["R_oracle"].append(res[k]["per_variant"][v]["R_oracle"]["weighted_f1"])
            for t in MI_THRESHOLDS:
                mi_sweep_acc[k][str(t)].append(res[k]["mi_sweep"][str(t)])
                sel_counts[k][str(t)].append(res[k]["selected_counts"][str(t)])
        with open(f"{OUT_PREFIX}_raw.json", "w") as fh:
            json.dump(raw, fh)
        print(f"  split {i}/{N_SPLITS} (seed {seed}) | KS-stable={len(stable_v[FPREP])} | "
              f"elapsed {time.time()-t0:.0f}s")

    # ------------------------------ summary ------------------------------
    fmt = lambda a: "{:.4f} [{:.4f},{:.4f}]".format(*full.ci95(a))
    summary = {"config": {"n_splits": N_SPLITS, "split_seeds": SPLIT_SEEDS,
                          "model_seed": MODEL_SEED, "per_class": PER_CLASS,
                          "holm_alpha": HOLM_ALPHA, "drop_identifiers": DROP_IDENTIFIERS,
                          "apply_drop_unmodellable": APPLY_DROP_UNMODELLABLE,
                          "unmodellable_dropped": dropped,
                          "apply_mi_selection": APPLY_MI_SELECTION,
                          "mi_thresholds": MI_THRESHOLDS, "mi_headline": MI_HEADLINE,
                          "fprep": FPREP, "variants": variants,
                          "n_features_full": n_full,
                          "n_candidate_features": len(cand_features),
                          "ks_stable_mean": float(np.mean(stable_sizes)),
                          "shifted_consistent": len(set(shifted_sets)) == 1,
                          "n_shifted": len(shifted_sets[0])},
               "classifiers": {}}

    for kind, label, _ in full.CLASSIFIERS:
        if kind not in kinds:
            continue
        A = acc[kind]; C1 = np.array(A["C1"])
        print("\n" + "=" * 74 + f"\n### {label}  (n={N_SPLITS})\n" + "-" * 74)
        print(f"  C1 baseline: {fmt(A['C1'])}")

        # MI-threshold sweep (matched footprint) — the B2 deliverable
        print(f"\n  MI-THRESHOLD SWEEP (C3 on prep footprint; recovery gap vs C1)")
        print(f"  {'thr':>6s}{'n selected':>12s}{'C3 wF1':>22s}{'C3-C1 gap':>22s}"
              f"{'within d=0.01':>14s}")
        sweep_out = {}
        for t in MI_THRESHOLDS:
            arr = np.array(mi_sweep_acc[kind][str(t)])
            gap = arr - C1
            nsel = float(np.mean(sel_counts[kind][str(t)]))
            within = bool(abs(full.ci95(gap)[0]) < 0.01 and full.ci95(gap)[1] > -0.01 and full.ci95(gap)[2] < 0.01)
            print(f"  {t:>6}{nsel:>12.1f}{fmt(arr):>22s}{fmt(gap):>22s}{str(within):>14s}")
            sweep_out[str(t)] = {"n_selected_mean": nsel, "C3": full.ci95(arr),
                                 "gap_vs_C1": full.ci95(gap), "within_delta": within}

        # headline transfer matrix
        print(f"\n  TRANSFER (C3 at MI>={MI_HEADLINE}; prep={FPREP})")
        if RUN_TRANSFER:
            print("  " + f"{'cond':6s}" + "".join(f"{v:>22s}" for v in variants))
            for cond in ("C2", "C3", "R_oracle"):
                print("  " + f"{cond:6s}" + "".join(f"{fmt(A[v][cond]):>22s}" for v in variants))
            c3 = np.array([A[v]["C3"] for v in variants])
            spread = float(np.max(c3.max(0) - c3.min(0)))
            print(f"  C3 invariance across footprints (max per-split spread): {spread:.2e} "
                  f"{'(near-zero as expected)' if spread < 5e-3 else '(report)'}")
        else:
            print(f"  C2 {fmt(A[FPREP]['C2'])}   C3 {fmt(A[FPREP]['C3'])}   R_oracle {fmt(A[FPREP]['R_oracle'])}")

        summary["classifiers"][kind] = {
            "label": label, "C1": full.ci95(A["C1"]),
            "mi_sweep": sweep_out,
            "transfer": {cond: {v: full.ci95(A[v][cond]) for v in variants}
                         for cond in ("C2", "C3", "R_oracle")},
            "raw_wf1": {"C1": A["C1"],
                        **{v: {"C2": A[v]["C2"], "C3": A[v]["C3"], "R_oracle": A[v]["R_oracle"]} for v in variants}}}

    with open(f"{OUT_PREFIX}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n" + "=" * 74)
    print(f"KS-stable set: mean {np.mean(stable_sizes):.1f} features "
          f"(from {len(cand_features)} modellable candidates); shifted consistent: "
          f"{len(set(shifted_sets))==1} ({len(shifted_sets[0])} features)")
    print(f"saved: {OUT_PREFIX}_summary.json (+ {OUT_PREFIX}_raw.json)  "
          f"total elapsed {time.time()-t0:.0f}s")
    return summary


if __name__ == "__main__":
    run_revised()
