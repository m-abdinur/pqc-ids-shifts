# ================================================================
# REVISED COMPONENT ABLATION + RANDOM-SUBSET CONTROL
# Final A1+B2 pipeline version
#
# Purpose:
#   1. Attribute C1 -> C2 degradation to footprint component groups.
#   2. Test whether the final revised C3 pipeline recovers after T4_full.
#   3. Compare C3_true against a random same-size feature subset.
#
# Key final-pipeline rules:
#   - C1/T1/T2/T3/T4 use the full feature set.
#   - C3 uses a reduced candidate pool.
#   - Unmodellable features are excluded from C3 candidacy.
#   - KS/Holm identifies stable features.
#   - MI selection is active.
#   - Headline MI threshold = 0.05.
#   - T4_full == C2.
# ================================================================

import json
import time
import warnings
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

# Make sure local code folder is importable
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import pilot_pqc_stable_pipeline as base
import pilot_option_b as ob
import option_b_full_experiment as full
import revised_pipeline_main_only as rev

warnings.filterwarnings("ignore")

# ============================ CONFIG ============================
N_SPLITS    = 10
SPLIT_SEEDS = list(range(N_SPLITS))
MODEL_SEED  = 42
PER_CLASS   = 30000
TEST_SIZE   = 0.30
HOLM_ALPHA  = 0.05

DROP_IDENTIFIERS = False  # main ablation belongs to the main 70-feature design
MI_THRESHOLD = 0.05

RAND_SEED_OFFSET = 1000
OUT_PREFIX = "revised_ablation_a1b2"

FILES = ob.FILES

# Use the exact final unmodellable-feature list from the revised pipeline
UNMODELLABLE = list(rev.UNMODELLABLE)

# Component families == exactly the 23 columns the validated transform touches
BYTES_F = [
    "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Subflow Fwd Byts", "Subflow Bwd Byts"
]

PKTS_HDRS_F = [
    "Tot Fwd Pkts", "Tot Bwd Pkts",
    "Subflow Fwd Pkts", "Subflow Bwd Pkts",
    "Fwd Header Len", "Bwd Header Len"
]

LENSTATS_F = [
    "Fwd Pkt Len Mean", "Fwd Seg Size Avg",
    "Bwd Pkt Len Mean", "Bwd Seg Size Avg",
    "Pkt Len Mean", "Pkt Size Avg",
    "Fwd Pkt Len Max", "Bwd Pkt Len Max", "Pkt Len Max"
]

RATES_F = [
    "Flow Byts/s", "Flow Pkts/s", "Fwd Pkts/s", "Bwd Pkts/s"
]

TIERS = [
    ("T1_bytes",     BYTES_F),
    ("T2_pkts_hdrs", BYTES_F + PKTS_HDRS_F),
    ("T3_len_stats", BYTES_F + PKTS_HDRS_F + LENSTATS_F),
    ("T4_full",      BYTES_F + PKTS_HDRS_F + LENSTATS_F + RATES_F),
]
# ================================================================


def apply_tier(X, tier_cols, cfg=ob.MLKEM):
    """
    Apply the full validated transform, then revert all non-tier columns.
    This keeps the ablation consistent with the main approximation.
    """
    Xa = ob.make_approx_mlkem(X, cfg)
    keep = {c for c in tier_cols if c in Xa.columns}
    revert = [c for c in Xa.columns if c not in keep]
    Xa[revert] = X[revert].values
    return Xa


def fit_clf(kind, scale, Xtr_df, cols, ytr):
    if scale:
        sc = StandardScaler().fit(Xtr_df[cols])
        m = full.make_model(kind).fit(sc.transform(Xtr_df[cols]), ytr)
    else:
        sc = None
        m = full.make_model(kind).fit(Xtr_df[cols].values, ytr)
    return m, sc


def eval_clf(m, sc, Xdf, cols, y, i2c):
    Xe = sc.transform(Xdf[cols]) if sc is not None else Xdf[cols].values
    return base.evaluate(m, Xe, y, i2c)


def compute_mi_scores(Xtr, ytr, features):
    vals = mutual_info_classif(
        Xtr[features].values,
        ytr,
        random_state=MODEL_SEED
    )
    return {f: float(v) for f, v in zip(features, vals)}


def select_stable_mi_features(Xtr, Xtr_full, ytr, candidate_features):
    """
    Final A1+B2 C3 logic:
    1. KS/Holm on candidate features.
    2. MI scores on stable candidate features.
    3. Select stable features with MI >= threshold.
    """
    stable, shifted, _ = base.stable_features_via_ks(
        Xtr[candidate_features],
        Xtr_full[candidate_features],
        candidate_features,
        HOLM_ALPHA
    )

    mi = compute_mi_scores(Xtr, ytr, stable)
    selected = [f for f in stable if mi.get(f, 0.0) >= MI_THRESHOLD]

    if len(selected) == 0:
        raise RuntimeError("No selected C3 features. Check MI threshold or candidate pool.")

    return selected, stable, shifted, mi


def one_split(seed, X, y, full_features, candidate_features, i2c):
    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y
    )

    # Full footprint for training-side stable-feature detection
    Xtr_full = ob.make_approx_mlkem(Xtr)

    selected, stable, shifted, mi = select_stable_mi_features(
        Xtr,
        Xtr_full,
        ytr,
        candidate_features
    )

    K = len(selected)

    rng = np.random.RandomState(RAND_SEED_OFFSET + seed)
    rand_feats = sorted(
        rng.choice(candidate_features, size=K, replace=False).tolist()
    )
    n_retained_shifted = len(set(rand_feats) & set(shifted))

    Xte_tiers = {name: apply_tier(Xte, cols) for name, cols in TIERS}

    res = {}

    for kind, label, scale in full.CLASSIFIERS:
        if kind == "xgb" and not full.HAVE_XGB:
            continue

        # C1/T1/T2/T3/T4: full-feature model
        m1, sc1 = fit_clf(kind, scale, Xtr, full_features, ytr)

        out = {
            "C1": eval_clf(m1, sc1, Xte, full_features, yte, i2c)
        }

        for name, _ in TIERS:
            out[name] = eval_clf(
                m1,
                sc1,
                Xte_tiers[name],
                full_features,
                yte,
                i2c
            )

        # Final revised C3_true: selected stable + MI features
        m3, sc3 = fit_clf(kind, scale, Xtr, selected, ytr)
        out["C3_true"] = eval_clf(
            m3,
            sc3,
            Xte_tiers["T4_full"],
            selected,
            yte,
            i2c
        )

        # Random same-size control from modellable C3 candidate pool
        mr, scr = fit_clf(kind, scale, Xtr, rand_feats, ytr)
        out["C3_random"] = eval_clf(
            mr,
            scr,
            Xte_tiers["T4_full"],
            rand_feats,
            yte,
            i2c
        )

        out["selected_features"] = selected
        out["random_features"] = rand_feats

        res[kind] = out

    meta = {
        "selected_count": K,
        "stable_count": len(stable),
        "shifted_count": len(shifted),
        "random_retained_shifted": n_retained_shifted,
        "selected_features": selected,
        "stable_features": sorted(stable),
        "shifted_features": sorted(shifted),
        "mi_scores_selected": {f: mi.get(f, 0.0) for f in selected},
        "random_features": rand_feats,
    }

    return res, meta


def ci_fmt(values):
    m, lo, hi = full.ci95(values)
    return f"{m:.4f} [{lo:.4f},{hi:.4f}]"


def run_ablation():
    t0 = time.time()

    kinds = [
        k for k, _, _ in full.CLASSIFIERS
        if not (k == "xgb" and not full.HAVE_XGB)
    ]

    conds = ["C1"] + [n for n, _ in TIERS] + ["C3_true", "C3_random"]

    print("=" * 78)
    print(
        f"REVISED ABLATION A1+B2 | {N_SPLITS} splits | "
        f"per_class={PER_CLASS} | MI>={MI_THRESHOLD} | "
        f"drop_identifiers={DROP_IDENTIFIERS}"
    )
    print(f"Unmodellable excluded from C3 candidacy: {len(UNMODELLABLE)} features")
    for name, cols in TIERS:
        print(f"  {name:14s} {len(cols)} columns")
    print("  note: T4_full == condition C2")
    print("=" * 78)

    if not full.HAVE_XGB:
        print("  !! xgboost not installed -> !pip install xgboost")

    data = base.load_and_clean(
        FILES,
        base.LABEL_COL,
        base.LABEL_NORMALISE,
        base.CLASSES,
        PER_CLASS,
        MODEL_SEED
    )

    X, y, features, class_to_int = base.prepare_xy(
        data,
        base.NON_FEATURE_COLS,
        base.CLASSES
    )

    full_features = list(features)

    if DROP_IDENTIFIERS:
        full_features = [c for c in full_features if c not in ob.IDENTIFIER_FEATURES]

    candidate_features = [
        c for c in full_features
        if c not in UNMODELLABLE
    ]

    if DROP_IDENTIFIERS:
        candidate_features = [
            c for c in candidate_features
            if c not in ob.IDENTIFIER_FEATURES
        ]

    i2c = {i: c for c, i in class_to_int.items()}

    print(f"  data: {len(X):,} rows")
    print(f"  full features for C1/T1/T2/T3/T4: {len(full_features)}")
    print(f"  C3 candidate features: {len(candidate_features)}")
    print()

    acc = {k: {c: [] for c in conds} for k in kinds}
    raw = {}
    meta_by_seed = {}

    for i, seed in enumerate(SPLIT_SEEDS, 1):
        res, meta = one_split(
            seed,
            X,
            y,
            full_features,
            candidate_features,
            i2c
        )

        raw[str(seed)] = res
        meta_by_seed[str(seed)] = meta

        for k in kinds:
            for c in conds:
                acc[k][c].append(res[k][c]["weighted_f1"])

        with open(f"{OUT_PREFIX}_raw.json", "w") as fh:
            json.dump(
                {
                    "raw": raw,
                    "meta": meta_by_seed
                },
                fh,
                indent=2
            )

        print(
            f"  split {i}/{N_SPLITS} seed={seed} | "
            f"stable={meta['stable_count']} | selected={meta['selected_count']} | "
            f"shifted={meta['shifted_count']} | "
            f"random retained shifted={meta['random_retained_shifted']} | "
            f"elapsed {time.time() - t0:.0f}s"
        )

    summary = {
        "config": {
            "n_splits": N_SPLITS,
            "split_seeds": SPLIT_SEEDS,
            "model_seed": MODEL_SEED,
            "per_class": PER_CLASS,
            "test_size": TEST_SIZE,
            "holm_alpha": HOLM_ALPHA,
            "mi_threshold": MI_THRESHOLD,
            "drop_identifiers": DROP_IDENTIFIERS,
            "full_features_count": len(full_features),
            "candidate_features_count": len(candidate_features),
            "unmodellable_features": UNMODELLABLE,
            "tiers": {n: c for n, c in TIERS},
            "footprint": ob.MLKEM,
        },
        "meta_by_seed": meta_by_seed,
        "classifiers": {}
    }

    for kind, label, _ in full.CLASSIFIERS:
        if kind not in kinds:
            continue

        A = acc[kind]
        C1 = np.array(A["C1"])

        print("\n" + "=" * 78)
        print(f"### {label}  (n={N_SPLITS} matched splits)")
        print("-" * 78)
        print(f"  C1 baseline: {ci_fmt(A['C1'])}")
        print(f"\n  {'condition':14s}{'weighted F1':>26s}{'delta vs C1':>26s}")

        for name, _ in TIERS:
            d = np.array(A[name]) - C1
            print(f"  {name:14s}{ci_fmt(A[name]):>26s}{ci_fmt(d):>26s}")

        d_true = np.array(A["C3_true"]) - C1
        d_rand = np.array(A["C3_random"]) - C1
        target_gap = np.array(A["C3_true"]) - np.array(A["C3_random"])

        print()
        print(f"  C3_true      {ci_fmt(A['C3_true'])}   delta vs C1 {ci_fmt(d_true)}")
        print(f"  C3_random    {ci_fmt(A['C3_random'])}   delta vs C1 {ci_fmt(d_rand)}")
        print(f"  targeting effect C3_true - C3_random: {ci_fmt(target_gap)}")

        summary["classifiers"][kind] = {
            "label": label,
            "means_ci": {c: full.ci95(A[c]) for c in conds},
            "raw_wf1": {c: A[c] for c in conds},
            "delta_vs_c1": {
                c: (np.array(A[c]) - C1).tolist()
                for c in conds
                if c != "C1"
            },
            "targeting_effect": target_gap.tolist()
        }

    with open(f"{OUT_PREFIX}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 78)
    print(
        f"saved: {OUT_PREFIX}_summary.json (+ {OUT_PREFIX}_raw.json) "
        f"total elapsed {time.time() - t0:.0f}s"
    )

    return summary


if __name__ == "__main__":
    run_ablation()
