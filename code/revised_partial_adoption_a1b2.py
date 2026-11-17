# ================================================================
# REVISED PARTIAL-ADOPTION ANALYSIS
# Final A1+B2 pipeline version
#
# Purpose:
#   Test gradual PQC-like adoption in the test set:
#     0%, 25%, 50%, 75%, 100% of test flows shifted
#
# Key final-pipeline rules:
#   - C1/full model is trained on all original features.
#   - C3 uses reduced candidate pool.
#   - Unmodellable features are excluded from C3 candidacy.
#   - KS/Holm identifies stable features.
#   - MI selection is active.
#   - Headline MI threshold = 0.05.
#   - 100% adoption should reproduce C2/C3-full-shift behaviour.
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
MI_THRESHOLD = 0.05

DROP_IDENTIFIERS = False
OUT_PREFIX = "revised_partial_adoption_a1b2"

ADOPTION_LEVELS = [0.00, 0.25, 0.50, 0.75, 1.00]

FILES = ob.FILES
UNMODELLABLE = list(rev.UNMODELLABLE)
# ================================================================


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
    stable, shifted, _ = base.stable_features_via_ks(
        Xtr[candidate_features],
        Xtr_full[candidate_features],
        candidate_features,
        HOLM_ALPHA
    )

    mi = compute_mi_scores(Xtr, ytr, stable)
    selected = [f for f in stable if mi.get(f, 0.0) >= MI_THRESHOLD]

    if len(selected) == 0:
        raise RuntimeError("No selected C3 features. Check MI threshold.")

    return selected, stable, shifted, mi


def make_partial_adoption_test(Xte, yte, adoption, seed):
    """
    Create a mixed test set where adoption proportion of each class is shifted.
    Selection is stratified by class to avoid class-composition artifacts.
    """
    if adoption == 0.0:
        return Xte.copy()

    X_full = ob.make_approx_mlkem(Xte)

    if adoption == 1.0:
        return X_full

    Xmixed = Xte.copy()
    rng = np.random.RandomState(5000 + seed + int(adoption * 1000))

    y_arr = np.asarray(yte)
    chosen_positions = []

    for cls in np.unique(y_arr):
        cls_positions = np.where(y_arr == cls)[0]
        n_pick = int(round(len(cls_positions) * adoption))
        if n_pick > 0:
            picked = rng.choice(cls_positions, size=n_pick, replace=False)
            chosen_positions.extend(picked.tolist())

    chosen_positions = sorted(chosen_positions)

    # Replace selected rows with their shifted versions using positional indexing.
    Xmixed.iloc[chosen_positions, :] = X_full.iloc[chosen_positions, :].values

    return Xmixed


def one_split(seed, X, y, full_features, candidate_features, i2c):
    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y
    )

    Xtr_full = ob.make_approx_mlkem(Xtr)

    selected, stable, shifted, mi = select_stable_mi_features(
        Xtr,
        Xtr_full,
        ytr,
        candidate_features
    )

    partial_tests = {
        str(adoption): make_partial_adoption_test(Xte, yte, adoption, seed)
        for adoption in ADOPTION_LEVELS
    }

    res = {}

    for kind, label, scale in full.CLASSIFIERS:
        if kind == "xgb" and not full.HAVE_XGB:
            continue

        # Full-feature C1 model
        m_full, sc_full = fit_clf(kind, scale, Xtr, full_features, ytr)

        # Revised C3 model
        m_c3, sc_c3 = fit_clf(kind, scale, Xtr, selected, ytr)

        out = {
            "selected_features": selected,
            "partial": {}
        }

        for adoption in ADOPTION_LEVELS:
            key = str(adoption)
            Xp = partial_tests[key]

            out["partial"][key] = {
                "full_feature_model": eval_clf(
                    m_full, sc_full, Xp, full_features, yte, i2c
                ),
                "c3_selected_model": eval_clf(
                    m_c3, sc_c3, Xp, selected, yte, i2c
                )
            }

        res[kind] = out

    meta = {
        "selected_count": len(selected),
        "stable_count": len(stable),
        "shifted_count": len(shifted),
        "selected_features": selected,
        "stable_features": sorted(stable),
        "shifted_features": sorted(shifted),
        "mi_scores_selected": {f: mi.get(f, 0.0) for f in selected},
    }

    return res, meta


def ci_fmt(values):
    m, lo, hi = full.ci95(values)
    return f"{m:.4f} [{lo:.4f},{hi:.4f}]"


def run_partial_adoption():
    t0 = time.time()

    kinds = [
        k for k, _, _ in full.CLASSIFIERS
        if not (k == "xgb" and not full.HAVE_XGB)
    ]

    print("=" * 78)
    print(
        f"REVISED PARTIAL ADOPTION A1+B2 | {N_SPLITS} splits | "
        f"per_class={PER_CLASS} | MI>={MI_THRESHOLD} | "
        f"levels={ADOPTION_LEVELS}"
    )
    print(f"Unmodellable excluded from C3 candidacy: {len(UNMODELLABLE)} features")
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
    print(f"  full features: {len(full_features)}")
    print(f"  C3 candidate features: {len(candidate_features)}")
    print()

    raw = {}
    meta_by_seed = {}

    acc = {
        k: {
            str(adoption): {
                "full_feature_model": [],
                "c3_selected_model": []
            }
            for adoption in ADOPTION_LEVELS
        }
        for k in kinds
    }

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
            for adoption in ADOPTION_LEVELS:
                key = str(adoption)
                acc[k][key]["full_feature_model"].append(
                    res[k]["partial"][key]["full_feature_model"]["weighted_f1"]
                )
                acc[k][key]["c3_selected_model"].append(
                    res[k]["partial"][key]["c3_selected_model"]["weighted_f1"]
                )

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
            "adoption_levels": ADOPTION_LEVELS,
            "full_features_count": len(full_features),
            "candidate_features_count": len(candidate_features),
            "unmodellable_features": UNMODELLABLE,
            "footprint": ob.MLKEM,
        },
        "meta_by_seed": meta_by_seed,
        "classifiers": {}
    }

    for kind, label, _ in full.CLASSIFIERS:
        if kind not in kinds:
            continue

        print("\n" + "=" * 78)
        print(f"### {label}  (n={N_SPLITS} matched splits)")
        print("-" * 78)
        print(f"{'Adoption':>10s} | {'Full-feature model':>25s} | {'C3 selected model':>25s}")

        model_summary = {}

        for adoption in ADOPTION_LEVELS:
            key = str(adoption)
            full_vals = acc[kind][key]["full_feature_model"]
            c3_vals = acc[kind][key]["c3_selected_model"]

            print(
                f"{int(adoption * 100):9d}% | "
                f"{ci_fmt(full_vals):>25s} | "
                f"{ci_fmt(c3_vals):>25s}"
            )

            model_summary[key] = {
                "full_feature_model": full_vals,
                "c3_selected_model": c3_vals,
                "full_feature_model_ci": full.ci95(full_vals),
                "c3_selected_model_ci": full.ci95(c3_vals),
            }

        summary["classifiers"][kind] = {
            "label": label,
            "adoption_results": model_summary
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
    run_partial_adoption()
