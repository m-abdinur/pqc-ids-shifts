"""Dependence-aware uncertainty review: corrected resampled t (Nadeau & Bengio, 2003).

Assistant-prepared sensitivity analysis. It does not replace the recorded
analysis. The ten 70:30 splits reuse the same rows, so the split-to-split
variance understates the uncertainty of the mean difference. The corrected
standard error multiplies the variance term by (1/k + n_test/n_train) instead
of 1/k.

Inputs: locked main raw, locked B3 raw, grouped-evaluation metrics.
Outputs: verified_outputs/uncertainty/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "verified_outputs/uncertainty"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["rf", "xgb", "mlp"]
K = 10
N = 119571
N_TEST = int(np.ceil(0.30 * N))          # sklearn train_test_split with test_size=0.30
RATIO_RANDOM = N_TEST / (N - N_TEST)
RATIO_GROUPED = 0.30 / 0.70               # grouped fill targets 30% per class stratum


def load_locked(path):
    raw = json.loads(path.read_text())
    out = {}
    for m in MODELS:
        blk = [raw[str(s)]["results"][m] for s in range(K)]
        out[m] = {"C1": np.array([b["C1"]["weighted_f1"] for b in blk]),
                  "C2": np.array([b["per_variant"]["hybrid768"]["C2"]["weighted_f1"] for b in blk]),
                  "C3": np.array([b["per_variant"]["hybrid768"]["C3"]["weighted_f1"] for b in blk])}
    return out


def load_grouped(path):
    met = pd.read_csv(path)
    out = {}
    for m in MODELS:
        sub = met[met.model == m]
        out[m] = {c: sub[sub.condition == c].sort_values("seed").weighted_f1.to_numpy() for c in ["C1", "C2", "C3"]}
    return out


def load_timing_free():
    import zipfile
    z = zipfile.ZipFile(ROOT / "reference/restrictions_sources_20260923/table410_sources.zip")
    tf = json.loads(z.read("timing_free_v6/outputs/final_10_splits_xgboost_3_2_0/timing_free_raw_10_splits.json"))
    return {m: {c: np.array([tf["splits"][str(s)]["classifiers"][m][c]["weighted_f1"] for s in range(K)])
                for c in ["C1", "C2", "C3"]} for m in MODELS}


def load_e2_b3():
    e2 = json.loads((ROOT / "reference/exposure_sources_20260923/tls_likely_exposure_v6/outputs/"
                     "final_10_splits_xgboost_3_2_0/e2_tls_likely_raw_10_splits.json").read_text())
    key = {"C1": "C1", "C2": "E2_C2", "C3": "E2_C3"}
    return {m: {c: np.array([e2["splits"][str(s)]["designs"]["E2_B3"]["classifiers"][m][key[c]]["weighted_f1"]
                             for s in range(K)]) for c in key} for m in MODELS}


def load_retraining():
    raw = json.loads((ROOT / "reference/packet_retrain_sources_20260923/retraining_v6/00_original_locked_v5/"
                      "extracted_reference/final_experiment_package_2026_07_02_v5_merged/01_raw_results/"
                      "r_oracle_main_raw.json").read_text())
    # "C1" slot holds the retrained reference so that "C3-C1" is C3 minus retrained
    out = {}
    for m in MODELS:
        blk = [raw[str(s)]["results"][m]["per_variant"]["hybrid768"] for s in range(K)]
        out[m] = {"C1": np.array([b["R_oracle"]["weighted_f1"] for b in blk]),
                  "C2": np.array([b["C2"]["weighted_f1"] for b in blk]),
                  "C3": np.array([b["C3"]["weighted_f1"] for b in blk])}
    return out


DESIGNS = {
    "main": (load_locked(ROOT / "reference/main_results/revised_pipeline_raw.json"), RATIO_RANDOM),
    "B3_identifier_free": (load_locked(ROOT / "reference/b3_results/revised_pipeline_b3_raw.json"), RATIO_RANDOM),
    "grouped": (load_grouped(ROOT / "reference/grouped_evaluation_20260923/results/metrics.csv"), RATIO_GROUPED),
    "timing_free": (load_timing_free(), RATIO_RANDOM),
    "E2_B3_tls_likely": (load_e2_b3(), RATIO_RANDOM),
    "retraining_ref (C1 slot = retrained)": (load_retraining(), RATIO_RANDOM),
}


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); n = len(p)
    adj = np.empty(n); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (n - k) * p[i])); adj[i] = run
    return adj


def analyse(d, ratio, corrected):
    s2 = d.var(ddof=1)
    se = np.sqrt(s2 * ((1 / K + ratio) if corrected else 1 / K))
    t95 = stats.t.ppf(0.95, K - 1)
    res = dict(mean=d.mean(), se=se, lo90=d.mean() - t95 * se, hi90=d.mean() + t95 * se,
               p_two_sided=2 * stats.t.sf(abs(d.mean()) / se, K - 1))
    for margin in (0.01, 0.005):
        res[f"tost_p_{margin}"] = max(stats.t.sf((d.mean() + margin) / se, K - 1),
                                      stats.t.cdf((d.mean() - margin) / se, K - 1))
    return res


rows = []
for design, (data, ratio) in DESIGNS.items():
    for corrected in (False, True):
        for m in MODELS:
            for contrast, (a, b) in {"C2-C1": ("C2", "C1"), "C3-C1": ("C3", "C1")}.items():
                d = data[m][a] - data[m][b]
                rows.append(dict(design=design, se_type="corrected" if corrected else "naive", model=m,
                                 contrast=contrast, **analyse(d, ratio, corrected)))
R = pd.DataFrame(rows)
for _, grp in R.groupby(["design", "se_type", "contrast"]):
    R.loc[grp.index, "holm_p_two_sided"] = holm(grp.p_two_sided)
    R.loc[grp.index, "holm_tost_0.01"] = holm(grp["tost_p_0.01"])
    R.loc[grp.index, "holm_tost_0.005"] = holm(grp["tost_p_0.005"])
R["degradation_significant"] = np.where(R.contrast == "C2-C1", (R.holm_p_two_sided < 0.05) & (R["mean"] < 0), np.nan)
R["equiv_0.01"] = np.where(R.contrast == "C3-C1", R["holm_tost_0.01"] < 0.05, np.nan)
R["equiv_0.005"] = np.where(R.contrast == "C3-C1", R["holm_tost_0.005"] < 0.05, np.nan)
R.to_csv(OUT / "corrected_resampled_results.csv", index=False)

meta = {"method": "Nadeau & Bengio (2003) corrected resampled t; variance factor (1/k + n_test/n_train)",
        "k": K, "ratio_random_split": RATIO_RANDOM, "ratio_grouped_split": RATIO_GROUPED,
        "se_inflation_random": float(np.sqrt((1 / K + RATIO_RANDOM) / (1 / K))),
        "note": "Sensitivity analysis only; recorded Shapiro/t/Wilcoxon procedure is unchanged."}
(OUT / "method.json").write_text(json.dumps(meta, indent=2))
pd.set_option("display.width", 250)
print(json.dumps(meta, indent=2))
cols = ["design", "se_type", "model", "contrast", "mean", "lo90", "hi90", "holm_p_two_sided",
        "holm_tost_0.01", "holm_tost_0.005", "degradation_significant", "equiv_0.01", "equiv_0.005"]
print(R[cols].to_string(float_format=lambda x: f"{x:.4g}"))
