#!/usr/bin/env python3
"""
Revised analysis pack for A1+B2 pipeline outputs.

Inputs:
- revised_pipeline_raw.json
- revised_pipeline_b3_raw.json
- revised_pipeline_b3_mlp_sweep_raw.json

Outputs:
- results/revised_analysis_pack/revised_equivalence_table.csv
- results/revised_analysis_pack/revised_pairwise_tests.csv
- results/revised_analysis_pack/revised_summary_markdown.md
"""

import json
import math
from pathlib import Path
from statistics import mean
import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
DELTA = 0.01

OUTDIR = Path("results/revised_analysis_pack")
OUTDIR.mkdir(parents=True, exist_ok=True)

INPUTS = {
    "main_revised": "revised_pipeline_raw.json",
    "b3_revised": "revised_pipeline_b3_raw.json",
    "b3_mlp_sweep": "revised_pipeline_b3_mlp_sweep_raw.json",
}


def load_json(path):
    return json.loads(Path(path).read_text())


def ci95(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(np.mean(x))
    if n < 2:
        return m, math.nan, math.nan
    se = stats.sem(x)
    lo, hi = stats.t.interval(0.95, n - 1, loc=m, scale=se)
    return m, float(lo), float(hi)


def cohen_dz(diff):
    diff = np.asarray(diff, dtype=float)
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return math.inf if np.mean(diff) != 0 else 0.0
    return float(np.mean(diff) / sd)


def shapiro_p(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return math.nan
    try:
        return float(stats.shapiro(x).pvalue)
    except Exception:
        return math.nan


def paired_test(a, b):
    """
    Returns paired t-test and Wilcoxon as robustness check.
    a and b are paired arrays.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = b - a

    out = {}
    out["mean_a"] = float(np.mean(a))
    out["mean_b"] = float(np.mean(b))
    out["mean_diff_b_minus_a"] = float(np.mean(diff))
    out["diff_ci95_low"], out["diff_ci95_high"] = ci95(diff)[1], ci95(diff)[2]
    out["shapiro_p_diff"] = shapiro_p(diff)
    out["cohen_dz"] = cohen_dz(diff)

    try:
        t = stats.ttest_rel(b, a)
        out["paired_t_p"] = float(t.pvalue)
        out["paired_t_stat"] = float(t.statistic)
    except Exception:
        out["paired_t_p"] = math.nan
        out["paired_t_stat"] = math.nan

    try:
        w = stats.wilcoxon(b, a, zero_method="wilcox", alternative="two-sided")
        out["wilcoxon_p"] = float(w.pvalue)
        out["wilcoxon_stat"] = float(w.statistic)
    except Exception:
        out["wilcoxon_p"] = math.nan
        out["wilcoxon_stat"] = math.nan

    return out


def tost_equivalence(a, b, delta=DELTA, alpha=ALPHA):
    """
    Paired TOST for equivalence of b-a within [-delta, +delta].
    Equivalent if both one-sided tests pass.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = b - a
    n = len(diff)
    md = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    se = sd / math.sqrt(n)

    if se == 0:
        p_lower = 0.0 if md > -delta else 1.0
        p_upper = 0.0 if md < delta else 1.0
    else:
        # H01: diff <= -delta, test diff > -delta
        t_lower = (md - (-delta)) / se
        p_lower = 1 - stats.t.cdf(t_lower, df=n - 1)

        # H02: diff >= +delta, test diff < +delta
        t_upper = (md - delta) / se
        p_upper = stats.t.cdf(t_upper, df=n - 1)

    equivalent = (p_lower < alpha) and (p_upper < alpha)

    return {
        "delta": delta,
        "tost_p_lower_diff_gt_minus_delta": float(p_lower),
        "tost_p_upper_diff_lt_plus_delta": float(p_upper),
        "tost_p_max": float(max(p_lower, p_upper)),
        "tost_equivalent": bool(equivalent),
    }


def holm_adjust(pvals):
    """
    Holm-Bonferroni adjusted p-values.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(n)
    running_max = 0.0

    for rank, idx in enumerate(order):
        adj = (n - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted


def get_runs(obj):
    """
    Locate split/run records.

    The revised raw files are stored as:
    {
      "0": {...},
      "1": {...},
      ...
      "9": {...}
    }

    This function also supports older list/container formats.
    """
    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        # Revised pipeline format: numeric seed keys as strings.
        numeric_keys = []
        for k in obj.keys():
            try:
                numeric_keys.append(int(k))
            except Exception:
                pass

        if numeric_keys and len(numeric_keys) == len(obj):
            return [obj[str(k)] for k in sorted(numeric_keys)]

        # Older/common formats.
        for key in ["runs", "splits", "per_split", "records", "raw"]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]

    raise ValueError(
        f"Could not find run list. Top-level keys: {list(obj.keys()) if isinstance(obj, dict) else type(obj)}"
    )


def metric_from_dict(d):
    """
    Extract weighted F1 from a metric dictionary.
    """
    candidates = [
        "weighted_f1", "wF1", "wf1", "f1_weighted",
        "weighted_f1_score", "f1"
    ]
    for c in candidates:
        if isinstance(d, dict) and c in d and isinstance(d[c], (int, float)):
            return float(d[c])
    return None


def find_model_block(run, model):
    """
    Find model block for rf/xgb/mlp inside one run.
    """
    possible_keys = [model, model.upper(), model.lower()]
    if model == "xgb":
        possible_keys += ["XGBoost", "xgboost"]
    if model == "rf":
        possible_keys += ["Random Forest", "random_forest"]
    if model == "mlp":
        possible_keys += ["MLP", "MLP (scaled)", "mlp_scaled"]

    # Direct locations
    for k in possible_keys:
        if isinstance(run, dict) and k in run and isinstance(run[k], dict):
            return run[k]

    for container in ["models", "metrics", "results", "model_results"]:
        if container in run and isinstance(run[container], dict):
            for k in possible_keys:
                if k in run[container] and isinstance(run[container][k], dict):
                    return run[container][k]

    return None


def extract_condition(model_block, cond, threshold=None):
    """
    Extract weighted F1 for C1/C2/C3 from a model block.

    Revised raw structure:
    - C1 is stored directly under model_block["C1"].
    - C2 is stored under model_block["per_variant"]["hybrid768"]["C2"].
    - C3 headline is stored under model_block["mi_sweep"][threshold]
      and also under model_block["per_variant"]["hybrid768"]["C3"].
    """
    if model_block is None:
        return None

    # C2 is variant-specific. For the main analysis, use the prep/main footprint.
    if cond.lower() == "c2":
        pv = model_block.get("per_variant", {})
        if isinstance(pv, dict):
            for variant in ["hybrid768", "FPREP"]:
                if variant in pv and isinstance(pv[variant], dict):
                    if "C2" in pv[variant]:
                        val = pv[variant]["C2"]
                        if isinstance(val, dict):
                            m = metric_from_dict(val)
                            if m is not None:
                                return m
                        elif isinstance(val, (int, float)):
                            return float(val)

            # fallback: use first available variant if hybrid768 is absent
            for variant, block in pv.items():
                if isinstance(block, dict) and "C2" in block:
                    val = block["C2"]
                    if isinstance(val, dict):
                        m = metric_from_dict(val)
                        if m is not None:
                            return m
                    elif isinstance(val, (int, float)):
                        return float(val)

    cond_keys = [cond, cond.lower(), cond.upper()]
    if cond == "c3" and threshold is not None:
        thr_keys = [
            str(threshold),
            f"{threshold}",
            f"{float(threshold):.2f}",
            f"{float(threshold):.1f}",
        ]

        # common C3 containers
        for c3_container in ["c3_by_threshold", "c3_by_mi", "c3", "C3", "mi_sweep", "thresholds"]:
            if c3_container in model_block and isinstance(model_block[c3_container], dict):
                for tk in thr_keys:
                    if tk in model_block[c3_container]:
                        val = model_block[c3_container][tk]
                        if isinstance(val, dict):
                            m = metric_from_dict(val)
                            if m is not None:
                                return m
                        elif isinstance(val, (int, float)):
                            return float(val)

        # flattened keys
        for tk in thr_keys:
            for key in [f"c3_{tk}", f"C3_{tk}", f"c3_mi_{tk}", f"C3_mi_{tk}"]:
                if key in model_block:
                    val = model_block[key]
                    if isinstance(val, dict):
                        m = metric_from_dict(val)
                        if m is not None:
                            return m
                    elif isinstance(val, (int, float)):
                        return float(val)

    for ck in cond_keys:
        if ck in model_block:
            val = model_block[ck]
            if isinstance(val, dict):
                m = metric_from_dict(val)
                if m is not None:
                    return m
            elif isinstance(val, (int, float)):
                return float(val)

    return None


def extract_arrays(raw_path, thresholds=(0.05,), models=("rf", "xgb", "mlp")):
    obj = load_json(raw_path)
    runs = get_runs(obj)

    data = {}
    for model in models:
        c1, c2 = [], []
        c3 = {str(t): [] for t in thresholds}

        for run in runs:
            mb = find_model_block(run, model)
            if mb is None:
                continue

            v1 = extract_condition(mb, "c1")
            v2 = extract_condition(mb, "c2")
            if v1 is not None:
                c1.append(v1)
            if v2 is not None:
                c2.append(v2)

            for t in thresholds:
                v3 = extract_condition(mb, "c3", t)
                if v3 is not None:
                    c3[str(t)].append(v3)

        if c1 or c2 or any(c3[str(t)] for t in thresholds):
            data[model] = {"c1": c1, "c2": c2, "c3": c3}

    return data


def analyse_dataset(label, raw_path, thresholds):
    arrays = extract_arrays(raw_path, thresholds=thresholds)
    pairwise_rows = []
    equiv_rows = []

    for model, d in arrays.items():
        c1 = d["c1"]
        c2 = d["c2"]

        if len(c1) == 0:
            continue

        if len(c2) == len(c1) and len(c2) > 0:
            test = paired_test(c1, c2)
            row = {
                "dataset": label,
                "model": model,
                "comparison": "C1_vs_C2",
                "threshold": "",
                **test,
            }
            pairwise_rows.append(row)

        for t, c3 in d["c3"].items():
            if len(c3) != len(c1) or len(c3) == 0:
                continue

            test13 = paired_test(c1, c3)
            eq13 = tost_equivalence(c1, c3)

            pairwise_rows.append({
                "dataset": label,
                "model": model,
                "comparison": "C1_vs_C3",
                "threshold": t,
                **test13,
            })

            equiv_rows.append({
                "dataset": label,
                "model": model,
                "comparison": "C1_equiv_C3",
                "threshold": t,
                "n": len(c1),
                "c1_mean": float(np.mean(c1)),
                "c3_mean": float(np.mean(c3)),
                "c3_minus_c1": float(np.mean(np.asarray(c3) - np.asarray(c1))),
                "ci95_low": ci95(np.asarray(c3) - np.asarray(c1))[1],
                "ci95_high": ci95(np.asarray(c3) - np.asarray(c1))[2],
                **eq13,
            })

            if len(c2) == len(c3) and len(c2) > 0:
                test23 = paired_test(c2, c3)
                pairwise_rows.append({
                    "dataset": label,
                    "model": model,
                    "comparison": "C2_vs_C3",
                    "threshold": t,
                    **test23,
                })

    return pairwise_rows, equiv_rows


def main():
    all_pairwise = []
    all_equiv = []

    # Main and B3 only need headline 0.05.
    p, e = analyse_dataset("main_revised", INPUTS["main_revised"], thresholds=(0.05,))
    all_pairwise.extend(p)
    all_equiv.extend(e)

    p, e = analyse_dataset("b3_revised", INPUTS["b3_revised"], thresholds=(0.05,))
    all_pairwise.extend(p)
    all_equiv.extend(e)

    # MLP sweep needs all thresholds.
    p, e = analyse_dataset("b3_mlp_sweep", INPUTS["b3_mlp_sweep"], thresholds=(0.0, 0.01, 0.05, 0.10, 0.20))
    all_pairwise.extend(p)
    all_equiv.extend(e)

    pairwise = pd.DataFrame(all_pairwise)
    equiv = pd.DataFrame(all_equiv)

    if not pairwise.empty and "paired_t_p" in pairwise:
        valid = pairwise["paired_t_p"].notna()
        pairwise["paired_t_p_holm"] = np.nan
        pairwise.loc[valid, "paired_t_p_holm"] = holm_adjust(pairwise.loc[valid, "paired_t_p"].to_numpy())

    if not equiv.empty and "tost_p_max" in equiv:
        valid = equiv["tost_p_max"].notna()
        equiv["tost_p_max_holm"] = np.nan
        equiv.loc[valid, "tost_p_max_holm"] = holm_adjust(equiv.loc[valid, "tost_p_max"].to_numpy())
        equiv["equivalent_after_holm"] = equiv["tost_p_max_holm"] < ALPHA

    pairwise.to_csv(OUTDIR / "revised_pairwise_tests.csv", index=False)
    equiv.to_csv(OUTDIR / "revised_equivalence_table.csv", index=False)

    md = []
    md.append("# Revised Analysis Pack Summary\n")
    md.append(f"Equivalence margin: ±{DELTA} weighted F1\n")
    md.append("## Equivalence table\n")
    if not equiv.empty:
        show = equiv[[
            "dataset", "model", "threshold", "n",
            "c1_mean", "c3_mean", "c3_minus_c1",
            "ci95_low", "ci95_high",
            "tost_p_max", "tost_equivalent",
            "tost_p_max_holm", "equivalent_after_holm"
        ]].copy()
        md.append(show.to_markdown(index=False))
    else:
        md.append("No equivalence rows extracted. Check raw JSON structure.")

    md.append("\n\n## Pairwise tests\n")
    if not pairwise.empty:
        show2 = pairwise[[
            "dataset", "model", "comparison", "threshold",
            "mean_a", "mean_b", "mean_diff_b_minus_a",
            "diff_ci95_low", "diff_ci95_high",
            "shapiro_p_diff", "paired_t_p", "paired_t_p_holm",
            "wilcoxon_p", "cohen_dz"
        ]].copy()
        md.append(show2.to_markdown(index=False))
    else:
        md.append("No pairwise rows extracted. Check raw JSON structure.")

    (OUTDIR / "revised_summary_markdown.md").write_text("\n".join(md))

    print("Saved:")
    print(OUTDIR / "revised_pairwise_tests.csv")
    print(OUTDIR / "revised_equivalence_table.csv")
    print(OUTDIR / "revised_summary_markdown.md")


if __name__ == "__main__":
    main()
