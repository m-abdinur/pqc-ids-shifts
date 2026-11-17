
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

RAW_PATH = Path("r_oracle_main_raw.json")
OUT_DIR = Path("results/chapter4_tables/r_oracle_analysis_pack")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
DELTA = 0.01
FOOTPRINT = "hybrid768"

MODEL_NAMES = {
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "mlp": "MLP",
}

METRICS = {
    "weighted_f1": "Weighted F1",
    "roc_auc_ovr_weighted": "ROC-AUC",
}

COMPARISONS = [
    ("C2", "R_oracle", "C2 vs R_oracle"),
    ("C3", "R_oracle", "C3 vs R_oracle"),
    ("C1", "R_oracle", "C1 vs R_oracle"),
]

def load_runs(path):
    if not path.exists():
        raise FileNotFoundError(path)
    obj = json.loads(path.read_text())
    if isinstance(obj, dict) and "raw" in obj:
        obj = obj["raw"]
    keys = sorted(obj.keys(), key=lambda x: int(x))
    return [obj[k] for k in keys]

def get_condition_values(runs, model_key, metric_key, condition):
    vals = []
    for run in runs:
        block = run["results"][model_key]
        if condition == "C1":
            vals.append(block["C1"][metric_key])
        else:
            vals.append(block["per_variant"][FOOTPRINT][condition][metric_key])
    return np.asarray(vals, dtype=float)

def ci95(vals):
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    mean = float(np.mean(vals))
    if n <= 1:
        return mean, mean, mean
    se = stats.sem(vals)
    if se == 0 or np.isnan(se):
        return mean, mean, mean
    lo, hi = stats.t.interval(0.95, n - 1, loc=mean, scale=se)
    return mean, float(lo), float(hi)

def fmt_ci(vals):
    m, lo, hi = ci95(vals)
    return f"{m:.4f} [{lo:.4f}, {hi:.4f}]"

def safe_shapiro(diff):
    diff = np.asarray(diff, dtype=float)
    if len(diff) < 3:
        return np.nan
    if np.allclose(diff, diff[0]):
        return np.nan
    return float(stats.shapiro(diff).pvalue)

def safe_wilcoxon(diff):
    diff = np.asarray(diff, dtype=float)
    if np.allclose(diff, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return np.nan

def cohen_dz(diff):
    diff = np.asarray(diff, dtype=float)
    sd = np.std(diff, ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(np.mean(diff) / sd)

def paired_tost(diff, delta=DELTA):
    """
    TOST on paired differences where diff = condition_a - condition_b.
    Equivalence means mean(diff) is inside [-delta, +delta].
    """
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    mean_diff = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    se = sd / np.sqrt(n)

    if se == 0 or np.isnan(se):
        p_lower = 0.0 if mean_diff > -delta else 1.0
        p_upper = 0.0 if mean_diff < delta else 1.0
        return p_lower, p_upper, max(p_lower, p_upper)

    df = n - 1

    # Lower test: H0 mean <= -delta vs H1 mean > -delta
    t_lower = (mean_diff + delta) / se
    p_lower = float(1 - stats.t.cdf(t_lower, df))

    # Upper test: H0 mean >= +delta vs H1 mean < +delta
    t_upper = (mean_diff - delta) / se
    p_upper = float(stats.t.cdf(t_upper, df))

    return p_lower, p_upper, max(p_lower, p_upper)

def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        adj = min(float(p_values[idx]) * factor, 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max

    return adjusted

runs = load_runs(RAW_PATH)

pairwise_rows = []
equiv_rows = []

for metric_key, metric_label in METRICS.items():
    for model_key, model_name in MODEL_NAMES.items():
        for cond_a, cond_b, comparison in COMPARISONS:
            a = get_condition_values(runs, model_key, metric_key, cond_a)
            b = get_condition_values(runs, model_key, metric_key, cond_b)
            diff = a - b

            mean_a, lo_a, hi_a = ci95(a)
            mean_b, lo_b, hi_b = ci95(b)
            mean_diff, lo_diff, hi_diff = ci95(diff)

            t_res = stats.ttest_rel(a, b)
            wilcox_p = safe_wilcoxon(diff)
            shapiro_p = safe_shapiro(diff)
            dz = cohen_dz(diff)

            pairwise_rows.append({
                "Metric": metric_label,
                "Model": model_name,
                "Comparison": comparison,
                "Condition A": cond_a,
                "Condition B": cond_b,
                "A mean": round(mean_a, 4),
                "B mean": round(mean_b, 4),
                "A 95% CI": f"{mean_a:.4f} [{lo_a:.4f}, {hi_a:.4f}]",
                "B 95% CI": f"{mean_b:.4f} [{lo_b:.4f}, {hi_b:.4f}]",
                "Mean difference A-B": round(mean_diff, 4),
                "Difference 95% CI": f"{mean_diff:.4f} [{lo_diff:.4f}, {hi_diff:.4f}]",
                "Paired t p": float(t_res.pvalue),
                "Wilcoxon p": wilcox_p,
                "Shapiro p for differences": shapiro_p,
                "Cohen dz": dz,
                "n": len(diff),
            })

            p_lower, p_upper, p_tost = paired_tost(diff, DELTA)
            within_ci_margin = bool(lo_diff > -DELTA and hi_diff < DELTA)

            equiv_rows.append({
                "Metric": metric_label,
                "Model": model_name,
                "Comparison": comparison,
                "Condition A": cond_a,
                "Condition B": cond_b,
                "Equivalence margin": DELTA,
                "Mean difference A-B": round(mean_diff, 4),
                "Difference 95% CI": f"{mean_diff:.4f} [{lo_diff:.4f}, {hi_diff:.4f}]",
                "TOST lower p": p_lower,
                "TOST upper p": p_upper,
                "TOST p raw": p_tost,
                "95% CI within margin": within_ci_margin,
                "n": len(diff),
            })

pairwise = pd.DataFrame(pairwise_rows)
equiv = pd.DataFrame(equiv_rows)

# Holm adjustment is applied to the TOST p-values across this R_oracle add-on table.
equiv["TOST p Holm"] = holm_adjust(equiv["TOST p raw"].values)
equiv["Equivalent after Holm"] = (equiv["TOST p Holm"] < ALPHA) & (equiv["95% CI within margin"])

pairwise_path = OUT_DIR / "r_oracle_pairwise_tests.csv"
equiv_path = OUT_DIR / "r_oracle_equivalence_table.csv"
summary_path = OUT_DIR / "r_oracle_summary_markdown.md"

pairwise.to_csv(pairwise_path, index=False)
equiv.to_csv(equiv_path, index=False)

# Human-readable summary
lines = []
lines.append("# R_oracle formal statistical add-on")
lines.append("")
lines.append("This analysis uses the already-generated R_oracle results. No model training is rerun.")
lines.append("")
lines.append(f"Equivalence margin: ±{DELTA:.2f}")
lines.append("")
lines.append("## Main reading")
lines.append("")
lines.append("- C2 vs R_oracle tests whether oracle retraining recovers the degradation seen under the shifted-test condition.")
lines.append("- C3 vs R_oracle tests whether the stable-feature pipeline is practically equivalent to oracle retraining.")
lines.append("- C1 vs R_oracle checks whether oracle retraining returns close to the original baseline.")
lines.append("")
lines.append("## Weighted F1: C3 vs R_oracle")
lines.append("")

wf1_equiv = equiv[(equiv["Metric"] == "Weighted F1") & (equiv["Comparison"] == "C3 vs R_oracle")]
for _, row in wf1_equiv.iterrows():
    verdict = "equivalent" if row["Equivalent after Holm"] else "not equivalent"
    lines.append(
        f"- {row['Model']}: mean difference {row['Mean difference A-B']:.4f}, "
        f"95% CI {row['Difference 95% CI']}, Holm-adjusted TOST p={row['TOST p Holm']:.4g}; {verdict}."
    )

lines.append("")
lines.append("## Weighted F1: C2 vs R_oracle")
lines.append("")

wf1_pair = pairwise[(pairwise["Metric"] == "Weighted F1") & (pairwise["Comparison"] == "C2 vs R_oracle")]
for _, row in wf1_pair.iterrows():
    lines.append(
        f"- {row['Model']}: C2 mean {row['A mean']:.4f}, R_oracle mean {row['B mean']:.4f}, "
        f"mean difference {row['Mean difference A-B']:.4f}, paired t p={row['Paired t p']:.4g}."
    )

summary_path.write_text("\n".join(lines))

print("Saved:")
print(pairwise_path)
print(equiv_path)
print(summary_path)

print("\nPairwise tests:")
print(pairwise.to_string(index=False))

print("\nEquivalence table:")
print(equiv.to_string(index=False))
