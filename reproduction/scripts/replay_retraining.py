"""Replay the retraining reference (R_oracle; thesis Table 4.21).

Assistant-prepared check. Inputs: the V5 R_oracle raw results and analysis pack
(reference/packet_retrain_sources_20260923/retraining_v6/...) and the locked main
raw results. Outputs: verified_outputs/retraining/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
V5 = (ROOT / "reference/packet_retrain_sources_20260923/retraining_v6/00_original_locked_v5/"
      "extracted_reference/final_experiment_package_2026_07_02_v5_merged")
OUT = ROOT / "verified_outputs/retraining"
OUT.mkdir(parents=True, exist_ok=True)
RAW = json.loads((V5 / "01_raw_results/r_oracle_main_raw.json").read_text())
MAIN = json.loads((ROOT / "reference/main_results/revised_pipeline_raw.json").read_text())
MODELS = ["rf", "xgb", "mlp"]
NAMES = {"rf": "Random Forest", "xgb": "XGBoost", "mlp": "MLP"}
METRICS = {"weighted_f1": "Weighted F1", "roc_auc_ovr_weighted": "ROC-AUC"}
SEEDS = [str(s) for s in range(10)]
summary = {}


def get(raw, s, m, cond, metric):
    b = raw[s]["results"][m]
    return (b["C1"] if cond == "C1" else b["per_variant"]["hybrid768"][cond])[metric]


# 1. C1/C2/C3 in the R_oracle run against the locked main results
rows = []
for s in SEEDS:
    for m in MODELS:
        for c in ["C1", "C2", "C3"]:
            for met in METRICS:
                rows.append(dict(seed=int(s), model=m, condition=c, metric=met,
                                 diff=get(RAW, s, m, c, met) - get(MAIN, s, m, c, met)))
A = pd.DataFrame(rows)
A.to_csv(OUT / "agreement_with_locked_main.csv", index=False)
summary["weighted_f1_equal_to_locked_main"] = bool((A[A.metric == "weighted_f1"]["diff"] == 0).all())
summary["max_abs_roc_auc_diff_vs_locked_main"] = float(A[A.metric != "weighted_f1"]["diff"].abs().max())


# 2. Table 4.21 means
def arr(m, c, met):
    return np.array([get(RAW, s, m, c, met) for s in SEEDS])


tab = []
for met, label in METRICS.items():
    for m in MODELS:
        c1, c2, c3, ro = (arr(m, c, met).mean() for c in ["C1", "C2", "C3", "R_oracle"])
        tab.append(dict(metric=label, model=NAMES[m], C1=c1, C2=c2, C3=c3, retrained=ro,
                        retr_minus_C1=ro - c1, C3_minus_retr=c3 - ro))
T = pd.DataFrame(tab)
T.to_csv(OUT / "table_4_21_replay.csv", index=False)
thesis = pd.DataFrame([
    ("Weighted F1", "Random Forest", .8491, .7344, .8505, .8486, -.0005, .0019),
    ("Weighted F1", "XGBoost", .8603, .7691, .8590, .8598, -.0005, -.0008),
    ("Weighted F1", "MLP", .8437, .5923, .8415, .8428, -.0009, -.0013),
    ("ROC-AUC", "Random Forest", .9625, .9521, .9633, .9625, -.0000, .0008),
    ("ROC-AUC", "XGBoost", .9678, .9509, .9674, .9678, -.0001, -.0004),
    ("ROC-AUC", "MLP", .9623, .8434, .9610, .9617, -.0006, -.0007)],
    columns=["metric", "model", "C1", "C2", "C3", "retrained", "retr_minus_C1", "C3_minus_retr"])
cmp = T.merge(thesis, on=["metric", "model"], suffixes=("", "_thesis"))
cols = ["C1", "C2", "C3", "retrained", "retr_minus_C1", "C3_minus_retr"]
summary["table_4_21_matches_thesis_4dp"] = bool(all(
    abs(round(r[c], 4) - r[c + "_thesis"]) < 1e-9 or abs(r[c] - r[c + "_thesis"]) < 5e-5
    for _, r in cmp.iterrows() for c in cols))


# 3. Statistics: recorded procedure and the paper's procedure
def tost(d, margin=0.01):
    se = d.std(ddof=1) / np.sqrt(len(d)); df = len(d) - 1
    return max(stats.t.sf((d.mean() + margin) / se, df), stats.t.cdf((d.mean() - margin) / se, df))


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); n = len(p)
    adj = np.empty(n); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (n - k) * p[i])); adj[i] = run
    return adj


rec = pd.read_csv(V5 / "02_analysis_tables/r_oracle_analysis_pack/r_oracle_equivalence_table.csv")
rows = []
for met, label in METRICS.items():
    for m in MODELS:
        for a in ["C2", "C3", "C1"]:
            d = arr(m, a, met) - arr(m, "R_oracle", met)
            se = d.std(ddof=1) / np.sqrt(10)
            lo95, hi95 = stats.t.interval(0.95, 9, loc=d.mean(), scale=se)
            lo90, hi90 = stats.t.interval(0.90, 9, loc=d.mean(), scale=se)
            rows.append(dict(metric=label, model=NAMES[m], comparison=f"{a} vs R_oracle", mean=d.mean(),
                             lo95=lo95, hi95=hi95, lo90=lo90, hi90=hi90, p=tost(d)))
S = pd.DataFrame(rows)
S["holm_all18"] = holm(S.p)
S["equiv_recorded_rule"] = (S.holm_all18 < 0.05) & (S.lo95 > -0.01) & (S.hi95 < 0.01)
c3 = S.comparison == "C3 vs R_oracle"
S.loc[c3, "holm_c3_family6"] = holm(S.loc[c3, "p"])
S.loc[c3, "equiv_paper_rule"] = (S.loc[c3, "holm_c3_family6"] < 0.05) & (S.loc[c3, "lo90"] > -0.01) & (S.loc[c3, "hi90"] < 0.01)
S.to_csv(OUT / "equivalence_replay.csv", index=False)
mm = S.merge(rec, left_on=["metric", "model", "comparison"], right_on=["Metric", "Model", "Comparison"])
summary["recorded_rows_matched"] = f"{len(mm)}/{len(rec)}"
summary["max_abs_holm_p_diff_vs_recorded"] = float((mm.holm_all18 - mm["TOST p Holm"]).abs().max())
summary["verdicts_match_recorded"] = bool((mm.equiv_recorded_rule == mm["Equivalent after Holm"]).all())
summary["recorded_rule"] = "Holm across all 18 add-on tests (C1, C2, C3 vs R_oracle, two metrics); equivalent if Holm p < 0.05 and 95% CI within margin"
summary["c3_vs_retrained_equivalent_recorded_rule"] = int(S.loc[c3, "equiv_recorded_rule"].sum())
summary["c3_vs_retrained_equivalent_paper_rule_90ci_holm6"] = int(S.loc[c3, "equiv_paper_rule"].sum())
summary["c3_vs_retrained_max_holm_p_paper_rule"] = float(S.loc[c3, "holm_c3_family6"].max())
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print(S.round(5).to_string())
