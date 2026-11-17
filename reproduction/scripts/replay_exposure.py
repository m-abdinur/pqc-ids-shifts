"""Replay the E2 (TLS-likely) statistics and the partial-adoption checks.

Inputs: reference/exposure_sources_20260923/ (unpacked exposure_sources.zip)
and the locked main/B3 raw results. Outputs: verified_outputs/exposure/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reference/exposure_sources_20260923"
OUT = ROOT / "verified_outputs/exposure"
OUT.mkdir(parents=True, exist_ok=True)

E2 = json.loads((SRC / "tls_likely_exposure_v6/outputs/final_10_splits_xgboost_3_2_0/"
                 "e2_tls_likely_raw_10_splits.json").read_text())
MAIN = json.loads((ROOT / "reference/main_results/revised_pipeline_raw.json").read_text())
B3 = json.loads((ROOT / "reference/b3_results/revised_pipeline_b3_raw.json").read_text())
MODELS = ["rf", "xgb", "mlp"]


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def tost(d, margin):
    se = d.std(ddof=1) / np.sqrt(len(d)); df = len(d) - 1
    return max(stats.t.sf((d.mean() + margin) / se, df), stats.t.cdf((d.mean() - margin) / se, df))


# 1. E2 statistics, both designs
rows = []
for design, ref in [("E2_main", MAIN), ("E2_B3", B3)]:
    for m in MODELS:
        def wf1(s, k):
            return E2["splits"][str(s)]["designs"][design]["classifiers"][m][k]["weighted_f1"]
        c1 = np.array([wf1(s, "C1") for s in range(10)])
        c2 = np.array([wf1(s, "E2_C2") for s in range(10)])
        c3 = np.array([wf1(s, "E2_C3") for s in range(10)])
        locked = np.array([ref[str(s)]["results"][m]["C1"]["weighted_f1"] for s in range(10)])
        for name, d in [("C2-C1", c2 - c1), ("C3-C1", c3 - c1)]:
            se = d.std(ddof=1) / np.sqrt(10); t = stats.t.ppf(0.95, 9)
            rows.append(dict(design=design, model=m, contrast=name, C1=c1.mean(),
                             C1_max_abs_diff_vs_locked=np.abs(c1 - locked).max(),
                             mean=d.mean(), lo90=d.mean() - t * se, hi90=d.mean() + t * se,
                             tost_p_001=tost(d, 0.01), tost_p_0005=tost(d, 0.005)))
S = pd.DataFrame(rows)
for _, grp in S.groupby(["design", "contrast"]):
    S.loc[grp.index, "holm_tost_001"] = holm(grp.tost_p_001)
    S.loc[grp.index, "holm_tost_0005"] = holm(grp.tost_p_0005)
S.to_csv(OUT / "e2_replay.csv", index=False)

# 2. Changed features classed stable and selected under E2
mech = pd.read_csv(SRC / "v6_tables_e2/v6_e2_feature_mechanical_classification.csv")
mech.to_csv(OUT / "e2_feature_classification.csv", index=False)
missed = {}
# Full-exposure screen (main pipeline) against the features E2 mechanically changes
changed = {d: set(mech[(mech.Design == d) & (mech["Mechanically Changed"] == True)].Feature)
           for d in ["E2-main", "E2-B3"]}
full_exposure_check = []
for s in range(10):
    shifted_main = set(MAIN[str(s)]["shifted_prep"]); stable_main = set(MAIN[str(s)]["stable_prep"])
    full_exposure_check.append({"split": s,
                                "full_exposure_shifted_equals_E2_changed": shifted_main == changed["E2-main"],
                                "stable_features_changed_under_E2": len(stable_main & changed["E2-main"]),
                                "n_shifted": len(shifted_main), "n_stable": len(stable_main)})
pd.DataFrame(full_exposure_check).to_csv(OUT / "full_exposure_screen_check.csv", index=False)
for design in ["E2_main", "E2_B3"]:
    sets = [set(E2["splits"][str(s)]["designs"][design]["shifted_features"]) for s in range(10)]
    missed[design] = {"shifted_counts": sorted({len(x) for x in sets}),
                      "shifted_same_in_all_splits": all(x == sets[0] for x in sets)}

# 3. Partial adoption (Table D.1)
PA = json.loads((SRC / "revised_partial_adoption_a1b2_raw.json").read_text())["raw"]
D1 = pd.read_csv(SRC / "table_revised_partial_adoption_a1b2.csv")
NAMES = {"rf": "Random Forest", "xgb": "XGBoost", "mlp": "MLP"}
pa_rows, pa_checks = [], {}
for m in MODELS:
    for level in ["0.0", "0.25", "0.5", "0.75", "1.0"]:
        full = np.array([PA[str(s)][m]["partial"][level]["full_feature_model"]["weighted_f1"] for s in range(10)])
        c3 = np.array([PA[str(s)][m]["partial"][level]["c3_selected_model"]["weighted_f1"] for s in range(10)])
        row = D1[(D1.Model == NAMES[m]) & (D1["Adoption proportion"] == float(level))].iloc[0]
        pa_rows.append(dict(model=m, level=float(level), full=full.mean(), c3=c3.mean(),
                            table_full=row["Full-feature model wF1"], table_c3=row["C3 selected model wF1"],
                            matches_table=bool(round(full.mean(), 4) == row["Full-feature model wF1"]
                                               and round(c3.mean(), 4) == row["C3 selected model wF1"])))
    def main(k):
        blocks = [MAIN[str(s)]["results"][m] for s in range(10)]
        return np.array([(b["C1"] if k == "C1" else b["per_variant"]["hybrid768"][k])["weighted_f1"]
                         for b in blocks])
    full0 = np.array([PA[str(s)][m]["partial"]["0.0"]["full_feature_model"]["weighted_f1"] for s in range(10)])
    full1 = np.array([PA[str(s)][m]["partial"]["1.0"]["full_feature_model"]["weighted_f1"] for s in range(10)])
    c3_all = np.array([[PA[str(s)][m]["partial"][lv]["c3_selected_model"]["weighted_f1"]
                        for lv in ["0.0", "0.25", "0.5", "0.75", "1.0"]] for s in range(10)])
    c3_diff = c3_all[:, 0] - main("C3")
    pa_checks[m] = {"0pct_equals_main_C1": bool(np.array_equal(full0, main("C1"))),
                    "100pct_equals_main_C2": bool(np.array_equal(full1, main("C2"))),
                    "C3_constant_across_levels": bool((c3_all == c3_all[:, :1]).all()),
                    "C3_vs_main_C3_splits_differing": [int(s) for s in np.flatnonzero(c3_diff != 0)],
                    "C3_vs_main_C3_diffs": [float(x) for x in c3_diff[c3_diff != 0]]}
pd.DataFrame(pa_rows).to_csv(OUT / "partial_adoption_replay.csv", index=False)

summary = {"e2_statistics_rows": len(S),
           "e2_c1_equals_locked": bool((S.C1_max_abs_diff_vs_locked == 0).all()),
           "all_within_0_01": bool(((S.lo90 > -0.01) & (S.hi90 < 0.01)).all()),
           "all_within_0_005": bool(((S.lo90 > -0.005) & (S.hi90 < 0.005)).all()),
           "e2_shifted_sets": missed,
           "e2_changed_feature_count": {k: len(v) for k, v in changed.items()},
           "full_exposure_screen_flags_all_E2_changed_features_all_splits":
               all(r["full_exposure_shifted_equals_E2_changed"] for r in full_exposure_check),
           "full_exposure_stable_features_changed_under_E2_max":
               max(r["stable_features_changed_under_E2"] for r in full_exposure_check),
           "table_d1_rows_match": all(r["matches_table"] for r in pa_rows),
           "partial_adoption_checks": pa_checks}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
