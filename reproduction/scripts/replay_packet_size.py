"""Replay the packet-size sensitivity analysis (thesis Table 4.13).

Assistant-prepared check. Inputs:
  reference/packet_retrain_sources_20260923/  (unpacked packet_retrain_sources.zip)
  reference/main_results/revised_pipeline_raw.json  (locked main results)
  the verified prepared matrix (provenance pickle; path given as argv[1])
Outputs: verified_outputs/packet_size/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reference/packet_retrain_sources_20260923/packet_size_sensitivity"
OUT = ROOT / "verified_outputs/packet_size"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "reference/locked_v5"))

RAW = json.loads((SRC / "full_10_splits/packet_size_raw_10_splits.json").read_text())
MAIN = json.loads((ROOT / "reference/main_results/revised_pipeline_raw.json").read_text())
MODELS = ["rf", "xgb", "mlp"]
NAMES = {"rf": "Random Forest", "xgb": "XGBoost", "mlp": "MLP"}
REF, ALTS = "packet_size_1460_reference", ["packet_size_1200", "packet_size_1330"]
SIZE = {"packet_size_1200": 1200, "packet_size_1330": 1330, REF: 1460}
SEEDS = [str(s) for s in range(10)]
summary = {}

# 1. Configuration: only added_pkt_size differs
cfg = RAW["config"]["variants"]
summary["only_packet_size_differs"] = all(
    [k for k in cfg[REF] if cfg[a][k] != cfg[REF][k]] == ["added_pkt_size"] for a in ALTS)


def wf1(seed, variant, model, cond):
    return RAW["splits"][seed]["variants"][variant]["results"][model][cond]["weighted_f1"]


def locked(seed, model, cond):
    block = MAIN[seed]["results"][model]
    return (block["C1"] if cond == "C1" else block["per_variant"]["hybrid768"][cond])["weighted_f1"]


# 2. Metric sources and agreement with the locked main results
summary["metrics_source"] = {v: sorted({RAW["splits"][s]["variants"][v]["metrics_source"] for s in SEEDS})
                             for v in [REF] + ALTS}
rows = []
for s in SEEDS:
    for v in [REF] + ALTS:
        for m in MODELS:
            for c in ["C1", "C2", "C3"]:
                rows.append(dict(seed=int(s), variant=v, model=m, condition=c, value=wf1(s, v, m, c),
                                 locked_main=locked(s, m, c), diff=wf1(s, v, m, c) - locked(s, m, c)))
agree = pd.DataFrame(rows)
agree.to_csv(OUT / "agreement_with_locked_main.csv", index=False)
summary["reference_1460_equals_locked_main_all_conditions"] = bool((agree[agree.variant == REF]["diff"] == 0).all())
summary["alternative_C1_equals_locked_C1"] = bool((agree[(agree.variant != REF) & (agree.condition == "C1")]["diff"] == 0).all())
nz = agree[(agree.variant != REF) & (agree.condition == "C3") & (agree["diff"] != 0)]
summary["alternative_C3_differs_from_locked_C3"] = nz[["seed", "variant", "model", "diff"]].to_dict("records")

# 3. Feature screening and identity checks
sel_rows = []
for s in SEEDS:
    ref = RAW["splits"][s]["variants"][REF]
    for a in ALTS:
        alt = RAW["splits"][s]["variants"][a]
        ks_diff = [f for f in ref["ks_distances"] if ref["ks_distances"][f] != alt["ks_distances"][f]]
        sel_rows.append(dict(seed=int(s), variant=a,
                             stable_equal=set(alt["stable_features"]) == set(ref["stable_features"]),
                             shifted_equal=set(alt["shifted_features"]) == set(ref["shifted_features"]),
                             selected_equal=alt["selected_features"] == ref["selected_features"],
                             mi_equal=alt["mi_scores"] == ref["mi_scores"],
                             ks_changed_features=";".join(sorted(ks_diff))))
    for v in [REF] + ALTS:
        chk = RAW["splits"][s]["variants"][v]["identity"]["checks"]
        sel_rows[-1].setdefault("identity_checks_all_true", True)
        sel_rows[-1]["identity_checks_all_true"] &= all(chk.values())
sel = pd.DataFrame(sel_rows)
sel.to_csv(OUT / "selection_comparison.csv", index=False)
summary["selection"] = {k: int(sel[k].sum()) for k in ["stable_equal", "shifted_equal", "selected_equal", "mi_equal"]}
summary["selection"]["comparisons"] = len(sel)
summary["ks_changed_feature_sets"] = sorted(set(sel.ks_changed_features))
summary["identity_checks_all_true"] = bool(sel.identity_checks_all_true.all())


# 4. Statistics: three predefined families, Holm within family, TOST at +/-0.01
def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); n = len(p)
    adj = np.empty(n); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (n - k) * p[i])); adj[i] = run
    return adj


def tost(d, margin=0.01):
    se = d.std(ddof=1) / np.sqrt(len(d)); df = len(d) - 1
    if se == 0:
        return 0.0 if abs(d.mean()) < margin else 1.0
    return max(stats.t.sf((d.mean() + margin) / se, df), stats.t.cdf((d.mean() - margin) / se, df))


fam = []
for m in MODELS:
    for a in ALTS:
        for cond, family in [("C2", "PACKET_SIZE_C2_TOST_001"), ("C3", "PACKET_SIZE_C3_TOST_001")]:
            d = np.array([wf1(s, a, m, cond) - wf1(s, REF, m, cond) for s in SEEDS])
            fam.append(dict(family=family, model=m, variant=a, mean=d.mean(), sd=d.std(ddof=1), p=tost(d)))
    for v in [REF] + ALTS:
        d = np.array([wf1(s, v, m, "C3") - wf1(s, v, m, "C1") for s in SEEDS])
        fam.append(dict(family="PACKET_SIZE_C3_RECOVERY_TOST_001", model=m, variant=v, mean=d.mean(),
                        sd=d.std(ddof=1), p=tost(d)))
F = pd.DataFrame(fam)
for _, grp in F.groupby("family"):
    F.loc[grp.index, "holm_p"] = holm(grp.p)
F["equivalent"] = F.holm_p < 0.05
F.to_csv(OUT / "equivalence_replay.csv", index=False)

rec = pd.read_csv(SRC / "statistical_results/packet_size_all_equivalence_results.csv")
rec["variant"] = rec.comparison_variant
m = F.merge(rec, left_on=["family", "model", "variant"], right_on=["family_id", "model_key", "variant"])
summary["recorded_rows_matched"] = f"{len(m)}/{len(rec)}"
summary["max_abs_mean_diff_vs_recorded"] = float((m["mean"] - m.comparison_minus_reference_mean).abs().max())
summary["verdicts_match_recorded"] = bool((m.equivalent == m.equivalent_after_holm).all())
summary["max_abs_holm_p_diff_vs_recorded"] = float((m.holm_p - m.tost_holm_p).abs().max())
summary["zero_variance_comparisons"] = F[F.sd == 0][["family", "model", "variant"]].to_dict("records")
summary["family_equivalent_counts"] = F.groupby("family").equivalent.sum().astype(int).to_dict()

# 5. Table 4.13 means
tab = []
for v in ALTS + [REF]:
    for mdl in MODELS:
        c1 = np.mean([wf1(s, v, mdl, "C1") for s in SEEDS])
        c2 = np.mean([wf1(s, v, mdl, "C2") for s in SEEDS])
        c3 = np.mean([wf1(s, v, mdl, "C3") for s in SEEDS])
        tab.append(dict(packet_size=SIZE[v], model=NAMES[mdl], C1=c1, C2=c2, C3=c3, C3_minus_C1=c3 - c1))
T = pd.DataFrame(tab)
T.to_csv(OUT / "table_4_13_replay.csv", index=False)
thesis = {(1200, "Random Forest"): (0.724488, 0.850486, 0.001402), (1200, "XGBoost"): (0.764213, 0.858989, -0.001289),
          (1200, "MLP"): (0.601116, 0.841515, -0.002177), (1330, "Random Forest"): (0.720909, 0.850486, 0.001402),
          (1330, "XGBoost"): (0.768883, 0.858989, -0.001289), (1330, "MLP"): (0.597832, 0.841515, -0.002177),
          (1460, "Random Forest"): (0.734407, 0.850483, 0.001399), (1460, "XGBoost"): (0.769109, 0.858989, -0.001289),
          (1460, "MLP"): (0.592295, 0.841515, -0.002177)}
summary["table_4_13_matches_thesis_6dp"] = all(
    (round(r.C2, 6), round(r.C3, 6), round(r.C3_minus_C1, 6)) == thesis[(r.packet_size, r.model)]
    for r in T.itertuples())

# 6. Transformation check on the verified matrix: which columns differ between packet sizes
if len(sys.argv) > 1:
    from seed0_reproduction import array_hash
    import revised_pipeline_main_only as pipeline
    df = pd.read_pickle(sys.argv[1])
    feats = json.loads((ROOT / "reference/seed0_reproduction_20260920/results/seed0_status.json").read_text())[
        "prepared_data"]["features"]
    X = df[feats]
    summary["matrix_hash_ok"] = array_hash(X.values) == "b852fe0225df0c7b3a3407c3d2c98bccfa2a6252b144f3e2e6d6ba5549df5e6e"
    shifted = {v: pipeline.ob.make_approx_mlkem(X, cfg[v]) for v in [REF] + ALTS}
    audit = json.loads((ROOT / "reference/feature_consistency_20260921/results/operator_integrity.json").read_text())
    expected = next(r["shifted_matrix_sha256"] for r in audit if r["variant"] == "hybrid768")
    summary["reference_transform_equals_historical_hybrid768"] = array_hash(shifted[REF].values) == expected
    diff_cols = {}
    for a in ALTS:
        a_vals, r_vals = shifted[a].to_numpy(), shifted[REF].to_numpy()
        cols = [c for j, c in enumerate(feats) if not np.array_equal(a_vals[:, j], r_vals[:, j], equal_nan=True)]
        diff_cols[a] = {c: int((shifted[a][c].to_numpy() != shifted[REF][c].to_numpy()).sum()) for c in cols}
    summary["columns_differing_from_1460"] = diff_cols

(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary, indent=2, default=str))
