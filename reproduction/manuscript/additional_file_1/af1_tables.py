"""Build every Additional file 1 table directly from locked results and verified replay outputs.

Each table is a dict: id, title, cols, rows (list of lists of strings), note.
No value is typed by hand except where a source is the verified thesis table (flagged SOURCE_THESIS).
"""
import ast
import json
import math
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

WC = Path("/home/claude/wc/PQC_IDS_Reproduction_Working_Copy")
VO = WC / "verified_outputs"
REF = WC / "reference"
THESIS = Path(__file__).resolve().parent / "thesis_extract.txt"
MODELS = [("rf", "RF"), ("xgb", "XGBoost"), ("mlp", "MLP")]
MNAME = {"rf": "RF", "xgb": "XGBoost", "mlp": "MLP", "Random Forest": "RF", "XGBoost": "XGBoost", "MLP": "MLP"}
PROVENANCE = {}  # table id -> source description (for the internal source map)


# ---------- formatting ----------
def f4(x, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    s = f"{x:+.4f}" if sign else f"{x:.4f}"
    if s in ("-0.0000", "+0.0000") or s == "-0.0000":
        s = "+0.0000" if sign else "0.0000"
    return s.replace("-", "−")


def f3(x, sign=False):
    s = f"{x:+.3f}" if sign else f"{x:.3f}"
    return s.replace("-", "−")


def ci(lo, hi):
    return f"[{f4(lo, True)}, {f4(hi, True)}]"


def pv(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "–"
    if p < 0.001:
        return "<0.001"
    if p > 0.999:
        return "1.000"
    if 0.045 < p < 0.055:  # show a fourth decimal near the 0.05 threshold
        return f"{p:.4f}"
    return f"{p:.3f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def yn(b):
    return "Yes" if b else "No"


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[idx]))
        adj[idx] = running
    return adj


def paired_tost(d, margin):
    d = np.asarray(d, float)
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / math.sqrt(n)
    if se == 0:
        return m, m, m, 0.0 if abs(m) < margin else 1.0
    t_lo = (m + margin) / se
    t_hi = (m - margin) / se
    p = max(1 - stats.t.cdf(t_lo, n - 1), stats.t.cdf(t_hi, n - 1))
    q = stats.t.ppf(0.95, n - 1)
    return m, m - q * se, m + q * se, p


def table(tid, title, cols, rows, note="", source=""):
    PROVENANCE[tid] = source
    return dict(id=tid, title=title, cols=cols, rows=rows, note=note)


# ---------- locked main raw ----------
RAW = json.load(open(REF / "main_results/revised_pipeline_raw.json"))
SEEDS = [str(s) for s in range(10)]


def raw_series(model, path):
    out = []
    for s in SEEDS:
        x = RAW[s]["results"][model]
        for k in path:
            x = x[k]
        out.append(x)
    return np.array(out, float)


C1 = {m: raw_series(m, ["C1", "weighted_f1"]) for m, _ in MODELS}


# ---------- S1 environment ----------
def t_env():
    env = json.load(open(WC / "environment/main_known_versions.json"))
    rows = [["Python", env["python"]]] + [[k, v] for k, v in env["packages"].items()]
    return table("S1", "Software versions of the reproduction environment", ["Component", "Version"], rows,
                 "These versions were used for the reproduction and verification runs, which reproduced the "
                 "stored weighted-F1 scores of the original experiment. statsmodels was used only for the "
                 "re-analysis. The original July 2026 runs recorded Python, scikit-learn, XGBoost and SciPy "
                 "versions but did not lock every dependency.",
                 "environment/main_known_versions.json")


# ---------- S2 statistical families ----------
def t_families():
    rows = [
        ["Main and B3: C2 − C1 and C3 − C1 differences (weighted F1)", "3 models per design", "–", "Primary (main) / robustness (B3)", "A"],
        ["Main and B3: C3 versus C1 TOST (weighted F1)", "3 models per design and margin", "±0.01; ±0.005", "Primary ±0.01; ±0.005 sensitivity", "A"],
        ["Main and B3: ROC-AUC differences", "3 models per design", "None", "Supporting; no equivalence test", "A"],
        ["Main and B3: per-attack F1 differences", "9 model–attack pairs", "–", "Supporting", "A"],
        ["Main, B3 and combined: per-attack C3 versus C1 TOST", "9 model–attack pairs", "±0.01 (weighted-F1 margin reused)", "Secondary, exploratory", "A and B; test added after per-class results existed"],
        ["Footprint, MI-threshold, partial-adoption results", "–", "–", "Descriptive", "A"],
        ["Retraining reference: C3 versus retrained TOST", "3 models", "±0.01; ±0.005", "Supplementary upper reference", "A (reported with the paper's rule)"],
        ["Component controls and repeated random-stable subsets", "3 models per contrast", "±0.01; ±0.005 where stated", "Supporting control", "B"],
        ["TLS-likely exposure (E2)", "3 models", "±0.01; ±0.005", "Supporting", "B"],
        ["Packet-size sensitivity", "3 families (6, 6, 9 tests)", "±0.01", "Sensitivity", "B"],
        ["Timing-free and combined restrictions", "3 models per design and margin", "±0.01; ±0.005", "Robustness", "B"],
        ["Source-day holdout", "6 per metric (2 directions × 3 models)", "±0.01", "Robustness", "B"],
        ["Efficiency (training and inference time)", "3 models per design and comparison", "–", "Supporting", "Experiment package"],
        ["Corrected transformation operators", "–", "–", "Descriptive sensitivity", "C"],
        ["Duplicate-aware grouped split", "3 models per contrast and margin", "±0.01; ±0.005", "Robustness", "C"],
        ["Corrected resampled t (all designs)", "As in the original family", "±0.01; ±0.005", "Sensitivity to split overlap", "C"],
        ["Unanticipated timing shift", "3 models per delay, margin and variant", "±0.01; ±0.005", "Boundary test", "C (plan fixed before the run)"],
        ["Simulation-augmented baseline", "3 models per test set and contrast", "±0.01; ±0.005", "Comparison of defences", "C (plan fixed before the run)"],
    ]
    return table("S2", "Statistical families, their role and when they were added",
                 ["Analysis", "Holm family", "Equivalence margin", "Role", "Runs produced"], rows,
                 "Runs produced: A, the locked experiment package (2 July 2026); B, later supporting analyses run "
                 "on the same prepared data and code after the locked package; C, analyses added during "
                 "verification in September 2026, after all earlier results were known. The column gives when the "
                 "model runs were produced; several statistical tables were regenerated in a later pass. Holm "
                 "adjustment is applied within each listed family only.",
                 "registry/statistical_family_registry.csv; DECISION_REGISTER.md; verification reports")


# ---------- S3 MI threshold ----------
def t_mi():
    taus = ["0.0", "0.01", "0.05", "0.1", "0.2"]
    rows = []
    csv = pd.read_csv(REF / "main_results/main_mi_threshold_split_values.csv")
    for t in taus:
        k = np.mean([RAW[s]["results"]["rf"]["selected_counts"][t] for s in SEEDS])
        r = [f"{float(t):.2f}", f"{k:.1f}"]
        for m, _ in MODELS:
            v = raw_series(m, ["mi_sweep", t]).mean()
            vc = csv[(csv.model == m) & (np.isclose(csv.mi_threshold, float(t)))].c3_weighted_f1.mean()
            assert abs(v - vc) < 1e-9, (m, t)
            r.append(f4(v))
        rows.append(r)
    return table("S3", "Mutual-information threshold sensitivity (main design, C3 weighted F1)",
                 ["MI threshold τ", "Mean selected features", "RF", "XGBoost", "MLP"], rows,
                 "Means over ten splits. Candidates are the 32 stable features; τ = 0 keeps all of them. The "
                 "primary threshold is τ = 0.05. The sweep refits C3 at each threshold, so its τ = 0.05 values "
                 "can differ from the main C3 scores in the fifth decimal (RF split 6 is a known near-tie).",
                 "reference/main_results/revised_pipeline_raw.json (mi_sweep); main_mi_threshold_split_values.csv")


# ---------- S4–S6 component controls ----------
def t_components():
    sm = pd.read_csv(zipfile.ZipFile(REF / "component_sources_20260923/component_sources.zip").open("component_controls_v6/statistics/component_control_split_metrics.csv"))
    order = [("C1", "C1"), ("C2", "C2"), ("KS_only", "KS-only"), ("MI_only_threshold", "MI-only (τ = 0.05)"),
             ("MI_only_top_k", "MI-only (equal size)"), ("KS_plus_MI", "KS + MI (C3)"),
             ("random_stable_k_mean", "Random stable"), ("random_candidate_k_mean", "Random candidate")]
    keys = sorted(sm.condition_key.unique())
    rows = []
    for key, label in order:
        if key not in keys:
            raise KeyError(f"{key} not in {keys}")
        r = [label]
        for m, _ in MODELS:
            r.append(f4(sm[(sm.condition_key == key) & (sm.model_key == m)].weighted_f1.mean()))
        rows.append(r)
    t4 = table("S4", "Pipeline component controls: mean weighted F1 (main design)",
               ["Condition", "RF", "XGBoost", "MLP"], rows,
               "KS-only keeps all 32 stable candidates. MI-only (τ = 0.05) applies the MI threshold to all 55 "
               "candidates without screening and keeps 47 features, including all 23 shifted ones. MI-only "
               "(equal size) keeps the top-ranked candidates up to the size of the C3 subset (24 or 25) and "
               "retains 13–14 shifted features. Random stable and random candidate are the means of five random "
               "subsets of C3 size per split, drawn from the stable pool and from all candidates (6–13 shifted "
               "features). Table S6 gives the larger random-stable control.",
               "component_sources.zip: component_controls_v6/statistics/component_control_split_metrics.csv")
    cc = pd.read_csv(VO / "component_controls/component_contrasts_replay.csv")
    rows = []
    keep = {"KS-only − C1": True, "KS+MI − KS-only": True, "KS+MI − MI-only top-k": False, "KS+MI − Random stable": False}
    for contrast, eqq in keep.items():
        assert contrast in set(cc.contrast), contrast
        for m, mn in MODELS:
            x = cc[(cc.contrast == contrast) & (cc.model == m)].iloc[0]
            rows.append([contrast.replace("MI-only top-k", "MI-only (equal size)").replace("KS+MI", "KS + MI"), mn,
                         f4(x["mean"], True), pv(x["holm"]),
                         pv(x["tost01_holm"]) if eqq else "–", pv(x["tost005_holm"]) if eqq else "–"])
    t5 = table("S5", "Pipeline component contrasts (weighted F1, paired over ten splits)",
               ["Contrast", "Model", "Mean difference", "Difference Holm p", "TOST ±0.01 Holm p", "TOST ±0.005 Holm p"], rows,
               "Holm adjustment is across the three models within each contrast. TOST is shown only where "
               "equivalence is the question; a TOST p below 0.05 means the two conditions are equivalent within "
               "that margin. Random stable here is the five-subset mean from Table S4.",
               "verified_outputs/component_controls/component_contrasts_replay.csv")
    d6 = pd.read_csv(VO / "component_controls/d6_statistics_replay.csv")
    rows = []
    for m, mn in MODELS:
        x = d6[d6.model == m].iloc[0]
        rows.append([mn, str(int(x.total_draws // 10)), f4(x["true"]), f4(x["rand"]), f4(x["diff"], True), ci(x.lo, x.hi),
                     f"{int(x.pos)}/10", pv(max(x.holm_two, x.holm_one))])
    t6 = table("S6", "MI-selected subset versus equal-size random stable subsets",
               ["Model", "Random draws per split", "C3 (MI subset)", "Random mean", "Difference", "95% CI", "Splits with positive difference", "Holm p"], rows,
               "Each random subset has the same size as the C3 subset in that split and is drawn only from that "
               "split's stable pool. The difference compares the MI subset with the mean of the random subsets "
               "in the same split. The Holm p-value is below 0.001 for both the one-sided and the two-sided test.",
               "verified_outputs/component_controls/d6_statistics_replay.csv")
    return [t4, t5, t6]


# ---------- S7 footprints ----------
def t_footprints():
    import importlib.util
    src = (REF / "locked_v5/revised_pipeline_main_only.py").read_text()
    variants = ast.literal_eval(re.search(r"VARIANTS = (\{.*?\n\})", src, re.S).group(1))
    labels = {"hybrid512": "Hybrid ML-KEM-512-like", "pure768": "Pure ML-KEM-768-like",
              "hybrid768": "Hybrid ML-KEM-768-like (main)", "hybrid768_p2": "Hybrid 768-like, 2 packets per direction"}
    rows = []
    for v in ["hybrid512", "pure768", "hybrid768", "hybrid768_p2"]:
        cfg = variants[v]
        r = [labels[v], f"{int(cfg['d_fwd_bytes']):,} / {int(cfg['d_bwd_bytes']):,}", f"{int(cfg['d_fwd_pkts'])}"]
        for m, _ in MODELS:
            c2 = raw_series(m, ["per_variant", v, "C2", "weighted_f1"])
            c3 = raw_series(m, ["per_variant", v, "C3", "weighted_f1"])
            r += [f4((c2 - C1[m]).mean(), True), f4((c3 - C1[m]).mean(), True)]
        rows.append(r)
    return table("S7", "Transformation-footprint sensitivity (weighted F1, mean paired differences)",
                 ["Footprint", "Added bytes fwd / bwd", "Added packets per direction", "RF C2−C1", "RF C3−C1",
                  "XGBoost C2−C1", "XGBoost C3−C1", "MLP C2−C1", "MLP C3−C1"], rows,
                 "All footprints use 40 header bytes per added packet and the 1,460-byte maximum-length reference. "
                 "C3 used the subset selected under the main footprint. Its inputs are unchanged under every "
                 "footprint, so C3 − C1 is identical across rows.",
                 "reference/main_results/revised_pipeline_raw.json (per_variant); locked_v5/revised_pipeline_main_only.py (VARIANTS)")


# ---------- S8 packet size ----------
def t_packet():
    t = pd.read_csv(VO / "packet_size/table_4_13_replay.csv")
    eq = pd.read_csv(VO / "packet_size/equivalence_replay.csv")
    rows = []
    for size in [1200, 1330, 1460]:
        for mfull in ["Random Forest", "XGBoost", "MLP"]:
            x = t[(t.packet_size == size) & (t.model == mfull)].iloc[0]
            mk = {"Random Forest": "rf", "XGBoost": "xgb", "MLP": "mlp"}[mfull]
            if size == 1460:
                eqs = "Reference"
            else:
                e = eq[(eq.family == "PACKET_SIZE_C2_TOST_001") & (eq.model == mk) & (eq.variant == f"packet_size_{size}")].iloc[0]
                eqs = f"{f4(e['mean'], True)}; Holm p {pv(e.holm_p)}; {'equivalent' if e.equivalent else 'not equivalent'}"
            rows.append([f"{size:,}", MNAME[mfull], f4(x.C2), f4(x.C2 - x.C1, True), f4(x.C3_minus_C1, True), eqs])
    return table("S8", "Packet-size sensitivity (weighted F1)",
                 ["Added-packet size (bytes)", "Model", "C2", "C2 − C1", "C3 − C1", "C2 versus 1,460-byte C2 (±0.01 TOST)"], rows,
                 "Packet size changes only Fwd Pkt Len Max, Bwd Pkt Len Max and Pkt Len Max. The screen classes all "
                 "three as shifted at every size, so the C3 inputs are identical and C3 does not change. C2 was "
                 "below C1 in every split at every size.",
                 "verified_outputs/packet_size/table_4_13_replay.csv; equivalence_replay.csv")


# ---------- S9 operators ----------
def t_operators():
    o = pd.read_csv(VO / "operator_sensitivity/c2_cross_arm_summary.csv")
    lab = {"historical": "Original", "byte_lengths": "Byte lengths", "zero_byte_rate": "Zero-byte rate", "combined": "Both corrections"}
    rows = []
    for m, mn in MODELS:
        for arm in ["historical", "byte_lengths", "zero_byte_rate", "combined"]:
            x = o[(o.model == m) & (o.arm == arm)].iloc[0]
            rows.append([mn, lab[arm], f4(x.mean_C2), f4(x.mean_C2_minus_C1, True),
                         f"{f4(x.min_C2_minus_C1, True)} to {f4(x.max_C2_minus_C1, True)}",
                         f"{int(x.splits_C2_below_C1)}/10", f4(x.mean_C2_Infiltration_f1)])
    return table("S9", "C2 under corrected transformation operators (weighted F1)",
                 ["Model", "Operator", "Mean C2", "Mean C2 − C1", "C2 − C1 range", "Splits with C2 < C1", "C2 Infiltration F1"], rows,
                 "Byte lengths: added packets take the allocated byte lengths, so a directional maximum cannot "
                 "exceed its total. Zero-byte rate: the byte rate is recomputed for flows whose original byte total "
                 "was zero. Feature selection was identical under every operator, and C3 predictions were unchanged "
                 "(mean C3: RF 0.8505, XGBoost 0.8590, MLP 0.8415).",
                 "verified_outputs/operator_sensitivity/c2_cross_arm_summary.csv; verification.json")


# ---------- S10–S13 exposure ----------
def t_exposure():
    cov = pd.read_csv(REF / "exposure_sources_20260923/tls_likely_exposure_v6/outputs/exposure_coverage_by_class.csv")
    rows = [[r.Class, f"{int(r['Total Flows']):,}", f"{int(r['E1 Bidirectional Count']):,} ({pct(r['E1 Percentage of Class'], 2)})",
             f"{int(r['E2 TLS-Likely Count']):,} ({pct(r['E2 Percentage of Class'], 2)})"] for _, r in cov.iterrows()]
    t10 = table("S10", "Transformation exposure by class",
                ["Class", "Flows", "Bidirectional flows (main exposure)", "TLS-likely flows"], rows,
                "Bidirectional: at least one forward and one backward packet. TLS-likely: bidirectional TCP flows to "
                "destination port 443.",
                "exposure_sources: tls_likely_exposure_v6/outputs/exposure_coverage_by_class.csv")
    e = pd.read_csv(VO / "exposure/e2_replay.csv")
    rows = []
    for design, lab in [("E2_B3", "Identifier-free (primary)"), ("E2_main", "Main features (descriptive)")]:
        for m, mn in MODELS:
            a = e[(e.design == design) & (e.model == m) & (e.contrast == "C2-C1")].iloc[0]
            b = e[(e.design == design) & (e.model == m) & (e.contrast == "C3-C1")].iloc[0]
            rows.append([lab, mn, f4(a.C1), f"{f4(a['mean'], True)} {ci(a.lo90, a.hi90)}", f"{f4(b['mean'], True)} {ci(b.lo90, b.hi90)}",
                         pv(b.holm_tost_001), pv(b.holm_tost_0005)])
    t11 = table("S11", "TLS-likely exposure: degradation and recovery (weighted F1)",
                ["Feature set", "Model", "C1", "C2 − C1 [90% CI]", "C3 − C1 [90% CI]", "C3 TOST ±0.01 Holm p", "C3 TOST ±0.005 Holm p"], rows,
                "Only the 6,576 TLS-likely rows (5.5%) were transformed. Because destination port defines this "
                "exposure, the identifier-free feature set is the primary analysis; the version that keeps Dst Port "
                "as a model input is descriptive. C1 values equal the corresponding main or B3 C1.",
                "verified_outputs/exposure/e2_replay.csv")
    fc = pd.read_csv(VO / "exposure/e2_feature_classification.csv")
    miss = fc[(fc.Design == "E2-B3") & (fc["Mechanically Changed"]) & (fc["Stable Splits"] > 0)]
    rows = [[r.Feature, f"{int(r['Stable Splits'])}/10", f"{int(r['Selected Splits'])}/10"] for _, r in miss.sort_values("Feature").iterrows()]
    chk = pd.read_csv(VO / "exposure/full_exposure_screen_check.csv")
    assert chk.full_exposure_shifted_equals_E2_changed.all() and (chk.stable_features_changed_under_E2 == 0).all()
    t12 = table("S12", "Altered features that passed the screen at TLS-likely exposure",
                ["Feature", "Classed stable", "Selected for C3"], rows,
                "Identifier-free design. At TLS-likely exposure the screen flagged 15 of the 23 altered features. When "
                "the whole simulated training copy was shifted, as in the main pipeline, the screen flagged exactly "
                "these 23 features in all ten splits, and none of the 32 stable features was altered.",
                "verified_outputs/exposure/e2_feature_classification.csv; full_exposure_screen_check.csv")
    pa = pd.read_csv(VO / "exposure/partial_adoption_replay.csv")
    rows = []
    for lvl in sorted(pa.level.unique()):
        r = [f"{int(lvl * 100)}%"]
        for m, _ in MODELS:
            x = pa[(pa.model == m) & (pa.level == lvl)].iloc[0]
            r += [f4(x.full), f4(x.c3)]
        rows.append(r)
    t13 = table("S13", "Partial adoption: share of transformed test rows (weighted F1)",
                ["Transformed share", "RF full features", "RF C3", "XGBoost full features", "XGBoost C3", "MLP full features", "MLP C3"], rows,
                "Transformed rows were mixed into the test set only; 0% equals C1 and 100% equals C2 for the "
                "full-feature model. Selection used a fully shifted training copy, so the C3 inputs, and therefore "
                "C3, do not change with the share. This script computed MI on the full training partition, not a "
                "20,000-row subsample; in split 1 this dropped Down/Up Ratio, which is why these C3 values differ "
                "slightly from the main C3.",
                "verified_outputs/exposure/partial_adoption_replay.csv")
    return [t10, t11, t12, t13]


# ---------- S14 retraining ----------
def t_retraining():
    t = pd.read_csv(VO / "retraining/table_4_21_replay.csv")
    eq = pd.read_csv(VO / "retraining/equivalence_replay.csv")
    wf = eq[(eq.metric == "Weighted F1") & (eq.comparison == "C3 vs R_oracle")].copy()
    wf["holm3"] = holm(wf.p.values)
    unc = pd.read_csv(VO / "uncertainty/corrected_resampled_results.csv")
    rows = []
    for _, x in t.iterrows():
        r = [x.metric, MNAME[x.model], f4(x.C1), f4(x.C2), f4(x.C3), f4(x.retrained), f4(x.retr_minus_C1, True), f4(x.C3_minus_retr, True)]
        if x.metric == "Weighted F1":
            w = wf[wf.model == x.model].iloc[0]
            r += [ci(w.lo90, w.hi90), pv(w.holm3)]
        else:
            r += ["–", "–"]
        rows.append(r)
    return table("S14", "Retraining reference: all 70 features, trained and tested on transformed data",
                 ["Metric", "Model", "C1", "C2", "C3", "Retrained", "Retrained − C1", "C3 − retrained", "C3 − retrained 90% CI", "TOST ±0.01 Holm p"], rows,
                 "The retrained model was fitted on the original labels with transformed training values only, so it "
                 "needs the simulated transformation but no captured post-quantum traffic. Because the same deterministic transformation was applied to training and test data, "
                 "retraining is expected to reach C1-level performance. Equivalence is tested for weighted F1 only, "
                 "with Holm adjustment across the three models; ROC-AUC differences are descriptive. Under the "
                 "corrected resampled t statistic, MLP was not equivalent to the retrained reference at ±0.005 "
                 "(Table S20).",
                 "verified_outputs/retraining/table_4_21_replay.csv; equivalence_replay.csv")


# ---------- S15–S17 duplicates, grouped, source-day ----------
def t_duplicates():
    tm = pd.read_csv(VO / "duplicates_sourceday/train_test_match_replay.csv")
    g = tm.groupby("cls", sort=False).mean(numeric_only=True)
    rows = [[c, f"{g.loc[c, 'n']:,.0f}", pct(g.loc[c, "any"]), pct(g.loc[c, "same"]), pct(g.loc[c, "diff"])]
            for c in ["Benign", "DoS", "Bot", "Infiltration", "All"]]
    t15 = table("S15", "Test rows with an exact copy of their feature vector in the training partition",
                ["Class", "Test rows per split", "Any label", "Same label", "Different label"], rows,
                "Means over the ten random stratified splits. A row can match training rows with the same and with a "
                "different label, so the last two columns can sum to more than the first. All different-label "
                "matches involve Benign and Infiltration: the prepared data contain 1,156 groups (3,140 rows) of "
                "identical vectors labelled both Benign and Infiltration.",
                "verified_outputs/duplicates_sourceday/train_test_match_replay.csv; dup_global_replay.json")
    gs = pd.read_csv(VO / "grouped_evaluation/grouped_stats_replay.csv")
    rows = []
    for m, mn in MODELS:
        a = gs[(gs.model == m) & (gs.contrast == "C2-C1")].iloc[0]
        b = gs[(gs.model == m) & (gs.contrast == "C3-C1")].iloc[0]
        rows.append([mn, f4(a.C1), f4(a.C2), f4(a.C3), f"{f4(a['mean'], True)}", pv(a.holm), f"{f4(b['mean'], True)} {ci(b.lo90, b.hi90)}",
                     pv(b["holm_tost0.01"]), pv(b["holm_tost0.005"])])
    t16 = table("S16", "Duplicate-aware grouped split (weighted F1)",
                ["Model", "C1", "C2", "C3", "C2 − C1", "Degradation Holm p", "C3 − C1 [90% CI]", "TOST ±0.01 Holm p", "TOST ±0.005 Holm p"], rows,
                "Every exact 70-feature vector was kept on one side of the train–test boundary (89,064 groups); "
                "label-conflict groups were kept whole; 30% of each class went to the test set; same ten seeds and "
                "model settings.",
                "verified_outputs/grouped_evaluation/grouped_stats_replay.csv")
    cd = pd.read_csv(VO / "duplicates_sourceday/crossday_replay.csv")
    dlab = {"feb28_to_mar01": "28\u00a0Feb\u00a0→\u00a01\u00a0Mar", "mar01_to_feb28": "1\u00a0Mar\u00a0→\u00a028\u00a0Feb"}
    rows = []
    for metric, mlab in [("wf1", "Weighted"), ("inf_f1", "Infiltration")]:
        for d in ["feb28_to_mar01", "mar01_to_feb28"]:
            for m, mn in MODELS:
                x = cd[(cd.metric == metric) & (cd.dir == d) & (cd.model == m)].iloc[0]
                lo, hi = [float(v) for v in re.findall(r"float64\(([-0-9.e]+)\)", x.ci90)]
                verdict = "Equivalent" if x.tost_holm < 0.05 else ("No; C3 above C1" if lo > 0 else "No")
                rows.append([mlab, dlab[d], mn, f4(x.C1), f4(x.C2), f4(x.C3), f4(x.c3_c1, True), f"[{f4(lo, True)}, {f4(hi, True)}]",
                             f"{int(x.c3_above_c1_splits)}/10", pv(x.tost_holm), verdict])
    t17 = table("S17", "Infiltration source-day holdout",
                ["F1 metric", "Train → test day", "Model", "C1", "C2", "C3", "C3 − C1", "90% CI", "Splits C3 > C1", "TOST ±0.01 Holm p", "Equivalent"], rows,
                "Infiltration training rows come only from one capture day and test rows only from the other. "
                "Benign, DoS and Bot keep their original random split. Holm adjustment is across the six "
                "direction–model comparisons for each metric. The 90% CIs are rounded to four decimals in the "
                "replay output.",
                "verified_outputs/duplicates_sourceday/crossday_replay.csv")
    return [t15, t16, t17]


# ---------- S18–S19 restrictions ----------
def t_restrictions():
    pa = pd.read_csv(VO / "per_attack/per_attack_tost_replay.csv")
    rows = []
    for _, x in pa[pa.design == "B3"].iterrows():
        rows.append(["Identifier-free (B3)", MNAME[x.model], x.attack, f4(x.c1), f4(x.c3), f"{f4(x.c3_c1, True)} {ci(x.ci90_lo, x.ci90_hi)}", pv(x.tost_holm)])
    # combined design: replay per-class TOST from split-level values
    cm = pd.read_csv(zipfile.ZipFile(REF / "restrictions_sources_20260923/table410_sources.zip").open("combined_timing_identifier/full_10_splits/combined_metrics_split_values.csv"))
    recs = []
    for m, mn in MODELS:
        for att in ["DoS", "Bot", "Infiltration"]:
            c1 = cm[(cm.model_key == m) & (cm.condition == "C1")].sort_values("seed")[f"{att}_f1"].values
            c3 = cm[(cm.model_key == m) & (cm.condition == "C3")].sort_values("seed")[f"{att}_f1"].values
            d = c3 - c1
            mean, lo, hi, p = paired_tost(d, 0.01)
            recs.append([mn, att, c1.mean(), c3.mean(), mean, lo, hi, p])
    adj = holm([r[-1] for r in recs])
    for r, a in zip(recs, adj):
        rows.append(["Combined", r[0], r[1], f4(r[2]), f4(r[3]), f"{f4(r[4], True)} {ci(r[5], r[6])}", pv(a)])
    t18 = table("S18", "Per-attack F1 under the feature restrictions (secondary, exploratory ±0.01 comparison)",
                ["Design", "Model", "Attack", "C1", "C3", "C3 − C1 [90% CI]", "TOST ±0.01 Holm p"], rows,
                "The ±0.01 margin was defined for weighted F1 and is reused here; no per-class margin was fixed before "
                "the per-class results. Holm adjustment is across the nine model–attack pairs within each design. "
                "Combined: identifier-free features for C1 and C2; C3 candidates also exclude timing and rate "
                "features. For DoS and Bot the baseline F1 is close to 1, so these tests carry little information.",
                "verified_outputs/per_attack/per_attack_tost_replay.csv; combined run split values (t410)")
    # B3 MLP sweep: verified thesis Table 4.12 (restrictions verification: all five rows match the raw file)
    txt = THESIS.read_text()
    blk = txt.split("Table ‎4.12 Identifier-Free MLP MI-Threshold Sweep\n[TABLE]\n")[1].split("[/TABLE]")[0].strip().splitlines()[1:]
    rows = []
    for line in blk:
        t, c3, d, eqv = [s.strip() for s in line.split("|")]
        rows.append([f"{float(t):.2f}", c3, d.replace("-", "−"), eqv])
    t19 = table("S19", "Identifier-free MLP: MI-threshold sweep (weighted F1)",
                ["MI threshold τ", "MLP C3", "C3 − C1", "Equivalent within ±0.01"], rows,
                "B3 C1 for MLP is 0.8381. No tested threshold brought MLP within ±0.01 of its baseline.",
                "thesis Table 4.12 (verified against the V5 sweep raw file in PQC_IDS_Restrictions_Verification.md)")
    return [t18, t19]


# ---------- S20 corrected resampled t ----------
def t_uncertainty():
    u = pd.read_csv(VO / "uncertainty/corrected_resampled_results.csv")
    dlab = {"main": "Main", "B3_identifier_free": "Identifier-free (B3)", "grouped": "Grouped split",
            "timing_free": "Timing-free", "E2_B3_tls_likely": "TLS-likely (B3)",
            "retraining_ref (C1 slot = retrained)": "C3 versus retrained"}
    rows = []
    for d in ["main", "B3_identifier_free", "grouped", "timing_free", "E2_B3_tls_likely", "retraining_ref (C1 slot = retrained)"]:
        for m, mn in MODELS:
            def g(se, c):
                q = u[(u.design == d) & (u.model == m) & (u.se_type == se) & (u.contrast == c)]
                return q.iloc[0] if len(q) else None
            n2, c2 = g("naive", "C2-C1"), g("corrected", "C2-C1")
            n3, c3 = g("naive", "C3-C1"), g("corrected", "C3-C1")
            deg = "–" if d.startswith("retraining") or c2 is None else pv(c2.holm_p_two_sided)
            rows.append([dlab[d], mn, deg, f"{f4(n3['mean'], True)}", ci(c3.lo90, c3.hi90),
                         f"{pv(n3['holm_tost_0.01'])} → {pv(c3['holm_tost_0.01'])}",
                         f"{pv(n3['holm_tost_0.005'])} → {pv(c3['holm_tost_0.005'])}"])
    return table("S20", "Sensitivity to overlapping splits: corrected resampled t statistic (weighted F1)",
                 ["Design", "Model", "C2 − C1 corrected Holm p", "C3 − C1", "Corrected 90% CI", "TOST ±0.01 Holm p (recorded → corrected)",
                  "TOST ±0.005 Holm p (recorded → corrected)"], rows,
                 "The correction of Nadeau and Bengio (2003) replaces 1/k with 1/k + n_test/n_train (k = 10; "
                 "n_test/n_train = 35,872/83,699, or 0.30/0.70 for the grouped split), which multiplies every "
                 "standard error by about 2.3. Recorded values use the uncorrected paired t statistic for TOST. "
                 "For C3 versus retrained, the difference is C3 − retrained. Holm adjustment is across the three "
                 "models within each design.",
                 "verified_outputs/uncertainty/corrected_resampled_results.csv")


# ---------- S21 error analysis ----------
def t_errors():
    s = pd.read_csv(VO / "error_analysis/error_analysis_summary.csv")
    rows = []
    for m, mn in MODELS:
        for c in ["C1", "C2", "C3"]:
            x = s[(s.model == m) & (s.condition == c)].iloc[0]
            assert abs(x.weighted_f1 - {"C1": C1[m].mean(), "C2": raw_series(m, ["per_variant", "hybrid768", "C2", "weighted_f1"]).mean(),
                                        "C3": raw_series(m, ["per_variant", "hybrid768", "C3", "weighted_f1"]).mean()}[c]) < 1e-9
            rows.append([mn, c, f4(x.macro_f1), f4(x.benign_false_alarm), f4(x.attack_to_benign)])
    return table("S21", "Error analysis: macro F1 and the direction of errors (main design)",
                 ["Model", "Condition", "Macro F1", "Benign false-alarm rate", "Attack-to-Benign rate"], rows,
                 "Benign false-alarm rate: share of Benign test rows predicted as any attack. Attack-to-Benign rate: "
                 "share of DoS, Bot and Infiltration test rows predicted as Benign. Means over ten splits, computed "
                 "from the saved predictions, which reproduce the stored weighted-F1 scores exactly. The high "
                 "baseline false-alarm rates are consistent with the Benign–Infiltration overlap described in "
                 "Table S15.",
                 "replayed from operator-sensitivity saved predictions (historical arm) by af1/replay_error_analysis.py")


# ---------- S22 efficiency ----------
def t_efficiency():
    txt = THESIS.read_text()
    blk = txt.split("Table D.13 Efficiency Statistics for Training and Inference Time\n[TABLE]\n")[1].split("[/TABLE]")[0].strip().splitlines()[1:]
    rows = []
    for line in blk:
        d, m, comp, ratio, c, ch, p, concl = [s.strip() for s in line.split("|")]
        rows.append([d, MNAME.get(m, m) if m != "XGB" else "XGBoost", comp, ratio, c, ch.replace("-", "−"), p, concl])
    return table("S22", "Training and inference time: C3 relative to the full-feature model",
                 ["Design", "Model", "Comparison", "Time ratio", "95% CI", "Change", "Holm p", "Conclusion"], rows,
                 "Ratios are C3 time divided by the full-feature time (C1 for training, C2 for inference), from "
                 "paired split-level measurements on the original execution hardware; they will differ on other "
                 "hardware. Holm adjustment is across the three models within each design and comparison.",
                 "thesis Table D.13 (source: statistical_results/efficiency/efficiency_revised_statistics_v1.csv in the original package; not replayed here)")


# ---------- S23–S24 timing shift (Colab run 20260926T080230Z_e9f1ad6e, primary record) ----------
TS = VO / "timing_shift_colab/results/analysis"


def t_timing():
    r = pd.read_csv(TS / "timing_shift_statistics.csv")
    delays = sorted(r.delay_us.unique())
    dl = lambda d: "0" if d == 0 else f"{d / 1000:g}"
    rows = []
    for d in delays:
        for m, mn in MODELS:
            def g(cond, var):
                return r[(r.condition == cond) & (r.variant == var) & (r.delay_us == d) & (r.model == m)].iloc[0]
            a, f_, c2 = g("C3u", "additive_max"), g("C3u", "floor_max"), g("C2", "additive_max")
            rows.append([dl(d), mn, f4(c2["mean_difference"], True),
                         f"{f4(a['mean_difference'], True)} {ci(a.ci90_lower, a.ci90_upper)}", pv(a["tost_0.01_holm"]),
                         f"{f4(f_['mean_difference'], True)} {ci(f_.ci90_lower, f_.ci90_upper)}", pv(f_["tost_0.01_holm"])])
    t23 = table("S23", "Unanticipated timing shift: the paper's C3 when the screen did not see the delay (weighted F1)",
                ["Delay (ms)", "Model", "C2 − C1", "C3 − C1, additive [90% CI]", "TOST ±0.01 Holm p",
                 "C3 − C1, floor [90% CI]", "TOST ±0.01 Holm p"], rows,
                "The delay is added on top of the main transformation, to eligible flows only. Additive: the delay is "
                "added to the largest gap (primary). Floor: the largest gap becomes at least the delay (sensitivity). "
                "C3 uses the subset selected in the main experiment. Holm adjustment is across the three models at "
                "each delay and variant.",
                "verified_outputs/timing_shift_colab/results/analysis/timing_shift_statistics.csv")
    sel = pd.read_csv(TS / "anticipated_selection_summary.csv")
    rows = []
    for d in delays:
        ks = sel[sel.delay_us == d].n_selected
        kept = f"{ks.min()}" if ks.min() == ks.max() else f"{ks.min()}–{ks.max()}"
        for m, mn in MODELS:
            x = r[(r.condition == "C3a") & (r.variant == "additive_max") & (r.delay_us == d) & (r.model == m)].iloc[0]
            rows.append([dl(d), kept, mn, f"{f4(x['mean_difference'], True)} {ci(x.ci90_lower, x.ci90_upper)}",
                         pv(x["tost_0.01_holm"]), pv(x["tost_0.005_holm"]), pv(x["corr_tost_0.01_holm"])])
    t24 = table("S24", "Anticipated timing shift: screen run against a copy that includes the delay (weighted F1)",
                ["Delay (ms)", "Features kept", "Model", "C3 − C1 [90% CI]", "TOST ±0.01 Holm p", "TOST ±0.005 Holm p",
                 "Corrected ±0.01 Holm p"], rows,
                "The screen compares the training data with a copy shifted by the main transformation plus the delay, "
                "then MI selects from the stable features and each model is refitted. Features kept is the range across "
                "the ten splits. Corrected: resampled t statistic of Nadeau and Bengio (2003).",
                "verified_outputs/timing_shift_colab/results/analysis/timing_shift_statistics.csv; anticipated_selection_summary.csv")
    return [t23, t24]


# ---------- S25 augmentation baseline (Colab run 20260926T103406Z_24f10c33, primary record) ----------
AUG = VO / "augmentation_colab/results/analysis"
AUG_ORDER = [("original", "Original data"), ("well_specified", "Simulated shift (as assumed)"),
             ("hybrid512", "Footprint: hybrid 512-like"), ("pure768", "Footprint: pure 768-like"),
             ("hybrid768_p2", "Footprint: 2 packets per direction"), ("packet_1200", "Packet size 1,200 B"),
             ("packet_1330", "Packet size 1,330 B"), ("adoption_25", "Adoption 25%"), ("adoption_50", "Adoption 50%"),
             ("adoption_75", "Adoption 75%"), ("delay_100us", "Delay 0.1 ms (unsimulated)"),
             ("delay_1000us", "Delay 1 ms (unsimulated)"), ("delay_10000us", "Delay 10 ms (unsimulated)"),
             ("delay_100000us", "Delay 100 ms (unsimulated)")]


def t_augmentation():
    st = pd.read_csv(AUG / "augmentation_statistics.csv")
    rows = []
    for ts, label in AUG_ORDER:
        for m, mn in MODELS:
            g = lambda c: st[(st.test_set == ts) & (st.contrast == c) & (st.model == m)].iloc[0]
            a, c3, d = g("A-C1"), g("C3-C1"), g("C3-A")
            rows.append([label, mn, f4(a["mean_difference"], True), "Yes" if a["equiv_0.01"] else "No",
                         f4(c3["mean_difference"], True), "Yes" if c3["equiv_0.01"] else "No",
                         f"{f4(d['mean_difference'], True)} {ci(d.ci90_lower, d.ci90_upper)}", pv(d["diff_holm"])])
    return [table("S25", "Training augmentation (A) versus stable-feature screening (C3), weighted F1",
                  ["Test set", "Model", "A − C1", "A within ±0.01", "C3 − C1", "C3 within ±0.01", "C3 − A [90% CI]",
                   "Difference Holm p"], rows,
                  "A is trained on the original training partition stacked with its simulated (hybrid-768, 1,460-byte) "
                  "copy, with the same labels and all 70 features. C3 is the paper's selection. Both assume the same "
                  "simulated shift; the footprint and packet-size rows test a wrong simulated size, and the delay rows "
                  "test a change neither anticipated. Within ±0.01: Holm-adjusted TOST p < 0.05 with the 90% interval "
                  "inside the margin, Holm across the three models per test set. Positive C3 − A favours screening.",
                  "verified_outputs/augmentation_colab/results/analysis/augmentation_statistics.csv")]


def all_tables():
    out = [t_env(), t_families(), t_mi()]
    out += t_components()
    out += [t_footprints(), t_packet(), t_operators()]
    out += t_exposure()
    out += [t_retraining()]
    out += t_duplicates()
    out += t_restrictions()
    out += [t_uncertainty(), t_errors(), t_efficiency()]
    out += t_timing()
    out += t_augmentation()
    return out


if __name__ == "__main__":
    for t in all_tables():
        print(f"\n=== Table {t['id']}. {t['title']}")
        print(" | ".join(t["cols"]))
        for r in t["rows"]:
            print(" | ".join(r))
