"""Independently verify a returned operator-sensitivity results folder.

Usage: python verify_operator_sensitivity_return.py <extracted results root> <output dir>

Recomputes every metric from the saved prediction/probability arrays, checks
fit sharing and feature selections, compares the historical arm with the
locked main record, and writes descriptive cross-arm summaries. It adds no
p-values, confidence intervals or equivalence decisions (see the plan).
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from seed0_reproduction import array_hash

CLASSES = ["Benign", "DoS", "Bot", "Infiltration"]
ARMS = ["historical", "byte_lengths", "zero_byte_rate", "combined"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(run_root, out):
    run_root, out = Path(run_root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    R = run_root / "results"
    report = {}

    manifest = json.loads((run_root / "result_manifest.json").read_text())
    bad = [f["path"] for f in manifest["files"] if sha(run_root / f["path"]) != f["sha256"]]
    report["result_manifest"] = {"run_id": manifest["run_id"], "files": len(manifest["files"]), "mismatches": bad}
    status = json.loads((R / "sensitivity_status.json").read_text())
    report["status"] = {k: status.get(k) for k in ("status", "fits_completed", "runner_revision", "plan_sha256",
                                                     "metric_rows", "completed_splits", "structural_gate_passed")}

    checks = json.loads((R / "historical_selection_checks.json").read_text())
    report["historical_selection_checks"] = {"splits": len(checks), "all_passed": all(c["passed"] for c in checks),
        "max_ks_diff": max(c["max_abs_ks_D_difference"] for c in checks),
        "max_mi_diff": max(c["max_abs_mi_difference"] for c in checks)}

    selections = json.loads((R / "feature_selection.json").read_text())
    same = {}
    for seed in sorted({s["seed"] for s in selections}):
        arms = {s["arm"]: s for s in selections if s["seed"] == seed}
        h = arms["historical"]
        same[seed] = all(arms[a]["selected"] == h["selected"] and arms[a]["stable"] == h["stable"]
                         and arms[a]["shifted"] == h["shifted"] for a in ARMS)
    report["selection_identical_across_arms_every_split"] = all(same.values())

    metrics = pd.read_csv(R / "metrics.csv")
    diffs = {"weighted_f1": 0.0, "per_class_f1": 0.0, "roc_auc": 0.0}
    positions = {}
    for _, r in metrics.iterrows():
        with np.load(R / r.prediction_file, allow_pickle=False) as a:
            yt, yp, P, c, pos = a["y_true"], a["y_pred"], a["probability"], a["classes"], a["test_row_positions"]
        assert list(c) == [0, 1, 2, 3] and np.isfinite(P).all() and np.array_equal(yp, c[P.argmax(1)])
        assert array_hash(yp) == r.prediction_sha256 and array_hash(P) == r.probability_sha256
        diffs["weighted_f1"] = max(diffs["weighted_f1"], abs(f1_score(yt, yp, average="weighted") - r.weighted_f1))
        pc = f1_score(yt, yp, labels=c, average=None)
        diffs["per_class_f1"] = max(diffs["per_class_f1"], max(abs(pc[i] - r[n + "_f1"]) for i, n in enumerate(CLASSES)))
        auc = roc_auc_score(yt, P, labels=c, multi_class="ovr", average="weighted")
        diffs["roc_auc"] = max(diffs["roc_auc"], abs(auc - r.roc_auc_ovr_weighted))
        positions.setdefault(int(r.seed), set()).add(array_hash(pos))
    test_hash = {s["seed"]: s["test_row_positions_sha256"] for s in selections}
    report["replay"] = {"records": int(len(metrics)), "max_abs_differences": diffs,
        "test_positions_consistent": all(len(v) == 1 and next(iter(v)) == test_hash[k] for k, v in positions.items())}

    fits = {f["fit_id"]: f for f in json.loads((R / "fit_records.json").read_text())}
    sharing = True
    for (seed, model), g in metrics.groupby(["seed", "model"]):
        base = g[g.condition.isin(["C1", "C2"])].fit_id.unique()
        c3 = g[g.condition == "C3"].fit_id.unique()
        sharing &= len(base) == 1 and len(c3) == 1 and fits[base[0]]["feature_count"] == 70
    report["fit_sharing"] = {"fits": len(fits), "one_baseline_and_one_C3_fit_per_split_model": bool(sharing)}

    locked = json.loads((ROOT / "reference/main_results/revised_pipeline_raw.json").read_text())
    exact, per_class, auc_max = 0, 0.0, 0.0
    hist = metrics[metrics.arm.isin(["original", "historical"])]
    for _, r in hist.iterrows():
        res = locked[str(r.seed)]["results"][r.model]
        src = res["C1"] if r.condition == "C1" else res["per_variant"]["hybrid768"][r.condition]
        exact += r.weighted_f1 == src["weighted_f1"]
        per_class = max(per_class, max(abs(r[n + "_f1"] - src["per_class_f1"][n]) for n in CLASSES))
        auc_max = max(auc_max, abs(r.roc_auc_ovr_weighted - src["roc_auc_ovr_weighted"]))
    report["historical_arm_vs_locked_main"] = {"weighted_f1_values": int(len(hist)), "exactly_equal": int(exact),
        "max_per_class_f1_difference": per_class, "max_roc_auc_difference": auc_max}

    c3 = metrics[metrics.condition == "C3"]
    ident = {"predictions": {}, "probabilities": {}}
    for a in ARMS:
        x = c3[c3.arm == a].set_index(["seed", "model"])
        h = c3[c3.arm == "historical"].set_index(["seed", "model"])
        ident["predictions"][a] = int((x.prediction_sha256 == h.prediction_sha256).sum())
        ident["probabilities"][a] = int((x.probability_sha256 == h.probability_sha256).sum())
    report["C3_identity_with_historical_arm_out_of_30"] = ident

    c1 = metrics[metrics.condition == "C1"].set_index(["seed", "model"])
    rows = []
    for (model, arm), g in metrics[metrics.condition == "C2"].groupby(["model", "arm"]):
        g = g.set_index(["seed", "model"])
        hc2 = metrics[(metrics.condition == "C2") & (metrics.arm == "historical") & (metrics.model == model)].set_index(["seed", "model"])
        d1 = g.weighted_f1 - c1.loc[g.index].weighted_f1
        dh = g.weighted_f1 - hc2.weighted_f1
        row = {"model": model, "arm": arm, "mean_C1": c1.loc[g.index].weighted_f1.mean(), "mean_C2": g.weighted_f1.mean(),
               "mean_C2_minus_C1": d1.mean(), "min_C2_minus_C1": d1.min(), "max_C2_minus_C1": d1.max(),
               "splits_C2_below_C1": int((d1 < 0).sum()),
               "mean_C2_minus_hist_C2": dh.mean(), "min_C2_minus_hist_C2": dh.min(), "max_C2_minus_hist_C2": dh.max(),
               "splits_below_hist_C2": int((dh < 0).sum()), "splits_above_hist_C2": int((dh > 0).sum()),
               "mean_C2_roc_auc": g.roc_auc_ovr_weighted.mean()}
        for n in CLASSES:
            row["mean_C2_" + n + "_f1"] = g[n + "_f1"].mean()
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["arm"] = pd.Categorical(summary["arm"], ARMS)
    summary = summary.sort_values(["model", "arm"])
    summary.to_csv(out / "c2_cross_arm_summary.csv", index=False)
    c3s = c3.groupby(["model", "arm"]).weighted_f1.mean().unstack()[ARMS]
    c3s.to_csv(out / "c3_cross_arm_means.csv")

    report["passed"] = (not bad and report["status"]["status"] == "sensitivity_completed"
        and report["historical_selection_checks"]["all_passed"] and report["replay"]["test_positions_consistent"]
        and diffs["weighted_f1"] <= 1e-12 and diffs["per_class_f1"] <= 1e-12 and diffs["roc_auc"] <= 5e-5
        and sharing and exact == len(hist))
    report["scope"] = ("Saved-array replay and locked-record comparison. Original CSVs were not reloaded locally. "
                       "Descriptive only; the overlapping-split uncertainty review remains open.")
    (out / "verification.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps(report, indent=2, default=float))
    print(summary.round(6).to_string())
    print(c3s.round(6).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
