"""Group-aware C1/C2/C3 evaluation of the main hybrid768 experiment.

Exact-duplicate feature vectors are kept together, so no vector appears in both
training and test data. Everything else follows the locked main pipeline.
Fixed plan: environment/grouped_evaluation_plan.json. Prepared with assistant
assistance; this is a separate new experiment, not a replacement of the locked
results.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback
import warnings

import numpy as np
import pandas as pd

from common import ROOT, holm, paired_summary, sha256, tost, verify_reference_code, write_csv, write_json
from seed0_reproduction import array_hash, check_environment, now, text_hash
from run_operator_sensitivity import load_verified_data

RUNNER_REVISION = "2026-09-23-grouped-v1"
CLASS_NAMES = ["Benign", "DoS", "Bot", "Infiltration"]
LOCKED_MEANS = {  # locked random-split main results, for descriptive comparison only
    "rf": {"C1": 0.849084, "C2": 0.734407, "C3": 0.850483},
    "xgb": {"C1": 0.860278, "C2": 0.769109, "C3": 0.858989},
    "mlp": {"C1": 0.843691, "C2": 0.592295, "C3": 0.841515},
}


def exact_groups(X):
    """One integer group id per byte-identical 70-feature row."""
    rows = np.ascontiguousarray(X.to_numpy())
    return pd.factorize(pd.Series([r.tobytes() for r in rows]))[0]


def grouped_split(group_ids, y, seed, test_fraction=0.30):
    """Group-aware 70:30 split: exact-duplicate groups never cross train/test.

    Groups are assigned to a stratum: their single class, or 'conflict' when a
    group holds more than one label. Within each stratum, groups are visited in
    a seeded random order and added to the test set while the stratum's test
    row count stays at or below round(test_fraction x stratum rows). Groups that
    would overshoot are skipped. Returns shuffled train and test positions.
    """
    rng = np.random.RandomState(seed)
    frame = pd.DataFrame({"g": group_ids, "y": y})
    labels = frame.groupby("g")["y"].agg(lambda s: tuple(sorted(set(s))))
    sizes = frame.groupby("g").size()
    stratum = labels.map(lambda t: str(t[0]) if len(t) == 1 else "conflict")
    test_groups = []
    for name in sorted(stratum.unique()):
        members = stratum.index[stratum == name].to_numpy()
        members = members[rng.permutation(len(members))]
        target = int(round(test_fraction * sizes[members].sum()))
        used = 0
        for g in members:
            if used + sizes[g] <= target:
                test_groups.append(g)
                used += sizes[g]
            if used == target:
                break
    in_test = np.isin(group_ids, np.asarray(test_groups))
    test = np.flatnonzero(in_test)
    train = np.flatnonzero(~in_test)
    return train[rng.permutation(len(train))], test[rng.permutation(len(test))]


def check_split(group_ids, y, train, test, seed):
    """Plan gates. Returns a record; raises if any gate fails."""
    again = grouped_split(group_ids, y, seed)
    shares = {CLASS_NAMES[c]: float((y[test] == c).sum() / (y == c).sum()) for c in range(4)}
    record = {
        "seed": seed, "train_rows": int(len(train)), "test_rows": int(len(test)),
        "disjoint_and_complete": bool(len(np.intersect1d(train, test)) == 0 and len(train) + len(test) == len(y)),
        "groups_crossing": int(np.intersect1d(group_ids[train], group_ids[test]).size),
        "class_test_shares": shares,
        "deterministic": bool(np.array_equal(again[0], train) and np.array_equal(again[1], test)),
        "train_positions_sha256": array_hash(train), "test_positions_sha256": array_hash(test),
    }
    largest = np.bincount(group_ids).argmax()
    record["largest_group_in_test"] = bool(np.isin(largest, group_ids[test]))
    record["passed"] = (record["disjoint_and_complete"] and record["groups_crossing"] == 0
                        and record["deterministic"] and all(.29 <= v <= .31 for v in shares.values()))
    if not record["passed"]:
        raise ValueError(f"Grouped split gate failed for seed {seed}: {record}")
    return record


def experiment(X, y, features, shifted_X, pipeline, plan, output, status):
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_info

    write_json(output / "threadpools.json", threadpool_info())
    predictions = output / "predictions"
    predictions.mkdir()
    groups = exact_groups(X)
    write_json(output / "group_summary.json", {
        "groups": int(groups.max() + 1), "rows": int(len(groups)),
        "rows_in_duplicate_groups": int((np.bincount(groups)[groups] > 1).sum())})
    candidates = [f for f in features if f not in pipeline.UNMODELLABLE]
    split_records, selections, fits, rows = [], [], [], []

    def save():
        write_json(output / "split_checks.json", split_records)
        write_json(output / "feature_selection.json", selections)
        write_json(output / "fit_records.json", fits)
        if rows:
            write_csv(output / "metrics.csv", rows)
        write_json(output / "grouped_status.json", status)

    def fit(kind, cols, train, ytrain, seed, role):
        print(f"Fit {len(fits) + 1}/{plan['fit_count']} | seed={seed} | model={kind} | {role}", flush=True)
        scaler = StandardScaler().fit(train[cols]) if kind == "mlp" else None
        values = scaler.transform(train[cols]) if scaler is not None else train[cols].values
        model = pipeline.full.make_model(kind)
        params = repr(model.get_params())
        model.fit(values, ytrain)
        if kind == "rf":
            model.set_params(n_jobs=1)
        fits.append({"fit_id": len(fits) + 1, "seed": seed, "model": kind, "role": role,
                     "feature_count": len(cols), "feature_order_sha256": text_hash(cols),
                     "constructor_parameters": params})
        status["fits_completed"] = len(fits)
        return model, scaler, cols, len(fits)

    def evaluate(fitted, frame, ytest, positions, seed, kind, condition):
        model, scaler, cols, fit_id = fitted
        values = scaler.transform(frame[cols]) if scaler is not None else frame[cols].values
        prob = model.predict_proba(values)
        classes = model.classes_
        if not np.isfinite(prob).all() or list(classes) != [0, 1, 2, 3]:
            raise ValueError("Invalid probabilities or class order.")
        pred = classes[prob.argmax(axis=1)]
        per = f1_score(ytest, pred, labels=classes, average=None)
        row = {"seed": seed, "model": kind, "condition": condition, "fit_id": fit_id,
               "feature_count": len(cols),
               "weighted_f1": float(f1_score(ytest, pred, average="weighted")),
               "roc_auc_ovr_weighted": float(roc_auc_score(ytest, prob, labels=classes, multi_class="ovr", average="weighted")),
               "prediction_sha256": array_hash(pred), "probability_sha256": array_hash(prob)}
        for c, name in enumerate(CLASS_NAMES):
            row[name + "_f1"] = float(per[c])
        name = f"seed{seed}_{kind}_{condition}.npz"
        np.savez_compressed(predictions / name, y_true=ytest, y_pred=pred, probability=prob,
                            classes=classes, test_row_positions=positions)
        row["prediction_file"] = "predictions/" + name
        rows.append(row)

    for seed in plan["split"]["split_seeds"]:
        status["current_seed"] = seed
        status["phase"] = "grouped_split_and_selection"
        save()
        tr, te = grouped_split(groups, y, seed)
        split_records.append(check_split(groups, y, tr, te, seed))
        train, test, ytrain, ytest = X.iloc[tr], X.iloc[te], y[tr], y[te]
        stable, shifted, D = pipeline.ks_holm_stable(train[candidates], shifted_X.iloc[tr][candidates], candidates, .05)
        mi = pipeline.mi_rank(train, stable, ytrain, seed)
        selected = [f for f in stable if mi.get(f, 0) >= .05] or stable[:1]
        selections.append({"seed": seed, "stable": stable, "shifted": shifted, "KS_D": D, "MI": mi,
                           "selected": selected, "train_positions_sha256": array_hash(tr),
                           "test_positions_sha256": array_hash(te)})
        status["phase"] = "classifier_fits"
        save()
        shifted_test = shifted_X.iloc[te]
        for kind in ("rf", "xgb", "mlp"):
            base = fit(kind, features, train, ytrain, seed, "C1_and_C2")
            evaluate(base, test, ytest, te, seed, kind, "C1")
            evaluate(base, shifted_test, ytest, te, seed, kind, "C2")
            c3 = fit(kind, selected, train, ytrain, seed, "C3")
            evaluate(c3, shifted_test, ytest, te, seed, kind, "C3")
            save()
        status["completed_splits"].append(seed)
        save()
    return pd.DataFrame(rows)


def statistics(metrics, output):
    """Descriptive summary plus the unchanged registered weighted-F1 procedure."""
    summary, tests = [], []
    for kind in ("rf", "xgb", "mlp"):
        m = metrics[metrics.model == kind].pivot(index="seed", columns="condition", values="weighted_f1")
        row = {"model": kind, "splits": len(m)}
        for c in ("C1", "C2", "C3"):
            row["mean_" + c] = float(m[c].mean())
            row["locked_random_split_mean_" + c] = LOCKED_MEANS[kind][c]
            row["grouped_minus_random_" + c] = float(m[c].mean() - LOCKED_MEANS[kind][c])
        for name, d in (("C2_minus_C1", m.C2 - m.C1), ("C3_minus_C1", m.C3 - m.C1)):
            row["mean_" + name], row["min_" + name], row["max_" + name] = float(d.mean()), float(d.min()), float(d.max())
            s = paired_summary(d.to_numpy())
            entry = {"model": kind, "contrast": name, **s}
            if name == "C3_minus_C1":
                for margin in (.01, .005):
                    t = tost(s, margin)
                    entry[f"tost_raw_p_{margin}"] = t["raw_p"]
            tests.append(entry)
        summary.append(row)
    tests = pd.DataFrame(tests)
    for name in ("C2_minus_C1", "C3_minus_C1"):
        k = tests.contrast == name
        tests.loc[k, "difference_holm_p"] = holm(tests.loc[k, "selected_p"].to_numpy())
    k = tests.contrast == "C3_minus_C1"
    for margin in (.01, .005):
        tests.loc[k, f"tost_holm_p_{margin}"] = holm(tests.loc[k, f"tost_raw_p_{margin}"].to_numpy())
        tests.loc[k, f"equivalent_{margin}"] = tests.loc[k, f"tost_holm_p_{margin}"] < .05
    decision = []
    for kind in ("rf", "xgb", "mlp"):
        deg = tests[(tests.model == kind) & (tests.contrast == "C2_minus_C1")].iloc[0]
        rec = tests[(tests.model == kind) & (tests.contrast == "C3_minus_C1")].iloc[0]
        decision.append({"model": kind,
                         "degradation_supported": bool(deg.mean_difference < 0 and deg.difference_holm_p < .05),
                         "recovery_equivalent_0.01": bool(rec["equivalent_0.01"]),
                         "recovery_equivalent_0.005": bool(rec["equivalent_0.005"]),
                         "main_claim_supported_under_grouped_split": bool(deg.mean_difference < 0 and deg.difference_holm_p < .05 and rec["equivalent_0.01"])})
    write_csv(output / "descriptive_summary.csv", summary)
    tests.to_csv(output / "registered_procedure_tests.csv", index=False)
    write_csv(output / "decision_rule.csv", decision)
    per_class = metrics.groupby(["model", "condition"])[[n + "_f1" for n in CLASS_NAMES]].mean().reset_index()
    per_class.to_csv(output / "per_class_f1_means.csv", index=False)
    return summary, decision


def run(args):
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan_path = ROOT / "environment/grouped_evaluation_plan.json"
    plan = json.loads(plan_path.read_text())
    status = {"status": "checking", "started_utc": now(), "runner_revision": RUNNER_REVISION,
              "plan_sha256": sha256(plan_path), "fits_completed": 0, "completed_splits": [],
              "historical_results_modified": False}
    old_filters, old_warning = list(warnings.filters), warnings.showwarning
    with (output / "warnings.log").open("w") as warning_file:
        def record_warning(message, category, filename, lineno, file=None, line=None):
            warning_file.write(warnings.formatwarning(message, category, filename, lineno, line))
            warning_file.flush()
            old_warning(message, category, filename, lineno, file=file, line=line)
        try:
            write_json(output / "grouped_status.json", status)
            env = check_environment(json.loads((ROOT / "environment/main_known_versions.json").read_text()))
            write_json(output / "runtime_environment.json", env)
            if env["problems"]:
                raise ValueError("Environment mismatch: " + "; ".join(env["problems"]))
            status["reference_sources_verified"] = verify_reference_code()
            sys.path.insert(0, str(ROOT / "reference/locked_v5"))
            import revised_pipeline_main_only as pipeline
            warnings.filters[:] = old_filters
            warnings.simplefilter("default")
            warnings.showwarning = record_warning
            if not pipeline.HAVE_SM or not pipeline.full.HAVE_XGB:
                raise ValueError("Required model/statistics dependency unavailable.")
            X, y, features, class_map = load_verified_data(args, pipeline, output)
            if class_map != {n: i for i, n in enumerate(CLASS_NAMES)}:
                raise ValueError("Class encoding differs from the verified record.")
            original_hash = array_hash(X.values)
            shifted_X = pipeline.ob.make_approx_mlkem(X, pipeline.VARIANTS["hybrid768"])
            audit = json.loads((ROOT / "reference/feature_consistency_20260921/results/operator_integrity.json").read_text())
            expected = next(r["shifted_matrix_sha256"] for r in audit if r["variant"] == "hybrid768")
            if array_hash(shifted_X.values) != expected:
                raise ValueError("Historical operator output differs from the verified record.")
            metrics = experiment(X, y, features, shifted_X, pipeline, plan, output, status)
            if array_hash(X.values) != original_hash:
                raise ValueError("Original matrix changed during analysis.")
            if len(metrics) != 90 or status["fits_completed"] != plan["fit_count"]:
                raise ValueError("Unexpected number of metric rows or fits.")
            status["metric_rows"] = len(metrics)
            status["status"] = "grouped_completed"
            status["phase"] = "statistics"
            write_json(output / "grouped_status.json", status)
            try:   # fits are complete; a statistics problem must not hide them
                summary, decision = statistics(metrics, output)
                status["statistics_status"] = "completed"
                print(pd.DataFrame(summary)[["model", "mean_C1", "mean_C2", "mean_C3", "mean_C2_minus_C1", "mean_C3_minus_C1"]].round(4).to_string(index=False))
                print(pd.DataFrame(decision).to_string(index=False))
            except Exception as error:
                status["statistics_status"] = "failed: " + str(error)
                (output / "statistics_traceback.txt").write_text(traceback.format_exc())
            status["phase"] = "completed"
        except Exception as error:
            status["status"] = "grouped_failed"
            status["error"] = str(error)
            (output / "traceback.txt").write_text(traceback.format_exc())
            traceback.print_exc()
        finally:
            warnings.filters[:] = old_filters
            warnings.showwarning = old_warning
            status["finished_utc"] = now()
            write_json(output / "grouped_status.json", status)
    return 0 if status["status"] == "grouped_completed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    sys.exit(run(parser.parse_args()))
