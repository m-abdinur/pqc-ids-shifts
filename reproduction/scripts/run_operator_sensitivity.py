"""Test two declared operator changes separately and jointly, on main hybrid768.

Prospective diagnostic only. The fixed plan preserves the source experiment
and does not certify the alternatives as realistic network traffic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
import warnings

import numpy as np

from common import ROOT, sha256, verify_reference_code, write_csv, write_json
from seed0_reproduction import array_hash, check_environment, now, text_hash
from run_experiment import SOURCES
from audit_feature_consistency import rules, transitions
from operator_sensitivity import ARMS, transform


def load_verified_data(args, pipeline, output):
    seed0 = ROOT / "reference/seed0_reproduction_20260920/results"
    expected_files = {r["filename"]: r for r in json.loads((seed0 / "input_files.json").read_text())}
    cache = args.cache_dir.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=False)
    paths, inputs = {}, []
    for name, attack in SOURCES:
        source, local = args.data_dir / name, cache / name
        before = source.stat()
        digest = hashlib.sha256()
        print("Reading:", name, flush=True)
        with source.open("rb") as incoming, local.open("xb") as outgoing:
            for block in iter(lambda: incoming.read(1024 * 1024), b""):
                digest.update(block)
                outgoing.write(block)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Input changed during reading: " + name)
        if digest.hexdigest() != expected_files[name]["sha256"] or local.stat().st_size != expected_files[name]["bytes"]:
            raise ValueError("Input differs from the verified seed-0 data: " + name)
        inputs.append({"filename": name, "sha256": digest.hexdigest(), "bytes": local.stat().st_size})
        paths[str(local)] = attack
    write_json(output / "input_files.json", inputs)
    base = pipeline.base
    data = base.load_and_clean(paths, base.LABEL_COL, base.LABEL_NORMALISE, base.CLASSES, 30000, 42)
    X, y, features, class_map = base.prepare_xy(data, base.NON_FEATURE_COLS, base.CLASSES)
    expected = json.loads((seed0 / "seed0_status.json").read_text())["prepared_data"]
    if (array_hash(X.values) != expected["X_sha256"] or array_hash(y) != expected["y_sha256"]
            or features != expected["features"] or list(X.shape) != expected["X_shape"]):
        raise ValueError("Prepared matrix differs from the verified seed-0 input.")
    write_json(output / "prepared_data.json", {"X_sha256": array_hash(X.values), "y_sha256": array_hash(y),
        "features": features, "shape": list(X.shape), "class_encoding": class_map})
    return X, y, features, class_map


def structural_check(X, y, class_map, matrices, output):
    baseline = rules(X, 1e-5, 1e-6)
    rows, checks = [], []
    all_rows = np.ones(len(X), dtype=bool)
    eligible = ((X["Tot Fwd Pkts"] > 0) & (X["Tot Bwd Pkts"] > 0)).to_numpy()
    for arm, matrix in matrices.items():
        after = rules(matrix, 1e-5, 1e-6)
        for name in baseline:
            for label, mask in [("all", all_rows)] + [(c, y == code) for c, code in class_map.items()]:
                rows.append({"arm": arm, "rule": name, "class": label,
                             **transitions(baseline[name], after[name], mask)})
        check = {"arm": arm, "X_sha256": array_hash(matrix.values),
                 "changed_rows_vs_historical": int(np.any(matrix.to_numpy() != matrices["historical"].to_numpy(), axis=1).sum()),
                 "ineligible_unchanged": bool(np.array_equal(X.to_numpy()[~eligible], matrix.to_numpy()[~eligible])),
                 "finite": bool(np.isfinite(matrix.to_numpy()).all())}
        checks.append(check)
        if not check["finite"] or not check["ineligible_unchanged"]:
            raise ValueError("Matrix integrity failed: " + arm)
    write_csv(output / "structural_counts.csv", rows)
    write_json(output / "matrix_checks.json", checks)
    # The combined change must add no violation on the nine bounds and four
    # rate-sign checks. The two mean-equality diagnostics remain descriptive.
    failures = [r for r in rows if r["arm"] == "combined" and r["class"] == "all"
                and "_equals_" not in r["rule"]
                and (r["introduced_violations"] or r["after_violations_previously_unevaluable"])]
    write_json(output / "structural_gate.json", {"passed": not failures, "failed_rules": failures,
        "meaning": "Targeted algebraic gate only; not a full physical-validity certificate"})
    if failures:
        raise ValueError("Combined alternative failed its predefined algebraic gate; no classifiers fitted.")


RUNNER_REVISION = "2026-09-23-archive-order-fix"


def check_historical_selection(seed, candidates, stable, shifted, D, mi, selected, old):
    """Compare the runtime historical-arm selection with the locked main record.

    The locked writer (run_revised in revised_pipeline_main_only.py) saves
    stable_prep in candidate order but saves shifted_prep as
    sorted(shifted). ks_holm_stable returns both lists in candidate order.
    So the runtime shifted list is checked two ways: it must be in candidate
    order, and its sorted form must equal the archived list exactly.
    Tolerances are unchanged: KS D 1e-12, MI 1e-8. Returns a record of every
    check; the caller stops the run if any check fails.
    """
    def exact_list(values):
        return len(values) == len(set(values))

    checks = {}
    checks["stable_exact_order"] = stable == old["stable_prep"]
    checks["stable_in_candidate_order"] = stable == [c for c in candidates if c in set(stable)]
    checks["shifted_in_candidate_order"] = shifted == [c for c in candidates if c in set(shifted)]
    checks["shifted_sorted_matches_archive"] = sorted(shifted) == old["shifted_prep"]
    checks["no_duplicates"] = all(exact_list(v) for v in (stable, shifted, old["stable_prep"], old["shifted_prep"]))
    checks["partition_covers_candidates"] = (not set(stable) & set(shifted)
                                             and set(stable) | set(shifted) == set(candidates)
                                             and len(stable) + len(shifted) == len(candidates))
    checks["ks_keys_match"] = list(D) == candidates and set(old["ks_D_prep"]) == set(candidates)
    checks["mi_keys_match"] = list(mi) == stable and set(old["mi_prep"]) == set(old["stable_prep"])
    finite = all(np.isfinite(v) for v in list(D.values()) + list(mi.values())
                 + list(old["ks_D_prep"].values()) + list(old["mi_prep"].values()))
    checks["statistics_finite"] = bool(finite)
    ks_diff = max((abs(D[f] - old["ks_D_prep"][f]) for f in D if f in old["ks_D_prep"]), default=float("inf"))
    mi_diff = max((abs(mi[f] - old["mi_prep"][f]) for f in mi if f in old["mi_prep"]), default=float("inf"))
    checks["ks_within_1e-12"] = bool(checks["ks_keys_match"] and finite and ks_diff <= 1e-12)
    checks["mi_within_1e-8"] = bool(checks["mi_keys_match"] and finite and mi_diff <= 1e-8)
    expected = [f for f in old["stable_prep"] if old["mi_prep"].get(f, 0) >= .05] or old["stable_prep"][:1]
    checks["selected_exact_order"] = selected == expected
    return {"seed": seed, "passed": all(checks.values()), "checks": checks,
            "max_abs_ks_D_difference": ks_diff if np.isfinite(ks_diff) else None,
            "max_abs_mi_difference": mi_diff if np.isfinite(mi_diff) else None,
            "runtime_stable": stable, "runtime_shifted_candidate_order": shifted,
            "archived_stable_prep": old["stable_prep"], "archived_shifted_prep_sorted": old["shifted_prep"],
            "runtime_selected": selected, "archived_expected_selected": expected}


def experiment(X, y, features, class_map, matrices, pipeline, plan, output, status, reference):
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_info

    prediction_dir = output / "predictions"
    prediction_dir.mkdir()
    write_json(output / "threadpools.json", threadpool_info())
    candidates = [f for f in features if f not in pipeline.UNMODELLABLE]
    rows, fits, selections, pairs, selection_checks = [], [], [], [], []
    status["fits_completed"] = 0
    status["planned_fits_for_prepared_splits"] = 0
    status["completed_splits"] = []

    def save_status():
        write_json(output / "sensitivity_status.json", status)

    def fit_model(kind, cols, train, ytrain, seed, role):
        print(f"Fit {len(fits)+1}/{status['maximum_fits']} maximum | seed={seed} | model={kind} | {role}", flush=True)
        scaler = StandardScaler().fit(train[cols]) if kind == "mlp" else None
        values = scaler.transform(train[cols]) if scaler is not None else train[cols].values
        estimator = pipeline.full.make_model(kind)
        parameters = repr(estimator.get_params())
        estimator.fit(values, ytrain)
        # Same setting across every diagnostic arm, chosen in advance in view
        # of the independently verified RF aggregation-order issue.
        if kind == "rf":
            estimator.set_params(n_jobs=1)
        fits.append({"fit_id": len(fits)+1, "seed": seed, "model": kind, "role": role,
                     "feature_count": len(cols), "feature_order_sha256": text_hash(cols),
                     "constructor_parameters": parameters,
                     "prediction_n_jobs": 1 if kind == "rf" else "constructor setting"})
        status["fits_completed"] = len(fits)
        write_json(output / "fit_records.json", fits)
        save_status()
        return estimator, scaler, cols, len(fits)

    def evaluate(fitted, frame, ytest, positions, seed, kind, arm, condition):
        estimator, scaler, cols, fit_id = fitted
        values = scaler.transform(frame[cols]) if scaler is not None else frame[cols].values
        probability = estimator.predict_proba(values)
        classes = estimator.classes_
        if not np.isfinite(probability).all() or set(classes) != set(class_map.values()):
            raise ValueError("Invalid probability output or missing class.")
        predicted = classes[np.argmax(probability, axis=1)]
        row = {"seed": seed, "model": kind, "arm": arm, "condition": condition, "fit_id": fit_id,
               "selected_count": len(cols), "weighted_f1": float(f1_score(ytest, predicted, average="weighted")),
               "roc_auc_ovr_weighted": float(roc_auc_score(ytest, probability, labels=classes, multi_class="ovr", average="weighted")),
               "prediction_sha256": array_hash(predicted), "probability_sha256": array_hash(probability)}
        per_class = f1_score(ytest, predicted, labels=classes, average=None)
        for name, code in class_map.items():
            row[name + "_f1"] = float(per_class[list(classes).index(code)])
        name = f"seed{seed}_{kind}_{arm}_{condition}.npz"
        np.savez_compressed(prediction_dir / name, y_true=ytest, y_pred=predicted,
                           probability=probability, classes=classes, test_row_positions=positions)
        row["prediction_file"] = "predictions/" + name
        rows.append(row)
        return row

    for seed in plan["split_seeds"]:
        status["current_seed"] = seed
        status["phase"] = "train_only_feature_selection"
        save_status()
        tr, te = train_test_split(np.arange(len(X)), test_size=.3, random_state=seed, stratify=y)
        train, test, ytrain, ytest = X.iloc[tr], X.iloc[te], y[tr], y[te]
        arm_columns, mi_cache = {}, {}
        for arm in ARMS:
            stable, shifted, D = pipeline.ks_holm_stable(train[candidates], matrices[arm].iloc[tr][candidates], candidates, .05)
            key = tuple(stable)
            if key not in mi_cache:
                mi_cache[key] = pipeline.mi_rank(train, stable, ytrain, seed)
            mi = mi_cache[key]
            selected = [f for f in stable if mi.get(f, 0) >= .05] or stable[:1]
            if not selected:
                raise ValueError("No selectable stable feature in arm: " + arm)
            arm_columns[arm] = selected
            selections.append({"seed": seed, "arm": arm, "candidate_features": candidates,
                               "stable": stable, "shifted": shifted, "KS_D": D,
                               "MI": mi, "selected": selected,
                               "test_row_positions_sha256": array_hash(te), "train_row_positions_sha256": array_hash(tr)})
        # Save every arm's lists and statistics before the historical check,
        # so a failed check still leaves the evidence needed to diagnose it.
        write_json(output / "feature_selection.json", selections)
        if reference is not None:
            hist = next(s for s in selections if s["seed"] == seed and s["arm"] == "historical")
            result = check_historical_selection(seed, candidates, hist["stable"], hist["shifted"],
                                                hist["KS_D"], hist["MI"], hist["selected"], reference[str(seed)])
            selection_checks.append(result)
            write_json(output / "historical_selection_checks.json", selection_checks)
            if not result["passed"]:
                failed = [name for name, ok in result["checks"].items() if not ok]
                raise ValueError("Historical feature selection differs from the locked main record: "
                                 + ", ".join(failed))
        groups = list(dict.fromkeys(tuple(arm_columns[a]) for a in ARMS))
        status["planned_fits_for_prepared_splits"] += 3 * (1 + len(groups))
        status["phase"] = "classifier_comparison"
        save_status()
        for kind in ("rf", "xgb", "mlp"):
            baseline = fit_model(kind, features, train, ytrain, seed, "C1_and_all_C2_arms")
            c1 = evaluate(baseline, test, ytest, te, seed, kind, "original", "C1")
            c3_fits = {}
            for selected in groups:
                c3_fits[selected] = fit_model(kind, list(selected), train, ytrain, seed, "C3_for_identical_ordered_subset")
            scores = {}
            for arm in ARMS:
                frame = matrices[arm].iloc[te]
                c2 = evaluate(baseline, frame, ytest, te, seed, kind, arm, "C2")
                c3 = evaluate(c3_fits[tuple(arm_columns[arm])], frame, ytest, te, seed, kind, arm, "C3")
                scores[arm] = (c2, c3)
            for arm in ARMS:
                c2, c3 = scores[arm]
                pairs.append({"seed": seed, "model": kind, "arm": arm, "C1": c1["weighted_f1"],
                    "C2": c2["weighted_f1"], "C3": c3["weighted_f1"],
                    "C2_minus_C1": c2["weighted_f1"]-c1["weighted_f1"],
                    "C3_minus_C1": c3["weighted_f1"]-c1["weighted_f1"],
                    "C2_minus_historical_C2": c2["weighted_f1"]-scores["historical"][0]["weighted_f1"],
                    "C3_minus_historical_C3": c3["weighted_f1"]-scores["historical"][1]["weighted_f1"]})
            write_csv(output / "metrics.csv", rows)
            write_csv(output / "paired_scores.csv", pairs)
        status["completed_splits"].append(seed)
        save_status()
    summaries = []
    for kind in ("rf", "xgb", "mlp"):
        for arm in ARMS:
            values = [r for r in pairs if r["model"] == kind and r["arm"] == arm]
            row = {"model": kind, "arm": arm, "splits": len(values)}
            for field in ("C1", "C2", "C3", "C2_minus_C1", "C3_minus_C1", "C2_minus_historical_C2", "C3_minus_historical_C3"):
                v = np.asarray([r[field] for r in values])
                row["mean_" + field] = float(v.mean())
                row["min_" + field] = float(v.min())
                row["max_" + field] = float(v.max())
            summaries.append(row)
    write_csv(output / "descriptive_summary.csv", summaries)
    assert len(rows) == len(plan["split_seeds"]) * 3 * 9
    assert len(fits) == status["planned_fits_for_prepared_splits"] <= status["maximum_fits"]
    status["metric_rows"] = len(rows)
    status["saved_prediction_files"] = len(rows)
    status["phase"] = "completed"


def run(args):
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "checking", "started_utc": now(), "fits_completed": 0,
              "runner_revision": RUNNER_REVISION,
              "historical_results_modified": False, "scope": "Separate main-hybrid768 sensitivity experiment"}
    plan_path = ROOT / "environment/operator_sensitivity_plan.json"
    plan = json.loads(plan_path.read_text())
    status["plan_sha256"] = sha256(plan_path)
    status["maximum_fits"] = plan["maximum_fits"]
    old_filters, old_warning = list(warnings.filters), warnings.showwarning
    with (output / "warnings.log").open("w") as warning_file:
        def record_warning(message, category, filename, lineno, file=None, line=None):
            warning_file.write(warnings.formatwarning(message, category, filename, lineno, line))
            warning_file.flush()
            old_warning(message, category, filename, lineno, file=file, line=line)
        try:
            write_json(output / "sensitivity_status.json", status)
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
            cfg = pipeline.VARIANTS["hybrid768"]
            if cfg != plan["footprint"] or pipeline.MI_HEADLINE != .05 or pipeline.MODEL_SEED != 42:
                raise ValueError("Plan does not match the archived core settings.")
            X, y, features, class_map = load_verified_data(args, pipeline, output)
            original_hash = array_hash(X.values)
            matrices = {arm: transform(X, cfg, pipeline.ob.make_approx_mlkem, arm) for arm in ARMS}
            audit = json.loads((ROOT / "reference/feature_consistency_20260921/results/operator_integrity.json").read_text())
            expected = next(r["shifted_matrix_sha256"] for r in audit if r["variant"] == "hybrid768")
            if array_hash(matrices["historical"].values) != expected or array_hash(X.values) != original_hash:
                raise ValueError("Historical operator output or original matrix identity differs.")
            structural_check(X, y, class_map, matrices, output)
            status["structural_gate_passed"] = True
            reference = json.loads((ROOT / "reference/main_results/revised_pipeline_raw.json").read_text())
            experiment(X, y, features, class_map, matrices, pipeline, plan, output, status, reference)
            if array_hash(X.values) != original_hash:
                raise ValueError("Original matrix changed during analysis.")
            status["status"] = "sensitivity_completed"
            print("Sensitivity completed; fits:", status["fits_completed"], flush=True)
        except Exception as error:
            status["status"] = "sensitivity_failed"
            status["error"] = str(error)
            (output / "traceback.txt").write_text(traceback.format_exc())
            traceback.print_exc()
        finally:
            warnings.filters[:] = old_filters
            warnings.showwarning = old_warning
            status["finished_utc"] = now()
            write_json(output / "sensitivity_status.json", status)
    return 0 if status["status"] == "sensitivity_completed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    sys.exit(run(parser.parse_args()))
