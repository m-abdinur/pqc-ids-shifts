"""Collect evidence for RF seed-6 variation using two fixed-seed fits.

The main experiment and its acceptance rules are unchanged. Prediction
threading and accumulation order are varied only as diagnostic controls.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import warnings

from common import ROOT, read_csv, sha256, verify_reference_code, write_csv, write_json
from run_experiment import SOURCES
from seed0_reproduction import array_hash, check_environment, now, text_hash


def forest_signature(model):
    """Hash named learned arrays, excluding uninitialised structured padding."""
    import numpy as np
    signatures = []
    for estimator in model.estimators_:
        state = estimator.tree_.__getstate__()
        digest = hashlib.sha256()
        for name in state["nodes"].dtype.names:
            values = np.ascontiguousarray(state["nodes"][name])
            if name in ("feature", "threshold", "missing_go_to_left"):
                values = values.copy()
                values[state["nodes"]["left_child"] == -1] = 0
            digest.update(name.encode())
            digest.update(values.dtype.str.encode())
            digest.update(values.tobytes())
        digest.update(np.ascontiguousarray(state["values"]).tobytes())
        digest.update(str(estimator.random_state).encode())
        digest.update(np.ascontiguousarray(estimator.classes_).tobytes())
        signatures.append(digest.hexdigest())
    return {"forest_sha256": text_hash(signatures), "tree_sha256": signatures}


def ordered_probabilities(model, matrix, reverse=False):
    import numpy as np
    probability = np.zeros((len(matrix), len(model.classes_)), dtype="float64")
    trees = reversed(model.estimators_) if reverse else model.estimators_
    for tree in trees:
        probability += tree.predict_proba(matrix)
    return probability / len(model.estimators_)


def collect_predictions(model, matrices, config, fit_index, capture):
    """Capture each call once, deriving its labels from the same probabilities."""
    try:
        model.set_params(n_jobs=config["native_prediction_jobs"])
        for repetition in range(config["native_repetitions_per_fit"]):
            capture(fit_index, "native_repeat", repetition, "hybrid768",
                    model.predict_proba(matrices["hybrid768"]), model.classes_)
        capture(fit_index, "native_p2", 0, "hybrid768_p2",
                model.predict_proba(matrices["hybrid768_p2"]), model.classes_)
        model.set_params(n_jobs=config["serial_prediction_jobs"])
        for repetition in range(config["serial_repetitions_per_fit"]):
            capture(fit_index, "serial_repeat", repetition, "hybrid768",
                    model.predict_proba(matrices["hybrid768"]), model.classes_)
        capture(fit_index, "serial_p2", 0, "hybrid768_p2",
                model.predict_proba(matrices["hybrid768_p2"]), model.classes_)
        for reverse in (False, True):
            capture(fit_index, "reverse_sum" if reverse else "forward_sum", 0, "hybrid768",
                    ordered_probabilities(model, matrices["hybrid768"], reverse), model.classes_)
    finally:
        model.set_params(n_jobs=config["native_prediction_jobs"])


def run(args):
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from threadpoolctl import threadpool_info
    from joblib import effective_n_jobs

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "environment/rf_seed6_diagnostic.json").read_text())
    status = {"status": "checking", "started_utc": now(), "config": config,
              "fits_completed": 0, "historical_results_modified": False,
              "strict_main_comparison_status": "needs_review"}
    status_path = output / "diagnostic_status.json"
    write_json(status_path, status)
    old_filters, old_warning = list(warnings.filters), warnings.showwarning
    warning_file = (output / "warnings.log").open("w")

    def record_warning(message, category, filename, lineno, file=None, line=None):
        warning_file.write(warnings.formatwarning(message, category, filename, lineno, line))
        warning_file.flush()
        old_warning(message, category, filename, lineno, file=file, line=line)

    try:
        target = json.loads((ROOT / "environment/main_known_versions.json").read_text())
        environment = check_environment(target)
        write_json(output / "runtime_environment.json", environment)
        if environment["problems"]:
            raise ValueError("Environment mismatch: " + "; ".join(environment["problems"]))
        status["reference_sources_verified"] = verify_reference_code()
        status["effective_prediction_workers"] = {
            "native": min(200, effective_n_jobs(config["native_prediction_jobs"])),
            "serial": min(200, effective_n_jobs(config["serial_prediction_jobs"]))}
        seed0 = ROOT / "reference/seed0_reproduction_20260920/results"
        expected_inputs = {r["filename"]: r for r in json.loads((seed0 / "input_files.json").read_text())}
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
            if digest.hexdigest() != expected_inputs[name]["sha256"] or local.stat().st_size != expected_inputs[name]["bytes"]:
                raise ValueError("Input does not match the verified seed-0 data: " + name)
            paths[str(local)] = attack
            inputs.append({"filename": name, "sha256": digest.hexdigest(), "bytes": local.stat().st_size})
        write_json(output / "input_files.json", inputs)
        sys.path.insert(0, str(ROOT / "reference/locked_v5"))
        import revised_pipeline_main_only as pipeline
        warnings.filters[:] = old_filters
        warnings.simplefilter("default")
        warnings.showwarning = record_warning
        base = pipeline.base
        data = base.load_and_clean(paths, base.LABEL_COL, base.LABEL_NORMALISE, base.CLASSES, 30000, 42)
        X, y, features, class_map = base.prepare_xy(data, base.NON_FEATURE_COLS, base.CLASSES)
        del data
        expected = json.loads((seed0 / "seed0_status.json").read_text())["prepared_data"]
        if (array_hash(X.values) != expected["X_sha256"] or array_hash(y) != expected["y_sha256"]
                or features != expected["features"] or list(X.shape) != expected["X_shape"]):
            raise ValueError("Prepared matrix differs from the verified seed-0 input.")
        old_path = ROOT / "reference/main_results/revised_pipeline_raw.json"
        new_path = ROOT / "reference/main_reproduction_20260920/results/revised_pipeline_raw.json"
        old_raw = json.loads(old_path.read_text())["6"]
        new_raw = json.loads(new_path.read_text())["6"]
        status["reference_input_sha256"] = {"locked_main": sha256(old_path), "returned_main": sha256(new_path)}
        selected = [f for f in old_raw["stable_prep"] if old_raw["mi_prep"].get(f, 0) >= .05]
        new_selected = [f for f in new_raw["stable_prep"] if new_raw["mi_prep"].get(f, 0) >= .05]
        if selected != new_selected:
            raise ValueError("Seed-6 selected features differ between the two main records.")
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=6, stratify=y)
        reference_rows = [r for r in read_csv(ROOT / "reference/c1sf_records/split_level_results/same_feature_c1_metrics_by_split.csv") if r["seed"] == "6"]
        signatures = {"train_index_sha256": text_hash(Xtr.index), "test_index_sha256": text_hash(Xte.index),
                      "selected_feature_order_sha256": text_hash(selected)}
        if len(reference_rows) != 3:
            raise ValueError("Expected three seed-6 split-reference rows.")
        for row in reference_rows:
            for key, value in signatures.items():
                if row[key] != value:
                    raise ValueError("Seed-6 signature mismatch: " + key)
        train_matrix = Xtr[selected].values
        matrices = {"original": Xte[selected].values}
        for name, footprint in pipeline.VARIANTS.items():
            matrices[name] = pipeline.ob.make_approx_mlkem(Xte, footprint)[selected].values
        matrix_rows = [{"condition": name, "dtype": str(matrix.dtype), "sha256": array_hash(matrix),
                        "equal_to_original": bool(np.array_equal(matrix, matrices["original"])),
                        "changed_elements_vs_original": int(np.sum(matrix != matrices["original"]))}
                       for name, matrix in matrices.items()]
        write_csv(output / "selected_matrix_comparison.csv", matrix_rows)
        status["data_and_selection"] = {"rows": len(X), "features": len(features), "selected": selected,
                                        "class_encoding": class_map, **signatures,
                                        "all_selected_test_matrices_equal": all(r["equal_to_original"] for r in matrix_rows),
                                        "training_matrix_sha256": array_hash(train_matrix)}
        write_json(output / "threadpools.json", threadpool_info())
        rows, calls, model_signatures, models = [], [], [], []
        baseline_predictions, baseline_probabilities = {}, {}

        def capture(fit_index, mode, repetition, condition, probabilities, classes):
            predictions = classes.take(np.argmax(probabilities, axis=1), axis=0)
            if not np.isfinite(probabilities).all():
                raise ValueError("Non-finite diagnostic probabilities.")
            if fit_index not in baseline_predictions:
                baseline_predictions[fit_index] = predictions.copy()
                baseline_probabilities[fit_index] = probabilities.copy()
            sorted_prob = np.sort(probabilities, axis=1)
            margins = sorted_prob[:, -1] - sorted_prob[:, -2]
            per_class = f1_score(yte, predictions, average=None, labels=classes)
            row = {"call": len(rows), "fit": fit_index, "mode": mode, "repetition": repetition,
                   "condition": condition, "weighted_f1": float(f1_score(yte, predictions, average="weighted")),
                   "roc_auc": float(roc_auc_score(yte, probabilities, average="weighted", multi_class="ovr", labels=classes)),
                   "prediction_sha256": array_hash(predictions), "probability_sha256": array_hash(probabilities),
                   "changed_predictions_vs_first_call": int(np.sum(predictions != baseline_predictions[fit_index])),
                   "max_probability_change_vs_first_call": float(np.max(np.abs(probabilities - baseline_probabilities[fit_index]))),
                   "minimum_top_two_margin": float(margins.min()),
                   "near_tie_rows": int(np.sum(margins <= config["near_tie_probability_margin"])),
                   **{f"{name}_f1": float(score) for name, score in zip(base.CLASSES, per_class)}}
            rows.append(row)
            calls.append({"predictions": predictions, "probabilities": probabilities, "near_tie": margins <= config["near_tie_probability_margin"]})
            write_csv(output / "prediction_calls.csv", rows)
            status["prediction_calls_completed"] = len(rows)
            write_json(status_path, status)
            print(f"fit={fit_index} {mode} {repetition}: F1={row['weighted_f1']:.16f}", flush=True)

        status["status"] = "collecting_diagnostic"
        for fit_index in range(config["fits"]):
            print("Fitting RF replica", fit_index + 1, "of", config["fits"], flush=True)
            model = pipeline.full.make_model("rf")
            (output / "fit_parameters.txt").write_text(repr(model.get_params()) + "\n")
            started = time.perf_counter()
            model.fit(train_matrix, ytr)
            signature = {"fit": fit_index, "fit_seconds": time.perf_counter() - started, **forest_signature(model)}
            model_signatures.append(signature)
            models.append(model)
            status["fits_completed"] += 1
            write_json(output / "forest_signatures.json", model_signatures)
            collect_predictions(model, matrices, config, fit_index, capture)

        prediction_stack = np.stack([c["predictions"] for c in calls])
        changed = np.any(prediction_stack != prediction_stack[0], axis=0)
        near = np.any(np.stack([c["near_tie"] for c in calls]), axis=0)
        relevant = np.flatnonzero(changed | near)
        probability_subset = np.stack([c["probabilities"][relevant] for c in calls])
        tree_probabilities = np.stack([np.stack([tree.predict_proba(matrices["hybrid768"][relevant]) for tree in model.estimators_])
                                       for model in models]) if len(relevant) else np.empty((config["fits"], 200, 0, 4))
        np.savez_compressed(output / "diagnostic_predictions.npz", test_index=Xte.index.to_numpy(),
                            y_true=yte, predictions=prediction_stack, relevant_positions=relevant,
                            relevant_test_index=Xte.index.to_numpy()[relevant], relevant_probabilities=probability_subset,
                            relevant_features=matrices["hybrid768"][relevant],
                            selected_features=np.asarray(selected), classes=models[0].classes_,
                            tree_probabilities_for_relevant_rows=tree_probabilities)
        # Keep one complete probability matrix for each distinct label vector.
        retained = set()
        for row, call in zip(rows, calls):
            key = row["prediction_sha256"]
            if key not in retained:
                np.savez_compressed(output / f"representative_call_{row['call']:02d}.npz",
                                    predictions=call["predictions"], probabilities=call["probabilities"], y_true=yte)
                retained.add(key)
        status["observations"] = {
            "unique_native_prediction_hashes_by_fit": {str(i): len({r["prediction_sha256"] for r in rows if r["fit"] == i and r["mode"] == "native_repeat"}) for i in range(config["fits"])},
            "unique_serial_prediction_hashes_by_fit": {str(i): len({r["prediction_sha256"] for r in rows if r["fit"] == i and r["mode"] == "serial_repeat"}) for i in range(config["fits"])},
            "learned_forests_identical": len({s["forest_sha256"] for s in model_signatures}) == 1,
            "changed_prediction_rows_across_all_calls": int(changed.sum()),
            "near_tie_rows_across_all_calls": int(near.sum()),
            "distinct_weighted_f1_values": sorted({r["weighted_f1"] for r in rows}),
            "interpretation": "Observations require review; diagnostic completion does not certify full-main equivalence or prove the historical cause."}
        status["status"] = "diagnostic_completed"
        return 0
    except Exception as error:
        status["status"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        (output / "failure_traceback.txt").write_text(traceback.format_exc())
        print(status["error"], file=sys.stderr, flush=True)
        return 2
    finally:
        status["finished_utc"] = now()
        write_json(status_path, status)
        warnings.filters[:] = old_filters
        warnings.showwarning = old_warning
        warning_file.close()
        print("Status:", status["status"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    sys.exit(run(parser.parse_args()))
