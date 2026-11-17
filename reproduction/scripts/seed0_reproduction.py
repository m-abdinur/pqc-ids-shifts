"""Reproduce the three recorded seed-0 C1-SF fits in a separate output folder.

This is a reproduction check, not a replacement experiment or a full ten-split
run. Feature selection is read from the verified original records.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings

from common import ROOT, read_csv, sha256, verify_reference_code, write_csv, write_json
from run_experiment import SOURCES


def now():
    return datetime.now(timezone.utc).isoformat()


def text_hash(values):
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def array_hash(values):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values)).tobytes()).hexdigest()


def check_environment(target):
    installed = {}
    for name in target["packages"]:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
    problems = []
    if platform.python_version() != target["python"]:
        problems.append(f"Python {platform.python_version()} != {target['python']}")
    for name, expected in target["packages"].items():
        if installed[name] != expected:
            problems.append(f"{name} {installed[name]} != {expected}")
    return {"python": platform.python_version(), "packages": installed,
            "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "problems": problems,
            "all_distributions": sorted(f"{d.metadata['Name']}=={d.version}"
                                        for d in importlib.metadata.distributions())}


def check_split(X_train, X_test, selected, references):
    signatures = {"train_index_sha256": text_hash(X_train.index.tolist()),
                  "test_index_sha256": text_hash(X_test.index.tolist()),
                  "selected_feature_order_sha256": text_hash(selected)}
    if len(X_train) != 83699 or len(X_test) != 35872 or len(selected) != 24:
        raise ValueError("The seed-0 split sizes or selected-feature count differ from the record.")
    for model, row in references.items():
        for field, value in signatures.items():
            if row[field] != value:
                raise ValueError(f"{model}: {field} differs from the recorded seed-0 split.")
    return signatures


def compare_result(model, evaluation, predictions, probabilities, reference, tolerance):
    errors = []
    wf1 = float(evaluation["weighted_f1"])
    auc = float(evaluation["roc_auc_ovr_weighted"])
    import math
    if not math.isfinite(wf1) or abs(wf1 - float(reference["c1_sf_weighted_f1"])) > tolerance["weighted_f1_absolute_tolerance"]:
        errors.append("weighted_f1")
    if not math.isfinite(auc) or abs(auc - float(reference["c1_sf_roc_auc"])) > tolerance["roc_auc_absolute_tolerance"]:
        errors.append("roc_auc")
    class_differences = {}
    expected_classes = {"Benign", "DoS", "Bot", "Infiltration"}
    if set(evaluation["per_class_f1"]) != expected_classes:
        raise ValueError("Expected per-class F1 for all four recorded classes.")
    for name, score in evaluation["per_class_f1"].items():
        delta = float(score) - float(reference[f"c1_sf_{name}_f1"])
        class_differences[name] = delta
        if not math.isfinite(delta) or abs(delta) > tolerance["per_class_f1_absolute_tolerance"]:
            errors.append(f"{name}_f1")
    pred_hash, proba_hash = array_hash(predictions), array_hash(probabilities)
    if pred_hash != reference["prediction_sha256"]:
        errors.append("prediction_hash")
    return {"seed": 0, "model": model, "weighted_f1": wf1,
            "reference_weighted_f1": float(reference["c1_sf_weighted_f1"]),
            "weighted_f1_difference": wf1 - float(reference["c1_sf_weighted_f1"]),
            "roc_auc": auc, "roc_auc_difference": auc - float(reference["c1_sf_roc_auc"]),
            "max_absolute_per_class_f1_difference": max(abs(d) for d in class_differences.values()),
            "prediction_sha256": pred_hash, "prediction_hash_match": pred_hash == reference["prediction_sha256"],
            "probability_sha256": proba_hash, "probability_hash_match": proba_hash == reference["probability_sha256"],
            "prediction_dtype": str(predictions.dtype), "probability_dtype": str(probabilities.dtype),
            "per_class_f1_differences": json.dumps(class_differences, sort_keys=True),
            "passed_scores_and_prediction_hashes": not errors, "review_fields": "; ".join(errors)}


def run(args):
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    target_path = ROOT / "environment/seed0_known_versions.json"
    target = json.loads(target_path.read_text())
    record = {"started_utc": now(), "status": "checking", "seed": 0,
              "condition": "C1-SF", "models_requested": ["rf", "xgb", "mlp"],
              "fits_completed": 0, "acceptance": target["acceptance"],
              "full_experiment_reproduced": False, "historical_results_modified": False,
              "environment_target_sha256": sha256(target_path)}
    record_path = output / "seed0_status.json"
    write_json(record_path, record)
    old_filters, old_warning = list(warnings.filters), warnings.showwarning
    warning_stream = (output / "warnings.log").open("w")

    def visible_warning(message, category, filename, lineno, file=None, line=None):
        warning_stream.write(warnings.formatwarning(message, category, filename, lineno, line))
        warning_stream.flush()
        old_warning(message, category, filename, lineno, file=file, line=line)

    try:
        env = check_environment(target)
        write_json(output / "runtime_environment.json", env)
        record["environment_matches_recorded_top_level_versions"] = not env["problems"]
        if env["problems"]:
            record["status"] = "environment_mismatch"
            record["problems"] = env["problems"]
            print("Environment check failed:", "; ".join(env["problems"]), flush=True)
            return 2
        record["reference_sources_verified"] = verify_reference_code()
        csv_path = ROOT / "reference/c1sf_records/split_level_results/same_feature_c1_metrics_by_split.csv"
        stats_config = json.loads((ROOT / "reference/c1sf_records/manifests/c1_sf_vs_c1_statistical_configuration.json").read_text())
        if sha256(csv_path) != stats_config["input_sha256"]:
            raise ValueError("The C1-SF reference CSV no longer matches its recorded hash.")
        references = {row["model_key"]: row for row in read_csv(csv_path) if int(row["seed"]) == 0}
        if set(references) != {"rf", "xgb", "mlp"}:
            raise ValueError("Expected exactly three seed-0 reference models.")
        raw_path = ROOT / "reference/main_results/revised_pipeline_raw.json"
        raw = json.loads(raw_path.read_text())["0"]
        selected = [f for f in raw["stable_prep"] if raw["mi_prep"].get(f, 0) >= .05]
        record["reference_input_sha256"] = {csv_path.name: sha256(csv_path), raw_path.name: sha256(raw_path)}
        # Stream each Drive file to local scratch once, computing its checksum.
        cache = args.cache_dir.expanduser().resolve() if args.cache_dir else output / "input_cache"
        cache.mkdir(parents=True, exist_ok=False)
        files, inputs = {}, []
        for filename, attack in SOURCES:
            source, local = args.data_dir / filename, cache / filename
            if not source.is_file():
                raise FileNotFoundError(source)
            before = source.stat()
            print("Reading input:", filename, flush=True)
            digest = hashlib.sha256()
            with source.open("rb") as incoming, local.open("xb") as outgoing:
                for block in iter(lambda: incoming.read(1024 * 1024), b""):
                    digest.update(block)
                    outgoing.write(block)
            after = source.stat()
            if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                raise ValueError(f"Input changed while being read: {filename}")
            inputs.append({"filename": filename, "attack": attack, "bytes": local.stat().st_size,
                           "sha256": digest.hexdigest(), "role": "Observed input identity for this reproduction attempt"})
            files[str(local)] = attack
        write_json(output / "input_files.json", inputs)
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from threadpoolctl import threadpool_info
        sys.path.insert(0, str(ROOT / "reference/locked_v5"))
        import pilot_pqc_stable_pipeline as base
        import option_b_full_experiment as full
        if not full.HAVE_XGB:
            raise RuntimeError("The pinned XGBoost package could not be imported.")
        warnings.filters[:] = old_filters
        warnings.simplefilter("default")
        warnings.showwarning = visible_warning
        record["status"] = "preparing_data"
        write_json(record_path, record)
        data = base.load_and_clean(files, base.LABEL_COL, base.LABEL_NORMALISE, base.CLASSES, 30000, 42)
        X, y, features, class_map = base.prepare_xy(data, base.NON_FEATURE_COLS, base.CLASSES)
        del data
        counts = {name: int(np.sum(y == code)) for name, code in class_map.items()}
        expected_counts = {"Benign": 29827, "DoS": 30000, "Bot": 30000, "Infiltration": 29744}
        if len(X) != 119571 or len(features) != 70 or counts != expected_counts:
            raise ValueError(f"Prepared data differ: rows={len(X)}, features={len(features)}, classes={counts}")
        if class_map != {"Benign": 0, "DoS": 1, "Bot": 2, "Infiltration": 3}:
            raise ValueError("Class encoding differs from the source experiment.")
        if any(f not in features for f in selected):
            raise ValueError("A recorded selected feature is absent from the prepared data.")
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=0, stratify=y)
        signatures = check_split(Xtr, Xte, selected, references)
        record["prepared_data"] = {"rows": len(X), "feature_count": len(features),
                                   "features": features, "class_counts": counts,
                                   "selected_features": selected, **signatures,
                                   "X_dtype": str(X.values.dtype), "X_shape": list(X.shape),
                                   "X_sha256": array_hash(X.values), "y_sha256": array_hash(y)}
        write_json(output / "threadpools.json", threadpool_info())
        record["status"] = "fitting_seed0"
        write_json(record_path, record)
        result_rows = []
        for kind in ("rf", "xgb", "mlp"):
            print("Fitting seed 0:", kind, flush=True)
            model = full.make_model(kind)
            (output / f"{kind}_parameters.txt").write_text(repr(model.get_params()) + "\n")
            if kind == "mlp":
                scaler = StandardScaler().fit(Xtr[selected])
                train_matrix = scaler.transform(Xtr[selected])
                test_matrix = scaler.transform(Xte[selected])
            else:
                train_matrix, test_matrix = Xtr[selected].values, Xte[selected].values
            started = time.perf_counter()
            model.fit(train_matrix, ytr)
            fit_seconds = time.perf_counter() - started
            predictions, probabilities = model.predict(test_matrix), model.predict_proba(test_matrix)
            evaluation = base.evaluate(model, test_matrix, yte, {v: k for k, v in class_map.items()})
            comparison = compare_result(kind, evaluation, predictions, probabilities, references[kind], target["acceptance"])
            comparison.update(fit_seconds=fit_seconds, completed_utc=now())
            result_rows.append(comparison)
            np.savez_compressed(output / f"seed0_{kind}_predictions.npz",
                                test_index=Xte.index.to_numpy(), y_true=yte,
                                predictions=predictions, probabilities=probabilities,
                                classes=model.classes_)
            write_json(output / f"seed0_{kind}_metrics.json", evaluation)
            write_csv(output / "seed0_comparison.csv", result_rows)
            record["fits_completed"] = len(result_rows)
            record["completed_models"] = [r["model"] for r in result_rows]
            write_json(record_path, record)
            print(kind, "weighted F1:", comparison["weighted_f1"],
                  "difference:", comparison["weighted_f1_difference"],
                  "prediction hash matches:", comparison["prediction_hash_match"], flush=True)
        passed = all(r["passed_scores_and_prediction_hashes"] for r in result_rows)
        record["status"] = "passed_seed0_gate" if passed else "needs_review"
        record["probability_hash_matches"] = sum(r["probability_hash_match"] for r in result_rows)
        return 0 if passed else 1
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        (output / "failure_traceback.txt").write_text(traceback.format_exc())
        print(record["error"], file=sys.stderr, flush=True)
        return 2
    finally:
        record["finished_utc"] = now()
        write_json(record_path, record)
        warnings.filters[:] = old_filters
        warnings.showwarning = old_warning
        warning_stream.close()
        print("Status:", record["status"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory, never reused")
    parser.add_argument("--cache-dir", type=Path, help="New local directory for copies of the CSV inputs")
    sys.exit(run(parser.parse_args()))
