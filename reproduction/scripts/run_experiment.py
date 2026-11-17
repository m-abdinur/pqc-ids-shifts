"""Run the unchanged V5 main or B3 experiment with explicit input/output paths.

The default action checks inputs. Training must be selected explicitly. A run
always uses a new output directory and never resumes a historical checkpoint.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import traceback
import warnings

from common import ROOT, sha256, verify_reference_code, write_json

SOURCES = (
    ("Friday-16-02-2018_TrafficForML_CICFlowMeter.csv", "DoS"),
    ("Friday-02-03-2018_TrafficForML_CICFlowMeter.csv", "Bot"),
    ("Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv", "Infiltration"),
    ("Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv", "Infiltration"),
)
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "statsmodels")


def now():
    return datetime.now(timezone.utc).isoformat()


def environment():
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages}


def preflight(data_dir, expected_environment=None):
    verified = verify_reference_code()
    files, problems = [], []
    for filename, attack in SOURCES:
        path = (data_dir / filename).resolve()
        item = {"path": str(path), "attack": attack, "exists": path.is_file()}
        if path.is_file():
            with path.open(newline="", encoding="utf-8-sig") as stream:
                columns = [c.strip() for c in next(csv.reader(stream))]
            if "Label" not in columns:
                problems.append(f"Missing Label column: {filename}")
            if len(columns) != len(set(columns)):
                problems.append(f"Duplicate column names: {filename}")
            item["bytes"] = path.stat().st_size
        else:
            problems.append(f"Missing input: {filename}")
        files.append(item)
    env = environment()
    for name in PACKAGES:
        if env["packages"][name] is None:
            problems.append(f"Missing package: {name}")
    environment_match = None
    if expected_environment:
        expected = json.loads(expected_environment.read_text())
        if set(expected.get("packages", {})) != set(PACKAGES) or not expected.get("python"):
            raise ValueError("Expected environment must name Python and all six packages.")
        environment_match = expected["python"] == env["python"] and expected["packages"] == env["packages"]
        if not environment_match:
            problems.append("The runtime differs from the supplied expected environment.")
    return {"reference_files_verified": verified, "inputs": files, "environment": env,
            "expected_environment_match": environment_match, "problems": problems,
            "note": "No historical training-environment lock is inferred from the final C1-SF runtime."}


def train(args, check):
    import numpy as np
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "starting", "started_utc": now(), "variant": args.variant,
              "source_check": check, "checkpoint_resume": False,
              "validation_scope": "New reproduction attempt, not a certified replacement for locked results"}
    record_path = output / "run_record.json"
    write_json(record_path, record)
    original_warning = warnings.showwarning
    original_filters = list(warnings.filters)
    warning_stream = (output / "warnings.log").open("w", encoding="utf-8")

    def show_warning(message, category, filename, lineno, file=None, line=None):
        warning_stream.write(warnings.formatwarning(message, category, filename, lineno, line))
        warning_stream.flush()
        original_warning(message, category, filename, lineno, file=file, line=line)

    try:
        expected_inputs = None
        if args.expected_inputs:
            expected_inputs = {row["filename"]: row for row in json.loads(args.expected_inputs.read_text())}
            if set(expected_inputs) != {name for name, _ in SOURCES}:
                raise ValueError("Expected-input record must contain the four original CSV basenames.")
        cache = args.cache_dir.expanduser().resolve() if args.cache_dir else None
        if cache:
            cache.mkdir(parents=True, exist_ok=False)
        for item in record["source_check"]["inputs"]:
            source = Path(item["path"])
            before = source.stat()
            if cache:
                local = cache / source.name
                digest = hashlib.sha256()
                with source.open("rb") as incoming, local.open("xb") as outgoing:
                    for block in iter(lambda: incoming.read(1024 * 1024), b""):
                        digest.update(block)
                        outgoing.write(block)
                item["sha256"] = digest.hexdigest()
                item["training_path"] = str(local)
            else:
                item["sha256"] = sha256(source)
                item["training_path"] = str(source)
            after = source.stat()
            if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                raise ValueError(f"Input changed while reading: {source.name}")
            if expected_inputs:
                expected = expected_inputs[source.name]
                if item["sha256"] != expected["sha256"] or after.st_size != expected["bytes"]:
                    raise ValueError(f"Input differs from the verified seed-0 run: {source.name}")
            print("Input verified:", source.name, flush=True)
        record["inputs_match_seed0_record"] = True if expected_inputs else None
        sys.path.insert(0, str(ROOT / "reference/locked_v5"))
        module_name = {"main": "revised_pipeline_main_only", "b3": "revised_pipeline_b3"}[args.variant]
        pipeline = importlib.import_module(module_name)
        if not pipeline.full.HAVE_XGB:
            raise RuntimeError("XGBoost failed to import. The runner will not silently omit this model.")
        if not pipeline.HAVE_SM:
            raise RuntimeError("statsmodels failed to import; resolve the environment before training.")
        # Imported historical files suppress warnings. Restore them for this run.
        warnings.filters[:] = original_filters
        warnings.simplefilter("default")
        warnings.showwarning = show_warning
        pipeline.FILES = {item["training_path"]: item["attack"] for item in check["inputs"]}
        prefix = "revised_pipeline" if args.variant == "main" else "revised_pipeline_b3"
        pipeline.OUT_PREFIX = str(output / prefix)
        original_prepare = pipeline.base.prepare_xy

        def checked_prepare(*positional, **keyword):
            X, y, features, class_map = original_prepare(*positional, **keyword)
            counts = {name: int(np.sum(y == code)) for name, code in class_map.items()}
            expected = {"Benign": 29827, "DoS": 30000, "Bot": 30000, "Infiltration": 29744}
            if len(X) != 119571 or len(features) != 70 or counts != expected:
                raise ValueError(f"Prepared data do not match the recorded experiment: {len(X)=}, {len(features)=}, {counts=}")
            record["prepared_data"] = {"rows": len(X), "features_before_B3_exclusions": features,
                                       "class_counts": counts, "class_encoding": class_map,
                                       "feature_order_sha256": hashlib.sha256("\n".join(features).encode()).hexdigest(),
                                       "X_dtype": str(X.values.dtype), "X_shape": list(X.shape),
                                       "X_sha256": hashlib.sha256(np.ascontiguousarray(X.values).tobytes()).hexdigest(),
                                       "y_sha256": hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()}
            if args.expected_prepared:
                previous = json.loads(args.expected_prepared.read_text())["prepared_data"]
                actual = record["prepared_data"]
                for key in ("rows", "class_counts", "X_dtype", "X_shape", "X_sha256", "y_sha256"):
                    if actual[key] != previous[key]:
                        raise ValueError(f"Prepared data differ from the verified seed-0 run: {key}")
                if features != previous["features"]:
                    raise ValueError("Prepared feature order differs from the seed-0 run.")
                record["prepared_matrix_matches_seed0_record"] = True
            write_json(record_path, record)
            return X, y, features, class_map

        pipeline.base.prepare_xy = checked_prepare
        record["configuration"] = {key: getattr(pipeline, key) for key in (
            "N_SPLITS", "SPLIT_SEEDS", "MODEL_SEED", "PER_CLASS", "TEST_SIZE", "HOLM_ALPHA",
            "DROP_IDENTIFIERS", "APPLY_DROP_UNMODELLABLE", "UNMODELLABLE", "MI_THRESHOLDS",
            "MI_HEADLINE", "MI_SUBSAMPLE", "FPREP", "VARIANTS")}
        record["configuration"]["run_transfer_argument"] = True
        record["all_distributions"] = sorted(f"{d.metadata['Name']}=={d.version}"
                                             for d in importlib.metadata.distributions())
        from threadpoolctl import threadpool_info
        write_json(output / "threadpools.json", threadpool_info())
        record["configuration"]["model_parameters"] = {
            kind: pipeline.full.make_model(kind).get_params() for kind in ("rf", "xgb", "mlp")}
        # Some estimators expose NaN-valued defaults. Keep their parameter record
        # as readable text, separate from the strict JSON run record.
        model_parameters = record["configuration"].pop("model_parameters")
        (output / "model_parameters.txt").write_text(
            "\n\n".join(f"{kind}\n{params!r}" for kind, params in model_parameters.items()))
        record["status"] = "running"
        record["expected_fits"] = pipeline.N_SPLITS * 3 * (1 + len(pipeline.MI_THRESHOLDS) + 2 * len(pipeline.VARIANTS))
        record["fits_completed"] = 0
        record["split_computations_completed"] = []
        write_json(record_path, record)
        original_fit_eval, original_one_split = pipeline.fit_eval, pipeline.one_split

        def logged_fit_eval(*positional, **keyword):
            number = record["fits_completed"] + 1
            print(f"Fit {number}/{record['expected_fits']} | seed={record.get('current_seed')} | model={positional[0]}", flush=True)
            value = original_fit_eval(*positional, **keyword)
            record["fits_completed"] = number
            write_json(record_path, record)
            return value

        def logged_one_split(*positional, **keyword):
            seed = int(positional[0]) if positional else int(keyword["seed"])
            record["current_seed"] = seed
            write_json(record_path, record)
            value = original_one_split(*positional, **keyword)
            record["split_computations_completed"].append(seed)
            write_json(record_path, record)
            return value

        pipeline.fit_eval, pipeline.one_split = logged_fit_eval, logged_one_split
        # Preserve the original fitting order, MI sweep and all four footprints.
        pipeline.run_revised(run_transfer=True)
        record["status"] = "completed"
        record["result_sha256"] = {p.name: sha256(p) for p in output.glob(f"{prefix}*.json")}
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        (output / "failure_traceback.txt").write_text(traceback.format_exc())
        raise
    finally:
        record["finished_utc"] = now()
        write_json(record_path, record)
        warnings.showwarning = original_warning
        warnings.filters[:] = original_filters
        warning_stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("main", "b3"), default="main")
    parser.add_argument("--stage", choices=("check", "train"), default="check")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="A new directory; required for training")
    parser.add_argument("--expected-environment", type=Path,
                        help="Optional verified environment JSON with python and packages keys")
    parser.add_argument("--expected-inputs", type=Path, help="CSV identity record from the verified seed-0 run")
    parser.add_argument("--expected-prepared", type=Path, help="Seed-0 status JSON containing prepared-matrix hashes")
    parser.add_argument("--cache-dir", type=Path, help="New local folder for read-once input copies")
    args = parser.parse_args()
    check = preflight(args.data_dir.expanduser().resolve(), args.expected_environment)
    print(json.dumps(check, indent=2))
    if check["problems"]:
        return 2
    if args.stage == "train":
        if args.output is None:
            parser.error("--output is required for training")
        train(args, check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
