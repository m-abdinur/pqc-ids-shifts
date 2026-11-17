"""Export the already completed B3 folder without starting any model fits.

The source folder is read only. A new destination holds the copied records,
a recovery receipt, and an archive with independently computed file hashes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

COMPLETED_RUN_ID = "20260920T163825Z_c7cff4c7"
EXPECTED_RAW_SHA256 = "b1efd7f9d112b8835334672323263ebac8c2b25a7b721b7f6b73d48f5b80e634"
EXPECTED_REFERENCE_SHA256 = "eefc4d863ffa0d9c321c687f141d72a465a0c32ad8c29150016eee88ffccdbed"
EXPECTED_PACKAGE_SHA256 = "595ffc452cc97841cb5466bebfb944de15c52af64700ffe21ebf7748fad4a4f1"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def recover(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Recovery destination must be separate from the original run folder.")
    destination.mkdir(parents=True, exist_ok=False)
    bundle = destination / "recovered_records"
    bundle.mkdir()
    issues = []
    receipt = {"export_kind": "existing_B3_run_recovery", "model_fits_started": 0,
               "source_directory": str(source), "expected_run_id": COMPLETED_RUN_ID,
               "original_records_modified": False,
               "exported_utc": datetime.now(timezone.utc).isoformat(),
               "independent_raw_result_comparison_performed": False,
               "expected_raw_sha256_from_returned_notebook": EXPECTED_RAW_SHA256}

    if source.name != COMPLETED_RUN_ID:
        issues.append("Source folder name differs from the completed run recorded in the notebook.")
    elif not source.is_dir():
        issues.append("The completed run folder was not found at the expected Drive location.")
    else:
        for original in sorted(source.rglob("*")):
            if original.is_symlink():
                issues.append("Skipped symbolic link: " + str(original.relative_to(source)))
                continue
            if not original.is_file() or original.suffix.lower() == ".zip":
                continue
            relative = original.relative_to(source)
            # Preserve any earlier export manifest instead of replacing its record.
            if relative.as_posix() in ("result_manifest.json", "recovery_receipt.json"):
                relative = Path("previous_export") / relative
            target = bundle / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                before = original.stat()
                shutil.copyfile(original, target)
                after = original.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    issues.append("Source changed while copying: " + str(original.relative_to(source)))
            except OSError as error:
                issues.append("Could not copy " + str(relative) + ": " + str(error))

    def read_record(relative):
        path = bundle / relative
        if not path.is_file():
            issues.append("Missing record: " + relative)
            return {}
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError("Expected a JSON object")
            return value
        except (ValueError, OSError) as error:
            issues.append("Unreadable record " + relative + ": " + str(error))
            return {}

    notebook = read_record("notebook_status.json")
    run = read_record("results/run_record.json")
    comparison = read_record("comparison/b3_comparison_status.json")
    raw = read_record("results/revised_pipeline_b3_raw.json")
    summary = read_record("results/revised_pipeline_b3_summary.json")
    raw_path = bundle / "results/revised_pipeline_b3_raw.json"
    actual_hash = file_hash(raw_path) if raw_path.is_file() else None
    receipt.update(recorded_notebook_status=notebook.get("status"),
                   recorded_training_status=run.get("status"),
                   recorded_comparison_status=comparison.get("status"),
                   recorded_fits_completed=run.get("fits_completed"),
                   actual_raw_sha256=actual_hash)
    checks = {
        "run_id_matches": notebook.get("run_id") == COMPLETED_RUN_ID,
        "package_matches": notebook.get("package_sha256") == EXPECTED_PACKAGE_SHA256,
        "training_complete": run.get("status") == "completed" and run.get("variant") == "b3",
        "all_300_fits_recorded": run.get("expected_fits") == 300 and run.get("fits_completed") == 300,
        "all_ten_splits_recorded": run.get("split_computations_completed") == list(range(10)),
        "all_ten_raw_splits_present": set(raw) == {str(i) for i in range(10)},
        "raw_matches_notebook_printed_hash": actual_hash == EXPECTED_RAW_SHA256,
        "comparison_refers_to_same_raw": actual_hash is not None and comparison.get("actual_sha256") == actual_hash,
        "comparison_refers_to_locked_reference": comparison.get("reference_sha256") == EXPECTED_REFERENCE_SHA256,
        "comparison_records_pass": comparison.get("status") == "passed_b3_comparison"
            and comparison.get("fields_compared") == 2530 and comparison.get("fields_outside_rules") == 0
            and comparison.get("configuration_fields_compared") == 42 and comparison.get("configuration_fields_outside_rules") == 0,
        "identifier_removal_recorded": summary.get("config", {}).get("drop_identifiers") is True
            and summary.get("config", {}).get("n_features_full") == 68
            and summary.get("config", {}).get("n_candidate_features") == 53,
    }
    issues.extend("Check did not pass: " + key for key, passed in checks.items() if not passed)
    receipt["checks"] = checks
    receipt["issues"] = issues
    receipt["status"] = "recovered_for_independent_review" if not issues else "recovery_needs_review"
    receipt["interpretation"] = "These checks identify the existing run; independent result verification remains the next review step."
    write_json(bundle / "recovery_receipt.json", receipt)
    paths = sorted(p for p in bundle.rglob("*") if p.is_file())
    entries = [{"path": p.relative_to(bundle).as_posix(), "bytes": p.stat().st_size,
                "sha256": file_hash(p)} for p in paths]
    manifest = bundle / "result_manifest.json"
    write_json(manifest, {"run_id": COMPLETED_RUN_ID, "export_kind": "recovery_only", "files": entries})
    archive_path = destination / "PQC_IDS_B3_Recovered_Results.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths + [manifest]:
            archive.write(path, path.relative_to(bundle).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("The recovered archive failed its CRC check.")
        for entry in entries:
            data = archive.read(entry["path"])
            if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise RuntimeError("Recovered archive content failed verification: " + entry["path"])
    return archive_path, receipt


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    archive, receipt = recover(args.source, args.destination)
    print(json.dumps(receipt, indent=2))
    print("Saved:", archive)
