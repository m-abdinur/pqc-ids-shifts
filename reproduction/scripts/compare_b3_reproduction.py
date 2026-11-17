"""Compare a new identifier-free B3 run against its recovered locked records.

Uses the same prospective screening tolerances as the main reproduction.
RF numerical variability is recorded, never automatically exempted.
"""
import argparse
import json
from pathlib import Path
import sys

from common import ROOT, sha256, write_csv, write_json
from compare_main_reproduction import compare_tree, core_means


def compare(actual_path, summary_path, output):
    output.mkdir(parents=True, exist_ok=False)
    reference_path = ROOT / "reference/b3_results/revised_pipeline_b3_raw.json"
    reference_summary_path = ROOT / "reference/b3_results/revised_pipeline_b3_summary.json"
    target_path = ROOT / "environment/b3_known_versions.json"
    actual, reference = (json.loads(p.read_text()) for p in (actual_path, reference_path))
    rules = json.loads(target_path.read_text())["acceptance"]
    rows = sorted(compare_tree(actual, reference, rules), key=lambda row: row["field"])
    failures = [row for row in rows if not row["passed"]]
    complete = set(actual) == {str(i) for i in range(10)}
    expected_config = json.loads(reference_summary_path.read_text())["config"]
    config = json.loads(summary_path.read_text()).get("config", {}) if summary_path.is_file() else {}
    config_rows = sorted(compare_tree(config, expected_config, rules), key=lambda row: row["field"])
    config_failures = [row for row in config_rows if not row["passed"]]
    passed = complete and not failures and not config_failures
    report = {"status": "passed_b3_comparison" if passed else "needs_review",
              "scope": "Identifier-free B3 raw fields and summary configuration; excludes the separate MLP threshold sweep",
              "complete_seed_keys": complete, "actual_sha256": sha256(actual_path),
              "reference_sha256": sha256(reference_path), "criteria_sha256": sha256(target_path),
              "criteria": rules, "fields_compared": len(rows), "fields_outside_rules": len(failures),
              "configuration_fields_compared": len(config_rows), "configuration_fields_outside_rules": len(config_failures),
              "prediction_equality_assessed": False, "scientific_validity_established": False,
              "RF_variability_policy": "Report every exception; do not choose a better repeat or relax tolerance."}
    write_csv(output / "b3_field_comparison.csv", rows)
    write_csv(output / "b3_configuration_comparison.csv", config_rows)
    if failures:
        write_csv(output / "b3_differences.csv", failures)
    if config_failures:
        write_csv(output / "b3_configuration_differences.csv", config_failures)
    try:
        write_csv(output / "b3_headline_means.csv", core_means(actual, reference))
    except (KeyError, TypeError, ValueError) as error:
        report["headline_means_unavailable"] = str(error)
    write_json(output / "b3_comparison_status.json", report)
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.exit(compare(args.actual, args.summary, args.output))
