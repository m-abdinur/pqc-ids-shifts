"""Compare a new full main run with every field in the locked main raw JSON.

The numerical tolerances are reproduction-screening rules, not hypothesis
tests or equivalence margins. Missing splits and changed feature order fail.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

from common import ROOT, sha256, write_csv, write_json


def compare_tree(actual, expected, rules, path=()):
    rows = []

    def record(value, reference, passed, tolerance=None, difference=None, reason=""):
        rows.append({"field": "/".join(path), "actual": value, "reference": reference,
                     "absolute_tolerance": tolerance, "difference": difference,
                     "passed": passed, "reason": reason})

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            record(str(type(actual).__name__), "object", False, reason="wrong_type")
            return rows
        missing, extra = set(expected) - set(actual), set(actual) - set(expected)
        for key in sorted(missing):
            rows.append({"field": "/".join(path + (key,)), "actual": "MISSING",
                         "reference": "required", "absolute_tolerance": None,
                         "difference": None, "passed": False, "reason": "missing_field"})
        for key in sorted(extra):
            rows.append({"field": "/".join(path + (key,)), "actual": "unexpected",
                         "reference": "absent", "absolute_tolerance": None,
                         "difference": None, "passed": False, "reason": "extra_field"})
        for key in expected.keys() & actual.keys():
            rows.extend(compare_tree(actual[key], expected[key], rules, path + (key,)))
    elif isinstance(expected, list):
        record(json.dumps(actual), json.dumps(expected), actual == expected,
               reason="" if actual == expected else "list_membership_or_order")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if isinstance(expected, int):
            tolerance = 0
        elif "mi_prep" in path:
            tolerance = rules["mi_score_absolute_tolerance"]
        elif "ks_D_prep" in path:
            tolerance = rules["ks_D_absolute_tolerance"]
        elif path[-1] == "roc_auc_ovr_weighted":
            tolerance = rules["roc_auc_absolute_tolerance"]
        else:
            tolerance = rules["f1_absolute_tolerance"]
        if (not isinstance(actual, (int, float)) or isinstance(actual, bool)
                or not math.isfinite(actual) or not math.isfinite(expected)):
            record(str(actual), str(expected), False, tolerance, reason="nonfinite_or_wrong_type")
        else:
            difference = actual - expected
            record(actual, expected, abs(difference) <= tolerance, tolerance, difference,
                   "" if abs(difference) <= tolerance else "outside_tolerance")
    else:
        record(actual, expected, actual == expected, reason="" if actual == expected else "not_equal")
    return rows


def core_means(actual, reference):
    rows = []
    for model in ("rf", "xgb", "mlp"):
        for condition in ("C1", "C2", "C3"):
            def values(raw):
                result = []
                for seed in range(10):
                    block = raw[str(seed)]["results"][model]
                    metric = block["C1"] if condition == "C1" else block["per_variant"]["hybrid768"][condition]
                    result.append(float(metric["weighted_f1"]))
                return result
            new, old = values(actual), values(reference)
            if not all(math.isfinite(v) for v in new + old):
                raise ValueError("Non-finite headline result")
            a, b = statistics.fmean(new), statistics.fmean(old)
            rows.append({"model": model, "condition": condition, "new_mean_weighted_f1": a,
                         "reference_mean_weighted_f1": b, "difference": a - b})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference_path = ROOT / "reference/main_results/revised_pipeline_raw.json"
    target_path = ROOT / "environment/main_known_versions.json"
    reference = json.loads(reference_path.read_text())
    rules = json.loads(target_path.read_text())["acceptance"]
    actual = json.loads(args.actual.read_text())
    rows = sorted(compare_tree(actual, reference, rules), key=lambda row: row["field"])
    failures = [row for row in rows if not row["passed"]]
    complete_seeds = set(actual) == {str(i) for i in range(10)}
    report = {"status": "passed_main_comparison" if complete_seeds and not failures else "needs_review",
              "scope": "Full main raw-result comparison only; excludes B3 and later extensions",
              "complete_seed_keys": complete_seeds, "actual_sha256": sha256(args.actual),
              "reference_sha256": sha256(reference_path), "criteria_sha256": sha256(target_path),
              "criteria": rules, "fields_compared": len(rows), "fields_outside_rules": len(failures),
              "prediction_equality_assessed": False,
              "scientific_validity_established": False}
    write_csv(args.output / "main_field_comparison.csv", rows)
    if failures:
        write_csv(args.output / "main_differences.csv", failures)
    try:
        write_csv(args.output / "main_headline_means.csv", core_means(actual, reference))
    except (KeyError, TypeError, ValueError) as error:
        report["headline_means_unavailable"] = str(error)
    write_json(args.output / "main_comparison_status.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed_main_comparison" else 1


if __name__ == "__main__":
    sys.exit(main())
