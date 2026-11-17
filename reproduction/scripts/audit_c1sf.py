"""Audit existing C1-SF files and retain a small review bundle. Never fit models."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

from common import (ROOT, MODELS, compare_number, holm, new_output, paired_summary,
                    read_csv, sha256, tost, write_csv, write_json)


def indexed_results(path):
    rows = read_csv(path)
    keys = [(int(row["seed"]), row["model_key"]) for row in rows]
    expected = {(seed, model) for seed in range(10) for model in MODELS}
    if len(keys) != 30 or set(keys) != expected:
        raise ValueError(f"Expected 30 unique seed/model rows: {path}")
    return dict(zip(keys, rows))


def same_value(left, right):
    if left == right:
        return True
    try:
        a, b = float(left), float(right)
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-12)
    except (TypeError, ValueError):
        return False


def audit(extension, output):
    paths = {
        "final": extension / "split_level_results/same_feature_c1_metrics_by_split.csv",
        "paired": extension / "statistical_results/c1_sf_vs_c1_weighted_f1_paired_tests.csv",
        "tost": extension / "statistical_results/c1_sf_vs_c1_weighted_f1_tost.csv",
        "stats_config": extension / "manifests/c1_sf_vs_c1_statistical_configuration.json",
        "run_config": extension / "manifests/full_run_configuration.json",
        "environment": extension / "manifests/environment_versions.txt",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing C1-SF inputs: " + "; ".join(missing))
    candidates = sorted((extension / "validation").glob(
        "preliminary_xgboost_3_3_0_*/same_feature_c1_metrics_by_split.csv"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one preliminary XGBoost 3.3.0 checkpoint; found {len(candidates)}.")
    paths["preliminary"] = candidates[0]
    final, preliminary = indexed_results(paths["final"]), indexed_results(paths["preliminary"])
    config = json.loads(paths["stats_config"].read_text())
    problems = []
    if config["input_sha256"] != sha256(paths["final"]):
        problems.append({"field": "statistical input hash", "problem": "Final CSV differs from statistical configuration"})
    raw_path = ROOT / "reference/main_results/revised_pipeline_raw.json"
    raw = json.loads(raw_path.read_text())
    branches, history = [], []
    for (seed, model), row in sorted(final.items()):
        block = raw[str(seed)]["results"][model]
        selected = [f for f in raw[str(seed)]["stable_prep"] if raw[str(seed)]["mi_prep"].get(f, 0) >= .05]
        selected_hash = hashlib.sha256("\n".join(selected).encode()).hexdigest()
        if row["selected_feature_order_sha256"] != selected_hash:
            problems.append({"field": f"{seed}.{model}.selected_feature_order", "problem": "Does not match original selection order"})
        for column, expected in (("train_rows", 83699), ("test_rows", 35872),
                                 ("full_feature_count", 70), ("selected_feature_count", len(selected))):
            compare_number(row[column], expected, f"{seed}.{model}.{column}", problems, atol=0, rtol=0)
        for column in ("train_index_sha256", "test_index_sha256"):
            if len({final[(seed, k)][column] for k in MODELS}) != 1:
                problems.append({"field": f"{seed}.{column}", "problem": "Models have different recorded split hashes"})
        sf, c1, c3 = map(float, (row["c1_sf_weighted_f1"], row["canonical_c1_weighted_f1"], row["canonical_c3_weighted_f1"]))
        for value in (sf, c1, c3):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Weighted-F1 values must be finite and between zero and one.")
        compare_number(c1, block["C1"]["weighted_f1"], f"{seed}.{model}.C1_reference", problems, atol=1e-12, rtol=0)
        compare_number(c3, block["mi_sweep"]["0.05"], f"{seed}.{model}.C3_MI_reference", problems, atol=1e-12, rtol=0)
        compare_number(sf-c1, row["c1_sf_minus_c1_weighted_f1"], f"{seed}.{model}.stored_difference", problems, atol=1e-12, rtol=0)
        variant = block["per_variant"]["hybrid768"]["C3"]["weighted_f1"]
        branches.append({"seed": seed, "model": model, "C1_SF": sf,
                         "per_variant_C3": variant, "MI_sweep_C3": block["mi_sweep"]["0.05"],
                         "C1_SF_minus_per_variant": sf-variant,
                         "C1_SF_minus_MI_reference": sf-c3,
                         "per_variant_agrees_within_1e_12": abs(sf-variant) <= 1e-12,
                         "MI_reference_agrees_within_1e_12": abs(sf-c3) <= 1e-12})
        old = preliminary[(seed, model)]
        changed = [key for key in sorted(set(row) | set(old)) if not same_value(row.get(key), old.get(key))]
        if model in ("rf", "mlp") and changed:
            problems.append({"field": f"{seed}.{model}.retained_checkpoint", "changed_fields": changed})
        if model == "xgb" and row["completed_utc"] == old["completed_utc"]:
            problems.append({"field": f"{seed}.xgb.completion_time", "problem": "No changed completion time"})
        history.append({"seed": seed, "model": model, "changed_fields": "; ".join(changed),
                        "preliminary_completed_utc": old["completed_utc"],
                        "final_completed_utc": row["completed_utc"],
                        "prediction_hash_equal": row["prediction_sha256"] == old["prediction_sha256"],
                        "probability_hash_equal": row["probability_sha256"] == old["probability_sha256"]})
    stored_paired_rows = read_csv(paths["paired"])
    stored_paired = {r["model_key"]: r for r in stored_paired_rows}
    stored_tost_rows = read_csv(paths["tost"])
    stored_tost = {(r["model_key"], float(r["equivalence_margin"])): r for r in stored_tost_rows}
    if len(stored_paired_rows) != 3 or set(stored_paired) != set(MODELS):
        raise ValueError("Expected three unique paired-test rows.")
    if len(stored_tost_rows) != 6 or set(stored_tost) != {(m,d) for m in MODELS for d in (.01,.005)}:
        raise ValueError("Expected six unique TOST rows.")
    recomputed = []
    for model in MODELS:
        original = np.array([float(final[(s, model)]["canonical_c1_weighted_f1"]) for s in range(10)])
        selected = np.array([float(final[(s, model)]["c1_sf_weighted_f1"]) for s in range(10)])
        summary = paired_summary(selected - original)
        summary.update(model=model, full_C1_mean=float(original.mean()), C1_SF_mean=float(selected.mean()))
        recomputed.append(summary)
    mapping = {"full_C1_mean": "full_c1_mean", "C1_SF_mean": "c1_sf_mean",
               "mean_difference": "mean_paired_difference", "difference_sd": "difference_sd",
               "difference_se": "difference_se", "ci95_lower": "mean_difference_95_ci_lower",
               "ci95_upper": "mean_difference_95_ci_upper", "shapiro_p": "shapiro_p_value",
               "t_statistic": "paired_t_statistic", "t_p": "paired_t_p_value",
               "wilcoxon_statistic": "wilcoxon_statistic", "wilcoxon_p": "wilcoxon_p_value",
               "selected_p": "selected_p_value_raw", "cohen_dz": "cohens_dz"}
    for summary, adjusted in zip(recomputed, holm([r["selected_p"] for r in recomputed])):
        summary["selected_holm_p"] = float(adjusted)
        stored = stored_paired[summary["model"]]
        for ours, theirs in {**mapping, "selected_holm_p": "selected_p_value_holm"}.items():
            compare_number(summary[ours], stored[theirs], f"{summary['model']}.paired.{theirs}", problems)
    equivalence = []
    for margin in (.01, .005):
        tests = [tost(s, margin) for s in recomputed]
        for summary, test, adjusted in zip(recomputed, tests, holm([t["raw_p"] for t in tests])):
            model = summary["model"]
            test.update(model=model, holm_p=float(adjusted), ci90_lower=summary["ci90_lower"],
                        ci90_upper=summary["ci90_upper"], equivalent=bool(adjusted < .05))
            stored = stored_tost[(model, margin)]
            fields = {"raw_p": "equivalence_p_value_raw", "holm_p": "equivalence_p_value_holm",
                      "ci90_lower": "tost_90_ci_lower", "ci90_upper": "tost_90_ci_upper",
                      "lower_t": "lower_test_t_statistic", "upper_t": "upper_test_t_statistic",
                      "lower_p": "lower_test_p_value", "upper_p": "upper_test_p_value"}
            for ours, theirs in fields.items():
                compare_number(test[ours], stored[theirs], f"{model}.TOST.{margin}.{theirs}", problems)
            if test["equivalent"] != (stored["equivalent_after_holm"].lower() == "true"):
                problems.append({"field": f"{model}.TOST.{margin}.decision"})
            equivalence.append(test)
    output = new_output(output)
    write_csv(output / "C1SF_statistics_recomputed.csv", recomputed)
    write_csv(output / "C1SF_TOST_recomputed.csv", equivalence)
    write_csv(output / "C1SF_C3_source_comparison.csv", branches)
    write_csv(output / "C1SF_checkpoint_comparison.csv", history)
    # Keep raw records small enough to review without transferring the complete
    # experiment archive. Filenames and bytes are preserved under input_records.
    retained = set(paths.values())
    retained.update(p for p in paths["preliminary"].parent.iterdir() if p.is_file())
    copied = []
    for source in sorted(retained):
        if source.stat().st_size > 5 * 1024 * 1024:
            raise ValueError(f"Unexpectedly large C1-SF record: {source.name}")
        relative = source.relative_to(extension)
        target = output / "input_records" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append({"path": str(relative), "sha256": sha256(source), "bytes": source.stat().st_size})
    report = {"status": "passed" if not problems else "needs_review", "training_performed": False,
              "final_rows_checked": len(final), "preliminary_rows_checked": len(preliminary),
              "selected_features_checked_against": "Original raw stable_prep order and MI >= 0.05",
              "source_raw_sha256": sha256(raw_path), "copied_inputs": copied,
              "per_variant_C3_matches_within_1e_12": sum(r["per_variant_agrees_within_1e_12"] for r in branches),
              "MI_reference_C3_matches_within_1e_12": sum(r["MI_reference_agrees_within_1e_12"] for r in branches),
              "prediction_identity_claimed": False,
              "environment_boundary": "Final runtime text does not certify every retained RF/MLP fit; review both environment files and row chronology.",
              "uncertainty_boundary": "Replays recorded paired tests/TOST; dependence between repeated splits remains a separate question.",
              "problems": problems}
    write_json(output / "C1SF_audit.json", report)
    print(json.dumps(report, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="A new output directory")
    args = parser.parse_args()
    sys.exit(audit(args.extension_root.expanduser().resolve(), args.output))
