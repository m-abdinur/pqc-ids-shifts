"""Rebuild the B3 weighted-F1 table from saved results; no model fitting."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from common import (ROOT, MODELS, compare_number, holm, new_output, paired_summary,
                    read_csv, sha256, tost, verify_reference_code, write_csv, write_json)


def rebuild(output, raw_path=None):
    reference = ROOT / "reference/b3_results"
    raw_path = Path(raw_path) if raw_path else reference / "revised_pipeline_b3_raw.json"
    decision_path = ROOT / "reference/main_results/weighted_f1_recovery_decision_table_v1.csv"
    raw = json.loads(raw_path.read_text())
    if set(raw) != {str(i) for i in range(10)}:
        raise ValueError("Expected the ten original split seeds 0–9.")
    sources_checked = verify_reference_code()
    decisions = {r["model"]: r for r in read_csv(decision_path) if r["design"] == "B3"}
    if set(decisions) != set(MODELS):
        raise ValueError("Expected one B3 decision-table row per model.")
    rows, branches, split_values = [], [], []
    for model in MODELS:
        c1, c2, c3 = [], [], []
        for seed in range(10):
            block = raw[str(seed)]["results"][model]
            a = block["C1"]["weighted_f1"]
            b = block["per_variant"]["hybrid768"]["C2"]["weighted_f1"]
            c = block["per_variant"]["hybrid768"]["C3"]["weighted_f1"]
            sweep = block["mi_sweep"]["0.05"]
            c1.append(a); c2.append(b); c3.append(c)
            split_values.append({"seed": seed, "model": model, "C1": a, "C2": b, "C3": c,
                                 "C3_source": "per_variant.hybrid768.C3.weighted_f1"})
            branches.append({"seed": seed, "model": model, "per_variant_C3": c,
                             "MI_sweep_C3": sweep, "sweep_minus_per_variant": sweep - c,
                             "agree_within_1e_12": abs(sweep - c) <= 1e-12})
        a, b, c = map(np.asarray, (c1, c2, c3))
        degradation, recovery = paired_summary(b - a), paired_summary(c - a)
        row = {"model": model, "mean_C1": float(a.mean()), "mean_C2": float(b.mean()),
               "mean_C3": float(c.mean()), "mean_C3_minus_C2": float((c-b).mean()),
               "positive_C3_minus_C2_splits": int(np.sum(c > b)),
               "minimum_C3_minus_C2": float((c-b).min()),
               "maximum_C3_minus_C2": float((c-b).max()),
               "C2_minus_C1": degradation["mean_difference"],
               "C2_minus_C1_ci95_lower": degradation["ci95_lower"],
               "C2_minus_C1_ci95_upper": degradation["ci95_upper"],
               "degradation_raw_p": degradation["selected_p"],
               "degradation_cohen_dz": degradation["cohen_dz"],
               "C3_minus_C1": recovery["mean_difference"],
               "C3_minus_C1_ci95_lower": recovery["ci95_lower"],
               "C3_minus_C1_ci95_upper": recovery["ci95_upper"],
               "recovery_difference_raw_p": recovery["selected_p"],
               "recovery_difference_cohen_dz": recovery["cohen_dz"]}
        for tag, margin in (("001", .01), ("0005", .005)):
            row[f"tost_{tag}_ci90_lower"] = recovery["ci90_lower"]
            row[f"tost_{tag}_ci90_upper"] = recovery["ci90_upper"]
            row[f"tost_{tag}_raw_p"] = tost(recovery, margin)["raw_p"]
        rows.append(row)
    for stem in ("degradation", "recovery_difference", "tost_001", "tost_0005"):
        for row, adjusted in zip(rows, holm([r[f"{stem}_raw_p"] for r in rows])):
            row[f"{stem}_holm_p"] = float(adjusted)
    mismatches, count = [], 0
    for row in rows:
        for key, value in row.items():
            if key != "model":
                compare_number(value, decisions[row["model"]][key], f"{row['model']}.{key}", mismatches)
                count += 1
        for tag in ("001", "0005"):
            row[f"equivalent_{tag}"] = row[f"tost_{tag}_holm_p"] < .05
            if row[f"equivalent_{tag}"] != (decisions[row["model"]][f"equivalent_{tag}"] == "True"):
                mismatches.append({"field": f"{row['model']}.equivalent_{tag}"})
    output = new_output(output)
    write_csv(output / "b3_weighted_f1_rebuilt.csv", rows)
    write_csv(output / "b3_weighted_f1_by_split.csv", split_values)
    write_csv(output / "C3_source_branch_comparison.csv", branches)
    trace = {
        "status": "passed" if not mismatches else "mismatches_found",
        "training_performed": False, "reference_code_files_verified": sources_checked,
        "C3_source": "per_variant.hybrid768.C3.weighted_f1",
        "b3_decision_rows_checked": 3, "numeric_decision_cells_checked": count,
        "equivalence_decisions_checked": 6, "MI_thresholds_in_B3": [0.05],
        "C3_branch_differences_over_1e_12": [r for r in branches if not r["agree_within_1e_12"]],
        "decision_comparison_tolerance": {"absolute": 1e-10, "relative": 1e-8},
        "holm_families": "Three models separately for degradation, recovery, and each TOST margin",
        "statistical_scope": "Replay of the stored procedure; no correction for overlapping splits",
        "excluded_from_verification": ["Main decision rows (separate experiment)",
                                       "C1-SF CSV and preliminary checkpoint", "per-attack equivalence tables"],
        "input_sha256": {p.name: sha256(p) for p in (raw_path, decision_path)},
        "mismatches": mismatches,
    }
    write_json(output / "verification.json", trace)
    print(json.dumps(trace, indent=2))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="A new output directory")
    parser.add_argument("--raw", type=Path, help="Optional new B3 raw JSON to compare with the locked tables")
    args = parser.parse_args()
    sys.exit(rebuild(args.output, args.raw))
