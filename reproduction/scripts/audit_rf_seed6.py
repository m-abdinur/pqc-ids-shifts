"""Independently audit the returned seed-6 RF diagnostic without refitting.

Recompute metrics from saved labels, verify exported probabilities and replay
tree summation at the record whose prediction changed. Historical forests
were not exported by the main run, so its precise execution is not recreated.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from common import ROOT, sha256, write_csv, write_json
from seed0_reproduction import array_hash, text_hash


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def audit(source, output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source / "result_manifest.json").read_text())
    for entry in manifest["files"]:
        path = (source / entry["path"]).resolve()
        check(path.is_relative_to(source.resolve()), "Unsafe manifest path")
        check(path.stat().st_size == entry["bytes"], "Size mismatch: " + entry["path"])
        check(sha256(path) == entry["sha256"], "Hash mismatch: " + entry["path"])
    listed = {e["path"] for e in manifest["files"]}
    actual = {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()}
    check(actual == listed | {"result_manifest.json"}, "Manifest coverage differs")
    result = source / "results"
    status = json.loads((result / "diagnostic_status.json").read_text())
    notebook = json.loads((source / "notebook_status.json").read_text())
    check(status["status"] == notebook["status"] == "diagnostic_completed", "Incomplete run")
    check(status["fits_completed"] == 2 and status["prediction_calls_completed"] == 54,
          "Unexpected fit/call counts")
    check(notebook["package_sha256"] == "9232cf27b45833c1ac1c77e73abdbfaa48b74fd73c38bb95dd31e2bde3f327ac",
          "Unexpected supplied package")
    check(sha256(source / "working_copy_manifest.json") == notebook["package_manifest_sha256"],
          "Working manifest identity differs")
    runtime = json.loads((result / "runtime_environment.json").read_text())
    target = json.loads((source / "main_known_versions.json").read_text())
    check(runtime["python"] == target["python"] and runtime["packages"] == target["packages"],
          "Declared package versions differ")
    inputs = json.loads((result / "input_files.json").read_text())
    reference_inputs = json.loads((ROOT / "reference/seed0_reproduction_20260920/results/input_files.json").read_text())
    for item in inputs:
        reference = next(r for r in reference_inputs if r["filename"] == item["filename"])
        check(all(item[k] == reference[k] for k in ("sha256", "bytes")), "Input identity mismatch")

    data = np.load(result / "diagnostic_predictions.npz", allow_pickle=False)
    classes, truth, predictions = data["classes"], data["y_true"], data["predictions"]
    check(predictions.shape == (54, 35872), "Unexpected prediction shape")
    check(np.array_equal(classes, [0, 1, 2, 3]), "Class encoding differs")
    check(text_hash(data["test_index"]) == status["data_and_selection"]["test_index_sha256"],
          "Test-index signature mismatch")
    check(text_hash(data["selected_features"]) == status["data_and_selection"]["selected_feature_order_sha256"],
          "Selected-feature signature mismatch")
    rows = read_rows(result / "prediction_calls.csv")
    check(len(rows) == 54 and [int(r["call"]) for r in rows] == list(range(54)), "Call sequence differs")
    names = ["Benign", "DoS", "Bot", "Infiltration"]
    rebuilt = []
    for i, row in enumerate(rows):
        weighted = float(f1_score(truth, predictions[i], average="weighted"))
        per_class = f1_score(truth, predictions[i], average=None, labels=classes)
        check(weighted == float(row["weighted_f1"]), "Weighted F1 differs at call " + str(i))
        check(all(float(score) == float(row[name + "_f1"]) for name, score in zip(names, per_class)),
              "Per-class F1 differs at call " + str(i))
        check(array_hash(predictions[i]) == row["prediction_sha256"], "Prediction hash differs")
        rebuilt.append({"call": i, "fit": int(row["fit"]), "mode": row["mode"],
                        "weighted_f1": weighted, **{name + "_f1": float(s) for name, s in zip(names, per_class)}})
    write_csv(output / "recalculated_call_metrics.csv", rebuilt)

    relevant = data["relevant_positions"]
    probability = data["relevant_probabilities"]
    trees = data["tree_probabilities_for_relevant_rows"]
    check(probability.shape == (54, 77, 4) and trees.shape == (2, 200, 77, 4), "Unexpected probability shapes")
    check(np.isfinite(probability).all() and np.isfinite(trees).all(), "Non-finite probabilities")
    check(np.allclose(probability.sum(axis=2), 1, rtol=0, atol=1e-14), "Invalid probability sums")
    check(np.array_equal(classes.take(probability.argmax(axis=2)), predictions[:, relevant]),
          "Saved predictions disagree with their probabilities")
    check(np.array_equal(data["relevant_test_index"], data["test_index"][relevant]), "Relevant index differs")
    check(np.array_equal(trees[0], trees[1]), "Exported per-tree contributions differ between fits")

    signatures = json.loads((result / "forest_signatures.json").read_text())
    check(len(signatures) == 2, "Expected two forest signatures")
    for item in signatures:
        check(len(item["tree_sha256"]) == 200, "Expected 200 trees")
        check(text_hash(item["tree_sha256"]) == item["forest_sha256"], "Forest digest differs")
    check(signatures[0]["tree_sha256"] == signatures[1]["tree_sha256"], "Recorded tree hashes differ")

    for fit in range(2):
        for mode, order in (("forward_sum", range(200)), ("reverse_sum", range(199, -1, -1))):
            summed = np.zeros((len(relevant), 4))
            for tree in order:
                summed += trees[fit, tree]
            summed /= 200
            call = next(i for i, r in enumerate(rows) if int(r["fit"]) == fit and r["mode"] == mode)
            check(np.array_equal(summed, probability[call]), "Saved tree-summation result differs")

    representative_records = []
    for path in sorted(result.glob("representative_call_*.npz")):
        i = int(path.stem.rsplit("_", 1)[1])
        representative = np.load(path, allow_pickle=False)
        proba = representative["probabilities"]
        check(np.array_equal(representative["y_true"], truth), "Representative truth differs")
        check(np.array_equal(representative["predictions"], predictions[i]), "Representative labels differ")
        check(array_hash(proba) == rows[i]["probability_sha256"], "Full probability hash differs")
        check(np.array_equal(classes.take(proba.argmax(axis=1)), predictions[i]), "Full argmax differs")
        check(np.array_equal(proba[relevant], probability[i]), "Relevant/full probability mismatch")
        auc = float(roc_auc_score(truth, proba, average="weighted", multi_class="ovr", labels=classes))
        check(auc == float(rows[i]["roc_auc"]), "Representative ROC-AUC differs")
        representative_records.append({"call": i, "weighted_f1": rebuilt[i]["weighted_f1"],
                                       "roc_auc": auc, "confusion_matrix": confusion_matrix(truth, predictions[i], labels=classes).tolist()})

    changed = np.flatnonzero(np.any(predictions != predictions[0], axis=0))
    check(changed.tolist() == [19813], "Changed-record finding differs")
    position = int(changed[0])
    relevant_offset = int(np.flatnonzero(relevant == position)[0])
    serial = [i for i, r in enumerate(rows) if r["mode"].startswith("serial_")]
    check(len(serial) == 8 and all(np.array_equal(predictions[i], predictions[serial[0]]) for i in serial),
          "Single-worker predictions differ")
    check(len({rows[i]["probability_sha256"] for i in serial}) == 1, "Single-worker probability hashes differ")
    native = [i for i, r in enumerate(rows) if r["mode"] == "native_repeat"]
    check(len(native) == 40, "Expected 40 native repetitions")
    counts = {str(fit): {name: int(sum(predictions[i, position] == c for i in native if int(rows[i]["fit"]) == fit))
                        for c, name in ((0, "Benign"), (3, "Infiltration"))} for fit in range(2)}

    order_rows = []
    # A retrospective arithmetic check on the same exported tree contributions.
    # These twenty fixed order seeds do not fit models or tune any paper result.
    for seed in range(20):
        proba = np.zeros(4)
        for tree in np.random.RandomState(seed).permutation(200):
            proba += trees[0, tree, relevant_offset]
        proba /= 200
        order_rows.append({"order_seed": seed, "Benign_probability": float(proba[0]),
                           "Infiltration_probability": float(proba[3]),
                           "predicted_class": names[int(proba.argmax())]})
    check({r["predicted_class"] for r in order_rows} == {"Benign", "Infiltration"},
          "Order-only replay did not reproduce both classes")
    write_csv(output / "changed_record_order_replay.csv", order_rows)
    write_csv(output / "changed_record_by_call.csv", [{"call": i, "fit": rows[i]["fit"], "mode": rows[i]["mode"],
              "predicted_class": names[int(predictions[i, position])], "Benign_probability": float(probability[i, relevant_offset, 0]),
              "Infiltration_probability": float(probability[i, relevant_offset, 3]),
              "Infiltration_minus_Benign": float(probability[i, relevant_offset, 3] - probability[i, relevant_offset, 0])}
              for i in range(54)])

    old = json.loads((ROOT / "reference/main_results/revised_pipeline_raw.json").read_text())["6"]["results"]["rf"]
    new = json.loads((ROOT / "reference/main_reproduction_20260920/results/revised_pipeline_raw.json").read_text())["6"]["results"]["rf"]
    observed = {r["weighted_f1"] for r in rebuilt}
    check(observed == {old["mi_sweep"]["0.05"], new["mi_sweep"]["0.05"]}, "F1 pair differs from main records")
    for record in (old, new):
        expected = record["per_variant"]["hybrid768_p2"]["C3"]
        check(any(r["weighted_f1"] == expected["weighted_f1"] and
                  all(r[name + "_f1"] == expected["per_class_f1"][name] for name in names)
                  for r in rebuilt), "Two-packet F1 vector was not reproduced")
    matrices = read_rows(result / "selected_matrix_comparison.csv")
    check(len(matrices) == 5 and len({r["sha256"] for r in matrices}) == 1 and
          all(r["equal_to_original"] == "True" and int(r["changed_elements_vs_original"]) == 0 for r in matrices),
          "Selected-matrix report differs")
    report = {"status": "verified_diagnostic", "scope": "Saved-array audit; no classifier refitted locally",
              "run_id": notebook["run_id"], "manifest_files_verified": len(manifest["files"]),
              "weighted_f1_values_recalculated_exactly": 54, "per_class_f1_values_recalculated_exactly": 216,
              "prediction_hashes_verified": 54, "representative_full_probability_arrays_verified": len(representative_records),
              "recorded_tree_signatures_match": True, "exported_per_tree_probabilities_identical": True,
              "matrix_identity_scope": "All five selected-matrix hashes match in returned report; full matrices not exported",
              "changed_test_positions": changed.tolist(), "changed_prepared_row_indices": data["test_index"][changed].tolist(),
              "changed_record_true_class": names[int(truth[position])], "native_repetition_counts_by_fit": counts,
              "serial_calls": len(serial), "serial_predictions_and_probability_hashes_identical": True,
              "serial_weighted_f1": rebuilt[serial[0]]["weighted_f1"], "observed_weighted_f1_pair": sorted(observed),
              "near_tie_records_exported": len(relevant), "order_replay_trials": len(order_rows),
              "order_replay_class_counts": {name: sum(r["predicted_class"] == name for r in order_rows) for name in ("Benign", "Infiltration")},
              "all_four_main_F1_exceptions_reproduced_by_diagnostic": True,
              "representatives": representative_records, "original_main_comparison_status": "needs_review",
              "conclusion": "Parallel prediction variability and order-sensitive accumulation reproduced the exact F1 pair on a near-tied record.",
              "historical_limit": "Original main forests and per-fit predictions were not retained; the exact historical scheduling/record change cannot be inspected."}
    write_json(output / "verification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.output), indent=2))
