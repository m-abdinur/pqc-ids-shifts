"""Audit the archived feature transformation on the verified prepared data.

No classifier is fitted. These algebraic diagnostics do not validate a TLS
handshake, packet trace, or the extraction semantics of every CICFlowMeter field.
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
from run_experiment import SOURCES
from seed0_reproduction import array_hash, check_environment, now


ALLOWED_CHANGES = {
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Byts",
    "Tot Fwd Pkts", "Tot Bwd Pkts", "Subflow Fwd Pkts", "Subflow Bwd Pkts",
    "Fwd Header Len", "Bwd Header Len", "Fwd Pkt Len Mean", "Fwd Seg Size Avg",
    "Bwd Pkt Len Mean", "Bwd Seg Size Avg", "Pkt Len Mean", "Pkt Size Avg",
    "Fwd Pkt Len Max", "Bwd Pkt Len Max", "Pkt Len Max", "Flow Byts/s",
    "Flow Pkts/s", "Fwd Pkts/s", "Bwd Pkts/s",
}


def rules(frame, atol, rtol):
    """Return each rule's evaluable mask, violation mask and compared values.

    Max/total and mean bounds assume nonnegative packet lengths measured on the
    same byte basis. Directional mean/total equality is explicitly a semantic
    diagnostic: a mismatch also requires review of the extractor convention.
    Positive-rate rules require positive duration but assume no duration units.
    """
    def col(name):
        return frame[name].to_numpy(dtype=np.float64)

    values = {}

    def put(name, left, right, valid, relation, category):
        valid = valid & np.isfinite(left) & np.isfinite(right)
        tolerance = atol + rtol * np.maximum(np.abs(left), np.abs(right))
        if relation == "le":
            bad = left > right + tolerance
        elif relation == "equal":
            bad = np.abs(left - right) > tolerance
        else:
            bad = left <= 0  # A positive numerator and duration require a positive rate.
        values[name] = {"evaluable": valid, "violated": valid & bad,
                        "left": left, "right": right, "category": category}

    totals = col("TotLen Fwd Pkts") + col("TotLen Bwd Pkts")
    counts = col("Tot Fwd Pkts") + col("Tot Bwd Pkts")
    duration = col("Flow Duration")
    for tag, prefix in (("fwd", "Fwd"), ("bwd", "Bwd"), ("flow", "")):
        length = (prefix + " " if prefix else "") + "Pkt Len "
        count = col("Tot " + prefix + " Pkts") if prefix else counts
        total = col("TotLen " + prefix + " Pkts") if prefix else totals
        maximum, minimum, mean = (col(length + suffix) for suffix in ("Max", "Min", "Mean"))
        valid = np.isfinite(count) & (count > 0)
        put(tag + "_max_le_total", maximum, total,
            valid & (maximum >= 0) & (total >= 0), "le", "packet_length_bound")
        put(tag + "_min_le_mean", minimum, mean,
            valid & (minimum >= 0) & (mean >= 0), "le", "packet_length_bound")
        put(tag + "_mean_le_max", mean, maximum,
            valid & (mean >= 0) & (maximum >= 0), "le", "packet_length_bound")
        if prefix:
            ratio = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
            put(tag + "_mean_equals_total_over_count", mean, ratio,
                valid & np.isfinite(total) & (total >= 0) & (mean >= 0),
                "equal", "extractor_semantics_diagnostic")
    for tag, rate, numerator in (
        ("flow_bytes", "Flow Byts/s", totals), ("flow_packets", "Flow Pkts/s", counts),
        ("fwd_packets", "Fwd Pkts/s", col("Tot Fwd Pkts")),
        ("bwd_packets", "Bwd Pkts/s", col("Tot Bwd Pkts")),
    ):
        put(tag + "_positive_rate", col(rate), np.zeros(len(frame)),
            np.isfinite(numerator) & (numerator > 0) & np.isfinite(duration) & (duration > 0),
            "positive", "rate_sign")
    return values


def transitions(before, after, mask):
    bv, av = before["evaluable"], after["evaluable"]
    bb, ab = before["violated"], after["violated"]
    n = lambda condition: int(np.sum(mask & condition))
    return {
        "rows": int(mask.sum()), "before_evaluable": n(bv), "after_evaluable": n(av),
        "both_evaluable": n(bv & av), "before_violations": n(bb), "after_violations": n(ab),
        "persistent_violations": n(bb & ab), "introduced_violations": n(bv & av & ~bb & ab),
        "resolved_violations": n(bv & av & bb & ~ab),
        "after_violations_previously_unevaluable": n(~bv & ab),
        "before_violations_now_unevaluable": n(bb & ~av),
    }


def audit_frames(X, y, class_map, variants, operator, output, plan):
    """Count transitions and save compact row masks for independent recounting."""
    baseline_hash = array_hash(X.values)
    eligible = ((X["Tot Fwd Pkts"] > 0) & (X["Tot Bwd Pkts"] > 0)).to_numpy()
    strata = [("all", np.ones(len(X), dtype=bool))]
    strata += [(name, np.asarray(y) == code) for name, code in class_map.items()]
    eligibility = [("all", np.ones(len(X), dtype=bool)), ("eligible", eligible), ("ineligible", ~eligible)]
    tolerances = [("primary", plan["absolute_tolerance"], plan["relative_tolerance"]),
                  ("strict_sensitivity", 0.0, 0.0)]
    before = {tag: rules(X, a, r) for tag, a, r in tolerances}
    names = list(before["primary"])
    write_json(output / "rule_catalog.json", {
        "rule_order_for_bitmasks": names,
        "categories": {name: before["primary"][name]["category"] for name in names},
        "bit_encoding": "Rule i is bit (1 << i); row order is the verified prepared matrix order.",
        "tolerances": tolerances, "plan": plan,
    })
    rows, examples, branch_rows, integrity = [], [], [], []
    label_names = {int(code): name for name, code in class_map.items()}
    row_masks = {"row_position": np.arange(len(X)), "prepared_index": np.asarray(X.index),
                 "label": np.asarray(y), "eligible": eligible}
    for variant, cfg in variants.items():
        print("Auditing footprint:", variant, flush=True)
        shifted = operator(X, cfg)
        if list(shifted.columns) != list(X.columns) or not shifted.index.equals(X.index):
            raise ValueError("The operator changed row or feature order.")
        changed = X.to_numpy() != shifted.to_numpy()
        untouched = [c for c in X if c not in ALLOWED_CHANGES]
        gate = {
            "variant": variant, "rows": len(X), "eligible_rows": int(eligible.sum()),
            "changed_rows": int(changed.any(axis=1).sum()),
            "changed_cells": int(changed.sum()),
            "ineligible_changed_cells": int(changed[~eligible].sum()),
            "untouched_feature_changed_cells": int(np.sum(X[untouched].to_numpy() != shifted[untouched].to_numpy())),
            "nonfinite_after_cells": int(np.sum(~np.isfinite(shifted.to_numpy()))),
            "input_matrix_unchanged": array_hash(X.values) == baseline_hash,
            "shifted_matrix_sha256": array_hash(shifted.values),
        }
        integrity.append(gate)
        if (gate["ineligible_changed_cells"] or gate["untouched_feature_changed_cells"]
                or gate["nonfinite_after_cells"] or not gate["input_matrix_unchanged"]):
            write_json(output / "operator_integrity.json", integrity)
            raise ValueError("Operator preservation gate failed: " + variant)
        after = {tag: rules(shifted, a, r) for tag, a, r in tolerances}
        for tag, _, _ in tolerances:
            for stage, table in (("before", before[tag]), ("after", after[tag])):
                for field in ("evaluable", "violated"):
                    bits = np.zeros(len(X), dtype=np.uint32)
                    for i, name in enumerate(names):
                        bits |= table[name][field].astype(np.uint32) << i
                    row_masks[f"{variant}__{tag}__{stage}__{field}"] = bits
            for name in names:
                b, a = before[tag][name], after[tag][name]
                for class_name, class_mask in strata:
                    for eligibility_name, e_mask in eligibility:
                        rows.append({"variant": variant, "tolerance": tag, "rule": name,
                                     "category": a["category"], "class": class_name,
                                     "eligibility": eligibility_name,
                                     **transitions(b, a, class_mask & e_mask)})
                if tag == "primary":
                    groups = {"introduced": b["evaluable"] & a["violated"] & ~b["violated"],
                              "persistent": b["violated"] & a["violated"],
                              "resolved": b["violated"] & a["evaluable"] & ~a["violated"],
                              "after_previously_unevaluable": ~b["evaluable"] & a["violated"]}
                    for transition, mask in groups.items():
                        for index in np.flatnonzero(mask)[:plan["examples_per_rule_transition"]]:
                            examples.append({"variant": variant, "rule": name, "transition": transition,
                                "row_position": int(index), "prepared_index": int(X.index[index]),
                                "class": label_names[int(np.asarray(y)[index])], "eligible": bool(eligible[index]),
                                "before_left": float(b["left"][index]), "before_right": float(b["right"][index]),
                                "after_left": float(a["left"][index]), "after_right": float(a["right"][index])})
        old_total = X["TotLen Fwd Pkts"].to_numpy(dtype="float64") + X["TotLen Bwd Pkts"].to_numpy(dtype="float64")
        new_total = shifted["TotLen Fwd Pkts"].to_numpy(dtype="float64") + shifted["TotLen Bwd Pkts"].to_numpy(dtype="float64")
        zero_branch = eligible & (old_total == 0)
        for class_name, class_mask in strata:
            n = lambda condition: int(np.sum(class_mask & condition))
            branch_rows.append({"variant": variant, "class": class_name,
                "eligible_original_zero_byte_rows": n(zero_branch),
                "positive_added_total_and_unchanged_rate": n(zero_branch & (new_total > 0) & (shifted["Flow Byts/s"].to_numpy() == X["Flow Byts/s"].to_numpy())),
                "positive_duration_and_zero_rate_after": n(zero_branch & (X["Flow Duration"].to_numpy() > 0) & (shifted["Flow Byts/s"].to_numpy() == 0)),
                "nonpositive_duration": n(zero_branch & (X["Flow Duration"].to_numpy() <= 0))})
    write_csv(output / "rule_counts.csv", rows)
    if examples:
        write_csv(output / "rule_examples.csv", examples)
    write_json(output / "example_status.json", {"examples_saved": len(examples), "selection": "First prepared row positions per rule/variant/transition; no outcome-based choice"})
    write_csv(output / "zero_byte_branch.csv", branch_rows)
    write_json(output / "operator_integrity.json", integrity)
    np.savez_compressed(output / "row_rule_masks.npz", **row_masks)
    overview = [r for r in rows if r["tolerance"] == "primary" and r["class"] == "all" and r["eligibility"] == "all"]
    write_json(output / "audit_overview.json", {"rules": len(names), "variants": len(variants),
        "counts_are_per_rule_not_unique_rows_across_rules": True, "primary_all_rows": overview})
    return {"rules": len(names), "variants": len(variants), "count_rows": len(rows),
            "eligible_rows": int(eligible.sum()), "ineligible_rows": int((~eligible).sum()),
            "class_counts": {name: int(mask.sum()) for name, mask in strata if name != "all"}}


def run(args):
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = {"started_utc": now(), "status": "checking", "classifier_fits": 0,
              "historical_results_modified": False, "physical_validity_established": False}
    old_filters, old_warning = list(warnings.filters), warnings.showwarning
    with (output / "warnings.log").open("w") as warning_file:
        def record_warning(message, category, filename, lineno, file=None, line=None):
            warning_file.write(warnings.formatwarning(message, category, filename, lineno, line))
            warning_file.flush()
            old_warning(message, category, filename, lineno, file=file, line=line)
        try:
            write_json(output / "audit_status.json", status)
            plan_path = ROOT / "environment/feature_consistency_plan.json"
            plan = json.loads(plan_path.read_text())
            status["plan_sha256"] = sha256(plan_path)
            environment = check_environment(json.loads((ROOT / "environment/main_known_versions.json").read_text()))
            write_json(output / "runtime_environment.json", environment)
            if environment["problems"]:
                raise ValueError("Environment mismatch: " + "; ".join(environment["problems"]))
            status["reference_sources_verified"] = verify_reference_code()
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
            if pipeline.VARIANTS != plan["footprints"]:
                raise ValueError("Archived footprints differ from the fixed audit plan.")
            status["prepared_data"] = {"rows": len(X), "features": features, "X_shape": list(X.shape),
                "X_dtype": str(X.values.dtype), "X_sha256": array_hash(X.values), "y_sha256": array_hash(y),
                "class_encoding": class_map}
            write_json(output / "audit_status.json", status)
            status["counts"] = audit_frames(X, y, class_map, pipeline.VARIANTS,
                                            pipeline.ob.make_approx_mlkem, output, plan)
            status["status"] = "audit_completed"
            print("Audit completed. Classifier fits: 0. Findings require interpretation.", flush=True)
        except Exception as error:
            status["status"] = "audit_failed"
            status["error"] = str(error)
            (output / "traceback.txt").write_text(traceback.format_exc())
            traceback.print_exc()
        finally:
            warnings.filters[:] = old_filters
            warnings.showwarning = old_warning
            status["finished_utc"] = now()
            write_json(output / "audit_status.json", status)
    return 0 if status["status"] == "audit_completed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    sys.exit(run(parser.parse_args()))
