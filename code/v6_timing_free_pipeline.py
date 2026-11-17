
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import revised_pipeline_main_only as core


PROJECT_ROOT = Path(
    "/content/drive/MyDrive/pqc_ids_dissertation"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "final_experiment_package_v6"
    / "01_timing_free"
    / "outputs"
)


# Original A1 exclusions.
UNMODELLABLE_FEATURES = [
    "Fwd IAT Std",
    "Flow IAT Std",
    "Fwd Pkt Len Std",
    "Bwd Pkt Len Std",
    "Bwd IAT Std",
    "Pkt Len Std",
    "Pkt Len Var",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]


# Additional timing and rate exclusions for this experiment.
TIMING_AND_RATE_FEATURES = [
    "Flow Duration",
    "Flow Byts/s",
    "Flow Pkts/s",

    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",

    "Fwd IAT Tot",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",

    "Bwd IAT Tot",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",

    "Fwd Pkts/s",
    "Bwd Pkts/s",

    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",

    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]


EXCLUSION_SET = set(
    UNMODELLABLE_FEATURES
    + TIMING_AND_RATE_FEATURES
)


def ci95(values):
    mean, low, high = core.full.ci95(values)

    return {
        "mean": float(mean),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def class_distribution(y, int_to_class):
    return {
        int_to_class[class_index]: int(
            np.sum(y == class_index)
        )
        for class_index in sorted(int_to_class)
    }


def main(n_splits: int, output_dir: Path) -> None:
    start_time = time.time()

    output_dir.mkdir(parents=True, exist_ok=True)

    if not 1 <= n_splits <= 10:
        raise ValueError(
            "n_splits must be between 1 and 10."
        )

    # Fixed v6 timing-free settings.
    core.MI_THRESHOLDS = [0.0, 0.05]
    core.MI_HEADLINE = 0.05
    core.MI_SUBSAMPLE = 20000
    core.APPLY_MI_SELECTION = True

    variants = {
        core.FPREP: core.VARIANTS[core.FPREP]
    }

    kinds = [
        kind
        for kind, _, _ in core.full.CLASSIFIERS
        if not (
            kind == "xgb"
            and not core.full.HAVE_XGB
        )
    ]

    print("=" * 78)
    print("V6 EXPERIMENT 1: TIMING-FREE C3 CANDIDATE PIPELINE")
    print(f"Splits: {n_splits}")
    print("C1/C2: original full feature space")
    print("C3: timing-free candidate feature space")
    print("MI thresholds: [0.0, 0.05]")
    print("=" * 78)

    data = core.base.load_and_clean(
        core.FILES,
        core.base.LABEL_COL,
        core.base.LABEL_NORMALISE,
        core.base.CLASSES,
        core.PER_CLASS,
        core.MODEL_SEED,
    )

    X, y, full_features, class_to_int = (
        core.base.prepare_xy(
            data,
            core.base.NON_FEATURE_COLS,
            core.base.CLASSES,
        )
    )

    int_to_class = {
        integer: class_name
        for class_name, integer
        in class_to_int.items()
    }

    excluded_features = [
        feature
        for feature in full_features
        if feature in EXCLUSION_SET
    ]

    candidate_features = [
        feature
        for feature in full_features
        if feature not in EXCLUSION_SET
    ]

    requested_but_missing = sorted(
        EXCLUSION_SET - set(full_features)
    )

    if len(full_features) != 70:
        raise RuntimeError(
            "Expected 70 cleaned model features, "
            f"but found {len(full_features)}."
        )

    if len(excluded_features) != 31:
        raise RuntimeError(
            "Expected 31 timing/unmodellable exclusions, "
            f"but found {len(excluded_features)}."
        )

    if len(candidate_features) != 39:
        raise RuntimeError(
            "Expected 39 timing-free candidates, "
            f"but found {len(candidate_features)}."
        )

    print(f"\nRows: {len(X):,}")
    print(f"Full C1/C2 features: {len(full_features)}")
    print(f"Excluded from C3 candidacy: {len(excluded_features)}")
    print(f"Timing-free C3 candidates: {len(candidate_features)}")
    print(f"Requested exclusions absent: {requested_but_missing}")

    pd.DataFrame(
        {"feature": full_features}
    ).to_csv(
        output_dir / "full_feature_list.csv",
        index=False,
    )

    pd.DataFrame(
        {"feature": excluded_features}
    ).to_csv(
        output_dir / "timing_free_excluded_features.csv",
        index=False,
    )

    pd.DataFrame(
        {"feature": candidate_features}
    ).to_csv(
        output_dir / "timing_free_candidate_features.csv",
        index=False,
    )

    raw_output = {
        "config": {
            "analysis": "v6_timing_free_candidate_only",
            "n_splits": n_splits,
            "split_seeds": list(range(n_splits)),
            "model_seed": core.MODEL_SEED,
            "per_class": core.PER_CLASS,
            "test_size": core.TEST_SIZE,
            "holm_alpha": core.HOLM_ALPHA,
            "mi_thresholds": core.MI_THRESHOLDS,
            "mi_headline": core.MI_HEADLINE,
            "mi_subsample": core.MI_SUBSAMPLE,
            "footprint_name": core.FPREP,
            "footprint": variants[core.FPREP],
            "full_feature_count": len(full_features),
            "excluded_feature_count": len(excluded_features),
            "candidate_feature_count": len(candidate_features),
            "full_features": full_features,
            "excluded_features": excluded_features,
            "candidate_features": candidate_features,
            "requested_exclusions_missing": requested_but_missing,
            "class_distribution": class_distribution(
                y,
                int_to_class,
            ),
        },
        "splits": {},
    }

    conditions = ["C1", "C2", "C3"]

    metric_accumulator = {
        kind: {
            condition: {
                "weighted_f1": [],
                "roc_auc_ovr_weighted": [],
                "per_class_f1": {
                    class_name: []
                    for class_name in core.base.CLASSES
                },
            }
            for condition in conditions
        }
        for kind in kinds
    }

    threshold_wf1 = {
        kind: {
            str(threshold): []
            for threshold in core.MI_THRESHOLDS
        }
        for kind in kinds
    }

    threshold_counts = {
        str(threshold): []
        for threshold in core.MI_THRESHOLDS
    }

    stable_counter = Counter()
    selected_counter = Counter()
    mi_values = defaultdict(list)

    raw_path = (
        output_dir
        / f"timing_free_raw_{n_splits}_splits.json"
    )

    for run_number, seed in enumerate(
        range(n_splits),
        start=1,
    ):
        (
            result,
            stable_by_variant,
            shifted_by_variant,
            ks_d_by_variant,
            mi_scores,
        ) = core.one_split(
            seed,
            X,
            y,
            full_features,
            candidate_features,
            int_to_class,
            kinds,
            variants,
        )

        stable_features = list(
            stable_by_variant[core.FPREP]
        )

        shifted_features = sorted(
            shifted_by_variant[core.FPREP]
        )

        selected_features = [
            feature
            for feature in stable_features
            if mi_scores.get(feature, 0.0)
            >= core.MI_HEADLINE
        ]

        if not selected_features:
            selected_features = stable_features[:1]

        stable_counter.update(stable_features)
        selected_counter.update(selected_features)

        for feature, value in mi_scores.items():
            mi_values[feature].append(float(value))

        split_record = {
            "seed": seed,
            "stable_features": stable_features,
            "shifted_features": shifted_features,
            "selected_features_tau_0_05": selected_features,
            "stable_count": len(stable_features),
            "shifted_count": len(shifted_features),
            "selected_count_tau_0_05": len(
                selected_features
            ),
            "mi_scores": {
                feature: float(value)
                for feature, value in mi_scores.items()
            },
            "ks_d": {
                feature: float(value)
                for feature, value
                in ks_d_by_variant[core.FPREP].items()
            },
            "classifiers": {},
        }

        for kind in kinds:
            c1 = result[kind]["C1"]
            c2 = result[kind]["per_variant"][
                core.FPREP
            ]["C2"]
            c3 = result[kind]["per_variant"][
                core.FPREP
            ]["C3"]

            condition_results = {
                "C1": c1,
                "C2": c2,
                "C3": c3,
            }

            split_record["classifiers"][kind] = {
                "C1": c1,
                "C2": c2,
                "C3": c3,
                "mi_sweep_weighted_f1": result[kind][
                    "mi_sweep"
                ],
                "selected_counts": result[kind][
                    "selected_counts"
                ],
            }

            for condition, metrics in (
                condition_results.items()
            ):
                metric_accumulator[kind][condition][
                    "weighted_f1"
                ].append(
                    metrics["weighted_f1"]
                )

                metric_accumulator[kind][condition][
                    "roc_auc_ovr_weighted"
                ].append(
                    metrics["roc_auc_ovr_weighted"]
                )

                for class_name in core.base.CLASSES:
                    metric_accumulator[kind][condition][
                        "per_class_f1"
                    ][class_name].append(
                        metrics["per_class_f1"][
                            class_name
                        ]
                    )

            for threshold in core.MI_THRESHOLDS:
                threshold_key = str(threshold)

                threshold_wf1[kind][threshold_key].append(
                    result[kind]["mi_sweep"][
                        threshold_key
                    ]
                )

        for threshold in core.MI_THRESHOLDS:
            threshold_key = str(threshold)

            # Feature counts are identical across models.
            first_kind = kinds[0]

            threshold_counts[threshold_key].append(
                result[first_kind]["selected_counts"][
                    threshold_key
                ]
            )

        raw_output["splits"][str(seed)] = split_record

        raw_path.write_text(
            json.dumps(raw_output, indent=2),
            encoding="utf-8",
        )

        elapsed = time.time() - start_time

        print(
            f"\nSplit {run_number}/{n_splits} "
            f"| seed={seed} "
            f"| stable={len(stable_features)} "
            f"| shifted={len(shifted_features)} "
            f"| selected={len(selected_features)} "
            f"| elapsed={elapsed:.1f}s"
        )

        for kind in kinds:
            c1_value = result[kind]["C1"][
                "weighted_f1"
            ]

            c2_value = result[kind]["per_variant"][
                core.FPREP
            ]["C2"]["weighted_f1"]

            c3_value = result[kind]["per_variant"][
                core.FPREP
            ]["C3"]["weighted_f1"]

            print(
                f"  {kind:4s}: "
                f"C1={c1_value:.4f} "
                f"C2={c2_value:.4f} "
                f"C3={c3_value:.4f} "
                f"C3-C1={c3_value-c1_value:+.4f}"
            )

    feature_frequency_rows = []

    for feature in candidate_features:
        values = mi_values.get(feature, [])

        feature_frequency_rows.append(
            {
                "feature": feature,
                "stable_splits": stable_counter[feature],
                "selected_splits_tau_0_05": (
                    selected_counter[feature]
                ),
                "mean_mi_when_stable": (
                    float(np.mean(values))
                    if values
                    else np.nan
                ),
            }
        )

    feature_frequency_df = pd.DataFrame(
        feature_frequency_rows
    ).sort_values(
        [
            "selected_splits_tau_0_05",
            "stable_splits",
            "mean_mi_when_stable",
        ],
        ascending=[False, False, False],
    )

    feature_frequency_df.to_csv(
        output_dir
        / "timing_free_feature_frequency.csv",
        index=False,
    )

    summary = {
        "config": raw_output["config"],
        "feature_selection": {
            "stable_count_mean": float(
                np.mean(
                    [
                        split["stable_count"]
                        for split in raw_output[
                            "splits"
                        ].values()
                    ]
                )
            ),
            "shifted_count_mean": float(
                np.mean(
                    [
                        split["shifted_count"]
                        for split in raw_output[
                            "splits"
                        ].values()
                    ]
                )
            ),
            "selected_count_tau_0_05_mean": float(
                np.mean(
                    [
                        split[
                            "selected_count_tau_0_05"
                        ]
                        for split in raw_output[
                            "splits"
                        ].values()
                    ]
                )
            ),
        },
        "classifiers": {},
        "mi_threshold_sweep": {},
        "runtime_seconds": float(
            time.time() - start_time
        ),
    }

    for kind in kinds:
        summary["classifiers"][kind] = {}

        for condition in conditions:
            condition_data = (
                metric_accumulator[kind][condition]
            )

            summary["classifiers"][kind][condition] = {
                "weighted_f1": ci95(
                    condition_data["weighted_f1"]
                ),
                "roc_auc_ovr_weighted": ci95(
                    condition_data[
                        "roc_auc_ovr_weighted"
                    ]
                ),
                "per_class_f1": {
                    class_name: ci95(values)
                    for class_name, values
                    in condition_data[
                        "per_class_f1"
                    ].items()
                },
                "raw_weighted_f1": condition_data[
                    "weighted_f1"
                ],
            }

        c1_values = np.asarray(
            metric_accumulator[kind]["C1"][
                "weighted_f1"
            ],
            dtype=float,
        )

        c2_values = np.asarray(
            metric_accumulator[kind]["C2"][
                "weighted_f1"
            ],
            dtype=float,
        )

        c3_values = np.asarray(
            metric_accumulator[kind]["C3"][
                "weighted_f1"
            ],
            dtype=float,
        )

        summary["classifiers"][kind]["deltas"] = {
            "C2_minus_C1": ci95(
                c2_values - c1_values
            ),
            "C3_minus_C1": ci95(
                c3_values - c1_values
            ),
            "C3_minus_C2": ci95(
                c3_values - c2_values
            ),
        }

        summary["mi_threshold_sweep"][kind] = {}

        for threshold in core.MI_THRESHOLDS:
            key = str(threshold)

            summary["mi_threshold_sweep"][kind][key] = {
                "selected_count": ci95(
                    threshold_counts[key]
                ),
                "weighted_f1": ci95(
                    threshold_wf1[kind][key]
                ),
                "delta_vs_C1": ci95(
                    np.asarray(
                        threshold_wf1[kind][key],
                        dtype=float,
                    )
                    - c1_values
                ),
            }

    summary_path = (
        output_dir
        / f"timing_free_summary_{n_splits}_splits.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("TIMING-FREE RUN COMPLETE")
    print("Raw output:")
    print(raw_path)
    print("Summary:")
    print(summary_path)
    print("Feature frequency:")
    print(
        output_dir
        / "timing_free_feature_frequency.csv"
    )
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-splits",
        type=int,
        default=1,
        help="Number of split seeds to run, from 1 to 10.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    arguments = parser.parse_args()

    main(
        n_splits=arguments.n_splits,
        output_dir=arguments.output_dir,
    )
