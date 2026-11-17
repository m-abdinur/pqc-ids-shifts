# Additional file 1: source map

Internal record (not for submission). Built 24 September 2026 by `build_af1.py`,
which reads every table from the files below through `af1_tables.py`.

| Table | Title | Source | Verification status |
|---|---|---|---|
| S1 | Software versions of the reproduction environment | `environment/main_known_versions.json` | Record file in the package |
| S2 | Statistical families, their role and when they were added | `registry/statistical_family_registry.csv; DECISION_REGISTER.md; verification reports` | Compiled from the family registry, decision register and verification reports |
| S3 | Mutual-information threshold sensitivity (main design, C3 weighted F1) | `reference/main_results/revised_pipeline_raw.json (mi_sweep); main_mi_threshold_split_values.csv` | Recomputed from the locked raw results; matches thesis Table 4.2 |
| S4 | Pipeline component controls: mean weighted F1 (main design) | `component_sources.zip: component_controls_v6/statistics/component_control_split_metrics.csv` | Recomputed from the component-control split metrics; matches thesis Table D.3 (verified in PQC_IDS_Component_Controls_Verification.md) |
| S5 | Pipeline component contrasts (weighted F1, paired over ten splits) | `verified_outputs/component_controls/component_contrasts_replay.csv` | Verified replay |
| S6 | MI-selected subset versus equal-size random stable subsets | `verified_outputs/component_controls/d6_statistics_replay.csv` | Verified replay (D6) |
| S7 | Transformation-footprint sensitivity (weighted F1, mean paired differences) | `reference/main_results/revised_pipeline_raw.json (per_variant); locked_v5/revised_pipeline_main_only.py (VARIANTS)` | Recomputed from the locked raw results; matches thesis Table D.10 |
| S8 | Packet-size sensitivity (weighted F1) | `verified_outputs/packet_size/table_4_13_replay.csv; equivalence_replay.csv` | Verified replay |
| S9 | C2 under corrected transformation operators (weighted F1) | `verified_outputs/operator_sensitivity/c2_cross_arm_summary.csv; verification.json` | Verified replay (operator-sensitivity run) |
| S10 | Transformation exposure by class | `exposure_sources: tls_likely_exposure_v6/outputs/exposure_coverage_by_class.csv` | Source audit file; matches the verified exposure counts |
| S11 | TLS-likely exposure: degradation and recovery (weighted F1) | `verified_outputs/exposure/e2_replay.csv` | Verified replay (E2) |
| S12 | Altered features that passed the screen at TLS-likely exposure | `verified_outputs/exposure/e2_feature_classification.csv; full_exposure_screen_check.csv` | Verified replay (E2 classification and full-exposure check) |
| S13 | Partial adoption: share of transformed test rows (weighted F1) | `verified_outputs/exposure/partial_adoption_replay.csv` | Verified replay; matches thesis Table D.1 |
| S14 | Retraining reference: all 70 features, trained and tested on transformed data | `verified_outputs/retraining/table_4_21_replay.csv; equivalence_replay.csv` | Verified replay; Holm across three models recomputed for the paper's rule |
| S15 | Test rows with an exact copy of their feature vector in the training partition | `verified_outputs/duplicates_sourceday/train_test_match_replay.csv; dup_global_replay.json` | Verified replay (train–test match rates) |
| S16 | Duplicate-aware grouped split (weighted F1) | `verified_outputs/grouped_evaluation/grouped_stats_replay.csv` | Verified replay (grouped run) |
| S17 | Infiltration source-day holdout | `verified_outputs/duplicates_sourceday/crossday_replay.csv` | Verified replay (cross-day) |
| S18 | Per-attack F1 under the feature restrictions (secondary, exploratory ±0.01 comparison) | `verified_outputs/per_attack/per_attack_tost_replay.csv; combined run split values (t410)` | B3 rows: verified replay. Combined rows: new replay from the combined run's split values in this step; all nine match thesis Table D.8 |
| S19 | Identifier-free MLP: MI-threshold sweep (weighted F1) | `thesis Table 4.12 (verified against the V5 sweep raw file in PQC_IDS_Restrictions_Verification.md)` | Thesis Table 4.12, verified against the V5 sweep raw file (PQC_IDS_Restrictions_Verification.md) |
| S20 | Sensitivity to overlapping splits: corrected resampled t statistic (weighted F1) | `verified_outputs/uncertainty/corrected_resampled_results.csv` | Verified replay (uncertainty review) |
| S21 | Error analysis: macro F1 and the direction of errors (main design) | `replayed from operator-sensitivity saved predictions (historical arm) by af1/replay_error_analysis.py` | NEW replay in this step from the operator-sensitivity saved predictions (historical arm). All 27 values match thesis Table 4.20 |
| S22 | Training and inference time: C3 relative to the full-feature model | `thesis Table D.13 (source: statistical_results/efficiency/efficiency_revised_statistics_v1.csv in the original package; not replayed here)` | Thesis Table D.13. NOT replayed: the timing records are not in this working copy. Source in the original package: statistical_results/efficiency/efficiency_revised_statistics_v1.csv |

| S23 | Unanticipated timing shift: the paper's C3 when the screen did not see the delay | `verified_outputs/timing_shift_colab/results/analysis/timing_shift_statistics.csv` | Colab run 20260926T080230Z_e9f1ad6e (primary record); replayed and re-analysed independently; workspace run agrees on every ±0.01 decision |
| S24 | Anticipated timing shift: screen run against a copy that includes the delay | same, plus `anticipated_selection_summary.csv` | As S23 |

## Hand-written numbers in the running text

All other running-text numbers are computed from the same sources at build time. The ones typed by hand are:

- 58,242 / 74,127 (78.57%) and 17,354: Section 3 of the manuscript and `PQC_IDS_Operator_Sensitivity_Verification.md`.
- 53.6% DoS and 48.9% Bot duplicate shares; 1,745 Infiltration and 1,395 Benign label-conflict rows: duplicate audit (`duplicate_summary_by_class.csv`, `conflicting_label_duplicate_groups.csv`), checked in this step.
- 28.5% of test rows with an exact training copy: Table S15 source.
- 89,064 groups: `dup_global_replay.json`.
- 2.7 × 10⁻⁵ (RF split 6): locked raw results, 0.8511709277 − 0.8511436987.
- 47 features, 13–14 and 6–13 shifted features: component-control subset audit (verification report).
- Mean C3 under the operators (0.8505 / 0.8590 / 0.8415): operator verification report.

## Correction found while building

The duplicate share is 36,527 / 119,571 = 30.55%, which rounds to **30.5%**, not 30.6%.
The manuscript v2 (Section 4.1) is corrected as a tracked change. The earlier verification reports carry a correction note.

## Not in the working copy

- The operator-sensitivity saved predictions (about 125 MB) are needed to rerun `replay_error_analysis.py`. Their hashes are recorded in the operator-sensitivity verification.
- The efficiency timing records (Table S22).
