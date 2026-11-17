# PQC-IDS shift: code and results

Code, run records and derived results for:

> Mohamed Abdullahi Abdinur and Vaithegy Doraisamy. *What drives intrusion detection loss under a simulated ML-KEM-768 flow-feature shift? A stress test of stable-feature screening and training augmentation.* Submitted to *Frontiers of Computer Science*, 2026.

The study edits stored CSE-CIC-IDS2018 flow summaries to simulate the larger TLS handshakes of ML-KEM-768. It does not generate packets. It then tests whether two defences prepared on training data, stable-feature screening (C3) and training augmentation (A), restore weighted F1 for Random Forest (RF), XGBoost and a multilayer perceptron (MLP).

> **Two files are not in this release.** `calibration/decisions_log.md` (the October 2026 rules fixed before each run) and `reproduction/PQC_IDS_Reproduction_Working_Copy.zip` (the September 2026 working copy). Other files refer to them: the `calibration/` scripts and result summaries cite the log, and the September notebooks and `reproduction/check_split0.py` need the ZIP, so they cannot run from this repository alone. Where a table below points to one of these files, it is marked "not here yet". The run A script in `code/`, the October notebooks and the Table 6 scripts do not need either file.

## Repository map

| Path | What it holds |
|---|---|
| `code/` | Run A's `04_code/` (11 files, unchanged) and `v6_timing_free_pipeline.py` from run B (`01_timing_free/code/`) |
| `notebooks/` | Google Colab notebooks. The October 2026 notebooks (`2026-10_*.ipynb`) are saved with their outputs. The September 2026 notebooks (`PQC_IDS_*.ipynb`) were unpacked from the working copy ZIP and have no outputs, except `PQC_IDS_Operator_Sensitivity.ipynb` (see "Notebooks") |
| `reproduction/` | `PQC_IDS_Reproduction_Working_Copy.zip`, the September 2026 working copy (the September notebooks ask for it to be uploaded; not here yet). Its `scripts/`, `environment/` and `manuscript/` folders are unpacked next to it for reading only. `check_split0.py` reruns split 0 of the main design and compares it with run A (see "How to run") |
| `calibration/` | The October 2026 analyses as they ran: `decisions_log.md` (rules fixed before each run; not here yet), the Colab scripts (`my_c2_code/`), the Table 6 scripts (`scripts/`) and a summary of each run (`results/`) |
| `figures/code/` | The Fig. 1 script (`fig1_overview.py`), its style module, fonts and two input CSV files |
| `figures/tableau/` | One Tableau workbook per figure for Figs. 2 to 7, and the Excel files they read |
| `results/A_2026-07-02/` | Run A package ZIP, its `.sha256` file and its per-file manifest |
| `results/C_2026-09/` | `PQC_IDS_Main_Reproduction_Results.zip` (run C rerun of the main design) |
| `data/` | `SHA256SUMS` for the four CSE-CIC-IDS2018 input files. The data themselves are not included |
| `requirements.txt` | The pinned Python 3.12.13 environment (Supplementary Table S15) |
| `LICENSE` | MIT licence for the code |

**Not in this repository yet:**

- The run B results (the folder `results/B_2026-07/`).
- The other run C archives: `PQC_IDS_Augmentation_Results.zip`, `PQC_IDS_Timing_Shift_Results.zip`, `PQC_IDS_Operator_Sensitivity_Results.zip` and the other September results ZIPs.
- The run D output folders (`pqc_oav_calibration_<name>/`, written to Google Drive by the October notebooks). Their summaries are in `calibration/results/`.
- The figure image files (PDF, SVG, PNG, TIFF) and `figure_legends.md`.
- `calibration/decisions_log.md` and `reproduction/PQC_IDS_Reproduction_Working_Copy.zip` (see the note at the top).

Some table rows below point to these items. They are marked "not here yet".

## Runs

The results come from four groups of runs. Supplementary Table S16 assigns every analysis to one of them.

| Run | Date | What it is |
|---|---|---|
| A | 2 July 2026 | Original experiment package (`results/A_2026-07-02/final_experiment_package_2026_07_02_v5_merged.zip`) |
| B | July 2026 | Supporting analyses on the same prepared data and code (the v6 extension) |
| C | September 2026 | Verification reruns and analyses added after the earlier results were known |
| D | October 2026 | Analyses added after the earlier results were known; rules fixed in `calibration/decisions_log.md` (not here yet) before each run |

## Data

### CSE-CIC-IDS2018 (needed for every run except Table 6)

Download the four processed CSV files from the public AWS bucket. No AWS account is needed. The AWS command-line tool is needed. If it is not installed, install it with `python -m pip install awscli`.

```bash
mkdir -p cicids2018
for f in Friday-16-02-2018 Friday-02-03-2018 Wednesday-28-02-2018 Thursday-01-03-2018; do
  aws s3 cp --no-sign-request --region ca-central-1 \
    "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/${f}_TrafficForML_CICFlowMeter.csv" cicids2018/
done
cd cicids2018 && sha256sum -c ../data/SHA256SUMS
```

| File | Attack class taken from it | Size (bytes) |
|---|---|---|
| `Friday-16-02-2018_TrafficForML_CICFlowMeter.csv` | DoS (Hulk, SlowHTTPTest) | 333,723,605 |
| `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv` | Bot | 352,368,373 |
| `Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv` | Infiltration | 209,249,758 |
| `Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv` | Infiltration | 107,842,858 |

Benign flows are pooled across the four files. The notebooks expect the files in `MyDrive/cicids2018` on Google Drive. Change `DATA_DIR` at the top of a notebook if the files are kept elsewhere. The September notebooks check the checksums before any fitting.

The providers allow redistribution, but the files are not copied here because they are freely available. Please cite the dataset as its providers ask: Sharafaldin, Lashkari and Ghorbani (ICISSP 2018) and the [dataset page](https://www.unb.ca/cic/datasets/ids-2018.html).

### CIC-PQC_OAV v1 (Table 6 only)

Not included. The dataset is published without a licence that allows redistribution. Download `raw_sessions.jsonl` and `scenario_manifest.csv` from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/pqc-oav-dataset-2025.html) into `calibration/data/raw/`, and cite the dataset as its page asks (`calibration/CITATION.md`). Table 6 needs only these two files. The audit script `calibration/scripts/01_audit.py` and the setup check `calibration/scripts/00_check_setup.py` also need `ml_features_and_labels.csv` (see `calibration/data/raw/PUT_DATASET_FILES_HERE.txt`). This repository holds only the numbers derived from the dataset.

## Environment

Create the Python 3.12.13 environment of Supplementary Table S15 with uv:

```bash
python -m pip install uv==0.12.15
python -m uv venv --python 3.12.13 --managed-python .venv
python -m uv pip install --python .venv/bin/python --only-binary :all: -r requirements.txt
```

Call uv as `python -m uv`. A plain `uv` command may run another uv version that is already installed on the computer.

`requirements.txt` pins numpy 2.0.2, pandas 2.2.2, SciPy 1.16.3, scikit-learn 1.6.1, xgboost 3.2.0 and statsmodels 0.15.0, plus the 15 transitive packages resolved in the September runs (21 pins in total). The September notebooks build this environment themselves, in a separate interpreter inside Colab. `reproduction/environment/main_requirements.txt` has the same 21 pins, in another order, without comments and without the platform condition on `nvidia-nccl-cu12`.

The reported runs used Google Colab on Linux x86_64 with CPU runtimes. The Colab runtimes had 2 CPUs, and OpenBLAS (the maths library numpy uses) chose its "Haswell" code path.

Results on other computers can differ, mainly for the MLP. We reran split 0 of the main design on a Linux x86_64 cloud machine with 4 CPUs, a newer Intel processor and the pinned environment:

- With the machine's default settings, MLP scores differed from run A by up to 0.018 in weighted F1 and by up to 0.08 in per-class F1. RF, the KS distances, the MI values and XGBoost matched, except one XGBoost field (the MI-sweep score at threshold 0.01), which differed by 1.8 × 10⁻⁴.
- With Colab-like settings, 280 of 281 fields matched within the notebooks' tolerances. Only the same XGBoost field still differed by 1.8 × 10⁻⁴. Set them before running:

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENBLAS_CORETYPE=Haswell
```

A full ten-split rerun on the same machine with its default settings changed none of the paper's equivalence verdicts at the 0.01 margin. Only one flag in the XGBoost MI sweep changed ("within 0.01" at MI ≥ 0.2, from true to false), while its loss left moved from 0.00909 to 0.00916.

The runs did not all use the same versions:

| Run | Python | numpy | pandas | SciPy | scikit-learn | xgboost | statsmodels | Where recorded |
|---|---|---|---|---|---|---|---|---|
| A | 3.12.13 | 2.0.2 | 2.2.2 | 1.16.3 | 1.6.1 | 3.2.0 | 0.14.6 | `05_environment/` inside the run A package ZIP (a record, not a full lock) |
| B | not recorded centrally | | | | | 3.2.0 | | output folder names (run B results not here yet) |
| C | 3.12.13 | 2.0.2 | 2.2.2 | 1.16.3 | 1.6.1 | 3.2.0 | 0.15.0 | `requirements.txt`; install logs inside each results ZIP |
| D | 3.13.16 | 2.1.3 | 2.2.3 | 1.16.3 | 1.6.1 | 3.2.0 | not used | printed by each notebook; `calibration/results/` |

Run D used the default Colab runtime of October 2026, with xgboost reinstalled as 3.2.0. The bytes-only and port-443 notebook did not print versions and did not pin xgboost. Its XGBoost results were therefore rerun under xgboost 3.2.0 (Table 7, below).

The Fig. 1 script and the Table 6 scripts need only common packages; see "How to run" below.

## Seeds and splits

- **Sample.** 30,000 flows per class are drawn without replacement with seed 42 (`MODEL_SEED`). Removing rows with missing or infinite values and eight constant columns leaves 119,571 flows (29,827 Benign, 30,000 DoS, 30,000 Bot, 29,744 Infiltration) and 70 features.
- **Splits.** Ten stratified 70:30 splits with seeds 0 to 9: `train_test_split(X, y, test_size=0.30, random_state=s, stratify=y)`. Each split has 83,699 training and 35,872 test flows.
- **Models.** RF, XGBoost and the MLP use seed 42. MLP scalers are fitted on training data only.
- These constants are set in `code/revised_pipeline_main_only.py` (`N_SPLITS = 10`, `TEST_SIZE = 0.30`, `PER_CLASS = 30000`, `MODEL_SEED = 42`). Most October scripts assert them before fitting.
- The Table 6 bootstrap (handshake duration only) uses 10,000 resamples with seed 42.

## Reproduce the main-text tables and figures

Run times are wall-clock times recorded by the notebooks on Colab CPU runtimes. "Run A package" means `results/A_2026-07-02/final_experiment_package_2026_07_02_v5_merged.zip`. "Working copy" means `reproduction/PQC_IDS_Reproduction_Working_Copy.zip` (not here yet). Paths given "inside" a ZIP are paths within that archive.

| Paper item | Run | How to reproduce | Check against |
|---|---|---|---|
| Table 1 | | Comparison with prior work; no computation | |
| Table 2 | A | Constants in `code/revised_pipeline_main_only.py`; model settings in `make_model` in `code/option_b_full_experiment.py` | `00_data_provenance/` inside the run A package (input checksums, class counts, feature list) |
| Table 3 | A, C, D | Footprints in `VARIANTS` (`code/revised_pipeline_main_only.py`); bytes-only footprint and port-443 scope in `calibration/decisions_log.md` (8 October rows; not here yet) and `calibration/my_c2_code/colab_bytes_only_and_port443.py`; delays in `reproduction/environment/timing_shift_plan.json` | |
| Tables 4 and 5 | A, checked in C | `notebooks/PQC_IDS_Main_Reproduction.ipynb` (420 fits, about 3 h). It reruns `reproduction/scripts/run_experiment.py` for splits 0 to 9 and compares every stored score with `reproduction/scripts/compare_main_reproduction.py`. Run A's own script also works: `python code/revised_pipeline_main_only.py` in Colab with Drive mounted (it reads the CSV files from `MyDrive/cicids2018` and writes `revised_pipeline_raw.json`). Statistical columns: `reproduction/scripts/rebuild_main_results.py` (from the saved raw results, no fitting) and `reproduction/scripts/uncertainty_corrected_resampled.py` (uncorrected and overlap-corrected tests) | Inside the run A package: `01_raw_results/revised_pipeline_raw.json` and `02_analysis_tables/`. Inside the working copy (not here yet): `verified_outputs/uncertainty/corrected_resampled_results.csv`. `results/C_2026-09/PQC_IDS_Main_Reproduction_Results.zip`. Run B's `07_revised_statistics/`: not here yet |
| Table 6 | D | `calibration/scripts/03_real_shifts.py`, then `calibration/scripts/04_compare.py` (seconds). Needs the CIC-PQC_OAV v1 files in `calibration/data/raw/` | `calibration/results/03_real_shifts.csv`, `calibration/results/04_comparison_numbers.csv` |
| Table 7 | A, D | Hybrid and pure rows for all flows: the main design (Table 4). Bytes-only and port-443 rows for RF and MLP: `notebooks/2026-10_bytes_only_port443.ipynb` (about 2 h for bytes-only; allow 2.5 to 3 h for port 443; see the note under "Notebooks"). XGBoost column, all rows: `notebooks/2026-10_table7_xgboost_rerun.ipynb` (about 20 min; it also refits RF for all six rows as a check) | Summaries in `calibration/results/` (`07_bytes_only_and_port443.*`, `08_port443_only.*`, `10_table7_xgb.md`). The run D output folders (`pqc_oav_calibration_bytes_only/`, `pqc_oav_calibration_port443/`, `pqc_oav_calibration_table7_xgb/table7_xgb_result.json`): not here yet |
| Table 8 | C | `notebooks/PQC_IDS_Augmentation_Baseline.ipynb` (about 1.6 h). It runs `reproduction/scripts/run_augmentation_baseline.py` and `reproduction/scripts/analyse_augmentation_baseline.py` with the plan `reproduction/environment/augmentation_baseline_plan.json` | `PQC_IDS_Augmentation_Results.zip`: not here yet |
| Table 9 | A, B | Identifier-free design: `notebooks/PQC_IDS_B3_Reproduction.ipynb` (about 2.5 h; "B3" is the code's name for this design), or run A's script `python code/revised_pipeline_b3.py`. Timing-free design: `python code/v6_timing_free_pipeline.py --n-splits 10 --output-dir <folder>` (the default is one split). Combined design: the script `run_combined_full_10_splits.py` and its protocol `analysis_plan.md` are in run B's results, which are not here yet | Inside the run A package: `01_raw_results/revised_pipeline_b3_raw.json`, `02_analysis_tables/table_4_4_b3_identifier_free.csv`. Run B's timing-free and combined results: not here yet |
| Table 10 | C | `notebooks/PQC_IDS_Timing_Shift.ipynb` (about 1.1 h). It runs `reproduction/scripts/run_timing_shift.py` and `reproduction/scripts/analyse_timing_shift.py` with `reproduction/environment/timing_shift_plan.json` and `reproduction/environment/timing_shift_plan_amendment_1.md` | `PQC_IDS_Timing_Shift_Results.zip`: not here yet |
| Fig. 1 | A, C | `figures/code/fig1_overview.py` (see "Figures"). Data-flow check: `notebooks/2026-10_figure1_trace.ipynb` (5 to 10 min) asserts that test rows never enter the KS screen, MI selection, scaling or model fitting | The notebook's saved output. Its log file `figure1_trace_log.csv` (run D output folder): not here yet |
| Figs. 2 to 7 | | Made in Tableau from `figures/tableau/` (see "Figures") | |

## Figures

The figure image files are not in this repository yet.

### Fig. 1 (Python)

`figures/code/fig1_overview.py` draws Fig. 1. It uses `figures/code/style.py`, the fonts in `figures/code/fonts/` and two CSV files in `figures/code/data/` (`features_ks.csv` and `weighted_f1_by_split.csv`). It checks its counts against the two CSV files before it draws. It writes `out/Figure_1.pdf`, `.svg`, `.png` and `.tif` in `figures/code/`.

```bash
cd figures/code
python -m pip install -r requirements.txt
python fig1_overview.py
```

`figures/code/requirements.txt` pins matplotlib 3.11.2, numpy 2.5.3, pandas 3.0.5 and Pillow 12.3.0 under Python 3.13. The Liberation Sans fonts in `fonts/` load automatically.

### Figs. 2 to 7 (Tableau)

Figs. 2 to 7 were made in Tableau. `figures/tableau/` holds one workbook per figure and the Excel files they read:

| Figure | Workbook | Excel file |
|---|---|---|
| Fig. 2 | `figures/tableau/Figure_2.twb` | `figures/tableau/FCS_figures_2-7_data.xlsx` |
| Fig. 3 | `figures/tableau/Figure_3.twb` | `figures/tableau/FCS_figures_2-7_data.xlsx` |
| Fig. 4 | `figures/tableau/Figure_4.twb` | `figures/tableau/FCS_figures_2-7_data.xlsx` |
| Fig. 5 | `figures/tableau/Figure_5.twb` | `figures/tableau/fig5_tableau.xlsx` |
| Fig. 6 | `figures/tableau/Figure_6.twb` | `figures/tableau/fig6_tableau.xlsx` |
| Fig. 7 | `figures/tableau/Figure_7.twb` | `figures/tableau/fig7_tableau.xlsx` |

To open a workbook, use Tableau Desktop (the free Tableau Desktop Free Edition opens them). Each workbook points to its Excel file by a Windows path on the authors' laptop. Tableau will say it cannot find the file and ask for it. Pick the `.xlsx` file named above, in the same folder.

Each Excel file has a README sheet. In short:

- `FCS_figures_2-7_data.xlsx` has one sheet per figure panel: `fig2a`, `fig2b`, `fig3`, `fig4`, `fig5`, `fig6`, `fig7a` and `fig7b`. Its README sheet says the numbers are the values plotted in the paper's figures. Columns named Order, Condition order, Class order and X order keep the paper's order; sort by them, or Tableau sorts A to Z.
- `fig5_tableau.xlsx` holds Fig. 5 as one row per heatmap cell (54 cells), with values copied from sheet `fig5` of `FCS_figures_2-7_data.xlsx`, plus a colour key (`fig5_key`).
- `fig6_tableau.xlsx` holds the data of sheet `fig6` of `FCS_figures_2-7_data.xlsx`, reshaped into lines, bands (mean ± 1 SD) and axis ticks for Tableau.
- `fig7_tableau.xlsx` holds the bars of Fig. 7, copied from sheets `fig7a` and `fig7b` of `FCS_figures_2-7_data.xlsx`.

## Notebooks

**September 2026 notebooks** (`notebooks/PQC_IDS_*.ipynb`, run C). They were unpacked from `reproduction/PQC_IDS_Reproduction_Working_Copy.zip` and have no outputs. The exception is `PQC_IDS_Operator_Sensitivity.ipynb`: this copy came from the laptop. It is saved with its outputs and has five extra cells after the two cells of the ZIP copy. These cells split the results ZIP and the notebook into 20 MB parts and collect source files from Drive; they do not rerun the experiment.

| Notebook | What it does |
|---|---|
| `PQC_IDS_Main_Reproduction.ipynb` | Main design, all ten splits (Tables 4 and 5) |
| `PQC_IDS_B3_Reproduction.ipynb` | Identifier-free design (Table 9) |
| `PQC_IDS_B3_Recover_Results.ipynb` | Retrieves a finished B3 run from Drive; fits no models |
| `PQC_IDS_Seed0_Reproduction.ipynb` | Refits RF, XGBoost and MLP for split seed 0 only (C1-SF) |
| `PQC_IDS_Augmentation_Baseline.ipynb` | Training augmentation against screening (Table 8) |
| `PQC_IDS_Timing_Shift.ipynb` | Unanticipated timing shift (Table 10) |
| `PQC_IDS_Operator_Sensitivity.ipynb` | Corrected operators |
| `PQC_IDS_Grouped_Evaluation.ipynb` | Duplicate-aware grouped split |
| `PQC_IDS_Feature_Consistency_Audit.ipynb` | Feature-consistency audit; fits no models |
| `PQC_IDS_RF_Seed6_Diagnostic.ipynb` | RF split-6 diagnostic |

**October 2026 notebooks** (`notebooks/2026-10_*.ipynb`, run D). They are saved with their outputs. The same code is in `calibration/my_c2_code/`:

| Notebook | Paper item | Script in `calibration/my_c2_code/` |
|---|---|---|
| `2026-10_bytes_only_port443.ipynb` | Table 7, RF and MLP rows | Cell 1: `colab_bytes_only_and_port443.py`. Cell 2: `colab_port443_only.py` |
| `2026-10_table7_xgboost_rerun.ipynb` | Table 7, XGBoost column | `colab_table7_xgb.py` |
| `2026-10_shifted_flows_and_xgboost_check.ipynb` | Table S30 and the XGBoost reproducibility check | Cell 1: `colab_B1_B4.py`. Cell 2: an earlier version of `colab_table7_xgb.py` |
| `2026-10_mismatched_preparation.ipynb` | Table S31 | `colab_B2_mismatch.py` |
| `2026-10_tuned_mlp.ipynb` | Section A19 | `colab_B3_tuned_mlp.py` |
| `2026-10_figure1_trace.ipynb` | Fig. 1 data-flow check | `colab_trace_figure1.py` |

Notes on the October notebooks:

- The notebook cells and the scripts match, except for the wording of some comments and printed messages and two kinds of exception. In `colab_B1_B4.py` and `colab_table7_xgb.py`, only the check on an already-imported xgboost was changed after the runs. Cell 2 of `2026-10_shifted_flows_and_xgboost_check.ipynb` has an earlier form of that check, written differently.
- `2026-10_bytes_only_port443.ipynb`: the second cell (port-443-only run) shows a failed attempt from 7 October (RecursionError after 149 s). The port-443-only numbers in `calibration/results/08_port443_only.json` and `calibration/results/08_port443_only.md` come from the successful run of 8 October, which is not in this notebook.
- `2026-10_tuned_mlp.ipynb` holds the same cell three times, with the same output.
- The Table 7 XGBoost rerun was made twice, in separate sessions: `2026-10_table7_xgboost_rerun.ipynb` and cell 2 of `2026-10_shifted_flows_and_xgboost_check.ipynb`. The two outputs give identical numbers in all six cells; only the run times differ.

## How to run

**September notebooks.** Open one in Colab and choose Runtime > Run all. When asked, upload `reproduction/PQC_IDS_Reproduction_Working_Copy.zip` (not here yet, so these notebooks cannot be run from this repository for now). The notebook checks every file in it against its SHA-256 manifest, mounts Google Drive, checks the four CSV files, builds the pinned environment, runs, compares the new scores with the stored ones and writes a results ZIP to Google Drive. Differences are reported, never overwritten.

**October notebooks.** Use a fresh runtime. Each cell mounts Drive and looks for the four modules in `code/` anywhere in Google Drive, matching them by file size. Copy `code/` into Google Drive unchanged, or set `CODE_DIR` at the top of the cell. Each writes to its own folder, `MyDrive/pqc_oav_calibration_<name>/`. If xgboost is already imported, restart the session first.

**Main-design scripts outside Colab.** The scripts in `reproduction/scripts/` are there for reading only. They need other files from the working copy (`evidence/`, `reference/` and `environment/`), so they must run from the unpacked working-copy ZIP, as the notebooks do. Unpack `reproduction/PQC_IDS_Reproduction_Working_Copy.zip` (not here yet) into a scratch folder and run the scripts from there. Do not repack or change the ZIP: its SHA-256 is checked. `scripts/run_experiment.py --stage train --variant main` always runs all ten splits (420 fits). It has no option for fewer splits. On a machine with 4 CPUs, loading the data took 1 to 2 minutes, each split then took about 3.5 minutes, and all ten splits took about 41 minutes. `scripts/compare_main_reproduction.py` also needs all ten splits.

**Split-0 check** (about 6 minutes on 4 CPUs; unpacking and checking the working copy takes a few seconds). `reproduction/check_split0.py` reruns split 0 of the main design (42 fits) and compares it with run A. It unpacks the working copy into a new scratch folder and checks every file against the working copy's manifest. Then it runs the unchanged `scripts/run_experiment.py` with the arguments the main notebook uses, with the locked pipeline set to split 0 in memory. Last, it compares every split-0 field with run A, using the tolerances of the main notebook. Run it with the environment from "Environment" and the data from "Data":

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENBLAS_CORETYPE=Haswell
.venv/bin/python reproduction/check_split0.py --data-dir cicids2018
```

It prints the number of fields within tolerance and lists every field outside it. It writes the run and `split0_comparison.csv` to the scratch folder it names (or to a new folder given with `--workdir`). It exits with code 0 only if every field is within tolerance. Outside Colab, even with the Colab-like settings, expect 280 of 281 fields within tolerance and exit code 1, because the XGBoost MI-sweep score at threshold 0.01 differed by 1.8 × 10⁻⁴ on our test machine. Without the `export` line, expect the MLP differences described under "Environment".

**Table 6** (any computer, seconds):

```bash
cd calibration
python scripts/03_real_shifts.py   # writes results/03_real_shifts.csv and its log
python scripts/04_compare.py       # writes results/04_comparison_numbers.csv
```

A rerun with Python 3.10.12, pandas 2.3.3 and numpy 2.2.6 reproduced both CSV files and the log exactly.

**Fig. 1** (any computer): see "Figures".

**Supplementary tables.** `reproduction/manuscript/additional_file_1/af1_tables.py` builds the September supplementary tables from the working copy. It uses an earlier numbering of the supplement; `reproduction/manuscript/additional_file_1/Additional_file_1_Source_Map.md` gives each table's source. `reproduction/manuscript/additional_file_1/replay_error_analysis.py` computes Supplementary Table S9 from the saved predictions of the operator-sensitivity run (`PQC_IDS_Operator_Sensitivity_Results.zip`, not here yet). Both scripts have hard-coded folder paths at the top. Set them to local folders before running.

## Checking a run

- The run A package has its own checksum (`results/A_2026-07-02/final_experiment_package_2026_07_02_v5_merged.zip.sha256`, SHA-256 `47b7cd77b6a2af83c17f4220607597af835890851b4a4fdaa860fe34bb8c4987`) and a per-file manifest (`results/A_2026-07-02/final_experiment_package_2026_07_02_v5_merged_file_manifest.csv`). To check the ZIP:

  ```bash
  cd results/A_2026-07-02
  sha256sum -c final_experiment_package_2026_07_02_v5_merged.zip.sha256
  ```

- The 11 run A files in `code/` match the `04_code/` rows of that manifest.
- Each September results ZIP holds a manifest of its files, the install and run logs, and `working_copy_manifest.json`, which lists the code the run used.
- The comparison tolerances are fixed in the notebooks before each run and saved with the comparison (for example `comparison/main_comparison_status.json` inside `results/C_2026-09/PQC_IDS_Main_Reproduction_Results.zip`).
- The September rerun of the main design matched 2,806 of 2,810 stored fields within its tolerances (F1 within 10⁻¹²), and every headline mean exactly. The four exceptions are RF fields in split 6 that differ by at most 7 × 10⁻⁵ (Supplementary Section A1).
- `reproduction/PQC_IDS_Reproduction_Working_Copy.zip` (not here yet) is the working copy of 26 September 2026 without its 45 manuscript files (drafts, figure copies and manuscript-editing tools). Its `package_manifest.json` says so and keeps every other file's original SHA-256, so the notebooks' checks pass. Every script and run plan in it is byte-identical to what the grouped, timing-shift and augmentation runs used (`working_copy_manifest.json` in their results ZIPs). For the main run it matches 9 of 10 code files; `rebuild_main_results.py` was revised after that run. Do not change or repack this ZIP: the notebooks check its SHA-256 (`ca03febf6f4cc5c76131ff57fb25feba2745f58445a49c26b935ca05e6d607a7`).
- `.gitattributes` makes Git store and check out every file byte for byte. Keep it, or the SHA-256 checks and the October notebooks' file-size check can fail on Windows.

## Not included

- The CSE-CIC-IDS2018 CSV files and all CIC-PQC_OAV v1 files (download them as above).
- Model objects. The original forests and predictions of run A were not retained (Supplementary Section A1).
- Probability arrays for the timing and augmentation runs. Their archives hold class-label predictions, so weighted F1 can be replayed but ROC-AUC cannot.
- Draft notebooks, viva demonstration notebooks, manuscript-editing scripts and analyses that appear only in the dissertation.
- The items listed under "Not in this repository yet" at the top.

## Licence

- Code (`code/`, `reproduction/`, `notebooks/`, `calibration/scripts/`, `calibration/my_c2_code/`, `figures/code/*.py`): MIT, see `LICENSE`.
- Derived results, figures and documentation (everything else, unless stated): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- The Liberation Sans fonts in `figures/code/fonts/` keep their SIL Open Font License 1.1 (`figures/code/fonts/LICENSE_Liberation_fonts_OFL-1.1.txt`).
- The datasets keep their providers' terms.

## How to cite

Please cite the paper above. To cite this code, use: Abdinur M A, Doraisamy V. PQC-IDS shift: code and results. GitHub, 2026. https://github.com/m-abdinur/pqc-ids-shifts

## Contact

Mohamed Abdullahi Abdinur (m.abdinur@student.usm.my) and Vaithegy Doraisamy (vaithegy@usm.my), School of Computer Sciences, Universiti Sains Malaysia, Penang 11800, Malaysia.
