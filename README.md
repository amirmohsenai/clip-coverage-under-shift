# Prompt- and Shift-Aware Conformal Reliability of Zero-Shot Vision-Language Models

## Overview

This project implements an evaluation pipeline for measuring the conformal prediction reliability of a frozen OpenAI CLIP `ViT-B/32` zero-shot image classifier on CIFAR-10 and CIFAR-10-C under controlled prompt variations and visual distribution shifts. The codebase is designed primarily for Google Colab with persistent storage backed by Google Drive (and supports a local filesystem fallback).

The implementation addresses the problem of quantifying how split-conformal prediction sets behave when the evaluation distribution departs from the calibration distribution through wording changes in the text prompt, image corruptions across multiple severity levels, or both simultaneously. It also implements a matched-shift calibration regime in which calibration and evaluation subsets are drawn disjointly from the same corrupted distribution.

In execution order, the program:
1. Installs and verifies the `clip`, `ftfy`, `regex`, and `tqdm` packages.
2. Initializes protocol constants, directory paths, deterministic random-number generator settings, and environment metadata files.
3. Loads or downloads the official CIFAR-10 training and test splits, verifies or downloads the CIFAR-10-C archive from Zenodo with HTTP Range resume support, extracts the six configured corruption arrays and shared label file, and constructs deterministic class-balanced index splits across five calibration seeds.
4. Verifies or downloads the frozen OpenAI CLIP `ViT-B/32` checkpoint, encodes and L2-normalizes class text embeddings for four prompt templates, and caches them to disk.
5. Extracts and caches L2-normalized 512-dimensional CLIP image embeddings for the clean test split, seed-specific clean calibration splits, and lazily evaluated CIFAR-10-C subsets, automatically halving the batch size on CUDA out-of-memory errors.
6. Computes scaled cosine-similarity logits for individual prompt templates and a deterministic arithmetic-mean logit ensemble, fits scalar post-hoc temperature scaling on calibration data via L-BFGS, and computes finite-sample conformal thresholds and prediction sets using deterministic Adaptive Prediction Sets (APS), Temperature-Scaled APS (`TS+APS`), and Regularized Adaptive Prediction Sets (RAPS).
7. Evaluates all 2,235 planned `(seed, experiment, corruption, severity, prompt, method)` configurations with incremental CSV checkpointing, aggregates metrics across the five calibration seeds with percentile bootstrap confidence intervals, generates summary tables and six Matplotlib figures, runs automated forensic and data-leakage assertions, and packages all generated files into an export archive.

## Project Structure

The reviewed project bundle contains the following files:

```text
.
|-- README.md
|-- report.tex
|-- report.pdf
|-- code_changes.py
|-- prompt_shift_conformal_vlm_persistent.py
`-- prompt_shift_conformal_vlm_persistent.ipynb
```

- `prompt_shift_conformal_vlm_persistent.ipynb`: The 43-cell Jupyter notebook (22 code cells and 21 Markdown cells) implementing the complete workflow for Google Colab.
- `prompt_shift_conformal_vlm_persistent.py`: The synchronized standalone Python script corresponding to the notebook cells.
- `code_changes.py`: A structured Python audit log recording every substantive modification made during code review and cleanup, along with retained compatibility behaviors and verification scope.
- `README.md`: Project documentation covering architecture, dependencies, setup, execution, input/output schemas, and implementation constraints.
- `report.tex` and `report.pdf`: Detailed technical report in LaTeX (and its compiled PDF) explaining the software architecture, algorithms, mathematical formulations, and data flow.

When executed, the code creates and populates the following workspace directory tree under `ROOT` (`/content/drive/MyDrive/prompt_shift_conformal_vlm` when `USE_GOOGLE_DRIVE = True`, or `/content/prompt_shift_conformal_vlm` when `USE_GOOGLE_DRIVE = False`):

```text
<ROOT>/
|-- data/
|   |-- CIFAR-10-C.tar
|   |-- cifar10/
|   `-- cifar10c/
|       |-- brightness.npy
|       |-- contrast.npy
|       |-- fog.npy
|       |-- gaussian_noise.npy
|       |-- jpeg_compression.npy
|       |-- motion_blur.npy
|       `-- labels.npy
|-- splits/
|   |-- shift_test_5000_seed_20270317.npz
|   |-- clean_calibration_seed_{13,42,123,456,765}.npz
|   `-- matched_shift_seed_{13,42,123,456,765}.npz
|-- cache/
|   |-- clip_weights/ViT-B-32.pt
|   |-- text_features.npz
|   |-- image_embeddings/
|   |-- cifar10c_embeddings/
|   |-- logits/
|   `-- calibration_states/
|-- metadata/
|   |-- config.json
|   |-- environment.json
|   |-- forensic_audit.json
|   `-- persistence_status.json
|-- logs/
|-- assets/
|   `-- pipeline_diagram.png
|-- results/
|   |-- experiment_manifest.csv
|   |-- results_checkpoint.csv
|   |-- results.csv
|   |-- results_aggregated_across_seeds.csv
|   |-- risk_coverage.csv
|   |-- figures/
|   `-- tables/
|-- final_artifacts/
`-- prompt_shift_conformal_vlm_final_artifacts.zip
```

External datasets (CIFAR-10 and CIFAR-10-C) and model weights (`ViT-B-32.pt`) are not bundled inside the source repository; they are verified in persistent storage or downloaded automatically on first execution when `AUTO_DOWNLOAD_MISSING = True`.

## Implementation Overview

The codebase is organized into 21 sequential sections that correspond directly between `prompt_shift_conformal_vlm_persistent.py` and the code/Markdown cells of `prompt_shift_conformal_vlm_persistent.ipynb`:

1. **Dependency Installation and Verification (Section 1 / Cells 1-2):** Invokes `pip install` via `subprocess.run` for OpenAI CLIP (`git+https://github.com/openai/CLIP.git`), `ftfy`, `regex`, and `tqdm`, and verifies each module with `importlib.import_module`.
2. **Configuration and Reproducibility Setup (Section 2 / Cells 3-4):** Defines protocol constants (`PRIMARY_SEED = 42`, `CALIBRATION_SEEDS = [13, 42, 123, 456, 765]`, `ALPHA = 0.10`, `TARGET_COVERAGE = 0.90`, `MODEL_NAME = "ViT-B/32"`, prompt templates `P0_canonical` through `P3_photo_wording`, corruption dictionary `CORRUPTIONS`, `SEVERITIES = [1, 3, 5]`, `METHODS = ["APS", "TS+APS", "RAPS"]`, `RAPS_K_REG = 5`, `RAPS_LAMBDA = 0.01`, `ECE_BINS = 15`), mounts Google Drive if enabled, creates workspace directories, seeds RNGs via `set_global_seed`, and writes `metadata/environment.json` and `metadata/config.json`.
3. **Filesystem, Hashing, and Split Validation Utilities (Section 3 / Cells 5-6):** Defines `sha256_file`, `md5_file`, `sanitize_name`, `atomic_write_json`, `stable_condition_seed`, `_bootstrap_condition_seed`, `stratified_indices`, `save_indices`, `load_valid_indices`, `finite_and_normalized_probabilities`, and `cache_npz_valid`.
4. **Dataset Acquisition and Split Construction (Section 4 / Cells 7-8):** Loads CIFAR-10 via `ensure_cifar10`, downloads or verifies `CIFAR-10-C.tar` via `download_with_progress` and `_verify_cifar10c_archive`, extracts required `.npy` files via `extract_selected_cifar10c_members`, verifies label alignment across all five severity blocks via `verify_cifar10c_alignment`, defines the memmap-backed `CIFAR10CSubset` dataset class, and builds or loads the fixed 5,000-image shifted test split, five 5,000-image clean calibration splits, and five disjoint `(2,000 calibration, 5,000 test)` matched-shift splits.
5. **CLIP Checkpoint Loading and Text Encoding (Section 5 / Cells 9-10):** Verifies `ViT-B-32.pt` against its SHA-256 hash (`ensure_clip_weights`), loads the model in evaluation mode with all parameters frozen (`requires_grad_(False)`), extracts `logit_scale`, and computes L2-normalized class text embeddings (`build_text_features` and `load_or_build_text_features`).
6. **Image Embedding Extraction and Caching (Sections 6-7 / Cells 11-14):** Defines `CleanCIFARSubset`, `encode_image_embeddings`, and `get_shift_embeddings`. Image features are encoded once per unique image subset and persisted as `.npz` archives containing `embeddings`, `labels`, and `sample_ids`.
7. **Zero-Shot Logits, Temperature Scaling, and Conformal Methods (Sections 8-10 / Cells 15-20):** Computes prompt-specific and ensemble logits (`logits_from_embeddings`, `logits_cached`), converts scaled logits to float64 probability matrices (`probability_from_logits`), fits constrained scalar temperature scaling on calibration data (`fit_temperature`), and implements deterministic conformal scoring and set construction (`conformal_quantile`, `sorted_probabilities`, `aps_calibration_scores`, `raps_calibration_scores`, `prediction_sets_aps`, `prediction_sets_raps`, `fit_conformal_state`, `build_prediction_sets`).
8. **Metrics, Experiment Runner, and Sanity Check (Sections 11-14 / Cells 21-28):** Implements evaluation metrics (`accuracy`, `coverage`, `expected_calibration_error`, `nll`, `entropy`, `evaluate_predictions`, `risk_coverage_curve`), generates the 2,235-row experiment manifest (`ExperimentSpec`, `build_experiment_specs`), runs a 128-calibration / 256-test sanity check, and executes all experiments with per-row CSV checkpointing (`run_single_experiment`, `run_all_experiments`).
9. **Statistical Aggregation, Visualization, Auditing, and Packaging (Sections 15-21 / Cells 29-42):** Aggregates metrics across the five calibration seeds (`bootstrap_seed_ci`, `aggregate_results`), generates six PNG figures and comparison/ablation tables, executes `run_forensic_audit`, writes repository metadata files, draws `assets/pipeline_diagram.png`, and packages all deliverables into `prompt_shift_conformal_vlm_final_artifacts.zip`.

## Requirements and Environment

### Python and Package Dependencies

The code relies on the following packages based on its actual import statements:

- **Installed explicitly in Section 1 (`REQUIRED_PIP`):**
  - `clip` (installed from `git+https://github.com/openai/CLIP.git`)
  - `ftfy`
  - `regex`
  - `tqdm`
- **Expected in the host environment (pre-installed in Google Colab):**
  - `torch` and `torchvision`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `Pillow` (`PIL`)
  - `google.colab` (required only when `USE_GOOGLE_DRIVE = True`; download triggers at the end of the script gracefully fall back when `google.colab` is absent)

No exact version pins are enforced in `requirements.txt`. In the recorded Colab execution metadata (`Cell 4`), the environment used Python 3.13.15, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, NumPy 2.1.3, and pandas 2.2.3 on Linux with an NVIDIA Tesla T4 GPU. Those versions represent the observed runtime environment rather than strict minimum version constraints; note, however, that the code uses `Path.unlink(missing_ok=True)` (Python 3.8+) and `zipfile` / `pathlib` standard library features.

### Hardware and Storage Considerations

- **CPU and GPU support:** Device selection is automatic (`DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`). `INITIAL_BATCH_SIZE` defaults to 128 when CUDA is available and 32 on CPU. If a CUDA out-of-memory error occurs during `encode_image_embeddings`, the batch size is halved down to `MIN_BATCH_SIZE = 8`.
- **Disk space:** Full execution requires approximately 2.92 GB for `CIFAR-10-C.tar`, approximately 900 MB for the six extracted corruption `.npy` files (`50,000 x 32 x 32 x 3` `uint8` arrays at ~153.6 MB each), ~170 MB for CIFAR-10, ~338 MB for the `ViT-B-32.pt` checkpoint, and ~500 MB for cached embeddings, logits, and results.

## Installation and Setup

### Running in Google Colab

1. Open `prompt_shift_conformal_vlm_persistent.ipynb` in Google Colab.
2. Select a GPU runtime if available (`Runtime -> Change runtime type -> Hardware accelerator: GPU`).
3. Ensure Google Drive has sufficient free space (~5 GB) if keeping `USE_GOOGLE_DRIVE = True`.

### Running Locally Outside Google Colab

To run the code on a local machine without Google Colab:

1. Create and activate a Python 3.10+ virtual environment, then install PyTorch, torchvision, scientific computing packages, and the CLIP dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy pandas matplotlib Pillow
pip install git+https://github.com/openai/CLIP.git ftfy regex tqdm
```

2. Before running `prompt_shift_conformal_vlm_persistent.py` or the notebook locally, set `USE_GOOGLE_DRIVE = False` in Section 2 (Cell 4). When `USE_GOOGLE_DRIVE = True`, the script imports `google.colab.drive` and raises a `RuntimeError` if not running inside Google Colab. You may also change `LOCAL_ROOT` from `"/content/prompt_shift_conformal_vlm"` to a local directory path where your user account has write permissions.

*(Note: The installation commands above are derived directly from the project's imports and setup cell; full end-to-end execution was not re-run during this documentation audit because the audit environment does not have internet access or a PyTorch/GPU runtime.)*

## How to Run the Project

### Notebook Execution

Run all cells in `prompt_shift_conformal_vlm_persistent.ipynb` sequentially from top to bottom (`Cell 0` through `Cell 42`). Because later cells depend on configuration constants, split dictionaries (`clean_calibration_indices`, `matched_splits`, `shift_test_indices`), the frozen `model` and `preprocess` objects, and cached embedding arrays created in earlier cells, out-of-order execution will fail with `NameError` or missing-file errors.

### Script Execution

After setting `USE_GOOGLE_DRIVE = False` and configuring `LOCAL_ROOT` for a local machine, run:

```bash
python3 prompt_shift_conformal_vlm_persistent.py
```

### Key Configuration Options

All primary settings are defined in Section 2 (Cell 4):
- `USE_GOOGLE_DRIVE`: Set to `True` on Google Colab to persist all datasets, weights, caches, and results in `DRIVE_ROOT`; set to `False` for local execution under `LOCAL_ROOT`.
- `AUTO_DOWNLOAD_MISSING`: When `True`, missing CIFAR-10 splits, the Zenodo `CIFAR-10-C.tar` archive, and the OpenAI CLIP `ViT-B-32.pt` checkpoint are downloaded automatically over the network. When `False`, all required datasets and checkpoints must already exist under `DATA_ROOT` and `CACHE_ROOT / "clip_weights"`.
- `DELETE_CIFAR10C_TAR_AFTER_EXTRACTION`: Set to `True` in Section 2 if you wish to delete the 2.92 GB `CIFAR-10-C.tar` archive after the six corruption `.npy` arrays and `labels.npy` have been extracted and validated.
- `RUN_FULL_EXPERIMENTS` (Section 14 / Cell 28): When `True`, executes any uncompleted rows of the 2,235-row experiment matrix and writes `results/results.csv`. When `False`, loads existing rows from `results/results_checkpoint.csv` without running new experiments (note that Sections 15-21 require all 2,235 rows to be present in `results_checkpoint.csv` / `results.csv` to pass completeness assertions).

## Inputs, Outputs, and Generated Files

### Required External Inputs (Downloaded or Persisted)

| Input Resource | Location Under `ROOT` | Format and Expected Structure |
| :--- | :--- | :--- |
| CIFAR-10 train and test splits | `data/cifar10/` | Standard `torchvision.datasets.CIFAR10` binary batches (50,000 train, 10,000 test RGB images of size `32 x 32`). |
| CIFAR-10-C archive | `data/CIFAR-10-C.tar` | 2,918,471,680-byte tar archive from Zenodo (`MD5: 56bf5dcef84df0e2308c6dcbcbbd8499`, `SHA-256: c72763e1...876a7`). |
| Extracted CIFAR-10-C arrays | `data/cifar10c/<corruption>.npy` and `labels.npy` | Six corruption `.npy` files of shape `(50000, 32, 32, 3)` (`uint8`) and `labels.npy` of shape `(50000,)` (`uint8` integers in `0..9`). |
| CLIP `ViT-B/32` checkpoint | `cache/clip_weights/ViT-B-32.pt` | TorchScript archive verified against `SHA-256: 40d36571...950af`. |

### Generated Caches and Intermediate Files

- **Index splits (`splits/*.npz`):** Compressed NumPy archives storing deterministic class-balanced sample indices (`indices`), validation label slices (`labels`), and JSON `metadata`. Matched-shift split files store `cal_indices`, `test_indices`, `cal_labels`, `test_labels`, and `metadata`.
- **Text feature cache (`cache/text_features.npz`):** Stores `(10, 512)` `float32` L2-normalized text embedding matrices keyed by prompt name (`P0_canonical`, `P1_paraphrase`, `P2_generic_visual`, `P3_photo_wording`).
- **Image embedding caches (`cache/image_embeddings/*.npz` and `cache/cifar10c_embeddings/<corruption>/severity_<1|3|5>/*.npz`):** Store `embeddings` (`(N, 512)` `float32`), `labels` (`(N,)` `int64`), and `sample_ids` (`(N,)` `int64`).
- **Logit caches (`cache/logits/<cache_tag>/<prompt_condition>.npz`):** Store `logits` (`(N, 10)` `float32`), `labels`, and `sample_ids`.
- **Calibration state caches (`cache/calibration_states/*.json`):** Store fitted `temperature`, conformal `quantile`, `n_calibration`, `method`, `seed`, `calibration_condition`, `prompt_condition`, `corruption`, `severity`, `alpha`, `raps_k_reg`, and `raps_lambda`.

### Final Output Files

- `results/experiment_manifest.csv`: Pre-execution manifest listing all 2,235 planned evaluation runs.
- `results/results_checkpoint.csv` and `results/results.csv`: Per-run evaluation table with 2,235 rows and 24 columns (`RESULT_COLUMNS`).
- `results/results_aggregated_across_seeds.csv`: Condition-level table with 447 rows and 73 columns summarizing 9 metrics across the 5 calibration seeds (`mean`, `std`, `median`, `min`, `max`, `ci95_low`, `ci95_high`).
- `results/risk_coverage.csv`: Selective top-1 risk-coverage curves (101 coverage points per curve) for clean and severe combined-shift (`motion_blur` severity 5 + `P1_paraphrase`) headline conditions.
- `results/figures/figure_1_coverage_vs_severity.png` through `figure_6_clean_vs_matched_shift.png`: Six 300-DPI diagnostic plots generated from the saved CSV tables.
- `results/tables/main_comparison_table.{csv,html,png}` and `ablation_summary_APS.csv`: Programmatically filtered summary tables.
- `metadata/{config.json,environment.json,forensic_audit.json,persistence_status.json}`: Provenance and verification records.
- `README.md`, `EXPERIMENT_PROTOCOL.md`, `LIMITATIONS.md`, `requirements.txt`, and `assets/pipeline_diagram.png`: Repository documentation and conceptual pipeline diagram generated inside `ROOT`.
- `final_artifacts/artifact_manifest.csv` and `prompt_shift_conformal_vlm_final_artifacts.zip`: Final export bundle with per-file byte counts and SHA-256 digests.

## Implementation Details and Limitations

- **Deterministic APS and Empty Prediction Sets:** The primary APS implementation (`prediction_sets_aps`) and RAPS implementation (`prediction_sets_raps`) are strictly deterministic and do not apply uniform random tie-breaking at the boundary label. A label is included in the prediction set if and only if its cumulative (or RAPS-penalized cumulative) sorted probability is at most `q + 1e-12`. Consequently, when the top-1 class probability exceeds `q + 1e-12`, the resulting prediction set is empty; the code preserves this behavior intentionally and records the fraction of empty sets in `empty_set_rate`.
- **Scope of Backbone and Datasets:** The implementation supports only the `ViT-B/32` model registered in `CLIP_MODEL_REGISTRY` and evaluates 10-class zero-shot classification on CIFAR-10 and six corruptions of CIFAR-10-C at severities 1, 3, and 5.
- **Seed-Variability Bootstrap Intervals:** `bootstrap_seed_ci` resamples the 5 calibration-seed values (`n = 5`) within each condition 4,000 times. As documented in the code, these intervals measure variability across random calibration splits rather than population-level image sampling uncertainty.
- **Aggregated Metric Columns:** While `RESULT_COLUMNS` records 11 numerical metrics per run, `aggregate_results` aggregates 9 of them (`accuracy`, `coverage`, `coverage_gap`, `average_set_size`, `singleton_rate`, `ECE`, `NLL`, `entropy`, `fitted_temperature`) and does not compute cross-seed summary columns for `empty_set_rate` or `conformal_quantile`.
- **Duplicate Clean Calibration State Files:** `calibration_state_path` incorporates `corruption` and `severity` into the JSON filename even when `calibration_condition == "clean"`. As a result, `fit_or_load_calibration_state` writes separate JSON files containing identical fitted parameters for each `(corruption, severity)` pair evaluated under clean calibration.
