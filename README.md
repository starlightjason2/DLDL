[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# DLDL: Disruption Labeling with Deep Learning

A 1D CNN that uses plasma current to predict disruption time. For labeling D-IIID shots with the 'ipspr15V' plasma current PTDATA signal.

## Environment Setup

1. **Create `.env`** from a template:
   ```bash
   cp .env.local.example .env.local && ln -s .env.local .env   # local
   # or
   cp .env.polaris.example .env.polaris && ln -s .env.polaris .env   # Polaris
   ```

2. **Install:**
   ```bash
   pip install -e .
   ```

`.env` is loaded when you call ``load_settings()`` from ``config.settings`` (typically at script startup).

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATA_DIR` | ✓ | Raw signal .txt files |
| `LABELS_PATH` | ✓ | Shot list with disruption times |
| `DATA_PATH` | ✓ | Full path to preprocessed dataset tensor (`.pt`) |
| `TRAIN_LABELS_PATH` | ✓ | Full path to preprocessed labels tensor (`.pt`) |
| `PROG_DIR` | ✓ | Logs, checkpoints, and `graph.py` plot output |
| `JOB_ID` | ✓ | Run identifier |
| `NORMALIZATION_TYPE` | | `scale`, `meanvar-whole`, or `meanvar-single` (default `meanvar-whole`). |
| `CPU_USE` | | Fraction of CPU cores for preprocessing, 0-1 (default `0.2`; use `0.5-1.0` on HPC with more RAM) |
| `PREPROCESSOR_MAX_WORKERS` | | Hard cap on preprocessing workers (default `4`; avoids fork/resource issues on HPC) |
| `DATALOADER_NUM_WORKERS` | | DataLoader workers for training (default `4`; 0 when no GPU) |
| `PMI_LOCAL_RANK` | | Distributed training (scheduler) |
| `PMI_RANK` | | Process rank (default 0) |
| `PMI_SIZE` | | World size (default 1) |

## Workflow

```
Raw .txt → preprocess_data.py → .pt files → train.py → Model + logs → graph.py → Visualizations
```

**Config:** Set `NORMALIZATION_TYPE` and `CPU_USE` in `.env` (or env). Point `DATA_PATH` / `TRAIN_LABELS_PATH` at the `.pt` files that match that normalization (e.g. filenames containing `meanvar-whole`). Scripts call `load_settings()` for hyperparameters from `dldl.json` (with env overrides).

### 1. Preprocessing

```bash
python src/preprocess_data.py
```

* Deletes the preprocessed files at `DATA_PATH` and `TRAIN_LABELS_PATH` (so the next build is fresh)
* Loads raw files from `DATA_DIR`, computes stats (max length, mean, std)
* Applies normalization, pads to uniform length, builds labels
* Writes tensors to `DATA_PATH` and `TRAIN_LABELS_PATH`
* Runs integrity check

**Normalization** (env `NORMALIZATION_TYPE`; default `meanvar-whole`):

| Value | Formula | Use case |
|-------|---------|----------|
| `scale` | `(x - min) / (max - min)` per shot | Relative patterns matter, magnitudes vary |
| `meanvar-whole` | `(x - μ) / σ` dataset-wide | Same scale across all shots |
| `meanvar-single` | `(x - μ) / σ` per shot | Per-shot standardization |

### 2. Validation (optional)

```bash
python src/validate.py
```

* Verifies preprocessed files exist
* Runs dataset integrity check (samples examples and compares to raw data)
* Options: `--num-checks N`, `--skip-integrity`, `--verbose`

### 3. Training

```bash
python src/train.py
```

* Loads preprocessed tensors at `DATA_PATH` and `TRAIN_LABELS_PATH` (must match preprocessing / `NORMALIZATION_TYPE`)
* Validates files exist, splits 80/10/10 train/dev/test
* Uses distributed training if `PMI_*` are set
* Saves checkpoints and logs to `PROG_DIR`

Hyperparameters (lr, epochs, etc.) are in `train.py`.

### 4. Visualization

```bash
python -m src.graph
```

* Loads training log CSV from `PROG_DIR/{JOB_ID}_training_log.csv`
* Creates a 2x2 subplot visualization with:
  - Training and Validation Loss
  - Validation Accuracy
  - Validation Precision and Recall
  - Validation F1 Score
* Saves plot to `PROG_DIR/{JOB_ID}_training_log_plot.png`

**Command-line Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--csv` | str | `PROG_DIR/{JOB_ID}_training_log.csv` | Path to the training log CSV file. If not specified, uses the default path based on `PROG_DIR` and `JOB_ID` from config. |
| `--output` | str | `PROG_DIR/{JOB_ID}_training_log_plot.png` | Path where the plot image will be saved. If not specified and `--show` is not used, defaults to that path. If `--show` is used without `--output`, the plot is only displayed and not saved. |
| `--show` | flag | False | Display the plot interactively using matplotlib's GUI. When used without `--output`, the plot is only displayed and not saved to disk. Can be combined with `--output` to both save and display. **Note:** Requires an interactive backend (PyQt5, PyQt6, or Tkinter). In headless environments, the plot will be saved to the default location instead. |

**Examples:**
```bash
# Use default paths from config
python -m src.graph

# Specify custom CSV file
python -m src.graph --csv path/to/training_log.csv

# Save plot to specific location
python -m src.graph --output path/to/output.png

# Display plot interactively (not saved)
python -m src.graph --show

# Save and display plot
python -m src.graph --output plot.png --show

# Use custom CSV and save to custom location
python -m src.graph --csv custom_log.csv --output plot.png
```

## Polaris Batch Jobs

Run preprocessing and training on compute nodes (not login nodes):

```bash
# From DLDL project root, with .env symlinked to .env.polaris
cd /path/to/DLDL

# Edit scripts/run_preprocess.sh and scripts/run_train.sh:
# - Uncomment and set your conda activation
# - Adjust #PBS -A if your account differs from fusiondl_aesp

# Submit preprocessing (CPU, ~2 hr walltime)
qsub scripts/run_preprocess.sh

# Submit training (GPU, ~4 hr walltime)
qsub scripts/run_train.sh

# Check status
qstat -u $USER
```

Output logs: `preprocess_<jobid>.out`, `train_<jobid>.out` (and `.err`).

### Bayesian Hyperparameter Tuning

```bash
# From project root (with .env.polaris configured)
./scripts/start_hptune.sh
```

Runs a chain of jobs: each controller picks or creates a trial (lr, epochs, dropout), submits a training job, then chains the next controller. Uses random sampling for the first 5 trials, then Bayesian optimization. Results in `scripts/hptune/trials_log.csv` and `scripts/hptune/trials/`.

## Quick Reference

* **Config:** `NORMALIZATION_TYPE` and `CPU_USE` in env (or `.env`) drive preprocessing and training.
* **Output files:** Whatever paths you set in `DATA_PATH` and `TRAIN_LABELS_PATH` (often filenames include the normalization mode, e.g. `processed_dataset_meanvar-whole.pt`).
* **Multiple configs:** Change `NORMALIZATION_TYPE` and update `DATA_PATH` / `TRAIN_LABELS_PATH` to the matching `.pt` files.
