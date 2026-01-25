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

`.env` is loaded when `constants` is imported.

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATA_DIR` | ✓ | Raw signal .txt files |
| `DATASET_DIR` | ✓ | Preprocessed datasets and labels |
| `LABELS_PATH` | ✓ | Shot list with disruption times |
| `PROG_DIR` | ✓ | Logs and checkpoints |
| `JOB_ID` | ✓ | Run identifier |
| `NORMALIZATION_TYPE` | | `scale`, `meanvar-whole`, or `meanvar-single` (default `meanvar-whole`). |
| `CPU_USE` | | Fraction of CPU cores for preprocessing, 0-1 (default `0.2`; use `0.2-0.3` for ~32GB RAM) |
| `PMI_LOCAL_RANK` | | Distributed training (scheduler) |
| `PMI_RANK` | | Process rank (default 0) |
| `PMI_SIZE` | | World size (default 1) |

## Workflow

```
Raw .txt → preprocess_data.py → .pt files → train.py → Model + logs
```

**Config:** Set `NORMALIZATION_TYPE` and `CPU_USE` in `.env` (or env). `NORMALIZATION_TYPE` is the normalization method and filename suffix. Both scripts use `constants`.

### 1. Preprocessing

```bash
python src/preprocess_data.py
```

* Deletes cached processed files for the current `NORMALIZATION_TYPE`
* Loads raw files from `DATA_DIR`, computes stats (max length, mean, std)
* Applies normalization, pads to uniform length, builds labels
* Writes `processed_dataset_{NORMALIZATION_TYPE}.pt` and `processed_labels_{NORMALIZATION_TYPE}.pt` to `DATASET_DIR`
* Runs integrity check

**Normalization** (env `NORMALIZATION_TYPE`; default `meanvar-whole`):

| Value | Formula | Use case |
|-------|---------|----------|
| `scale` | `(x - min) / (max - min)` per shot | Relative patterns matter, magnitudes vary |
| `meanvar-whole` | `(x - μ) / σ` dataset-wide | Same scale across all shots |
| `meanvar-single` | `(x - μ) / σ` per shot | Per-shot standardization |

### 2. Training

```bash
python src/train.py
```

* Loads preprocessed dataset/labels (same `NORMALIZATION_TYPE` as preprocessing)
* Validates files exist, splits 80/10/10 train/dev/test
* Uses distributed training if `PMI_*` are set
* Saves checkpoints and logs to `PROG_DIR`

Hyperparameters (lr, epochs, etc.) are in `train.py`.

## Quick Reference

* **Config:** `NORMALIZATION_TYPE` and `CPU_USE` in env (or `.env`) drive preprocessing and training.
* **Output files:** `processed_dataset_{NORMALIZATION_TYPE}.pt`, `processed_labels_{NORMALIZATION_TYPE}.pt` in `DATASET_DIR`.
* **Multiple configs:** Change `NORMALIZATION_TYPE` in env to switch; each value uses its own .pt files.
