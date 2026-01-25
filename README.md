[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

### DLDL: Disruption Labeling with Deep Learning

A 1D convolutional neural network that uses the plasma current to produce a time of disruption if a disruption occurs. To be used for labeling D-IIID shots using the 'ipspr15V' plasma current PTDATA signal.

## Environment Setup

This project uses environment variables for configuration to easily switch between different environments (local development, Polaris HPC, etc.).

### Quick Start

1. **For local development:**
Copy the example file into `.env.local` with your local paths.
   ```bash
   cp .env.local.example .env.local && ln -s .env.local .env
   ```

2. **For Polaris HPC:**
Copy the example file into `.env.polaris` with your Polaris paths.
   ```bash
   cp .env.polaris.example .env.polaris && ln -s .env.polaris .env
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .
   ```

The `.env` file will be automatically loaded when the `constants` module is imported.

### Environment Variables

**Required Variables:**
- `DATA_DIR`: Raw signal data directory containing .txt files
- `DATASET_DIR`: Directory for preprocessed datasets and labels
- `LABELS_PATH`: Path to labels file (shot list with disruption times)
- `PROG_DIR`: Training progress/output directory for logs and checkpoints
- `JOB_ID`: Training run identifier (used for naming logs and checkpoints)

**Optional Variables (for distributed training):**
- `PMI_LOCAL_RANK`: Local rank for distributed training (set by scheduler)
- `PMI_RANK`: Process rank (defaults to 0 if not set)
- `PMI_SIZE`: Total number of processes (defaults to 1 if not set)

## Workflow

### 1. Preprocessing (`preprocess_data.py`)

Preprocesses raw plasma current signals into PyTorch-ready datasets.

**Steps:**
1. Deletes existing cached files for the specified `dataset_id`
2. Loads raw signal files from `DATA_DIR`
3. Computes dataset statistics (max length, mean, std)
4. Applies normalization (if specified)
5. Pads sequences to uniform length
6. Creates binary classification labels + scaled disruption times
7. Saves processed dataset and labels to `DATASET_DIR`

**Configuration:**
- Set `DATASET_ID` and `NORMALIZATION` in `preprocess_data.py`
- Normalization options: `None`, `"scale"`, `"meanvar-whole"`, `"meanvar-single"`

**Normalization Methods:**
- `None`: No normalization (raw values, zero-padded)
- `"scale"`: Min-max scaling per shot to [0, 1] range: `(x - min(x)) / (max(x) - min(x))`
  - Each shot normalized independently using its own min/max
  - Preserves relative patterns within each shot
- `"meanvar-whole"`: Z-score normalization using dataset-wide statistics: `(x - μ_dataset) / σ_dataset`
  - All shots normalized using the same mean and standard deviation computed across entire dataset
  - Makes all shots directly comparable on the same scale
- `"meanvar-single"`: Z-score normalization per shot: `(x - μ_shot) / σ_shot`
  - Each shot normalized independently to have mean=0 and std=1
  - Removes magnitude differences between shots while preserving relative patterns

**Usage:**
```bash
python src/preprocess_data.py
```

### 2. Training (`main.py`)

Trains the IpCNN model using preprocessed data.

**Steps:**
1. Loads preprocessed dataset and labels (using same `DATASET_ID` as preprocessing)
2. Validates files exist before training
3. Splits data into train/dev/test sets (80/10/10)
4. Initializes distributed training (if `PMI_RANK`/`PMI_SIZE` set)
5. Trains model with validation and checkpointing
6. Saves logs and model checkpoints to `PROG_DIR`

**Configuration:**
- Set `DATASET_ID` to match preprocessing (must be identical)
- Training hyperparameters (lr, epochs, etc.) in `main.py`

**Usage:**
```bash
python src/main.py
```

### Workflow Summary

```
Raw Data (.txt files)
    ↓
[preprocess_data.py]
    ↓
Preprocessed Dataset (.pt files)
    ↓
[main.py]
    ↓
Trained Model + Logs
```

**Important:** The `DATASET_ID` in both scripts must match to ensure preprocessing and training use the same files.

### Dataset ID Approach

The project uses a `dataset_id` suffix system to manage multiple preprocessed dataset configurations. This allows different preprocessing methods (normalization strategies) to coexist without overwriting each other.

**How it works:**
- The `dataset_id` is a string suffix appended to processed filenames
- Example: `dataset_id="meanvar-whole"` creates:
  - `processed_dataset_meanvar-whole.pt`
  - `processed_labels_meanvar-whole.pt`
- Both `preprocess_data.py` and `main.py` use the same `DATASET_ID` constant to ensure consistency
- This enables easy experimentation with different preprocessing configurations (e.g., `"scale"`, `"meanvar-single"`, `""` for no normalization)

**Benefits:**
- Multiple preprocessed datasets can coexist in the same directory
- Automatic alignment between preprocessing and training workflows
- Easy switching between configurations by changing one constant
