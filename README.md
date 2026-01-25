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
- `DATA_PATH`: Path to preprocessed dataset file (.pt format)
- `TRAIN_LABELS_PATH`: Path to preprocessed labels file (.pt format)
- `SCALED_LABELS_FILENAME`: Filename for scaled labels (stored in DATASET_DIR)
- `PROG_DIR`: Training progress/output directory for logs and checkpoints
- `JOB_ID`: Training run identifier (used for naming logs and checkpoints)

**Optional Variables (for distributed training):**
- `PMI_LOCAL_RANK`: Local rank for distributed training (set by scheduler)
- `PMI_RANK`: Process rank (defaults to 0 if not set)
- `PMI_SIZE`: Total number of processes (defaults to 1 if not set)


