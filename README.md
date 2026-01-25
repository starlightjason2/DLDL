[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

### DL^2: Disruption Labeling with Deep Learning

A 1D convolutional neural network that uses the plasma current to produce a time of disruption if a disruption occurs. To be used for labeling D-IIID shots using the 'ipspr15V' plasma current PTDATA signal.

## Environment Setup

This project uses environment variables for configuration to easily switch between different environments (local development, Polaris HPC, etc.).

### Quick Start

1. **For local development:**
   ```bash
   cp .env.local.example .env.local
   # Edit .env.local with your local paths
   ln -s .env.local .env
   ```

2. **For Polaris HPC:**
   ```bash
   cp .env.polaris.example .env.polaris
   # Edit .env.polaris with your Polaris paths
   ln -s .env.polaris .env
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .
   ```

The `.env` file will be automatically loaded when the `constants` module is imported (requires `python-dotenv` package, which is included in dependencies).

### Environment Variables

All configuration uses the `DLDL_` prefix:
- `DLDL_DATA_DIR`: Raw signal data directory
- `DLDL_DATASET_DIR`: Preprocessed data directory  
- `DLDL_LABELS_PATH`: Path to labels file
- `DLDL_DATA_PATH`: Preprocessed dataset file
- `DLDL_TRAIN_LABELS_PATH`: Preprocessed labels file
- `DLDL_MAX_LENGTH_FILE`: Max length metadata file
- `DLDL_PROG_DIR`: Training progress/output directory
- `DLDL_JOB_ID`: Training run identifier

If no `.env` file is found, the code will use default values (pointing to `/data/...` paths).
