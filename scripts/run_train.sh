#!/bin/bash
# Train one model (single process, single GPU).
#PBS -N dldl_train
#PBS -l select=1:system=polaris,place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A fusiondl_aesp
#PBS -k doe
#PBS -j oe

set -euo pipefail

cd "$PROJECT_ROOT"

# Load every setting from .env into the environment.
set -a
source .env
set +a

# For a tuning trial, TRIAL_DIR/.env holds the hyperparameter overrides.
if [[ -f "${TRIAL_DIR:-}/.env" ]]; then
    set -a
    source "$TRIAL_DIR/.env"
    set +a
fi

# Activate the conda environment that has PyTorch etc.
source "$DLDL_CONDASH"
conda activate "$CONDA_ENV"
export PYTHONPATH="$PROJECT_ROOT/src"

python src/train.py
