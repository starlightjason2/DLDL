#!/bin/bash
#PBS -N dldl_preprocess
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -l filesystems=home:eagle
#PBS -q small
#PBS -A fusiondl_aesp
#PBS -o /lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL/data/processed_data
#PBS -e /lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL/data/processed_data
set -euo pipefail

# This script lives in <repo>/scripts, so the repo root is two levels up.
project_root="$(dirname "$(dirname "$(readlink -f "$0")")")"
cd "$project_root"

# Load every setting from .env into the environment.
set -a
source .env
set +a

# Activate the conda environment that has PyTorch etc.
source "$DLDL_CONDASH"
conda activate "$CONDA_ENV"

python src/preprocess_data.py
