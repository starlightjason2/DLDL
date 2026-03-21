#!/bin/bash
#PBS -N dldl_train
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A fusiondl_aesp
#PBS -o /eagle/fusiondl_aesp/starlightjason2/DLDL/train_%j.out
#PBS -e /eagle/fusiondl_aesp/starlightjason2/DLDL/train_%j.err

# DLDL training - runs on compute node (GPU)

set -e

cd "${PBS_O_WORKDIR:-$(pwd)}"

# Activate conda
source /soft/applications/conda/2025-09-25/mconda3/etc/profile.d/conda.sh
conda activate base

# .env is loaded by Python (config.settings.load_settings) when the script runs

# HPC-friendly settings
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Run training
python src/train.py
