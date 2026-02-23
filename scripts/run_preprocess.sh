#!/bin/bash
#PBS -N dldl_preprocess
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A fusiondl_aesp
#PBS -o preprocess_%j.out
#PBS -e preprocess_%j.err

# DLDL preprocessing - runs on compute node (CPU-only)

set -e

cd "${PBS_O_WORKDIR:-$(pwd)}"

# Activate conda
source /soft/applications/conda/2025-09-25/mconda3/etc/profile.d/conda.sh
conda activate base

# .env is loaded by Python (constants.py) when the script runs

# HPC-friendly settings (avoids thread exhaustion)
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Run preprocessing
python src/preprocess_data.py
