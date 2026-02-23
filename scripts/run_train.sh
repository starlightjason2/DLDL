#!/bin/bash
#PBS -N dldl_train
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=4:00:00
#PBS -l filesystems=home:eagle
#PBS -q prod
#PBS -A fusiondl_aesp
#PBS -o train_%j.out
#PBS -e train_%j.err

# DLDL training - runs on compute node (GPU)

set -e

cd "${PBS_O_WORKDIR:-$(pwd)}"

# Activate conda
source /soft/applications/conda/2025-09-25/mconda3/etc/profile.d/conda.sh
conda activate base

# .env is loaded by Python (constants.py) when the script runs

# HPC-friendly settings
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Run training
python src/train.py
