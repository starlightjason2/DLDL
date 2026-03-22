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

# DLDL preprocessing - runs on compute node (CPU-only)

set -e

cd "${PBS_O_WORKDIR:-$(pwd)}"

# Activate conda
source /soft/applications/conda/2025-09-25/mconda3/etc/profile.d/conda.sh
conda activate base

# .env is loaded by Python (config.settings.load_settings) when the script runs

# HPC-friendly settings (avoids thread exhaustion)
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Run preprocessing
python src/preprocess_data.py

_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "---- tail PBS stdout (last 100 lines): ${_REPO}/preprocess_${PBS_JOBID%%.*}.out ----"
tail -n 100 "${_REPO}/preprocess_${PBS_JOBID%%.*}.out" 2>/dev/null || tail -n 100 "${_REPO}/preprocess_${PBS_JOBID}.out" 2>/dev/null || true
echo "---- LIVE follow (run on login node): tail -F ${_REPO}/preprocess_${PBS_JOBID%%.*}.out ----"
