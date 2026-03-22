#!/bin/bash
#PBS -N dldl_hptune
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A fusiondl_aesp
#PBS -o /dev/null
#PBS -e /dev/null

# DLDL training - runs on compute node (GPU)

set -e

cd /lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL

# Activate conda
source /soft/applications/conda/2025-09-25/mconda3/etc/profile.d/conda.sh
conda activate base

set -a
source /lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL/scripts/hptune/trials/trial_1/.env
set +a
# Single merged log via tee; discard PBS -o/-e to avoid duplicating the same stream
exec > >(tee "$PROG_DIR/train_${PBS_JOBID}.log") 2>&1

# HPC-friendly settings
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Run training
# --- fast test: training commented out; restore for real runs ---
# python src/train.py
echo "[run_train.sh] fast test: skipped python src/train.py"

_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "---- tail PBS stdout (last 100 lines): ${_REPO}/train_${PBS_JOBID%%.*}.out ----"
tail -n 100 "${_REPO}/train_${PBS_JOBID%%.*}.out" 2>/dev/null || tail -n 100 "${_REPO}/train_${PBS_JOBID}.out" 2>/dev/null || true
echo "---- LIVE follow (run on login node): tail -F ${_REPO}/train_${PBS_JOBID%%.*}.out ----"
