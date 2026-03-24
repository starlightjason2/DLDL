#!/bin/bash
#PBS -N dldl_train
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -l filesystems=home:eagle
#PBS -q small
#PBS -A fusiondl_aesp
#PBS -k doe
#PBS -o /lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL/data/hptune/controller_logs/
#PBS -e /lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL/data/hptune/controller_logs/

# DLDL training - runs on compute node (GPU)
set -e

# __HPTUNE_ENV_INJECT__
# .env is loaded by Python (config.settings.load_settings) when the script runs

# __HPTUNE_CD_OVERRIDE__
cd "${PBS_O_WORKDIR:-$(pwd)}"

# Activate conda
source /soft/applications/conda/2025-09-25/mconda3/etc/profile.d/conda.sh
conda activate base

# HPC-friendly settings (PBS may allocate many CPUs; cap BLAS threads)
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"

# Run training
BEST_PARAMS_PATH="$PROG_DIR/${JOB_ID}_best_params.pt"
if [ -f "$BEST_PARAMS_PATH" ]; then
    echo "Existing checkpoint found at $BEST_PARAMS_PATH; skipping retraining."
else
    python src/train.py
fi

# Tail log for convenience
if [ -n "$PROG_DIR" ]; then
    echo "---- tail tee log: $PROG_DIR/train_${PBS_JOBID%%.*}.log ----"
    tail -n 100 "$PROG_DIR/train_${PBS_JOBID%%.*}.log" 2>/dev/null || true
    echo "---- LIVE follow: tail -F $PROG_DIR/train_${PBS_JOBID%%.*}.log ----"
else
    _REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    echo "---- tail PBS stdout: ${_REPO}/train_${PBS_JOBID%%.*}.out ----"
    tail -n 100 "${_REPO}/train_${PBS_JOBID%%.*}.out" 2>/dev/null || true
    echo "---- LIVE follow: tail -F ${_REPO}/train_${PBS_JOBID%%.*}.out ----"
fi