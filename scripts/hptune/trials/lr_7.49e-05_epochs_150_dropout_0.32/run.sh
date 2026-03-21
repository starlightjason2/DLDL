#!/bin/bash
#PBS -N dldl_hptune
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A fusiondl_aesp
#PBS -o /lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL/scripts/hptune/trials/lr_7.49e-05_epochs_150_dropout_0.32/train_%j.out
#PBS -e /lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL/scripts/hptune/trials/lr_7.49e-05_epochs_150_dropout_0.32/train_%j.err

# DLDL training - runs on compute node (GPU)

set -e

cd /lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL

# Activate conda
source /soft/applications/conda/2025-09-25/mconda3/etc/profile.d/conda.sh
conda activate base

set -a
source /lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL/scripts/hptune/trials/lr_7.49e-05_epochs_150_dropout_0.32/.env
set +a
# Write our own log (don't rely on PBS -o/-e)
exec > >(tee "$PROG_DIR/train_${PBS_JOBID}.log") 2>&1

# HPC-friendly settings
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Run training
python src/train.py
