#!/bin/bash
#PBS -N dldl_hp_mpi
#PBS -A fusiondl_aesp
#PBS -k doe
# Stdout/stderr paths come from qsub -o/-e in start_hptune_parallel.sh (not hardcoded here).

set -e

if [ -z "${PROJECT_ROOT:-}" ]; then
    echo "ERROR: PROJECT_ROOT is not set (pass via qsub -v from start_hptune_parallel.sh)"
    exit 1
fi
if [ -z "${HPTUNE_CONTROLLER_NODES:-}" ]; then
    echo "ERROR: HPTUNE_CONTROLLER_NODES is not set"
    exit 1
fi

HPTUNE_DIR="${DLDL_HPTUNE_DIR:-$PROJECT_ROOT/data/hptune}"
LOG_DIR="$HPTUNE_DIR/controller_logs"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

require_file() {
    local path="$1"
    local description="$2"
    if [ -f "$path" ]; then
        return 0
    fi
    echo "ERROR: $description not found: $path"
    exit 1
}

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

# Always leave a trace even if PBS .OU is empty or delayed.
if [ -n "${PBS_JOBID:-}" ]; then
    exec > >(tee -a "${LOG_DIR}/controller_${PBS_JOBID%%.*}.log") 2>&1
fi

# Cray GPU-aware MPI (use system mpiexec, not conda)
module load craype-accel-nvidia80
module load cray-mpich

source /soft/applications/conda/2025-09-25/mconda3/etc/profile.d/conda.sh
conda activate base

require_file "$PROJECT_ROOT/.env" "required env file"
source "$PROJECT_ROOT/.env"

export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"

export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
export GPUS_PER_NODE
export HPTUNE_MPI_SIZE=$((HPTUNE_CONTROLLER_NODES * GPUS_PER_NODE))

echo "controller_parallel: PROJECT_ROOT=$PROJECT_ROOT HPTUNE_DIR=$HPTUNE_DIR"
echo "controller_parallel: HPTUNE_CONTROLLER_NODES=$HPTUNE_CONTROLLER_NODES GPUS_PER_NODE=$GPUS_PER_NODE HPTUNE_MPI_SIZE=$HPTUNE_MPI_SIZE"
echo "controller_parallel: mpiexec=$(command -v mpiexec)"

mpiexec -n "$HPTUNE_MPI_SIZE" -ppn "$GPUS_PER_NODE" python -m hptune_mpi