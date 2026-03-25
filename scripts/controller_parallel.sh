#!/bin/bash -l
#PBS -N dldl_hp_mpi_decentralized

set -euo pipefail

cd "$PROJECT_ROOT"

source "$DLDL_CONDASH"
conda activate "$CONDA_ENV"

export PYTHONPATH="$PROJECT_ROOT/src"

export HPTUNE_MPI_SIZE=$(( HPTUNE_CONTROLLER_NODES * GPUS_PER_NODE ))

mpiexec \
    -n "$HPTUNE_MPI_SIZE" \
    -ppn "$GPUS_PER_NODE" \
    python -m hptune_mpi