#!/bin/bash
#PBS -N dldl_hp_mpi
#PBS -A fusiondl_aesp
#PBS -k doe
# Stdout/stderr: qsub -o/-e from start_hptune_parallel.sh

# Bootstrap: always write before set -e / before PROJECT_ROOT checks (PBS may drop .OU on instant failure).
umask 022
_BOOT="${TMPDIR:-/tmp}/dldl_parallel_${PBS_JOBID:-nojobid}.log"
{
  echo "=== bootstrap $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
  echo "PBS_JOBID=${PBS_JOBID:-}"
  echo "PBS_O_LOGNAME=${PBS_O_LOGNAME:-}"
  echo "PROJECT_ROOT=${PROJECT_ROOT:-<unset>}"
  echo "HPTUNE_CONTROLLER_NODES=${HPTUNE_CONTROLLER_NODES:-<unset>}"
  echo "DLDL_HPTUNE_DIR=${DLDL_HPTUNE_DIR:-<unset>}"
} >>"$_BOOT" 2>&1

# Lmod (non-login PBS batch often has no `module` until this is sourced)
if [ -f /opt/cray/pe/lmod/lmod/init/bash ]; then
  # shellcheck source=/dev/null
  source /opt/cray/pe/lmod/lmod/init/bash
fi

set -eo pipefail

if [ -z "${PROJECT_ROOT:-}" ]; then
  echo "ERROR: PROJECT_ROOT is not set" | tee -a "$_BOOT"
  exit 1
fi
if [ -z "${HPTUNE_CONTROLLER_NODES:-}" ]; then
  echo "ERROR: HPTUNE_CONTROLLER_NODES is not set" | tee -a "$_BOOT"
  exit 1
fi

HPTUNE_DIR="${DLDL_HPTUNE_DIR:-$PROJECT_ROOT/data/hptune}"
LOG_DIR="$HPTUNE_DIR/controller_logs"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

mkdir -p "$LOG_DIR"
# Mirror bootstrap into Eagle-backed controller_logs as soon as it exists
cp -f "$_BOOT" "${LOG_DIR}/bootstrap_${PBS_JOBID%%.*}.log" 2>/dev/null || cat "$_BOOT" >>"${LOG_DIR}/bootstrap_${PBS_JOBID%%.*}.log"

# Full trace: PBS stdout + controller_logs (tee duplicates to both)
exec > >(tee -a "${LOG_DIR}/controller_${PBS_JOBID%%.*}.log") 2>&1
echo "=== controller main $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
echo "bootstrap also at $_BOOT"

cd "$PROJECT_ROOT"

require_file() {
  local path="$1"
  local description="$2"
  if [ -f "$path" ]; then
    return 0
  fi
  echo "ERROR: $description not found: $path"
  exit 1
}

# Cray MPI + CUDA first, then conda — conda activate can reset LD_LIBRARY_PATH and break ~/.local mpi4py.
module load cuda/12.9
module load craype-accel-nvidia80
module load cray-mpich
_DLD_POST_CRAY="${LD_LIBRARY_PATH:-}"
_PST_POST_CRAY="${PATH:-}"

source /soft/applications/conda/2025-09-25/mconda3/etc/profile.d/conda.sh
conda activate base

export LD_LIBRARY_PATH="${_DLD_POST_CRAY}:${LD_LIBRARY_PATH:-}"
export PATH="${_PST_POST_CRAY}:${PATH:-}"
if [ -n "${CRAY_LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
fi

# Prepend Cray MPICH lib + lib-abi-mpich (conda/.env may drop PE_MPICH_*; find is robust).
_mpich_ld_from_find() {
  local abi abi_dir lib_dir
  abi="$(find /opt/cray/pe/mpich -path '*/lib-abi-mpich/libmpi.so.12' 2>/dev/null | head -1)"
  [ -n "$abi" ] || return 0
  abi_dir="$(dirname "$abi")"
  lib_dir="$(dirname "$abi_dir")/lib"
  export LD_LIBRARY_PATH="${lib_dir}:${abi_dir}:${LD_LIBRARY_PATH}"
}
_mpich_ld_from_find

require_file "$PROJECT_ROOT/.env" "required env file"
source "$PROJECT_ROOT/.env"

# .env may clobber LD_LIBRARY_PATH / set HPTUNE_TRIAL_NODES=2.
_mpich_ld_from_find
export HPTUNE_TRIAL_NODES=1

export HPTUNE_MAX_TRIALS="${HPTUNE_MAX_TRIALS:-10}"
# GPU-aware MPI transport requires GTL-linked binaries; conda python is not. Use device via CUDA in train.py.
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-0}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export GPUS_PER_NODE
export HPTUNE_TRIAL_NODES
export HPTUNE_MPI_SIZE=$((HPTUNE_CONTROLLER_NODES * GPUS_PER_NODE))

echo "controller_parallel: PROJECT_ROOT=$PROJECT_ROOT HPTUNE_DIR=$HPTUNE_DIR"
echo "controller_parallel: HPTUNE_CONTROLLER_NODES=$HPTUNE_CONTROLLER_NODES GPUS_PER_NODE=$GPUS_PER_NODE HPTUNE_TRIAL_NODES=$HPTUNE_TRIAL_NODES HPTUNE_MPI_SIZE=$HPTUNE_MPI_SIZE"
echo "controller_parallel: mpiexec=$(command -v mpiexec)"
echo "controller_parallel: python=$(command -v python)"
python -c "from mpi4py import MPI; print('mpi4py_smoketest', MPI.Get_library_version()[:80])" || {
  echo "ERROR: mpi4py cannot load libmpi; check LD_LIBRARY_PATH"
  exit 1
}

mpiexec -n "$HPTUNE_MPI_SIZE" -ppn "$GPUS_PER_NODE" python -m hptune_mpi
