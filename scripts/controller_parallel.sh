#!/bin/bash
#PBS -N dldl_hp_mpi
#PBS -A fusiondl_aesp
#PBS -k doe
umask 022
_BOOT="$TMPDIR/dldl_parallel_${PBS_JOBID}.log"
{
  echo "=== bootstrap $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
  echo "PBS_JOBID=$PBS_JOBID PBS_O_LOGNAME=$PBS_O_LOGNAME"
  echo "PROJECT_ROOT=$PROJECT_ROOT HPTUNE_CONTROLLER_NODES=$HPTUNE_CONTROLLER_NODES"
} >>"$_BOOT" 2>&1

if [ -f /opt/cray/pe/lmod/lmod/init/bash ]; then
  # shellcheck source=/dev/null
  source /opt/cray/pe/lmod/lmod/init/bash
fi

set -eo pipefail

cd "$PROJECT_ROOT"
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

module load cuda/12.9
module load craype-accel-nvidia80
module load cray-mpich
_DLD_POST_CRAY="$LD_LIBRARY_PATH"
_PST_POST_CRAY="$PATH"

set -a
# shellcheck source=/dev/null
source "$PROJECT_ROOT/.env"
set +a

# shellcheck source=/dev/null
source "$DLDL_CONDASH" && conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="${_DLD_POST_CRAY}:${LD_LIBRARY_PATH}"
export PATH="${_PST_POST_CRAY}:${PATH}"
[ -z "$CRAY_LD_LIBRARY_PATH" ] || export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"

_mpich_ld_from_find() {
  local abi abi_dir lib_dir
  abi="$(find /opt/cray/pe/mpich -path '*/lib-abi-mpich/libmpi.so.12' 2>/dev/null | head -1)"
  [ -n "$abi" ] || return 0
  abi_dir="$(dirname "$abi")"
  lib_dir="$(dirname "$abi_dir")/lib"
  export LD_LIBRARY_PATH="${lib_dir}:${abi_dir}:${LD_LIBRARY_PATH}"
}
_mpich_ld_from_find

export PYTHONPATH="$PROJECT_ROOT/src"
LOG_DIR="$HPTUNE_DIR/controller_logs"

mkdir -p "$LOG_DIR"
cp -f "$_BOOT" "${LOG_DIR}/bootstrap_${PBS_JOBID%%.*}.log" 2>/dev/null || cat "$_BOOT" >>"${LOG_DIR}/bootstrap_${PBS_JOBID%%.*}.log"

exec > >(tee -a "${LOG_DIR}/controller_${PBS_JOBID%%.*}.log") 2>&1
echo "=== controller main $(date -u '+%Y-%m-%dT%H:%M:%SZ') === bootstrap also at $_BOOT"

export GPUS_PER_NODE HPTUNE_TRIAL_NODES
export HPTUNE_MPI_SIZE=$((HPTUNE_CONTROLLER_NODES * GPUS_PER_NODE))
export TORCH_NCCL_TRACE_BUFFER_SIZE TORCH_NCCL_DUMP_ON_TIMEOUT

echo "controller_parallel: PROJECT_ROOT=$PROJECT_ROOT HPTUNE_DIR=$HPTUNE_DIR"
echo "controller_parallel: HPTUNE_CONTROLLER_NODES=$HPTUNE_CONTROLLER_NODES GPUS_PER_NODE=$GPUS_PER_NODE HPTUNE_TRIAL_NODES=$HPTUNE_TRIAL_NODES HPTUNE_MPI_SIZE=$HPTUNE_MPI_SIZE"
echo "controller_parallel: running mpiexec=$(command -v mpiexec) python=$(command -v python)"

mpiexec -n "$HPTUNE_MPI_SIZE" -ppn "$GPUS_PER_NODE" python -m hptune_mpi
