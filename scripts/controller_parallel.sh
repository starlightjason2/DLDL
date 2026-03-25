#!/usr/bin/env bash -l
#PBS -N dldl_hp_mpi
#PBS -A fusiondl_aesp
#PBS -k doe
# -q, -l select/walltime/filesystems are supplied by qsub from submit_hptune.sh

# ===========================================================================
# 0. Bootstrap log — written before set -e so failures here don't abort early
# ===========================================================================
umask 022
_BOOT="${TMPDIR}/dldl_parallel_${PBS_JOBID}.log"
{
    echo "=== bootstrap $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
    echo "PBS_JOBID=$PBS_JOBID  PBS_O_LOGNAME=$PBS_O_LOGNAME"
    echo "PROJECT_ROOT=$PROJECT_ROOT  HPTUNE_CONTROLLER_NODES=$HPTUNE_CONTROLLER_NODES"
} >> "$_BOOT" 2>&1

# ===========================================================================
# 1. Strict mode (after bootstrap block so its errors don't get swallowed)
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

# ===========================================================================
# 2. Cray/module environment
# ===========================================================================
# Source lmod if available (login-node initialisation may not carry over)
[[ -f /opt/cray/pe/lmod/lmod/init/bash ]] && \
    source /opt/cray/pe/lmod/lmod/init/bash  # shellcheck source=/dev/null

module load cuda/12.9
module load craype-accel-nvidia80   # required for GPU-aware MPICH (GTL library)
module load cray-mpich

# Snapshot Cray-modified paths *before* conda clobbers them
_CRAY_LD="$LD_LIBRARY_PATH"
_CRAY_PATH="$PATH"

# ===========================================================================
# 3. Project environment + conda
# ===========================================================================
cd "$PROJECT_ROOT"

set -a
source "$PROJECT_ROOT/.env"    # shellcheck source=/dev/null
set +a

source "$DLDL_CONDASH" && conda activate "$CONDA_ENV"  # shellcheck source=/dev/null

# Restore Cray paths on top so they win over conda's generic libs
export LD_LIBRARY_PATH="${_CRAY_LD}:${LD_LIBRARY_PATH}"
export PATH="${_CRAY_PATH}:${PATH}"
[[ -n "${CRAY_LD_LIBRARY_PATH:-}" ]] && \
    export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"

# ===========================================================================
# 4. MPICH ABI shim — add lib-abi-mpich paths if the module didn't expose them
# ===========================================================================
_add_mpich_abi_paths() {
    local abi
    abi="$(find /opt/cray/pe/mpich -name 'libmpi.so.12' -path '*/lib-abi-mpich/*' 2>/dev/null | head -1)"
    [[ -z "$abi" ]] && return 0
    local abi_dir lib_dir
    abi_dir="$(dirname "$abi")"
    lib_dir="$(dirname "$abi_dir")/lib"
    export LD_LIBRARY_PATH="${lib_dir}:${abi_dir}:${LD_LIBRARY_PATH}"
}
_add_mpich_abi_paths

# ===========================================================================
# 5. Python path + logging
# ===========================================================================
export PYTHONPATH="$PROJECT_ROOT/src"

LOG_DIR="$HPTUNE_DIR/controller_logs"
mkdir -p "$LOG_DIR"

# Copy bootstrap log to shared storage now that it's mounted
cp -f "$_BOOT" "${LOG_DIR}/bootstrap_${PBS_JOBID%%.*}.log" 2>/dev/null \
    || cat "$_BOOT" >> "${LOG_DIR}/bootstrap_${PBS_JOBID%%.*}.log"

# Tee all subsequent output to a persistent controller log
exec > >(tee -a "${LOG_DIR}/controller_${PBS_JOBID%%.*}.log") 2>&1

# ===========================================================================
# 6. Runtime info
# ===========================================================================
echo "=== controller main $(date -u '+%Y-%m-%dT%H:%M:%SZ') === (bootstrap: $_BOOT)"
echo "PROJECT_ROOT=$PROJECT_ROOT  HPTUNE_DIR=$HPTUNE_DIR"
echo "CONTROLLER_NODES=$HPTUNE_CONTROLLER_NODES  GPUS_PER_NODE=$GPUS_PER_NODE  TRIAL_NODES=$HPTUNE_TRIAL_NODES"

export HPTUNE_MPI_SIZE=$(( HPTUNE_CONTROLLER_NODES * GPUS_PER_NODE ))
echo "HPTUNE_MPI_SIZE=$HPTUNE_MPI_SIZE"
echo "mpiexec=$(command -v mpiexec)  python=$(command -v python)"

# Export vars the MPI dispatcher reads from the environment
export GPUS_PER_NODE HPTUNE_TRIAL_NODES TORCH_NCCL_TRACE_BUFFER_SIZE TORCH_NCCL_DUMP_ON_TIMEOUT

# ===========================================================================
# 7. Launch
# ===========================================================================
mpiexec \
    -n  "$HPTUNE_MPI_SIZE" \
    -ppn "$GPUS_PER_NODE" \
    python -m hptune_mpi