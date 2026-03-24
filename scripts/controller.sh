#!/bin/bash
#PBS -N dldl_hp_ctl
#PBS -l select=1:system=polaris,place=scatter,walltime=1:00:00,filesystems=home:eagle
#PBS -q small
#PBS -A fusiondl_aesp
#PBS -k doe
#PBS -o /lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL/data/hptune/controller_logs/
#PBS -e /lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL/data/hptune/controller_logs/
set -e

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(readlink -f "$SCRIPT_DIR/..")}"
HPTUNE_DIR="${DLDL_HPTUNE_DIR:-$PROJECT_ROOT/data/hptune}"
HPTUNE_QUEUE="${HPTUNE_QUEUE:-small}"
LOG_DIR="$HPTUNE_DIR/controller_logs"

require_file() {
    local path="$1"
    local description="$2"
    if [ -f "$path" ]; then
        return 0
    fi
    echo "ERROR: $description not found: $path"
    exit 1
}

parse_dispatch_trials() {
    # Anchor to line start: loguru also logs "Next trial -> ..." and .* would match twice.
    printf '%s\n' "$1" | sed -n 's/^[[:space:]]*Next trial -> //p' | sed 's/[[:space:]]//g' | sed '/^$/d'
}

run_dispatcher_pass() {
    local output
    if ! output="$(python -m hptune_serial 2>&1)"; then
        echo "=== Python output ==="
        echo "$output"
        echo "ERROR: hptune_serial failed. Aborting chain."
        exit 1
    fi
    printf '%s' "$output"
}

submit_trial() {
    local trial="$1"
    local trial_dir="$HPTUNE_DIR/trials/$trial"

    if [ ! -d "$trial_dir" ]; then
        echo "ERROR: Trial directory does not exist: $trial_dir"
        echo "  The optimizer suggested '$trial' but no matching directory was found."
        exit 1
    fi
    require_file "$trial_dir/run.sh" "trial launcher"

    qsub \
        -A fusiondl_aesp \
        -q "$HPTUNE_QUEUE" \
        -l "select=1:system=polaris,place=scatter,walltime=${HPTUNE_TRAIN_WALLTIME:-1:00:00},filesystems=home:eagle" \
        -v "PROJECT_ROOT=$PROJECT_ROOT,DLDL_HPTUNE_CHAIN_ID=$DLDL_HPTUNE_CHAIN_ID,TRIAL_ID=$trial,HPTUNE_QUEUE=$HPTUNE_QUEUE,DLDL_HPTUNE_DIR=$HPTUNE_DIR,HPTUNE_MAX_TRIALS=${HPTUNE_MAX_TRIALS:-10}" \
        -k doe \
        -o "$LOG_DIR/" \
        -e "$LOG_DIR/" \
        "$trial_dir/run.sh"
}

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

source /soft/applications/conda/2025-09-25/mconda3/etc/profile.d/conda.sh
conda activate base

require_file "$PROJECT_ROOT/.env" "required env file"
source "$PROJECT_ROOT/.env"

# PBS assigns many CPUs to one node; OpenBLAS/NumPy otherwise tries to spawn
# that many threads and pthread_create can fail (Exit_status=1, no Python output).
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export DLDL_HPTUNE_CHAIN_ID="${DLDL_HPTUNE_CHAIN_ID:-chain_${PBS_JOBID%%.*}}"

echo "======================================"
echo "Controller started"
echo "  Job ID    : $PBS_JOBID"
echo "  Chain ID  : $DLDL_HPTUNE_CHAIN_ID"
echo "  Timestamp : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "======================================"

echo "Running Bayesian Optimization..."
OUT="$(run_dispatcher_pass)"
echo "=== Python output ==="
echo "$OUT"

# NOTE: Python prints exactly "Next trial -> <dir_name>" as its own stdout line (loguru may duplicate text).
# parse_dispatch_trials matches only lines starting with that prefix.
mapfile -t TRIALS < <(parse_dispatch_trials "$OUT")

if [ "${#TRIALS[@]}" -gt 1 ]; then
    echo "ERROR: Serial controller received multiple trial suggestions: ${TRIALS[*]}"
    exit 1
fi

if [ "${#TRIALS[@]}" -eq 0 ]; then
    echo "No more trials suggested by optimizer. Chain complete."
    echo "Chain $DLDL_HPTUNE_CHAIN_ID finished at $(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
        >> "$LOG_DIR/chain_summary.log"
    exit 0
fi

TRIAL="${TRIALS[0]}"
echo "=== TRIAL extracted: '$TRIAL' ==="

echo "Submitting trial job: $TRIAL"
if ! TRIAL_JOB_ID="$(submit_trial "$TRIAL")"; then
    echo "ERROR: Trial qsub failed. Aborting chain."
    exit 1
fi

python -m hptune_serial --mark-running "$TRIAL"

echo "======================================"
echo "Chain step complete"
echo "  Trial job : $TRIAL_JOB_ID  ($TRIAL)"
echo "  Next controller will be submitted by run.sh after training completes."
echo "======================================"

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ'),$DLDL_HPTUNE_CHAIN_ID,$PBS_JOBID,$TRIAL,$TRIAL_JOB_ID" \
    >> "$LOG_DIR/chain_steps.csv"
