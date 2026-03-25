#!/bin/bash
#PBS -N dldl_hp_ctl
#PBS -l select=1:system=polaris,place=scatter,walltime=1:00:00,filesystems=home:eagle
# Queue: set by scripts/start_hptune.sh (qsub -q "$HPTUNE_QUEUE"). Do not use #PBS -q small —
# on Polaris, "small" is not a valid -q name (use prod, debug, debug-scaling, etc.).
#PBS -A fusiondl_aesp
#PBS -k doe
#PBS -o /lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL/data/hptune/controller_logs/
#PBS -e /lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL/data/hptune/controller_logs/
set -e

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

parse_dispatch_trials() {
  printf '%s\n' "$1" | sed -n 's/^[[:space:]]*Next trial -> //p' | sed 's/[[:space:]]//g' | sed '/^$/d'
}

run_dispatcher_pass() {
  local output
  if ! output="$(python -m hptune_serial 2>&1)"; then
    echo "=== Python output ===" && echo "$output" && echo "ERROR: hptune_serial failed. Aborting chain." && exit 1
  fi
  printf '%s' "$output"
}

submit_trial() {
  local trial="$1" trial_dir="$HPTUNE_DIR/trials/$1"
  qsub -A fusiondl_aesp -q "$HPTUNE_QUEUE" \
    -l "select=1:system=polaris,place=scatter,walltime=$HPTUNE_TRAIN_WALLTIME,filesystems=home:eagle" \
    -v "PROJECT_ROOT=$PROJECT_ROOT,HPTUNE_CHAIN_ID=$HPTUNE_CHAIN_ID,TRIAL_ID=$trial,HPTUNE_QUEUE=$HPTUNE_QUEUE,HPTUNE_MAX_TRIALS=$HPTUNE_MAX_TRIALS" \
    -k doe -o "$LOG_DIR/" -e "$LOG_DIR/" "$trial_dir/run.sh"
}

cd "$PROJECT_ROOT"
set -a
# shellcheck source=/dev/null
source "$PROJECT_ROOT/.env"
set +a
# shellcheck source=/dev/null
source "$DLDL_CONDASH" && conda activate "$CONDA_ENV"

export PYTHONPATH="$PROJECT_ROOT/src"
LOG_DIR="$HPTUNE_DIR/controller_logs"
mkdir -p "$LOG_DIR"

echo "======================================"
echo "Controller started  Job ID: $PBS_JOBID  Chain ID: $HPTUNE_CHAIN_ID  $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "======================================"

echo "Running Bayesian Optimization..."
OUT="$(run_dispatcher_pass)"
echo "=== Python output ===" && echo "$OUT"

mapfile -t TRIALS < <(parse_dispatch_trials "$OUT")

if [ "${#TRIALS[@]}" -gt 1 ]; then
  echo "ERROR: Serial controller received multiple trial suggestions: ${TRIALS[*]}"
  exit 1
fi

if [ "${#TRIALS[@]}" -eq 0 ]; then
  echo "No more trials suggested by optimizer. Chain complete."
  echo "Chain $HPTUNE_CHAIN_ID finished at $(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$LOG_DIR/chain_summary.log"
  exit 0
fi

TRIAL="${TRIALS[0]}"
echo "=== TRIAL extracted: '$TRIAL' ==="
echo "Submitting trial job: $TRIAL"
TRIAL_JOB_ID="$(submit_trial "$TRIAL")" || { echo "ERROR: Trial qsub failed. Aborting chain."; exit 1; }

python -m hptune_serial --mark-running "$TRIAL"

echo "======================================"
echo "Chain step complete  Trial job: $TRIAL_JOB_ID  ($TRIAL)  Next controller from run.sh after training."
echo "======================================"

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ'),$HPTUNE_CHAIN_ID,$PBS_JOBID,$TRIAL,$TRIAL_JOB_ID" >> "$LOG_DIR/chain_steps.csv"
