#!/bin/bash
#PBS -N dldl_hp_ctl
#PBS -l select=1:system=polaris,place=scatter,walltime=1:00:00,filesystems=home:eagle
#PBS -q small
#PBS -A fusiondl_aesp
#PBS -k doe
#PBS -o /lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL/data/hptune/controller_logs/
#PBS -e /lus/eagle/projects/fusiondl_aesp/starlightjason2/DLDL/data/hptune/controller_logs/
set -e

# --- 1. Setup ---
PROJECT_ROOT="${PROJECT_ROOT:-$HOME}"
HPTUNE_DIR="${DLDL_HPTUNE_DIR:-$PROJECT_ROOT/data/hptune}"
HPTUNE_QUEUE="${HPTUNE_QUEUE:-small}"
HPTUNE_PARALLELISM="${HPTUNE_PARALLELISM:-1}"
HPTUNE_DISPATCH_SLEEP_SECONDS="${HPTUNE_DISPATCH_SLEEP_SECONDS:-60}"
LOG_DIR="$HPTUNE_DIR/controller_logs"
USE_INTERNAL_WORKERS=0
# Prod parallel runs keep one shared allocation alive and launch workers inside
# it; other modes keep using external qsub submissions per trial.
if [ "$HPTUNE_PARALLELISM" -gt 1 ] && [ "$HPTUNE_QUEUE" = "prod" ]; then
    USE_INTERNAL_WORKERS=1
fi

cd "$PROJECT_ROOT"

source /soft/applications/conda/2025-09-25/mconda3/etc/profile.d/conda.sh
conda activate base

if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "ERROR: required env file not found: $PROJECT_ROOT/.env"
    exit 1
fi
source "$PROJECT_ROOT/.env"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export DLDL_HPTUNE_CHAIN_ID="${DLDL_HPTUNE_CHAIN_ID:-chain_${PBS_JOBID%%.*}}"

echo "======================================"
echo "Controller started"
echo "  Job ID    : $PBS_JOBID"
echo "  Chain ID  : $DLDL_HPTUNE_CHAIN_ID"
echo "  Timestamp : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "======================================"

submit_trial() {
    local trial="$1"
    local trial_dir="$HPTUNE_DIR/trials/$trial"

    if [ ! -d "$trial_dir" ]; then
        echo "ERROR: Trial directory does not exist: $trial_dir"
        return 1
    fi

    if [ ! -f "$trial_dir/run.sh" ]; then
        echo "ERROR: run.sh not found in trial directory: $trial_dir"
        return 1
    fi

    if [ "$USE_INTERNAL_WORKERS" -eq 1 ]; then
        local host=""
        local slot=""
        local idx
        # Reuse the first idle host slot so the dispatcher can keep a stable
        # pool of one-node trial workers inside the outer prod allocation.
        for idx in "${!WORKER_HOSTS[@]}"; do
            if [ -z "${WORKER_PIDS[$idx]:-}" ]; then
                host="${WORKER_HOSTS[$idx]}"
                slot="$idx"
                break
            fi
        done
        if [ -z "$host" ]; then
            echo "ERROR: No free worker slot available for $trial."
            return 1
        fi

        local hostfile="$trial_dir/mpiexec_hostfile.txt"
        printf '%s\n' "$host" > "$hostfile"
        echo "Launching internal worker: $trial on $host"
        # mpiexec binds this trial to a single allocated host while the
        # dispatcher continues refilling other free hosts asynchronously.
        mpiexec --hostfile "$hostfile" -n 1 -ppn 1 bash "$trial_dir/run.sh" \
            > "$trial_dir/launcher_${PBS_JOBID%%.*}.log" 2>&1 &
        local launcher_pid=$!

        WORKER_PIDS[$slot]="$launcher_pid"
        WORKER_TRIALS[$slot]="$trial"
        WORKER_HOSTFILES[$slot]="$hostfile"

        python -m bayesian_hp_tuning --mark-running "$trial"
        mkdir -p "$LOG_DIR"
        echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ'),$DLDL_HPTUNE_CHAIN_ID,$PBS_JOBID,$trial,internal:${host}:pid=${launcher_pid}" \
            >> "$LOG_DIR/chain_steps.csv"
        echo "Submitted: $trial -> internal:${host}:pid=${launcher_pid}"
        return 0
    fi

    echo "Submitting trial job: $trial"
    local trial_job_id
    trial_job_id=$(qsub \
        -A fusiondl_aesp \
        -q "$HPTUNE_QUEUE" \
        -l select=1:system=polaris,place=scatter,walltime=${HPTUNE_TRAIN_WALLTIME:-1:00:00},filesystems=home:eagle \
        -v "PROJECT_ROOT=$PROJECT_ROOT,DLDL_HPTUNE_CHAIN_ID=$DLDL_HPTUNE_CHAIN_ID,TRIAL_ID=$trial,HPTUNE_QUEUE=$HPTUNE_QUEUE,HPTUNE_PARALLELISM=$HPTUNE_PARALLELISM" \
        -k doe \
        -o "$LOG_DIR/" \
        -e "$LOG_DIR/" \
        "$trial_dir/run.sh") || {
            echo "ERROR: Trial qsub failed for $trial."
            return 1
        }

    python -m bayesian_hp_tuning --mark-running "$trial"
    mkdir -p "$LOG_DIR"
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ'),$DLDL_HPTUNE_CHAIN_ID,$PBS_JOBID,$trial,$trial_job_id" \
        >> "$LOG_DIR/chain_steps.csv"
    echo "Submitted: $trial -> $trial_job_id"
}

refresh_internal_workers() {
    local idx pid trial rc
    if [ "$USE_INTERNAL_WORKERS" -ne 1 ]; then
        return 0
    fi
    # Poll background launcher PIDs so finished slots become eligible for the
    # next dispatch pass without waiting for the entire allocation to drain.
    for idx in "${!WORKER_HOSTS[@]}"; do
        pid="${WORKER_PIDS[$idx]:-}"
        if [ -z "$pid" ]; then
            continue
        fi
        if kill -0 "$pid" 2>/dev/null; then
            continue
        fi
        trial="${WORKER_TRIALS[$idx]:-unknown}"
        wait "$pid"
        rc=$?
        echo "Internal worker finished: trial=$trial host=${WORKER_HOSTS[$idx]} pid=$pid exit_code=$rc"
        rm -f "${WORKER_HOSTFILES[$idx]:-}" 2>/dev/null || true
        unset "WORKER_PIDS[$idx]"
        unset "WORKER_TRIALS[$idx]"
        unset "WORKER_HOSTFILES[$idx]"
    done
}

internal_workers_active() {
    local idx active=0
    if [ "$USE_INTERNAL_WORKERS" -ne 1 ]; then
        echo 0
        return 0
    fi
    for idx in "${!WORKER_HOSTS[@]}"; do
        if [ -n "${WORKER_PIDS[$idx]:-}" ]; then
            active=$((active + 1))
        fi
    done
    echo "$active"
}

if [ "$HPTUNE_PARALLELISM" -le 1 ]; then
    # --- 2. Run Bayesian Optimizer ---
    echo "Running Bayesian Optimization..."
    OUT=$(python -m bayesian_hp_tuning 2>&1)
    EXIT_CODE=$?

    echo "=== Python output ==="
    echo "$OUT"
    echo "=== Python exit code: $EXIT_CODE ==="

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: bayesian_hp_tuning exited with code $EXIT_CODE. Aborting chain."
        exit 1
    fi

    # --- 3. Extract Trial ID ---
    # NOTE: Python prints exactly "Next trial -> <dir_name>" to stdout.
    # This grep/sed must stay in sync with _log_next_trial_marker in bayesian_hp_tuning.py.
    TRIAL=$(echo "$OUT" | grep 'Next trial ->' | tail -1 | sed 's/.*Next trial -> //; s/[[:space:]]//g')

    echo "=== TRIAL extracted: '$TRIAL' ==="

    if [ -z "$TRIAL" ]; then
        echo "No more trials suggested by optimizer. Chain complete."
        echo "Chain $DLDL_HPTUNE_CHAIN_ID finished at $(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
            >> "$LOG_DIR/chain_summary.log"
        exit 0
    fi

    submit_trial "$TRIAL" || exit 1

    echo "======================================"
    echo "Chain step complete"
    echo "  Trial      : $TRIAL"
    echo "  Next controller will be submitted by run.sh after training completes."
    echo "======================================"
    exit 0
fi

# --- Parallel dispatcher loop ---
echo "Running HPTune dispatcher with parallelism=$HPTUNE_PARALLELISM"
if [ "$USE_INTERNAL_WORKERS" -eq 1 ]; then
    if [ -z "${PBS_NODEFILE:-}" ] || [ ! -f "$PBS_NODEFILE" ]; then
        echo "ERROR: PBS_NODEFILE is required for internal prod workers."
        exit 1
    fi
    # One unique host per worker slot; the dispatcher only uses as many slots
    # as HPTUNE_PARALLELISM even if the outer allocation is larger.
    mapfile -t WORKER_HOSTS < <(sort -u "$PBS_NODEFILE")
    if [ "${#WORKER_HOSTS[@]}" -lt "$HPTUNE_PARALLELISM" ]; then
        echo "ERROR: Requested HPTUNE_PARALLELISM=$HPTUNE_PARALLELISM but only ${#WORKER_HOSTS[@]} unique hosts were allocated."
        exit 1
    fi
    declare -a WORKER_PIDS
    declare -a WORKER_TRIALS
    declare -a WORKER_HOSTFILES
    echo "Using internal workers on hosts: ${WORKER_HOSTS[*]}"
fi
while true; do
    refresh_internal_workers
    # The Python dispatcher is the single authority for trial creation and CSV
    # updates, which avoids races between parallel workers.
    OUT=$(python -m bayesian_hp_tuning 2>&1)
    EXIT_CODE=$?

    echo "=== Python output ==="
    echo "$OUT"
    echo "=== Python exit code: $EXIT_CODE ==="

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: bayesian_hp_tuning exited with code $EXIT_CODE. Aborting dispatcher."
        exit 1
    fi

    mapfile -t TRIALS < <(printf '%s\n' "$OUT" | sed -n 's/.*Next trial -> //p' | sed 's/[[:space:]]//g' | sed '/^$/d')
    COMPLETE=$(printf '%s\n' "$OUT" | sed -n 's/.*complete=\([01]\).*/\1/p' | tail -1)

    if [ ${#TRIALS[@]} -gt 0 ]; then
        for TRIAL in "${TRIALS[@]}"; do
            submit_trial "$TRIAL" || exit 1
        done
    fif

    if [ "$COMPLETE" = "1" ]; then
        if [ "$USE_INTERNAL_WORKERS" -eq 1 ] && [ "$(internal_workers_active)" -gt 0 ]; then
            echo "Dispatcher reports complete, waiting for internal workers to finish..."
        else
            echo "Dispatcher complete."
            echo "Chain $DLDL_HPTUNE_CHAIN_ID finished at $(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
                >> "$LOG_DIR/chain_summary.log"
            exit 0
        fi
    fi

    if [ "$USE_INTERNAL_WORKERS" -eq 1 ]; then
        echo "Internal workers active: $(internal_workers_active)"
    fi
    echo "Sleeping ${HPTUNE_DISPATCH_SLEEP_SECONDS}s before next dispatcher pass..."
    sleep "$HPTUNE_DISPATCH_SLEEP_SECONDS"
done