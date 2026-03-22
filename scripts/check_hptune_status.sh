#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(readlink -f "$(cd "$(dirname "$0")/.." && pwd)")"
HPTUNE_DATA_DIR="${DLDL_HPTUNE_DIR:-$PROJECT_ROOT/data/hptune}"
TRIALS_DIR="$HPTUNE_DATA_DIR/trials"
LOG_DIR="$HPTUNE_DATA_DIR/controller_logs"
TRIALS_LOG="$TRIALS_DIR/trials_log.csv"
TAIL_LINES=40
FOLLOW=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [--follow] [--lines N]

Checks the current HPTune workflow status and prints the latest controller/trial info.
By default, it shows the last ${TAIL_LINES} lines of the newest training log.

Options:
  -f, --follow     Follow the newest training log with tail -f
  -n, --lines N    Number of lines to show from the newest training log (default: ${TAIL_LINES})
  -h, --help       Show this help text
EOF
}

latest_file() {
    local pattern=$1
    shopt -s nullglob
    local files=($pattern)
    shopt -u nullglob
    if [ ${#files[@]} -eq 0 ]; then
        return 1
    fi
    ls -1t "${files[@]}" 2>/dev/null | head -n 1
}

status_label() {
    case "$1" in
        0) echo "completed" ;;
        -1) echo "running" ;;
        -2) echo "queued" ;;
        *) echo "unknown($1)" ;;
    esac
}

while [ $# -gt 0 ]; do
    case "$1" in
        -f|--follow)
            FOLLOW=1
            ;;
        -n|--lines)
            if [ $# -lt 2 ]; then
                echo "ERROR: --lines requires a value" >&2
                exit 1
            fi
            TAIL_LINES=$2
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

echo "== HPTune Status =="
echo "Project root : $PROJECT_ROOT"
echo "HPTune data  : $HPTUNE_DATA_DIR"
echo "Trials dir   : $TRIALS_DIR"
echo "Logs dir     : $LOG_DIR"
echo

echo "== PBS Jobs =="
if command -v qstat >/dev/null 2>&1; then
    qstat -u "${USER:-starlightjason2}" 2>/dev/null || true
else
    echo "qstat not found"
fi
echo

echo "== Trial Log =="
if [ -f "$TRIALS_LOG" ]; then
    IFS=' ' read -r row_count done_count running_count queued_count < <(
        awk -F, '
            NR > 1 && NF {
                rows++
                if ($13 == 0) done++
                else if ($13 == -1) running++
                else if ($13 == -2) queued++
            }
            END { print rows+0, done+0, running+0, queued+0 }
        ' "$TRIALS_LOG"
    )
    echo "Rows      : $row_count"
    echo "Completed : $done_count"
    echo "Running   : $running_count"
    echo "Queued    : $queued_count"
    if [ "$row_count" -gt 0 ]; then
        latest_row=$(tail -n 1 "$TRIALS_LOG")
        IFS=, read -r trial_id lr epochs dropout weight_decay batch_size gradient_clip lr_scheduler lr_scheduler_factor lr_scheduler_patience early_stopping_patience val_loss status <<< "$latest_row"
        echo "Latest    : $trial_id ($(status_label "$status"))"
        echo "Val loss  : $val_loss"
    fi
else
    echo "No trials log found at $TRIALS_LOG"
fi
echo

echo "== Latest Controller Output =="
latest_controller_ou=$(latest_file "$LOG_DIR/*.OU" || true)
if [ -n "${latest_controller_ou:-}" ]; then
    echo "$latest_controller_ou"
    tail -n 10 "$latest_controller_ou" || true
else
    echo "No controller stdout log found"
fi
echo

latest_training_log=$(latest_file "$TRIALS_DIR/trial_*/train_*.log" || true)
echo "== Latest Training Log =="
if [ -z "${latest_training_log:-}" ]; then
    echo "No training log found under $TRIALS_DIR"
    exit 0
fi

echo "$latest_training_log"
if [ "$FOLLOW" -eq 1 ]; then
    echo
    echo "Following latest training log..."
    tail -f "$latest_training_log"
else
    tail -n "$TAIL_LINES" "$latest_training_log"
fi
