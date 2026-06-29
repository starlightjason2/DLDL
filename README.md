[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# DLDL: Disruption Labeling with Deep Learning

A 1D CNN that uses plasma current to predict whether a shot disrupts (binary disruption classification). For labeling D-IIID shots with the 'ipspr15V' plasma current PTDATA signal.

Interactive polaris shell

```
qsub -I -A fusiondl_aesp -q debug \
  -l select=1:system=polaris:ngpus=4 \
  -l place=scatter \
  -l walltime=0:30:00 \
  -l filesystems=home:eagle
```

## Environment Setup

1. **Create `.env`** from a template. Each example file lists the **full** variable set required by the code (71+ keys); do not use a paths-only snippet as your only `.env`:
   ```bash
   cp .env.example .env              # generic template; edit paths
   cp .env.local.example .env        # local dev (relative paths, smaller HPTUNE_MAX_TRIALS)
   cp .env.polaris.example .env      # Polaris / Eagle paths
   # or symlink: ln -sf .env.polaris .env
   ```
   See [Environment Variables](#environment-variables) for descriptions.

2. **Install:**
   ```bash
   pip install -e .
   ```

`.env` is loaded by entry scripts (``load_dotenv``) before training or HPTune step jobs read ``os.environ``.

**Hyperparameter tuning:** Trial state is stored in **`{HPTUNE_DIR}/trials/trials.csv`** (see `service.trial_service`) with **Pydantic** models (`model.hp_trial.HPTuneTrial`). Trial directories live under `{HPTUNE_DIR}/trials/trial_*`.

### Environment Variables

Everything is read from the process environment (typically via a project-root `.env` loaded by `load_dotenv` in Python entry scripts, or `source`d by PBS shell scripts). There is no JSON config file. **Required** means the key must be set for that workflow.

#### Paths and run identity

| Variable | Required | Description |
|----------|----------|-------------|
| `PROJECT_ROOT` | ✓* | Absolute path to the repository root. Required by `scripts/*.sh`. |
| `DATA_DIR` | ✓ | Directory of raw signal `.txt` files (one per shot). |
| `LABELS_PATH` | ✓ | Shot list with disruption times. |
| `DATA_PATH` | ✓ | Full path to preprocessed dataset tensor (`.pt`). |
| `TRAIN_LABELS_PATH` | ✓ | Full path to preprocessed labels tensor (`.pt`). |
| `PROG_DIR` | ✓ | Training logs, checkpoints, and `graph.py` outputs. |
| `JOB_ID` | ✓ | Run identifier (filenames / logs). |

\*Set `PROJECT_ROOT` in `.env` for local and PBS runs. PBS scripts `source` the project `.env` and `cd` to `PROJECT_ROOT`.

#### Preprocessing and dataset

| Variable | Required | Description |
|----------|----------|-------------|
| `NORMALIZATION_TYPE` | ✓ | `scale`, `meanvar-whole`, or `meanvar-single`; must match how tensors were built and filename conventions. |
| `CPU_USE` | ✓ | Fraction of CPU cores for preprocessing workers (e.g. `0.2`). |
| `PREPROCESSOR_MAX_WORKERS` | ✓ | Max parallel preprocessing processes (e.g. `4`). |

#### Training hyperparameters (`train.py` / `model/cnn.py`)

| Variable | Required | Description |
|----------|----------|-------------|
| `EARLY_STOPPING_PATIENCE` | ✓ | Integer ≥ 1. |
| `LEARNING_RATE` | ✓ | Positive float. |
| `NUM_EPOCHS` | ✓ | Integer ≥ 1. |
| `LOG_INTERVAL` | ✓ | Integer ≥ 1. |
| `WEIGHT_DECAY` | ✓ | Float ≥ 0. |
| `DROPOUT_RATE` | ✓ | Float in [0, 1]. |
| `BATCH_SIZE` | ✓ | Integer ≥ 1. |
| `LR_SCHEDULER` | ✓ | `true` / `false` (or `1` / `0`, `yes` / `no`, `on` / `off`). |
| `LR_SCHEDULER_FACTOR` | ✓ | Float in (0, 1). |
| `LR_SCHEDULER_PATIENCE` | ✓ | Integer ≥ 1. |
| `GRADIENT_CLIP` | ✓ | Float ≥ 0. |
| `DATALOADER_NUM_WORKERS` | ✓ | DataLoader workers (forced to `0` when no GPU). |
| `CLS_POS_WEIGHT` | ✓ | Positive-class (disruptive) weight in BCE loss (float ≥ 0). `>1` favors recall over precision. Tunable via `HPTUNE_CLS_POS_WEIGHT_*`. |
| `DECISION_THRESHOLD` | ✓ | Sigmoid probability cutoff for the disruptive class (float in [0, 1]). Fixed (not tuned); typically `0.5`. |
| `FBETA` | optional (default `1.8`) | Beta for the F-beta score used for model selection (best checkpoint + early stopping) and as the HPTune objective, which is **maximized**. `beta>1` weights recall over precision (default `1.8`). This defines the objective and is not itself tuned. |

#### Architecture (`train.py` → `IpCNN`)

| Variable | Required | Description |
|----------|----------|-------------|
| `CONV1_FILTERS`, `CONV1_KERNEL`, `CONV1_PADDING` | ✓ | Conv1 layer (integers ≥ 0 where applicable). |
| `CONV2_FILTERS`, `CONV2_KERNEL`, `CONV2_PADDING` | ✓ | Conv2 layer. |
| `CONV3_FILTERS`, `CONV3_KERNEL`, `CONV3_PADDING` | ✓ | Conv3 layer. |
| `CONV4_FILTERS`, `CONV4_KERNEL`, `CONV4_PADDING` | ✓ | Conv4 layer. |
| `POOL_SIZE` | ✓ | Integer ≥ 1. |
| `FC1_SIZE`, `FC2_SIZE` | ✓ | Integer ≥ 1. |

#### Bayesian HPTune search space (`BayesianHPTuner`)

Comma-separated lists must not be empty (e.g. `HPTUNE_ALLOWED_EPOCHS=25,50,100`). All keys below are required when running `hptune_serial`.

| Variable | Required | Description |
|----------|----------|-------------|
| `HPTUNE_DIR` | ✓ | Absolute path to HPTune run root. Trials: `HPTUNE_DIR/trials/trial_*`. Log CSV: `HPTUNE_DIR/trials/trials.csv`. Controller logs: `HPTUNE_DIR/controller_logs/`. |
| `HPTUNE_LR_MIN`, `HPTUNE_LR_MAX` | ✓ | Learning-rate search bounds (float; min must be less than max). |
| `HPTUNE_DROPOUT_MIN`, `HPTUNE_DROPOUT_MAX` | ✓ | Dropout bounds (0–1; min must be less than max). |
| `HPTUNE_ALLOWED_EPOCHS` | ✓ | Comma-separated positive integers. |
| `HPTUNE_NUM_INITIAL_TRIALS` | ✓ | Integer ≥ 1. |
| `HPTUNE_WEIGHT_DECAY_LOG_MIN`, `HPTUNE_WEIGHT_DECAY_LOG_MAX` | ✓ | Log10 weight-decay bounds (min must be less than max). |
| `HPTUNE_ALLOWED_BATCH_SIZES` | ✓ | Comma-separated positive integers. |
| `HPTUNE_GRADIENT_CLIP_MIN`, `HPTUNE_GRADIENT_CLIP_MAX` | ✓ | `min` ≤ `max`. |
| `HPTUNE_LR_SCHEDULER_FACTOR_MIN`, `HPTUNE_LR_SCHEDULER_FACTOR_MAX` | ✓ | In (0, 1); min must be less than max. |
| `HPTUNE_LR_SCHEDULER_PATIENCE_MIN`, `HPTUNE_LR_SCHEDULER_PATIENCE_MAX` | ✓ | Integers ≥ 1, `min` ≤ `max`. |
| `HPTUNE_EARLY_STOPPING_PATIENCE_MIN`, `HPTUNE_EARLY_STOPPING_PATIENCE_MAX` | ✓ | Integers ≥ 1, `min` ≤ `max`. |
| `HPTUNE_CLS_POS_WEIGHT_MIN`, `HPTUNE_CLS_POS_WEIGHT_MAX` | ✓ | BCE positive-class weight bounds (float ≥ 0; `min` ≤ `max`). Higher values push toward recall. |
| `HPTUNE_RANDOM_INSERT_EVERY` | ✓ | Insert a random trial every N completed trials after warmup (integer ≥ 0). |
| `HPTUNE_EI_XI` | ✓ | Expected-improvement ξ for Bayesian optimization (float ≥ 0). |

#### HPTune step jobs and PBS

| Variable | Required | Description |
|----------|----------|-------------|
| `HPTUNE_MAX_TRIALS` | ✓ | Stop when this many trials exist and none are running or queued. |
| `HPTUNE_MAX_RETRIES` | ✓ | Requeue failed trials up to this many times. |
| `TRIAL_TIMEOUT` | ✓ | Seconds without log activity before a stale `RUNNING` trial (e.g. from a lost step) is requeued or failed. |
| `HPTUNE_QUEUE` | ✓† | PBS queue for the step jobs. Use `debug` (allows 1 running + 1 queued per user), not `debug-scaling` (1 job total). |
| `HPTUNE_TRAIN_WALLTIME` | ✓† | Walltime for each step job (passed to `qsub`). |
| `HPTUNE_CHAIN_ID` | ✓† |  |

†Required for the serial PBS chain (`scripts/start_hptune.sh`, `scripts/run_hptune.sh`).

#### Shell, conda, and runtime

| Variable | Required | Description |
|----------|----------|-------------|
| `DLDL_CONDASH` | ✓‡ | Path to `conda.sh` (sourced by PBS scripts). |
| `CONDA_ENV` | ✓‡ | Conda environment name. |
| `TMPDIR` | | Temp directory (recommended on HPC). |
| `RESET` | | Set to `1` when calling `scripts/start_hptune.sh` to wipe files under `HPTUNE_DIR`. |

‡Required by PBS training/preprocess/HPTune scripts.

#### Set by PBS or the tuner (not in `.env`)

| Variable | Description |
|----------|-------------|
| `PBS_JOBID` | PBS job id (set on compute nodes). |
| `WARM_START_CHECKPOINT` | Set by `run_step` for each `src/train.py` subprocess when `best_trial/` holds a checkpoint, so the trial warm-starts from the best model so far. Unset means a cold start. |
| `TRIAL_DIR` | Optional override path `HPTUNE_DIR/trials/<trial_id>` whose `.env` `scripts/run_train.sh` sources for a manual single-trial run. The HP-tune chain does **not** use it — `run_step` trains each trial in-process and passes its hyperparameters through the environment. |

#### Thread caps (recommended on HPC)

Set to `1` on crowded PBS nodes to avoid BLAS oversubscription (`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `TORCH_NUM_THREADS`). PBS scripts do not set these automatically; include them in `.env`.

## Workflow

```
Raw .txt → preprocess_data.py → .pt files → train.py → Model + logs → graph.py → Visualizations
```

**Config:** Set paths and every variable in the [Environment Variables](#environment-variables) tables in your project-root `.env`. Point `DATA_PATH` / `TRAIN_LABELS_PATH` at the `.pt` files that match `NORMALIZATION_TYPE` (e.g. filenames containing `meanvar-whole`).

### 1. Preprocessing

```bash
python src/preprocess_data.py
```

* Deletes the preprocessed files at `DATA_PATH` and `TRAIN_LABELS_PATH` (so the next build is fresh)
* Loads raw files from `DATA_DIR`, computes stats (max length, mean, std)
* Applies normalization, pads to uniform length, builds labels
* Writes tensors to `DATA_PATH` and `TRAIN_LABELS_PATH`
* Runs integrity check

**Normalization** (env `NORMALIZATION_TYPE`; typical value `meanvar-whole`):

| Value | Formula | Use case |
|-------|---------|----------|
| `scale` | `(x - min) / (max - min)` per shot | Relative patterns matter, magnitudes vary |
| `meanvar-whole` | `(x - μ) / σ` dataset-wide | Same scale across all shots |
| `meanvar-single` | `(x - μ) / σ` per shot | Per-shot standardization |

### 2. Validation (optional)

```bash
python src/validate.py
```

* Verifies preprocessed files exist
* Runs dataset integrity check (samples examples and compares to raw data)
* Options: `--num-checks N`, `--skip-integrity`, `--verbose`

### 3. Training

```bash
python src/train.py
```

* Loads preprocessed tensors at `DATA_PATH` and `TRAIN_LABELS_PATH` (must match preprocessing / `NORMALIZATION_TYPE`)
* Validates files exist, splits 80/10/10 train/dev/test
* Trains on a single process / single GPU
* Selects the best checkpoint and applies early stopping by **maximizing the validation F-beta** (`FBETA`, default `1.8`), favoring recall over precision
* Saves checkpoints and logs to `PROG_DIR`

Training hyperparameters (learning rate, epochs, architecture sizes, etc.) are read from environment variables (see [Environment Variables](#environment-variables)).

### 4. Visualization

```bash
python src/graph.py
```

* Loads training log CSV from `PROG_DIR/{JOB_ID}_training_log.csv`
* Creates a 2x2 subplot visualization with:
  - Training and Validation Loss
  - Validation Accuracy
  - Validation Precision and Recall
  - Validation F1 Score
* Saves plot to `PROG_DIR/{JOB_ID}_training_log_plot.png`

**Command-line Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--csv` | str | `PROG_DIR/{JOB_ID}_training_log.csv` | Path to the training log CSV file. If not specified, uses the default path based on `PROG_DIR` and `JOB_ID` from the environment. |
| `--output` | str | `PROG_DIR/{JOB_ID}_training_log_plot.png` | Path where the plot image will be saved. If not specified and `--show` is not used, defaults to that path. If `--show` is used without `--output`, the plot is only displayed and not saved. |
| `--show` | flag | False | Display the plot interactively using matplotlib's GUI. When used without `--output`, the plot is only displayed and not saved to disk. Can be combined with `--output` to both save and display. **Note:** Requires an interactive backend (PyQt5, PyQt6, or Tkinter). In headless environments, the plot will be saved to the default location instead. |

**Examples:**
```bash
# Use default paths from the environment
python src/graph.py

# Specify custom CSV file
python src/graph.py --csv path/to/training_log.csv

# Save plot to specific location
python src/graph.py --output path/to/output.png

# Display plot interactively (not saved)
python src/graph.py --show

# Save and display plot
python src/graph.py --output plot.png --show

# Use custom CSV and save to custom location
python src/graph.py --csv custom_log.csv --output plot.png
```

## Polaris Batch Jobs

Run preprocessing and training on compute nodes (not login nodes):

```bash
# From DLDL project root, with `.env` configured
cd /path/to/DLDL

# Conda activation is driven by DLDL_CONDASH / CONDA_ENV in `.env` (no edits needed).
# Both scripts default to #PBS -A fusiondl_aesp and walltime 1:00:00 — edit the
# #PBS headers in scripts/run_preprocess.sh / scripts/run_train.sh if your account,
# queue, or walltime differ.

# Submit preprocessing (CPU)
qsub scripts/run_preprocess.sh

# Submit training (GPU)
qsub scripts/run_train.sh

# Check status
qstat -u $USER
```

Output logs: `preprocess_<jobid>.out`, `train_<jobid>.out` (and `.err`).

### Bayesian Hyperparameter Tuning

HPTune reads the same project-root `.env` as the rest of the workflow. On Polaris, that is typically a symlink to `.env.polaris`.

The supported path is a **serial, self-resubmitting PBS chain**: each job plans the next trial, trains it in-process, records its score, then submits the next step job and exits. Exactly one trial runs per job, and at most one job is ever pending (the running job plus the queued next step), so it fits queues that cap you at one running + one queued job per user (Polaris `debug`). There is no separate controller job and no job dependency.

**Objective:** each trial's score (the `score` column in `trials.csv`) is its best validation F-beta (`FBETA`, default `1.8`). The optimizer **maximizes** this score, and `best_trial/` tracks the highest-scoring trial. Because F-beta with `beta>1` weights recall over precision, tuning pushes `cls_pos_weight` toward catching more disruptions.

#### Serial HPTune

Launch:

```bash
./scripts/start_hptune.sh
```

To wipe a previous run tree first:

```bash
RESET=1 ./scripts/start_hptune.sh
```

**Debug queue (smoke-test / short chain)** — point `HPTUNE_DIR` at an isolated tree, cap trials, and use the `debug` queue:

```bash
export HPTUNE_DIR=/path/to/data/hptune_debug
export HPTUNE_MAX_TRIALS=10
export HPTUNE_QUEUE=debug   # 1 running + 1 queued per user; debug-scaling allows only 1 total
./scripts/start_hptune.sh
```

Check the trial log on the login node after jobs start:

```bash
column -t -s, /path/to/data/hptune_debug/trials/trials.csv | head
```

Layout:
- `scripts/start_hptune.sh`: submits the first step job (and handles `RESET=1`).
- `scripts/run_hptune.sh`: the step job — sets up the environment (sources `.env`, activates conda) and `exec`s `python -m hptune_serial`. No logic lives in shell.
- `src/hptune_serial.py`: CLI entry point for `BayesianHPTuner.run_step()`.
- `src/model/bayesian_hptuner.py`: trial state, acquisition, retries, and `run_step` (plan → train in-process → record → submit next step).
- `src/util/pbs.py`: `submit_hptune_step` `qsub` helper; the job id is read directly from `qsub` stdout.
- `src/model/hp_trial.py` (`HPTuneTrial`): creates each trial directory and per-trial `.env`.
- `src/service/trial_service.py`: reads/writes `{HPTUNE_DIR}/trials/trials.csv`.
- `scripts/run_train.sh`: standalone training entrypoint (single trial / manual run); not used by the chain.

Step flow (all in-process within `run_step`; no stdout parsing, no job dependency):
1. The step job `exec`s `hptune_serial`, which refreshes trial status (ingesting any trial left over from a prior step) and plans a new trial if none is queued.
2. It marks the chosen trial `RUNNING` and trains it in-process by running `src/train.py` as a subprocess with that trial's hyperparameters. If `best_trial/` already holds a checkpoint, the trial **warm-starts** from it: it loads the best model's weights and optimizer state (the trial still uses its own `lr`/`weight_decay`), so a terminated run picks up off the best model found so far rather than from a random init. Each trial's epoch counter and early-stopping state still start fresh, keeping trial scores independent and comparable.
3. It reads the trial's best validation F-beta from its `training_log.csv` and records it (`COMPLETED` with a score, or retried/`FAILED`); `best_trial/` is refreshed.
4. Unless `HPTUNE_MAX_TRIALS` is reached, it submits the next `scripts/run_hptune.sh` step and exits. Because the queue allows one queued job, the next step waits until this one ends, then runs.
5. When the cap is reached and nothing is running or queued, the step logs "Chain complete." and submits nothing further.

If a step is ever lost (walltime kill, node failure) before it can submit the next one, just re-run `./scripts/start_hptune.sh` — `run_step` resumes from `trials.csv`, and because checkpoints are full-state (`model` + `optimizer` + `epoch` + selection counters), the next trial warm-starts from the best checkpoint kept in `best_trial/`. (`start_hptune.sh` preserves `trials.csv` and `*_best_params.pt`; only `RESET=1` clears the run, and even then it keeps the best checkpoints.)

Important serial env:
- `PROJECT_ROOT`, `HPTUNE_DIR`, `HPTUNE_QUEUE`, `HPTUNE_TRAIN_WALLTIME`
- `HPTUNE_MAX_TRIALS`, `HPTUNE_CHAIN_ID`
- `OPENBLAS_NUM_THREADS` / `OMP_NUM_THREADS` (set to `1` in `.env`) — PBS often allocates many CPUs to one process; uncapped BLAS can try to spawn that many threads and fail (`pthread_create` / `Exit_status=1` with little log output).

Data and logging layout:
- Trial state CSV: `{HPTUNE_DIR}/trials/trials.csv`
- Trial directories: `{HPTUNE_DIR}/trials/trial_*`
- Best-trial artifacts: `{HPTUNE_DIR}/trials/best_trial/`
- PBS stdout/stderr: `{HPTUNE_DIR}/controller_logs/` (`.OU`/`.ER`)
- Per-trial training logs/checkpoints: `{HPTUNE_DIR}/trials/trial_*/`

#### Interactive run (`qsub -I`)

To drive the workflow by hand on a compute node — useful for a quick smoke test or for debugging a single trial — first grab an interactive node:

```bash
qsub -I -A fusiondl_aesp -q debug \
  -l select=1:system=polaris:ngpus=4 \
  -l place=scatter \
  -l walltime=1:00:00 \
  -l filesystems=home:eagle
```

When the compute-node prompt appears, set up the environment once, from the repo root:

```bash
cd /eagle/fusiondl_aesp/starlightjason2/DLDL
set -a; source .env; set +a      # load and export every .env setting
source "$DLDL_CONDASH"           # make `conda` available
conda activate "$CONDA_ENV"
export PYTHONPATH="$PWD/src"
```

Then run whichever step you need directly:

```bash
python src/preprocess_data.py    # build the tensors (only if not already built)
python src/train.py              # train one model using the hyperparameters in .env
python -m hptune_serial          # one HPTune step: plan + train the next trial, then queue the next step
```

To launch the full serial Bayesian chain (each step job trains one trial and self-chains the next until `HPTUNE_MAX_TRIALS`):

```bash
./scripts/start_hptune.sh        # add RESET=1 to wipe the previous run first
```

> **Run the scripts; do not paste their lines.** Always execute a script with `./scripts/start_hptune.sh` (or `bash scripts/start_hptune.sh`). Pasting individual lines into an interactive shell breaks path detection: interactively `$0` is `-bash`, so `readlink -f "$0"` fails with `readlink: invalid option -- 'b'`, and under `set -u` a bare `${BASH_SOURCE[0]}` reports `unbound variable`.

## Quick Reference

* **Config:** No JSON—all settings come from `.env` / the process environment; see [Environment Variables](#environment-variables).
* **Output files:** Whatever paths you set in `DATA_PATH` and `TRAIN_LABELS_PATH` (often filenames include the normalization mode, e.g. `processed_dataset_meanvar-whole.pt`).
* **Changing normalization:** Update `NORMALIZATION_TYPE` and point `DATA_PATH` / `TRAIN_LABELS_PATH` at the matching `.pt` files.

See https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html