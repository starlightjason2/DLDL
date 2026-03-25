[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# DLDL: Disruption Labeling with Deep Learning

A 1D CNN that uses plasma current to predict disruption time. For labeling D-IIID shots with the 'ipspr15V' plasma current PTDATA signal.

## Environment Setup

1. **Create `.env`** from a template:
   ```bash
   cp .env.local.example .env.local && ln -s .env.local .env   # local
   # or
   cp .env.polaris.example .env.polaris && ln -s .env.polaris .env   # Polaris
   ```

2. **Install:**
   ```bash
   pip install -e .
   ```

`.env` is loaded when you call ``Settings.load()`` from ``config.settings`` (typically at script startup).

**Hyperparameter tuning:** Trial state (serial and parallel controllers) is stored in **SQLite** (URL from `DB_CONNECTION`, typically under `data/hptune/trials/trials_log.db`) using **SQLAlchemy** 2.0 ORM (`database.connection` + `database.tables.Trial`, same layout as [this FastAPI SQLAlchemy template](https://github.com/mdhishaamakhtar/fastapi-sqlalchemy-postgres-template/tree/master/database)) with **Pydantic** models (`schemas.trial_schema.HPTuneTrial`) and persistence helpers (`service.trial_service`: `get_trials`, `save_trials`).

HP-tune requires the stdlib **sqlite3** module. On some minimal or HPC Python builds it may be missing—check with `python -c "import sqlite3; print(sqlite3.sqlite_version)"` and install your OS package for `python-sqlite` / full Python if needed.

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATA_DIR` | ✓ | Raw signal .txt files |
| `LABELS_PATH` | ✓ | Shot list with disruption times |
| `DATA_PATH` | ✓ | Full path to preprocessed dataset tensor (`.pt`) |
| `TRAIN_LABELS_PATH` | ✓ | Full path to preprocessed labels tensor (`.pt`) |
| `PROG_DIR` | ✓ | Logs, checkpoints, and `graph.py` plot output |
| `JOB_ID` | ✓ | Run identifier |
| `NORMALIZATION_TYPE` | | `scale`, `meanvar-whole`, or `meanvar-single` (default `meanvar-whole`). |
| `CPU_USE` | | Fraction of CPU cores for preprocessing, 0-1 (default `0.2`; use `0.5-1.0` on HPC with more RAM) |
| `PREPROCESSOR_MAX_WORKERS` | | Hard cap on preprocessing workers (default `4`; avoids fork/resource issues on HPC) |
| `DATALOADER_NUM_WORKERS` | | DataLoader workers for training (default `4`; forced to `0` when no GPU) |
| `PMI_LOCAL_RANK` | | Distributed training (scheduler) |
| `PMI_RANK` | | Process rank (default 0) |
| `PMI_SIZE` | | World size (default 1) |

**Hyperparameter tuning (controllers):** `BayesianHPTuner` and the trial ORM expect these to be set in the environment (no silent defaults—misconfiguration fails at validation). Paths must match: `TRIALS_DIR` is the parent of `trial_*` folders; `DB_CONNECTION` is the SQLAlchemy URL for the same run (e.g. `sqlite:////.../trials_log.db`).

| Variable | Required | Description |
|----------|----------|-------------|
| `TRIALS_DIR` | ✓ | Root directory for per-trial folders `trial_*` (also used when resolving trial paths in `HPTuneTrial`) |
| `DB_CONNECTION` | ✓ | SQLAlchemy database URL for the trial log (SQLite file path in URL form) |

## Workflow

```
Raw .txt → preprocess_data.py → .pt files → train.py → Model + logs → graph.py → Visualizations
```

**Config:** Set `NORMALIZATION_TYPE` and `CPU_USE` in `.env` (or env). Point `DATA_PATH` / `TRAIN_LABELS_PATH` at the `.pt` files that match that normalization (e.g. filenames containing `meanvar-whole`). Scripts call `Settings.load()` for hyperparameters from `dldl.json` (defaults under `defaultTraining`, architecture under `architecture`, plus env overrides).

### 1. Preprocessing

```bash
python src/preprocess_data.py
```

* Deletes the preprocessed files at `DATA_PATH` and `TRAIN_LABELS_PATH` (so the next build is fresh)
* Loads raw files from `DATA_DIR`, computes stats (max length, mean, std)
* Applies normalization, pads to uniform length, builds labels
* Writes tensors to `DATA_PATH` and `TRAIN_LABELS_PATH`
* Runs integrity check

**Normalization** (env `NORMALIZATION_TYPE`; default `meanvar-whole`):

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
* Uses distributed training if `PMI_*` are set
* Saves checkpoints and logs to `PROG_DIR`

Hyperparameters (lr, epochs, etc.) are in `train.py`.

### 4. Visualization

```bash
python -m src.graph
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
| `--csv` | str | `PROG_DIR/{JOB_ID}_training_log.csv` | Path to the training log CSV file. If not specified, uses the default path based on `PROG_DIR` and `JOB_ID` from config. |
| `--output` | str | `PROG_DIR/{JOB_ID}_training_log_plot.png` | Path where the plot image will be saved. If not specified and `--show` is not used, defaults to that path. If `--show` is used without `--output`, the plot is only displayed and not saved. |
| `--show` | flag | False | Display the plot interactively using matplotlib's GUI. When used without `--output`, the plot is only displayed and not saved to disk. Can be combined with `--output` to both save and display. **Note:** Requires an interactive backend (PyQt5, PyQt6, or Tkinter). In headless environments, the plot will be saved to the default location instead. |

**Examples:**
```bash
# Use default paths from config
python -m src.graph

# Specify custom CSV file
python -m src.graph --csv path/to/training_log.csv

# Save plot to specific location
python -m src.graph --output path/to/output.png

# Display plot interactively (not saved)
python -m src.graph --show

# Save and display plot
python -m src.graph --output plot.png --show

# Use custom CSV and save to custom location
python -m src.graph --csv custom_log.csv --output plot.png
```

## Polaris Batch Jobs

Run preprocessing and training on compute nodes (not login nodes):

```bash
# From DLDL project root, with `.env` configured
cd /path/to/DLDL

# Edit scripts/run_preprocess.sh and scripts/run_train.sh:
# - Uncomment and set your conda activation
# - Adjust #PBS -A if your account differs from fusiondl_aesp

# Submit preprocessing (CPU, ~2 hr walltime)
qsub scripts/run_preprocess.sh

# Submit training (GPU, ~4 hr walltime)
qsub scripts/run_train.sh

# Check status
qstat -u $USER
```

Output logs: `preprocess_<jobid>.out`, `train_<jobid>.out` (and `.err`).

### Bayesian Hyperparameter Tuning

HPTune reads the same project-root `.env` as the rest of the workflow. On Polaris, that is typically a symlink to `.env.polaris`.

There are now two HPTune execution modes:
- Serial chain: one trial job at a time, with each completed trial requeueing the next controller.
- MPI controller: one controller allocation launches multiple distributed trials and keeps dispatching until the search budget is exhausted.

#### Serial HPTune

Launch:

```bash
./scripts/start_hptune.sh
```

**Debug queue (smoke-test SQLite / short chain)** — point `TRIALS_DIR` / `DB_CONNECTION` at an isolated tree, cap trials, and use your site’s debug queue (example: `debug-scaling`):

```bash
export TRIALS_DIR=/path/to/data/hptune_debug/trials
export DB_CONNECTION=sqlite:////path/to/data/hptune_debug/trials/trials_log.db
export HPTUNE_MAX_TRIALS=10
export HPTUNE_QUEUE=debug-scaling   # or your site's debug queue name
./scripts/start_hptune.sh
```

Check the DB on the login node after jobs start (adjust the path to match `DB_CONNECTION`):

```bash
sqlite3 /path/to/data/hptune_debug/trials/trials_log.db 'SELECT trial_id, status FROM trials;'
```

Layout:
- `scripts/start_hptune.sh`: submits the serial controller job.
- `scripts/controller.sh`: runs one Bayesian optimizer pass, submits at most one queued trial, then exits.
- `src/schemas/trial_schema.py` (`HPTuneTrial`): generates each trial directory, per-trial `.env`, and `run.sh`.
- `src/service/trial_service.py`: loads and persists trial rows (`get_trials`, `save_trials`).
- Trial `run.sh`: runs `python src/train.py` and, on success, submits the next `scripts/controller.sh`.

Important serial env:
- `HPTUNE_QUEUE`
- `HPTUNE_CONTROLLER_WALLTIME`
- `HPTUNE_TRAIN_WALLTIME`
- `HPTUNE_MAX_TRIALS`
- `OPENBLAS_NUM_THREADS` / `OMP_NUM_THREADS` (defaults: `1` in `controller.sh` and `run_train.sh`) — PBS often allocates many CPUs to one process; uncapped BLAS can try to spawn that many threads and fail (`pthread_create` / `Exit_status=1` with little log output).

#### MPI HPTune

Launch:

```bash
HPTUNE_CONTROLLER_NODES=4 HPTUNE_TRIAL_NODES=1 ./scripts/start_hptune_parallel.sh
```

The MPI path is designed for multi-GPU and multi-node trial execution. The controller allocation includes one controller node plus worker nodes. In the example above, one node runs the HPTune controller and the remaining three nodes are available for trial execution.

Top-level layout:
- `scripts/start_hptune_parallel.sh`: submits the outer PBS controller allocation.
- `scripts/controller_parallel.sh`: activates the environment, computes MPI world size, and launches `python -m hptune_mpi` under `mpiexec`.
- `src/hptune_mpi.py`: rank `0` is the controller; worker ranks on non-controller hosts act as distributed training processes.
- `src/model/bayesian_hptuner.py`: owns trial state, trial creation, status sync, retries, and Bayesian sampling.
- `src/schemas/trial_schema.py` (`HPTuneTrial`): materializes each trial directory, `.env`, and `run.sh`.
- `src/service/trial_service.py`: SQLite trial log access and snapshot writes.
- `src/train.py`: actual training entrypoint used by generated trial scripts.
- `src/util/distributed.py`: helper functions for PyTorch distributed initialization.

MPI workflow step by step:
1. `scripts/start_hptune_parallel.sh` submits `scripts/controller_parallel.sh` as a PBS job.
2. `scripts/controller_parallel.sh` sets `GPUS_PER_NODE=4` by default and derives `HPTUNE_MPI_SIZE = HPTUNE_CONTROLLER_NODES * GPUS_PER_NODE`.
3. `mpiexec` starts `python -m hptune_mpi` with one MPI rank per GPU across the whole allocation.
4. In `src/hptune_mpi.py`, rank `0` becomes the HPTune controller. All MPI ranks on the controller host are reserved from trial dispatch; full trial slots are formed only from non-controller hosts.
5. The controller calls `BayesianHPTuner.sync_and_load()` to refresh trial state from the SQLite DB at `DB_CONNECTION` (default layout `data/hptune/trials/trials_log.db`).
6. The controller calls `BayesianHPTuner.plan_and_enqueue()` to create new queued trials under `TRIALS_DIR` (default layout `data/hptune/trials/trial_*`).
7. Each trial directory contains:
   - a generated `.env` with trial-specific hyperparameters
   - a generated `run.sh` derived from `scripts/run_train.sh`
   - trial logs and checkpoints under that same directory
8. The controller groups worker hosts into fixed-size slots using:
   - `GPUS_PER_NODE`
   - `HPTUNE_TRIAL_NODES`
   - all GPUs on each worker node
9. A trial slot therefore consumes `HPTUNE_TRIAL_NODES` worker nodes, with one MPI rank per GPU participating in training.
10. When a slot is free, the controller assigns a queued trial to that slot and marks the trial as running in `trials_log.db`.
11. Worker ranks create an MPI subcommunicator for their assigned slot and call `run_distributed_trial(...)`.
12. `run_distributed_trial(...)` loads the generated trial `.env`, sets `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, and `NODE_RANK`, and launches `python src/train.py` directly.
13. When the slot leader finishes, it sends either `DONE` or `FAILED` back to the controller, and failed trials are requeued or marked permanently failed according to `HPTUNE_MAX_RETRIES`.
14. The controller frees that slot, refreshes trial state, and dispatches more queued trials until `HPTUNE_MAX_TRIALS` is reached and no trials remain active.

Data and logging layout:
- Trial state database: `data/hptune/trials/trials_log.db` (see `database.connection`, `database.tables.Trial`, `schemas.trial_schema.HPTuneTrial`, `service.trial_service`)
- Trial directories: `data/hptune/trials/trial_*`
- Best-trial artifacts: `data/hptune/best_trial`
- Controller logs: `data/hptune/controller_logs/`
- Per-rank MPI logs: `data/hptune/trials/trial_*/dist_rank_<world_rank>.log`
- Standard training logs: `data/hptune/trials/trial_*/train_<jobid>.log`
- Under PBS, Loguru also writes `data/hptune/controller_logs/hptune_<PBS_JOBID>.txt`

MPI-related env and sizing:
- `TRIALS_DIR`, `DB_CONNECTION`, 
- `HPTUNE_CONTROLLER_NODES`: nodes reserved for the outer controller allocation
- one controller node is reserved from trial execution
- `HPTUNE_TRIAL_NODES`: nodes consumed by each dispatched trial
- `GPUS_PER_NODE`: defaults to `4` in `scripts/controller_parallel.sh`
- `HPTUNE_MPI_SIZE`: derived as `HPTUNE_CONTROLLER_NODES * GPUS_PER_NODE`
- `HPTUNE_MAX_TRIALS`
- `HPTUNE_MAX_RETRIES`
- `HPTUNE_EI_XI`
- `HPTUNE_RANDOM_INSERT_EVERY`

Run directory (trials DB, `best_trial/`, `controller_logs/`): set `hptune.dir` in `dldl.json` (relative to project root or absolute). Default when unset or `null` is `data/hptune`.

## Quick Reference

* **Config:** `NORMALIZATION_TYPE` and `CPU_USE` in env (or `.env`) drive preprocessing and training.
* **Output files:** Whatever paths you set in `DATA_PATH` and `TRAIN_LABELS_PATH` (often filenames include the normalization mode, e.g. `processed_dataset_meanvar-whole.pt`).
* **Multiple configs:** Change `NORMALIZATION_TYPE` and update `DATA_PATH` / `TRAIN_LABELS_PATH` to the matching `.pt` files.

See https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html