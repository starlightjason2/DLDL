"""Bayesian hyperparameter tuning orchestration (trial log, acquisition, trial dirs)."""

import os
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
from loguru import logger

from config.settings import load_settings
from util.hptune import (
    create_run_script,
    load_env_template,
    load_trials,
    next_trial_numbered_id,
    parse_val_loss,
)

# Parameters sampled in BO / random search (extend by updating HPTuneTrial + create_trial + Settings.default_hptune_param_bounds)
_BO_PARAM_KEYS = frozenset(load_settings().default_hptune_param_bounds().keys())


@dataclass
class HPTuneTrial:
    """One hyperparameter trial: non-architecture training hparams, log status, identity."""

    lr: float
    epochs: int
    dropout: float
    weight_decay: float
    batch_size: int
    gradient_clip: float
    lr_scheduler: bool
    lr_scheduler_factor: float
    lr_scheduler_patience: int
    early_stopping_patience: int
    trial_id: Optional[str] = None
    val_loss: float = -1.0
    status: int = -1

    @property
    def dir_name(self) -> str:
        """Folder under ``trials/`` (``trial_1``, ``trial_2``, …). Requires ``trial_id``."""
        if not self.trial_id:
            raise ValueError("trial_id must be set before using dir_name or path_under")
        return self.trial_id

    def path_under(self, trials_dir: str) -> str:
        return os.path.join(trials_dir, self.dir_name)

    @classmethod
    def from_series(cls, row: pd.Series) -> "HPTuneTrial":
        ls = row["lr_scheduler"]
        lr_scheduler = bool(int(ls)) if not pd.isna(ls) else True
        raw_tid = row["trial_id"]
        if raw_tid is None or (isinstance(raw_tid, float) and pd.isna(raw_tid)):
            raise ValueError("trials_log.csv row is missing trial_id (required)")
        tid = str(raw_tid).strip()
        if not tid or tid.lower() in ("nan", "none"):
            raise ValueError("trials_log.csv row has empty trial_id (required)")
        return cls(
            lr=float(row["lr"]),
            epochs=int(row["epochs"]),
            dropout=float(row["dropout"]),
            weight_decay=float(row["weight_decay"]),
            batch_size=int(row["batch_size"]),
            gradient_clip=float(row["gradient_clip"]),
            lr_scheduler=lr_scheduler,
            lr_scheduler_factor=float(row["lr_scheduler_factor"]),
            lr_scheduler_patience=int(row["lr_scheduler_patience"]),
            early_stopping_patience=int(row["early_stopping_patience"]),
            trial_id=tid,
            val_loss=float(row["val_loss"]),
            status=int(row["status"]),
        )

    def to_csv_row(self) -> dict[str, float | int | str]:
        if not self.trial_id:
            raise ValueError("trial_id must be set before serializing to CSV")
        return {
            "trial_id": self.trial_id,
            "lr": self.lr,
            "epochs": self.epochs,
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "gradient_clip": self.gradient_clip,
            "lr_scheduler": int(self.lr_scheduler),
            "lr_scheduler_factor": self.lr_scheduler_factor,
            "lr_scheduler_patience": self.lr_scheduler_patience,
            "early_stopping_patience": self.early_stopping_patience,
            "val_loss": self.val_loss,
            "status": self.status,
        }

    def bayesian_params(self, batch_sizes: tuple[int, ...]) -> dict[str, float]:
        """Float-only parameter dict aligned with BayesianOptimization pbounds."""
        bi = self._batch_index(batch_sizes)
        return {
            "lr": self.lr,
            "dropout": self.dropout,
            "log_wd": float(np.log10(max(self.weight_decay, 1e-20))),
            "epochs": float(self.epochs),
            "gradient_clip": self.gradient_clip,
            "lr_scheduler_u": 1.0 if self.lr_scheduler else 0.0,
            "lr_scheduler_factor": self.lr_scheduler_factor,
            "lr_sched_patience": float(self.lr_scheduler_patience),
            "early_stop_patience": float(self.early_stopping_patience),
            "batch_idx": float(bi),
        }

    def _batch_index(self, batch_sizes: tuple[int, ...]) -> int:
        if self.batch_size in batch_sizes:
            return batch_sizes.index(self.batch_size)
        return min(
            range(len(batch_sizes)), key=lambda i: abs(batch_sizes[i] - self.batch_size)
        )


class BayesianHPTuner:
    """Bayesian search over non-architecture training hyperparameters."""

    def __init__(
        self,
        trials_dir: str | None = None,
        csv_path: str | None = None,
        project_root: str | None = None,
        num_initial_trials: int | None = None,
        param_bounds: Optional[dict[str, tuple[float, float]]] = None,
        allowed_epochs: Optional[tuple[int, ...]] = None,
        batch_sizes: Optional[tuple[int, ...]] = None,
        chain_id: Optional[str] = None,
    ) -> None:
        s = load_settings()
        r = s.project_root
        hd = s.cfg.hptune.dir
        hdir = (
            os.environ.get("DLDL_HPTUNE_DIR")
            or (
                None
                if not hd
                else (
                    hd if os.path.isabs(hd) else os.path.normpath(os.path.join(r, hd))
                )
            )
            or os.path.join(r, "scripts", "hptune")
        )
        tdir, logp = os.path.join(hdir, "trials"), os.path.join(hdir, "trials_log.csv")
        self.trials_dir = trials_dir if trials_dir is not None else tdir
        self.csv_path = csv_path if csv_path is not None else logp
        self.project_root = project_root if project_root is not None else s.project_root
        hp = s.cfg.hptune
        self.num_initial_trials = (
            num_initial_trials
            if num_initial_trials is not None
            else hp.num_initial_trials
        )
        self.allowed_epochs = (
            tuple(allowed_epochs)
            if allowed_epochs is not None
            else tuple(hp.allowed_epochs)
        )
        self.batch_sizes = (
            tuple(batch_sizes)
            if batch_sizes is not None
            else tuple(hp.allowed_batch_sizes)
        )
        base = s.default_hptune_param_bounds(self.allowed_epochs, self.batch_sizes)
        if param_bounds is not None:
            bad = set(param_bounds) - _BO_PARAM_KEYS
            if bad:
                raise ValueError(
                    f"param_bounds has unknown keys {sorted(bad)}; "
                    f"allowed: {sorted(_BO_PARAM_KEYS)}",
                )
            for k, bounds in param_bounds.items():
                low, high = bounds
                if low >= high:
                    raise ValueError(
                        f"param_bounds[{k!r}] must have min < max, got {bounds}"
                    )
            base = {**base, **param_bounds}
        self.bounds = base
        self.chain_id = chain_id
        self.logger = logger.bind(name=__name__)

    def _pbounds(self) -> dict[str, tuple[float, float]]:
        return dict(self.bounds)

    def _suggestion_to_trial(self, s: dict[str, float]) -> HPTuneTrial:
        bi = int(np.clip(round(s["batch_idx"]), 0, max(len(self.batch_sizes) - 1, 0)))
        lsu_low, lsu_high = self.bounds["lr_scheduler_u"]
        lr_sched = bool(s["lr_scheduler_u"] >= (lsu_low + lsu_high) / 2.0)
        lsp_low, lsp_high = self.bounds["lr_sched_patience"]
        esp_low, esp_high = self.bounds["early_stop_patience"]
        lsf_low, lsf_high = self.bounds["lr_scheduler_factor"]
        return HPTuneTrial(
            lr=float(s["lr"]),
            epochs=min(self.allowed_epochs, key=lambda x: abs(x - s["epochs"])),
            dropout=float(s["dropout"]),
            weight_decay=10 ** float(s["log_wd"]),
            batch_size=self.batch_sizes[bi],
            gradient_clip=float(s["gradient_clip"]),
            lr_scheduler=lr_sched,
            lr_scheduler_factor=float(
                np.clip(s["lr_scheduler_factor"], lsf_low, lsf_high)
            ),
            lr_scheduler_patience=int(
                np.clip(round(s["lr_sched_patience"]), int(lsp_low), int(lsp_high))
            ),
            early_stopping_patience=int(
                np.clip(round(s["early_stop_patience"]), int(esp_low), int(esp_high))
            ),
            trial_id=None,
            val_loss=-1.0,
            status=-1,
        )

    def _log_pass_hyperparameters(self, trial: HPTuneTrial, *, context: str) -> None:
        tid = trial.trial_id or "(pending id)"
        self.logger.info(
            "Hyperparameters for this pass ({}): trial_id={} lr={:.2e} epochs={} dropout={:.4f} "
            "weight_decay={:.2e} batch_size={} gradient_clip={:.3f} lr_scheduler={} "
            "lr_scheduler_factor={:.3f} lr_scheduler_patience={} early_stopping_patience={}",
            context,
            tid,
            trial.lr,
            trial.epochs,
            trial.dropout,
            trial.weight_decay,
            trial.batch_size,
            trial.gradient_clip,
            trial.lr_scheduler,
            trial.lr_scheduler_factor,
            trial.lr_scheduler_patience,
            trial.early_stopping_patience,
        )

    def _log_next_trial_marker(self, dir_name: str) -> None:
        self.logger.info("Next trial -> {}", dir_name)

    def update_trials(self, df: pd.DataFrame) -> pd.DataFrame:
        """Update in-progress trials (val_loss=-1) by checking training logs."""
        in_progress = df[df["val_loss"] == -1]
        self.logger.info(
            "Sync: checking {} in-progress trial(s) under {}",
            len(in_progress),
            self.trials_dir,
        )
        completed_n = 0
        for idx in in_progress.index:
            trial = HPTuneTrial.from_series(df.loc[idx])
            completed, val_loss = parse_val_loss(trial.path_under(self.trials_dir))
            if completed:
                df.loc[idx, ["val_loss", "status"]] = val_loss, 0
                completed_n += 1
                self.logger.info(
                    "Sync: trial complete dir={} val_loss={:.6f}",
                    trial.dir_name,
                    val_loss,
                )
        if completed_n == 0 and len(in_progress) > 0:
            self.logger.debug("Sync: no new completions from training logs yet")
        elif completed_n:
            self.logger.info(
                "Sync: wrote {} newly completed trial(s) to {}",
                completed_n,
                self.csv_path,
            )
        df.to_csv(self.csv_path, index=False)
        return df

    def sample_random(self) -> HPTuneTrial:
        """Uniform random over ``param_bounds`` (log-uniform for ``lr`` and ``log_wd``)."""
        lr_low, lr_high = self.bounds["lr"]
        dr_low, dr_high = self.bounds["dropout"]
        lw_low, lw_high = self.bounds["log_wd"]
        gc_low, gc_high = self.bounds["gradient_clip"]
        lsu_low, lsu_high = self.bounds["lr_scheduler_u"]
        lsf_low, lsf_high = self.bounds["lr_scheduler_factor"]
        lsp_low, lsp_high = self.bounds["lr_sched_patience"]
        esp_low, esp_high = self.bounds["early_stop_patience"]
        batch_sizes = self.batch_sizes
        return HPTuneTrial(
            lr=10 ** np.random.uniform(np.log10(lr_low), np.log10(lr_high)),
            epochs=int(np.random.choice(self.allowed_epochs)),
            dropout=float(np.random.uniform(dr_low, dr_high)),
            weight_decay=10 ** float(np.random.uniform(lw_low, lw_high)),
            batch_size=batch_sizes[int(np.random.randint(0, len(batch_sizes)))],
            gradient_clip=float(np.random.uniform(gc_low, gc_high)),
            lr_scheduler=bool(
                np.random.uniform(lsu_low, lsu_high) >= (lsu_low + lsu_high) / 2.0
            ),
            lr_scheduler_factor=float(np.random.uniform(lsf_low, lsf_high)),
            lr_scheduler_patience=int(
                np.random.randint(int(lsp_low), int(lsp_high) + 1)
            ),
            early_stopping_patience=int(
                np.random.randint(int(esp_low), int(esp_high) + 1)
            ),
            trial_id=None,
            val_loss=-1.0,
            status=-1,
        )

    def sample_bayesian(self, df: pd.DataFrame) -> HPTuneTrial:
        """Bayesian optimization over all registered float parameters."""
        completed = df[df["status"] == 0]
        if completed.empty:
            self.logger.warning(
                "BO: zero completed trials; falling back to random sampling"
            )
            return self.sample_random()

        optimizer = BayesianOptimization(
            f=None,
            pbounds=self._pbounds(),
            acquisition_function=acquisition.ExpectedImprovement(xi=0.0),
            verbose=0,
            random_state=42,
        )
        for _, row in completed.iterrows():
            trial = HPTuneTrial.from_series(row)
            optimizer.register(
                params=trial.bayesian_params(self.batch_sizes),
                target=-trial.val_loss,
            )
        self.logger.debug(
            "BO: registered {} observation(s) (ExpectedImprovement xi=0)",
            len(completed),
        )

        suggestion = optimizer.suggest()
        trial = self._suggestion_to_trial(suggestion)
        self.logger.debug("BO: raw suggestion (assign trial_id before materializing)")
        return trial

    def sample_hyperparameters(self, df: pd.DataFrame) -> HPTuneTrial:
        completed_count = int((df["status"] == 0).sum())
        total_rows = len(df)
        if completed_count < self.num_initial_trials:
            self.logger.info(
                "Sample strategy: random warmup (completed={} / {} initial, total_rows={})",
                completed_count,
                self.num_initial_trials,
                total_rows,
            )
            return self.sample_random()
        self.logger.info(
            "Sample strategy: Bayesian (completed={}, total_rows={})",
            completed_count,
            total_rows,
        )
        return self.sample_bayesian(df)

    def create_trial(self, trial: HPTuneTrial, env_lines: list[str]) -> str:
        """Create trial directory with run.sh and .env. ``trial.trial_id`` must be set for new trials."""
        if not trial.trial_id:
            raise ValueError(
                "HPTuneTrial.trial_id must be set (use next_trial_numbered_id()) before create_trial"
            )
        trial_dir = trial.path_under(self.trials_dir)
        os.makedirs(trial_dir, exist_ok=True)
        self.logger.info("Materialize trial: dir={} path={}", trial.dir_name, trial_dir)

        env_path = os.path.join(trial_dir, ".env")
        ls_str = "true" if trial.lr_scheduler else "false"
        env_content = (
            "\n".join(env_lines)
            + f"""
# HPTune overrides
LEARNING_RATE={trial.lr}
NUM_EPOCHS={trial.epochs}
DROPOUT_RATE={trial.dropout}
WEIGHT_DECAY={trial.weight_decay}
BATCH_SIZE={trial.batch_size}
GRADIENT_CLIP={trial.gradient_clip}
LR_SCHEDULER={ls_str}
LR_SCHEDULER_FACTOR={trial.lr_scheduler_factor}
LR_SCHEDULER_PATIENCE={trial.lr_scheduler_patience}
EARLY_STOPPING_PATIENCE={trial.early_stopping_patience}
PROG_DIR={trial_dir}
JOB_ID=training
# run.sh tee already writes full stderr to train_${{PBS_JOBID}}.log; skip duplicate training.log
TRAIN_LOGURU_FILE=0
"""
        )
        with open(env_path, "w") as f:
            f.write(env_content)

        template_path = os.path.join(self.project_root, "scripts", "run_train.sh")
        script = create_run_script(
            self.project_root, trial_dir, env_path, template_path
        )
        run_path = os.path.join(trial_dir, "run.sh")
        with open(run_path, "w") as f:
            f.write(script)
        os.chmod(run_path, 0o755)
        self.logger.info("Wrote .env and run.sh (chmod 755) under {}", trial_dir)
        return trial.dir_name

    def find_pending_trial(self, df: pd.DataFrame) -> HPTuneTrial | None:
        for status_val in (-1, -2):
            candidates = df[df["val_loss"] == status_val]
            if not candidates.empty:
                return HPTuneTrial.from_series(candidates.iloc[0])
        self.logger.debug(
            "No pending trial rows (val_loss in -1, -2); will propose a new trial"
        )
        return None

    def run(self) -> None:
        """One controller step: sync logs, resume pending work, or enqueue a new trial."""
        self.logger.info(
            "=== HPTune pass start chain_id={} csv={} trials_dir={} ===",
            self.chain_id,
            self.csv_path,
            self.trials_dir,
        )
        df = self.update_trials(load_trials(self.trials_dir, self.csv_path))
        done = int((df["status"] == 0).sum())
        running = int((df["val_loss"] == -1).sum())
        queued = int((df["val_loss"] == -2).sum())
        self.logger.info(
            "Trial log snapshot: rows={} done={} running={} queued={}",
            len(df),
            done,
            running,
            queued,
        )

        pending = self.find_pending_trial(df)
        if pending:
            self._log_pass_hyperparameters(
                pending, context="pending (resume / awaiting worker)"
            )
            self._log_next_trial_marker(pending.dir_name)
            self.logger.info("=== HPTune pass end (pending) ===")
            return

        trial = self.sample_hyperparameters(df)
        trial = replace(trial, trial_id=next_trial_numbered_id(self.trials_dir, df))
        self._log_pass_hyperparameters(trial, context="newly proposed (this pass)")

        new_row = pd.DataFrame([trial.to_csv_row()])
        pd.concat([df, new_row], ignore_index=True).to_csv(self.csv_path, index=False)
        self.logger.info("Appended new trial row to {}", self.csv_path)

        self.create_trial(trial, load_env_template(self.project_root))
        self._log_next_trial_marker(trial.dir_name)
        self.logger.info("=== HPTune pass end (new trial scheduled) ===")
