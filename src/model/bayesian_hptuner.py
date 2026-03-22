"""Bayesian hyperparameter tuning orchestration (trial log, acquisition, trial dirs)."""

import fcntl
import os
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
from loguru import logger

from config.settings import load_settings
from model.hptune_trial import HPTuneTrial
from util.hptune import (
    create_run_script,
    load_env_template,
    load_trials,
    next_trial_numbered_id,
    parse_val_loss,
    sync_best_trial_artifacts,
)


class BayesianHPTuner:
    """Bayesian search over non-architecture training hyperparameters."""

    def __init__(self) -> None:
        settings = load_settings()
        r = settings.project_root
        hd = settings.cfg.hptune.dir
        hdir = (
            os.environ.get("DLDL_HPTUNE_DIR")
            or (
                None
                if not hd
                else (
                    hd if os.path.isabs(hd) else os.path.normpath(os.path.join(r, hd))
                )
            )
            or os.path.join(r, "data", "hptune")
        )
        self.trials_dir = os.path.join(hdir, "trials")
        self.best_trial_dir = os.path.join(hdir, "best_trial")
        self.csv_path = os.path.join(self.trials_dir, "trials_log.csv")
        self.state_lock_path = os.path.join(hdir, ".state.lock")
        self.project_root = r
        hp = settings.cfg.hptune
        self.num_initial_trials = hp.num_initial_trials
        self.parallelism = int(os.environ.get("HPTUNE_PARALLELISM", "1"))
        if self.parallelism < 1:
            raise ValueError("HPTUNE_PARALLELISM must be >= 1")
        self.random_insert_every = int(os.environ.get("HPTUNE_RANDOM_INSERT_EVERY", "5"))
        if self.random_insert_every < 0:
            raise ValueError("HPTUNE_RANDOM_INSERT_EVERY must be >= 0")
        self.expected_improvement_xi = float(os.environ.get("HPTUNE_EI_XI", "0.05"))
        if self.expected_improvement_xi < 0:
            raise ValueError("HPTUNE_EI_XI must be >= 0")
        self.max_trials = int(os.environ.get("HPTUNE_MAX_TRIALS", "10"))
        if self.max_trials < 1:
            raise ValueError("HPTUNE_MAX_TRIALS must be >= 1")
        self.allowed_epochs = tuple(hp.allowed_epochs)
        self.batch_sizes = tuple(hp.allowed_batch_sizes)
        self.bounds = settings.default_hptune_param_bounds(
            self.allowed_epochs, self.batch_sizes
        )
        self.logger = logger.bind(name=__name__)
        self._configure_loguru_file()

    def _configure_loguru_file(self) -> None:
        """Under PBS, write loguru to data/hptune/controller_logs/hptune_<PBS_JOBID>.txt."""
        jid = os.environ.get("PBS_JOBID")
        if not jid:
            return
        log_dir = os.path.join(os.path.dirname(self.trials_dir), "controller_logs")
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"hptune_{jid}.txt")
        logger.add(
            path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
            level="DEBUG",
            enqueue=True,
        )

    def _pbounds(self) -> dict[str, tuple[float, float]]:
        return dict(self.bounds)

    @contextmanager
    def _state_lock(self, *, context: str):
        """Serialize shared HPTune state updates across controller invocations."""
        os.makedirs(os.path.dirname(self.state_lock_path), exist_ok=True)
        with open(self.state_lock_path, "a+", encoding="utf-8") as lock_file:
            self.logger.debug(
                "Waiting for HPTune state lock ({}) at {}",
                context,
                self.state_lock_path,
            )
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            self.logger.debug("Acquired HPTune state lock ({})", context)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                self.logger.debug("Released HPTune state lock ({})", context)

    @staticmethod
    def _trial_signature(trial: HPTuneTrial) -> tuple[object, ...]:
        """Normalize a trial into a hashable signature for duplicate detection."""
        return (
            f"{trial.lr:.12g}",
            int(trial.epochs),
            f"{trial.dropout:.12g}",
            f"{trial.weight_decay:.12g}",
            int(trial.batch_size),
            f"{trial.gradient_clip:.12g}",
            int(trial.lr_scheduler),
            f"{trial.lr_scheduler_factor:.12g}",
            int(trial.lr_scheduler_patience),
            int(trial.early_stopping_patience),
        )

    def _seen_trial_signatures(self, df: pd.DataFrame) -> set[tuple[object, ...]]:
        """Collect signatures for all trial rows so new suggestions stay unique."""
        seen: set[tuple[object, ...]] = set()
        for _, row in df.iterrows():
            seen.add(self._trial_signature(HPTuneTrial.from_series(row)))
        return seen

    def _sample_unique_random(
        self,
        seen_signatures: set[tuple[object, ...]],
        *,
        context: str,
        max_attempts: int = 25,
    ) -> HPTuneTrial:
        """Draw random trials until one is not already present in the trial log."""
        for attempt in range(1, max_attempts + 1):
            trial = self.sample_random()
            signature = self._trial_signature(trial)
            if signature not in seen_signatures:
                if attempt > 1:
                    self.logger.info(
                        "Random sampling found a unique candidate after {} attempt(s) ({})",
                        attempt,
                        context,
                    )
                return trial
            self.logger.warning(
                "Duplicate random candidate rejected on attempt {} ({})",
                attempt,
                context,
            )
        raise RuntimeError(
            f"Unable to find a unique random HPTune candidate after {max_attempts} attempts ({context})"
        )

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
        # NOTE: controller.sh parses this exact format via: grep 'Next trial ->'
        # Do not change this string without updating the grep in controller.sh.
        self.logger.info("Next trial -> {}", dir_name)
        print(f"Next trial -> {dir_name}", flush=True)

    def _emit_dispatch_status(self, *, done: int, active: int, total: int) -> None:
        complete = int(active == 0 and total >= self.max_trials)
        print(
            f"Dispatch status -> done={done} active={active} total={total} "
            f"parallelism={self.parallelism} complete={complete}",
            flush=True,
        )

    def update_trials(self, df: pd.DataFrame) -> pd.DataFrame:
        """Update in-progress trials (status=-1) by checking training logs."""
        in_progress = df[df["status"].isin([-1, -2])]
        self.logger.info(
            "Sync: checking {} active trial(s) under {}",
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
            sync_best_trial_artifacts(df, self.trials_dir, self.best_trial_dir)
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
            return self._sample_unique_random(
                self._seen_trial_signatures(df),
                context="Bayesian fallback with zero completed trials",
            )

        seen_signatures = self._seen_trial_signatures(df)

        optimizer = BayesianOptimization(
            f=None,
            pbounds=self._pbounds(),
            acquisition_function=acquisition.ExpectedImprovement(
                xi=self.expected_improvement_xi
            ),
            allow_duplicate_points=True,
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
            "BO: registered {} observation(s) (ExpectedImprovement xi={})",
            len(completed),
            self.expected_improvement_xi,
        )

        suggestion = optimizer.suggest()
        trial = self._suggestion_to_trial(suggestion)
        if self._trial_signature(trial) in seen_signatures:
            self.logger.warning(
                "BO suggested a duplicate trial; falling back to random exploration"
            )
            return self._sample_unique_random(
                seen_signatures,
                context="duplicate Bayesian suggestion",
            )
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
            return self._sample_unique_random(
                self._seen_trial_signatures(df),
                context="warmup",
            )
        post_warmup_completed = completed_count - self.num_initial_trials
        if (
            self.random_insert_every
            and post_warmup_completed > 0
            and post_warmup_completed % self.random_insert_every == 0
        ):
            self.logger.info(
                "Sample strategy: periodic random insertion "
                "(completed={} post_warmup={} every={} total_rows={})",
                completed_count,
                post_warmup_completed,
                self.random_insert_every,
                total_rows,
            )
            return self._sample_unique_random(
                self._seen_trial_signatures(df),
                context="periodic random insertion",
            )
        self.logger.info(
            "Sample strategy: Bayesian (completed={}, total_rows={})",
            completed_count,
            total_rows,
        )
        return self.sample_bayesian(df)

    def _post_run_checkpoint_cleanup_block(self) -> str:
        """Shell snippet that removes epoch checkpoints after best checkpoint exists."""
        return """

# Post-run cleanup: keep only the best checkpoint for this trial.
BEST_PARAMS_PATH="$PROG_DIR/${JOB_ID}_best_params.pt"
if [ -f "$BEST_PARAMS_PATH" ]; then
    for checkpoint in "$PROG_DIR/${JOB_ID}_params_epoch"*.pt; do
        [ -e "$checkpoint" ] || continue
        rm -f "$checkpoint"
    done
fi
"""

    def create_trial(self, trial: HPTuneTrial, env_lines: list[str]) -> str:
        """Create trial directory with run.sh and .env. ``trial.trial_id`` must be set."""
        if not trial.trial_id:
            raise ValueError(
                "HPTuneTrial.trial_id must be set (use next_trial_numbered_id()) before create_trial"
            )
        trial_dir = trial.path_under(self.trials_dir)
        os.makedirs(trial_dir, exist_ok=True)
        self.logger.info("Materialize trial: dir={} path={}", trial.dir_name, trial_dir)

        env_path = os.path.join(trial_dir, ".env")
        trial.write_env_file(env_path, env_lines)

        template_path = os.path.join(self.project_root, "scripts", "run_train.sh")
        script = create_run_script(
            self.project_root, trial_dir, env_path, template_path
        )
        script += self._post_run_checkpoint_cleanup_block()

        if self.parallelism <= 1:
            # Append the next-controller submission block only in serial chain mode.
            hptune_dir = os.path.dirname(self.trials_dir)
            log_dir = os.path.join(hptune_dir, "controller_logs")
            controller_path = os.path.join(self.project_root, "scripts", "controller.sh")
            script += f"""
    # --- Submit Next Controller (HPTune Job Chain) ---
    # Appended by create_trial in bayesian_hp_tuning.py. Not present in run_train.sh.
    # Submits the next controller after training completes, continuing the chain.
    if [ -n "$DLDL_HPTUNE_CHAIN_ID" ] && [ -n "$PROJECT_ROOT" ]; then
        echo "Training complete. Submitting next controller..."
        NEXT_CTL_JOB_ID=$(qsub \\
            -A fusiondl_aesp \\
            -q "${{HPTUNE_QUEUE:-small}}" \\
            -l select=1:system=polaris,place=scatter,walltime=1:00:00,filesystems=home:eagle \\
            -k doe \\
            -o "{log_dir}/" \\
            -e "{log_dir}/" \\
            -v "PROJECT_ROOT=$PROJECT_ROOT,DLDL_HPTUNE_CHAIN_ID=$DLDL_HPTUNE_CHAIN_ID,HPTUNE_QUEUE=${{HPTUNE_QUEUE:-small}}" \\
            "{controller_path}") || {{
                echo "ERROR: Next controller qsub failed. Chain will not continue."
                exit 1
            }}
        echo "Next controller queued: $NEXT_CTL_JOB_ID"
    fi
    """

        run_path = os.path.join(trial_dir, "run.sh")
        with open(run_path, "w") as f:
            f.write(script)
        os.chmod(run_path, 0o755)
        self.logger.info("Wrote .env and run.sh (chmod 755) under {}", trial_dir)
        return trial.dir_name

    def find_pending_trial(self, df: pd.DataFrame) -> HPTuneTrial | None:
        """Return the first trial with status -1 (running) or -2 (queued), if any."""
        candidates = df[df["status"].isin([-1, -2])]
        if not candidates.empty:
            return HPTuneTrial.from_series(candidates.iloc[0])
        self.logger.debug(
            "No pending trial rows (status in -1, -2); will propose a new trial"
        )
        return None

    def mark_trials_running(self, trial_ids: list[str]) -> None:
        """Promote prepared trials from queued (-2) to running/submitted (-1)."""
        if not trial_ids:
            return
        with self._state_lock(context="mark_trials_running"):
            df = load_trials(self.trials_dir, self.csv_path)
            mask = df["trial_id"].isin(trial_ids) & (df["status"] == -2)
            updated = int(mask.sum())
            df.loc[mask, "status"] = -1
            df.to_csv(self.csv_path, index=False)
            self.logger.info(
                "Marked {} queued trial(s) as running/submitted: {}",
                updated,
                ",".join(trial_ids),
            )

    def _plan_new_trials(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[HPTuneTrial]]:
        """Create enough queued trials to fill the configured parallelism target."""
        active = int(df["status"].isin([-1, -2]).sum())
        available_slots = max(self.parallelism - active, 0)
        remaining_trials = max(self.max_trials - len(df), 0)
        plan_count = min(available_slots, remaining_trials)
        if plan_count == 0:
            return df, []

        env_lines = load_env_template(self.project_root)
        planned: list[HPTuneTrial] = []
        for _ in range(plan_count):
            trial_id = next_trial_numbered_id(self.trials_dir, df)
            trial = replace(
                self.sample_hyperparameters(df),
                trial_id=trial_id,
                status=-2,
            )
            self._log_pass_hyperparameters(trial, context="newly proposed (this pass)")
            self.create_trial(trial, env_lines)
            df = pd.concat([df, pd.DataFrame([trial.to_csv_row()])], ignore_index=True)
            planned.append(trial)
        df.to_csv(self.csv_path, index=False)
        self.logger.info(
            "Appended {} queued trial row(s) to {}",
            len(planned),
            self.csv_path,
        )
        return df, planned

    def run(self) -> None:
        """One dispatcher step: sync results and queue enough trials to fill capacity."""
        with self._state_lock(context="dispatcher_run"):
            self.logger.info(
                "=== HPTune pass start csv={} trials_dir={} ===",
                self.csv_path,
                self.trials_dir,
            )
            df = self.update_trials(load_trials(self.trials_dir, self.csv_path))
            done = int((df["status"] == 0).sum())
            running = int((df["status"] == -1).sum())
            queued = int((df["status"] == -2).sum())
            self.logger.info(
                "Trial log snapshot: rows={} done={} running={} queued={}",
                len(df),
                done,
                running,
                queued,
            )

            if len(df) >= self.max_trials and (running + queued) == 0:
                self.logger.info(
                    "Reached HPTUNE_MAX_TRIALS={} with {} trial row(s); dispatcher is complete",
                    self.max_trials,
                    len(df),
                )
                self._emit_dispatch_status(
                    done=done, active=running + queued, total=len(df)
                )
                self.logger.info("=== HPTune pass end (max trials reached) ===")
                return

            queued_trials = [
                HPTuneTrial.from_series(row)
                for _, row in df[df["status"] == -2].iterrows()
            ]
            if queued_trials:
                self.logger.info(
                    "Dispatcher found {} queued trial(s) awaiting submission",
                    len(queued_trials),
                )

            df, planned_trials = self._plan_new_trials(df)
            dispatchable_trials = queued_trials + planned_trials
            for trial in dispatchable_trials:
                self._log_next_trial_marker(trial.dir_name)

            done = int((df["status"] == 0).sum())
            active = int(df["status"].isin([-1, -2]).sum())
            self._emit_dispatch_status(done=done, active=active, total=len(df))
            self.logger.info(
                "=== HPTune pass end (dispatchable={} active={} total={}) ===",
                len(dispatchable_trials),
                active,
                len(df),
            )
