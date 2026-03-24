"""Bayesian hyperparameter tuning orchestration (trial log, acquisition, trial dirs)."""

from __future__ import annotations

import glob
import os
import time
from dataclasses import replace
from typing import Optional

import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
from loguru import logger

from config.settings import load_settings
from model.hptune_trial import HPTuneTrial, ParallelTrial, SerialTrial
from schemas.trial_schema import TrialSchema
from util.hptune import (
    load_env_template,
    next_trial_numbered_id,
    parse_val_loss,
    sync_best_trial_artifacts,
)
from service.trial_service import TrialService


class BayesianHPTuner:
    """Bayesian search over non-architecture training hyperparameters.

    Serial vs parallel trial layout is selected from ``parallelism``: new proposals are
    :class:`~model.hptune_trial.SerialTrial` when ``parallelism <= 1`` (PBS chain) and
    :class:`~model.hptune_trial.ParallelTrial` otherwise (MPI-style dispatch).
    """

    @staticmethod
    def _validate_initialization(
        *,
        trial_nodes: int,
        controller_nodes: Optional[str],
        random_insert_every: int,
        expected_improvement_xi: float,
        max_trials: int,
    ) -> tuple[int, int, int, float, int]:
        """Validate env-derived HP-tune settings in one place; return values to assign to ``self``."""
        if trial_nodes < 1:
            raise ValueError("HPTUNE_TRIAL_NODES must be >= 1")
        if controller_nodes is None:
            parallelism = 1
        else:
            worker_nodes = int(controller_nodes) - 1
            if worker_nodes < trial_nodes:
                raise ValueError(
                    "HPTUNE_CONTROLLER_NODES must reserve at least one controller node "
                    "plus enough worker nodes for one trial slot",
                )
            parallelism = worker_nodes // trial_nodes
        if random_insert_every < 0:
            raise ValueError("HPTUNE_RANDOM_INSERT_EVERY must be >= 0")
        if expected_improvement_xi < 0:
            raise ValueError("HPTUNE_EI_XI must be >= 0")
        if max_trials < 1:
            raise ValueError("HPTUNE_MAX_TRIALS must be >= 1")
        return (
            trial_nodes,
            parallelism,
            random_insert_every,
            expected_improvement_xi,
            max_trials,
        )

    def __init__(self) -> None:
        settings = load_settings()
        root_settings = settings.project_root
        hp_tune_dir = settings.cfg.hptune.dir
        hdir = (
            os.environ.get("DLDL_HPTUNE_DIR")
            or (
                None
                if not hp_tune_dir
                else (
                    hp_tune_dir
                    if os.path.isabs(hp_tune_dir)
                    else os.path.normpath(os.path.join(root_settings, hp_tune_dir))
                )
            )
            or os.path.join(root_settings, "data", "hptune")
        )
        self.trials_dir = os.path.join(hdir, "trials")
        self.best_trial_dir = os.path.join(hdir, "best_trial")
        os.makedirs(self.trials_dir, exist_ok=True)
        self.trials_db_path = TrialService.default_path(self.trials_dir)
        self._trial_service = TrialService(self.trials_db_path)
        self.project_root = root_settings
        self.max_retries = int(os.environ.get("HPTUNE_MAX_RETRIES", "2"))

        hp_tune_settings = settings.cfg.hptune
        self.num_initial_trials = hp_tune_settings.num_initial_trials
        (
            self.trial_nodes,
            self.parallelism,
            self.random_insert_every,
            self.expected_improvement_xi,
            self.max_trials,
        ) = self._validate_initialization(
            trial_nodes=int(os.environ.get("HPTUNE_TRIAL_NODES", "1")),
            controller_nodes=os.environ.get("HPTUNE_CONTROLLER_NODES"),
            random_insert_every=int(
                os.environ.get("HPTUNE_RANDOM_INSERT_EVERY", "5"),
            ),
            expected_improvement_xi=float(os.environ.get("HPTUNE_EI_XI", "0.05")),
            max_trials=int(os.environ.get("HPTUNE_MAX_TRIALS", "10")),
        )
        self.allowed_epochs = tuple(hp_tune_settings.allowed_epochs)
        self.batch_sizes = tuple(hp_tune_settings.allowed_batch_sizes)
        self.bounds = settings.default_hptune_param_bounds(
            self.allowed_epochs, self.batch_sizes
        )
        self._trial_cls: type[HPTuneTrial] = (
            SerialTrial if self.parallelism <= 1 else ParallelTrial
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

    def _seen_trial_signatures(self, trials: list[HPTuneTrial]) -> set[tuple[object, ...]]:
        """Collect signatures for all trials so new suggestions stay unique."""
        return {t.trial_signature() for t in trials}

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
            signature = trial.trial_signature()
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
        return self._trial_cls(
            **TrialSchema(
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
            ).model_dump()
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

    def _emit_dispatch_status(self, *, done: int, active: int, total: int) -> None:
        complete = int(active == 0 and total >= self.max_trials)
        print(
            f"Dispatch status -> done={done} active={active} total={total} "
            f"parallelism={self.parallelism} complete={complete}",
            flush=True,
        )

    def update_trials(self, trials: list[HPTuneTrial]) -> list[HPTuneTrial]:
        """Refresh in-progress trials from training logs; persist snapshot."""
        trials = list(trials)
        for i in range(len(trials)):
            trial = trials[i]
            if trial.status != -1:
                continue

            completed, val_loss = parse_val_loss(trial.path_under(self.trials_dir))

            if completed:
                trials[i] = replace(trial, val_loss=val_loss, status=0)
                sync_best_trial_artifacts(
                    trials, self.trials_dir, self.best_trial_dir
                )
            else:
                trial_dir = trial.path_under(self.trials_dir)
                log_files = glob.glob(os.path.join(trial_dir, "*.log"))

                if log_files:
                    latest = max(log_files, key=os.path.getmtime)
                    age = time.time() - os.path.getmtime(latest)

                    if age > 1800:  # 30 min stale → assume crash
                        retries = trial.retries

                        if retries < self.max_retries:
                            trials[i] = replace(
                                trial, status=-2, retries=retries + 1
                            )
                            self.logger.warning(
                                "Retrying trial {} (attempt {}/{})",
                                trial.trial_id,
                                retries + 1,
                                self.max_retries,
                            )
                        else:
                            trials[i] = replace(trial, status=-3)
                            self.logger.error(
                                "Trial {} failed permanently",
                                trial.trial_id,
                            )

        self._trial_service.persist_snapshot(trials)
        return trials

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
        return self._trial_cls(
            **TrialSchema(
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
            ).model_dump()
        )

    def sample_bayesian(self, trials: list[HPTuneTrial]) -> HPTuneTrial:
        """Bayesian optimization over all registered float parameters."""
        completed = [t for t in trials if t.status == 0]
        if not completed:
            self.logger.warning(
                "BO: zero completed trials; falling back to random sampling"
            )
            return self._sample_unique_random(
                self._seen_trial_signatures(trials),
                context="Bayesian fallback with zero completed trials",
            )

        seen_signatures = self._seen_trial_signatures(trials)

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
        for trial in completed:
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
        if trial.trial_signature() in seen_signatures:
            self.logger.warning(
                "BO suggested a duplicate trial; falling back to random exploration"
            )
            return self._sample_unique_random(
                seen_signatures,
                context="duplicate Bayesian suggestion",
            )
        self.logger.debug("BO: raw suggestion (assign trial_id before materializing)")
        return trial

    def sample_hyperparameters(self, trials: list[HPTuneTrial]) -> HPTuneTrial:
        completed_count = sum(1 for t in trials if t.status == 0)
        total_rows = len(trials)
        if completed_count < self.num_initial_trials:
            self.logger.info(
                "Sample strategy: random warmup (completed={} / {} initial, total_rows={})",
                completed_count,
                self.num_initial_trials,
                total_rows,
            )
            return self._sample_unique_random(
                self._seen_trial_signatures(trials),
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
                self._seen_trial_signatures(trials),
                context="periodic random insertion",
            )
        self.logger.info(
            "Sample strategy: Bayesian (completed={}, total_rows={})",
            completed_count,
            total_rows,
        )
        return self.sample_bayesian(trials)

    def mark_trials_running(self, trial_ids: list[str]) -> None:
        """Promote prepared trials from queued (-2) to running/submitted (-1)."""
        if not trial_ids:
            return
        want = set(trial_ids)
        trials = self._trial_service.get_trials()
        out: list[HPTuneTrial] = []
        updated = 0
        for t in trials:
            if t.trial_id in want and t.status == -2:
                out.append(replace(t, status=-1))
                updated += 1
            else:
                out.append(t)
        self._trial_service.persist_snapshot(out)
        self.logger.info(
            "Marked {} queued trial(s) as running/submitted: {}",
            updated,
            ",".join(trial_ids),
        )

    def mark_trial_failed(self, trial_id: str, *, return_code: int) -> None:
        """Record an immediately failed dispatched trial and enqueue a retry if allowed."""
        trials = self._trial_service.get_trials()
        out: list[HPTuneTrial] = []
        found = False
        for t in trials:
            if t.trial_id != trial_id:
                out.append(t)
                continue
            found = True
            if t.retries < self.max_retries:
                out.append(replace(t, status=-2, retries=t.retries + 1))
                self.logger.warning(
                    "Trial {} failed with return code {} and will be retried ({}/{})",
                    trial_id,
                    return_code,
                    t.retries + 1,
                    self.max_retries,
                )
            else:
                out.append(replace(t, status=-3))
                self.logger.error(
                    "Trial {} failed with return code {} and exhausted retries ({}/{})",
                    trial_id,
                    return_code,
                    t.retries,
                    self.max_retries,
                )
        if not found:
            raise ValueError(f"Unknown trial_id for failure update: {trial_id}")
        self._trial_service.persist_snapshot(out)

    def _plan_new_trials(
        self, trials: list[HPTuneTrial]
    ) -> tuple[list[HPTuneTrial], list[HPTuneTrial]]:
        """Create enough queued trials to fill the configured parallelism target."""
        active = sum(1 for t in trials if t.status in (-1, -2))
        available_slots = max(self.parallelism - active, 0)
        remaining_trials = max(self.max_trials - len(trials), 0)
        plan_count = min(available_slots, remaining_trials)
        if plan_count == 0:
            return trials, []

        env_lines = load_env_template(self.project_root)
        planned: list[HPTuneTrial] = []
        trials = list(trials)
        for _ in range(plan_count):
            tid = next_trial_numbered_id(
                self.trials_dir,
                (t.trial_id for t in trials if t.trial_id),
            )
            trial = replace(
                self.sample_hyperparameters(trials),
                trial_id=tid,
                status=-2,
            )
            self._log_pass_hyperparameters(trial, context="newly proposed (this pass)")
            trial.materialize_trial_files(
                project_root=self.project_root,
                trials_dir=self.trials_dir,
                env_lines=env_lines,
            )
            self.logger.info(
                "Materialized trial files: dir={} path={}",
                trial.dir_name,
                trial.path_under(self.trials_dir),
            )
            trials.append(trial)
            planned.append(trial)
        self._trial_service.persist_snapshot(trials)
        self.logger.info(
            "Appended {} queued trial row(s) to {}",
            len(planned),
            self.trials_db_path,
        )
        return trials, planned

    def sync_and_load(self) -> list[HPTuneTrial]:
        """Load trial state from the trial service, sync running trials from logs, return trials."""
        trials = self._trial_service.get_trials()
        return self.update_trials(trials)

    def plan_and_enqueue(self) -> tuple[list[HPTuneTrial], list[str]]:
        """Create new trials to fill available parallel slots."""
        trials = self._trial_service.get_trials()
        trials = self.update_trials(trials)

        trials, planned_trials = self._plan_new_trials(trials)

        trial_ids = [t.trial_id for t in planned_trials if t.trial_id]
        return trials, trial_ids

    def get_dispatchable_trials(self, trials: list[HPTuneTrial]) -> list[str]:
        """Return queued trials (-2)."""
        return [t.trial_id for t in trials if t.status == -2 and t.trial_id]

    def get_status_counts(self, trials: list[HPTuneTrial]) -> dict[str, int]:
        return {
            "done": sum(1 for t in trials if t.status == 0),
            "running": sum(1 for t in trials if t.status == -1),
            "queued": sum(1 for t in trials if t.status == -2),
            "failed": sum(1 for t in trials if t.status == -3),
            "total": len(trials),
        }

    def is_complete(self, trials: list[HPTuneTrial]) -> bool:
        counts = self.get_status_counts(trials)

        return (
            counts["total"] >= self.max_trials
            and counts["running"] == 0
            and counts["queued"] == 0
        )

    def run(self) -> None:
        """Single dispatcher pass used by the serial controller and CLI."""
        trials = self._trial_service.get_trials()
        trials = self.update_trials(trials)
        queued_trials = self.get_dispatchable_trials(trials)
        if not queued_trials:
            trials, planned_trials = self._plan_new_trials(trials)
            queued_trials = [t.trial_id for t in planned_trials if t.trial_id]

        for trial_id in queued_trials:
            self.logger.info("Next trial -> {}", trial_id)
            print(f"Next trial -> {trial_id}", flush=True)

        counts = self.get_status_counts(trials)
        self._emit_dispatch_status(
            done=counts["done"],
            active=counts["running"] + counts["queued"],
            total=counts["total"],
        )
