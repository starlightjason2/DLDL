"""Bayesian hyperparameter tuning orchestration (trial log, acquisition, trial dirs)."""

from __future__ import annotations


import glob
import os
import time
from typing import Any, Optional, Tuple

import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from config.settings import Settings
from schemas.trial_schema import HPTuneTrial
from util.hptune import (
    next_trial_numbered_id,
    parse_val_loss,
    sync_best_trial_artifacts,
)
from service.trial_service import get_trials, save_trials, sql_to_csv
import math


class BayesianHPTuner(BaseModel):
    """Bayesian search over non-architecture training hyperparameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    trials_dir: str
    log_dir: str
    project_root: str
    best_trial_dir: str

    max_retries: int = Field(ge=0)
    num_initial_trials: int = Field(ge=1)

    trial_nodes: int = Field(ge=1)
    controller_nodes: Optional[str] = None
    random_insert_every: int = Field(ge=0)
    expected_improvement_xi: float = Field(ge=0)
    max_trials: int = Field(ge=1)

    allowed_epochs: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    bounds: dict[str, Tuple[float, float]]

    num_slots: int = Field(ge=1)

    _log: Any = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        os.makedirs(self.trials_dir, exist_ok=True)

        job_id = os.environ.get("PBS_JOBID")

        if job_id:
            logger.add(
                os.path.join(self.log_dir, f"hptune_{job_id}.txt"),
                format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
                level="DEBUG",
                enqueue=True,
            )

    @classmethod
    def create(cls) -> BayesianHPTuner:

        trials_dir = os.environ["TRIALS_DIR"]
        log_dir = os.path.join(os.path.dirname(trials_dir), "controller_logs")
        os.makedirs(trials_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        settings = Settings.load()
        hp = settings.cfg.hptune
        trial_nodes = int(os.environ.get("HPTUNE_TRIAL_NODES", "1"))
        controller_nodes = int(os.environ.get("HPTUNE_CONTROLLER_NODES", "0"))

        # Effective parallel trial capacity is `floor((HPTUNE_CONTROLLER_NODES - 1) / HPTUNE_TRIAL_NODES)`, because one node is dedicated to the controller.
        num_slots = math.floor((controller_nodes - 1) / trial_nodes)
        random_insert_every = int(os.environ.get("HPTUNE_RANDOM_INSERT_EVERY", "5"))
        expected_improvement_xi = float(os.environ.get("HPTUNE_EI_XI", "0.05"))
        max_trials = int(os.environ.get("HPTUNE_MAX_TRIALS", "10"))
        max_retries = int(os.environ.get("HPTUNE_MAX_RETRIES", "2"))
        allowed_epochs = tuple(hp.allowed_epochs)
        batch_sizes = tuple(hp.allowed_batch_sizes)
        bounds = settings.default_hptune_param_bounds(allowed_epochs, batch_sizes)

        return cls.model_validate(
            {
                "trials_dir": trials_dir,
                "project_root": settings.project_root,
                "best_trial_dir": os.path.join(trials_dir, "best_trial"),
                "max_retries": max_retries,
                "num_initial_trials": hp.num_initial_trials,
                "trial_nodes": trial_nodes,
                "controller_nodes": controller_nodes,
                "random_insert_every": random_insert_every,
                "expected_improvement_xi": expected_improvement_xi,
                "max_trials": max_trials,
                "allowed_epochs": allowed_epochs,
                "batch_sizes": batch_sizes,
                "bounds": bounds,
                "num_slots": num_slots,
            }
        )

    def _pbounds(self) -> dict[str, tuple[float, float]]:
        return dict(self.bounds)

    def _seen_trial_signatures(
        self, trials: list[HPTuneTrial]
    ) -> set[tuple[object, ...]]:
        """Collect signatures for all trials so new suggestions stay unique."""
        return {t.trial_signature() for t in trials}

    def _sample_unique_random(
        self,
        seen_signatures: set[tuple[object, ...]],
        *,
        context: str,
        max_attempts: int = 25,
    ) -> dict[str, Any]:
        """Draw random hyperparameter dicts until one is not already in the trial log."""
        for attempt in range(1, max_attempts + 1):
            proposal = self.sample_random()
            signature = HPTuneTrial.signature_from_proposal(proposal)
            if signature not in seen_signatures:
                if attempt > 1:
                    logger.info(
                        "Random sampling found a unique candidate after {} attempt(s) ({})",
                        attempt,
                        context,
                    )
                return proposal
            logger.warning(
                "Duplicate random candidate rejected on attempt {} ({})",
                attempt,
                context,
            )
        raise RuntimeError(
            f"Unable to find a unique random HPTune candidate after {max_attempts} attempts ({context})"
        )

    def _suggestion_to_trial(self, s: dict[str, float]) -> dict[str, Any]:
        """Map a Bayesian ``suggest()`` vector to hyperparameters (no ``trial_id`` yet)."""
        bi = int(np.clip(round(s["batch_idx"]), 0, max(len(self.batch_sizes) - 1, 0)))
        lsu_low, lsu_high = self.bounds["lr_scheduler_u"]
        lr_sched = bool(s["lr_scheduler_u"] >= (lsu_low + lsu_high) / 2.0)
        lsp_low, lsp_high = self.bounds["lr_sched_patience"]
        esp_low, esp_high = self.bounds["early_stop_patience"]
        lsf_low, lsf_high = self.bounds["lr_scheduler_factor"]
        return {
            "lr": float(s["lr"]),
            "epochs": min(self.allowed_epochs, key=lambda x: abs(x - s["epochs"])),
            "dropout": float(s["dropout"]),
            "weight_decay": 10 ** float(s["log_wd"]),
            "batch_size": self.batch_sizes[bi],
            "gradient_clip": float(s["gradient_clip"]),
            "lr_scheduler": lr_sched,
            "lr_scheduler_factor": float(
                np.clip(s["lr_scheduler_factor"], lsf_low, lsf_high)
            ),
            "lr_scheduler_patience": int(
                np.clip(round(s["lr_sched_patience"]), int(lsp_low), int(lsp_high))
            ),
            "early_stopping_patience": int(
                np.clip(round(s["early_stop_patience"]), int(esp_low), int(esp_high))
            ),
        }

    def _emit_dispatch_status(self, *, done: int, active: int, total: int) -> None:
        complete = int(active == 0 and total >= self.max_trials)
        print(
            f"Dispatch status -> done={done} active={active} total={total} "
            f"num_slots={self.num_slots} complete={complete}",
            flush=True,
        )

    def update_trials(self, trials: list[HPTuneTrial]) -> list[HPTuneTrial]:
        """Refresh in-progress trials from training logs; persist snapshot."""
        trials = list(trials)
        for i in range(len(trials)):
            trial = trials[i]
            if trial.status != -1:
                continue

            completed, val_loss = parse_val_loss(trial.dir_path)

            if completed:
                trials[i] = trial.model_copy(update={"val_loss": val_loss, "status": 0})
                sync_best_trial_artifacts(trials, self.best_trial_dir)
            else:
                trial_dir = trial.dir_path
                log_files = glob.glob(os.path.join(trial_dir, "*.log"))

                if log_files:
                    latest = max(log_files, key=os.path.getmtime)
                    age = time.time() - os.path.getmtime(latest)

                    if age > int(os.environ.get("TRIAL_TIMEOUT", "1800")):
                        retries = trial.retries

                        if retries < self.max_retries:
                            trials[i] = trial.model_copy(
                                update={"status": -2, "retries": retries + 1}
                            )
                            logger.warning(
                                "Retrying trial {} (attempt {}/{})",
                                trial.trial_id,
                                retries + 1,
                                self.max_retries,
                            )
                        else:
                            trials[i] = trial.model_copy(update={"status": -3})
                            logger.error(
                                "Trial {} failed permanently",
                                trial.trial_id,
                            )

        save_trials(trials)
        return trials

    def sample_random(self) -> dict[str, Any]:
        """Uniform random hyperparameters (no ``trial_id``); log-uniform ``lr`` / ``log_wd``."""
        lr_low, lr_high = self.bounds["lr"]
        dr_low, dr_high = self.bounds["dropout"]
        lw_low, lw_high = self.bounds["log_wd"]
        gc_low, gc_high = self.bounds["gradient_clip"]
        lsu_low, lsu_high = self.bounds["lr_scheduler_u"]
        lsf_low, lsf_high = self.bounds["lr_scheduler_factor"]
        lsp_low, lsp_high = self.bounds["lr_sched_patience"]
        esp_low, esp_high = self.bounds["early_stop_patience"]
        batch_sizes = self.batch_sizes
        return {
            "lr": 10 ** np.random.uniform(np.log10(lr_low), np.log10(lr_high)),
            "epochs": int(np.random.choice(self.allowed_epochs)),
            "dropout": float(np.random.uniform(dr_low, dr_high)),
            "weight_decay": 10 ** float(np.random.uniform(lw_low, lw_high)),
            "batch_size": batch_sizes[int(np.random.randint(0, len(batch_sizes)))],
            "gradient_clip": float(np.random.uniform(gc_low, gc_high)),
            "lr_scheduler": bool(
                np.random.uniform(lsu_low, lsu_high) >= (lsu_low + lsu_high) / 2.0
            ),
            "lr_scheduler_factor": float(np.random.uniform(lsf_low, lsf_high)),
            "lr_scheduler_patience": int(
                np.random.randint(int(lsp_low), int(lsp_high) + 1)
            ),
            "early_stopping_patience": int(
                np.random.randint(int(esp_low), int(esp_high) + 1)
            ),
        }

    def sample_bayesian(self, trials: list[HPTuneTrial]) -> dict[str, Any]:
        """Bayesian optimization over all registered float parameters."""
        completed = [t for t in trials if t.status == 0]
        if not completed:
            logger.warning("BO: zero completed trials; falling back to random sampling")
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
        logger.debug(
            "BO: registered {} observation(s) (ExpectedImprovement xi={})",
            len(completed),
            self.expected_improvement_xi,
        )

        suggestion = optimizer.suggest()
        proposal = self._suggestion_to_trial(suggestion)
        if HPTuneTrial.signature_from_proposal(proposal) in seen_signatures:
            logger.warning(
                "BO suggested a duplicate trial; falling back to random exploration"
            )
            return self._sample_unique_random(
                seen_signatures,
                context="duplicate Bayesian suggestion",
            )
        logger.debug("BO: raw suggestion (assign trial_id before materializing)")
        return proposal

    def sample_hyperparameters(self, trials: list[HPTuneTrial]) -> dict[str, Any]:
        completed_count = sum(1 for t in trials if t.status == 0)
        total_rows = len(trials)
        if completed_count < self.num_initial_trials:
            logger.info(
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
            logger.info(
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
        logger.info(
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
        trials = get_trials()
        out: list[HPTuneTrial] = []
        updated = 0
        for t in trials:
            if t.trial_id in want and t.status == -2:
                out.append(t.model_copy(update={"status": -1}))
                updated += 1
            else:
                out.append(t)
        save_trials(out)
        logger.info(
            "Marked {} queued trial(s) as running/submitted: {}",
            updated,
            ",".join(trial_ids),
        )

    def mark_trial_failed(self, trial_id: str, *, return_code: int) -> None:
        """Record an immediately failed dispatched trial and enqueue a retry if allowed."""
        trials = get_trials()
        out: list[HPTuneTrial] = []
        found = False
        for t in trials:
            if t.trial_id != trial_id:
                out.append(t)
                continue
            found = True
            if t.retries < self.max_retries:
                out.append(
                    t.model_copy(update={"status": -2, "retries": t.retries + 1})
                )
                logger.warning(
                    "Trial {} failed with return code {} and will be retried ({}/{})",
                    trial_id,
                    return_code,
                    t.retries + 1,
                    self.max_retries,
                )
            else:
                out.append(t.model_copy(update={"status": -3}))
                logger.error(
                    "Trial {} failed with return code {} and exhausted retries ({}/{})",
                    trial_id,
                    return_code,
                    t.retries,
                    self.max_retries,
                )
        if not found:
            raise ValueError(f"Unknown trial_id for failure update: {trial_id}")
        save_trials(out)

    def _plan_new_trials(
        self, trials: list[HPTuneTrial]
    ) -> tuple[list[HPTuneTrial], list[HPTuneTrial]]:
        """Create enough queued trials to fill the configured num_slots target."""
        active = sum(1 for t in trials if t.status in (-1, -2))
        available_slots = max(self.num_slots - active, 0)
        remaining_trials = max(self.max_trials - len(trials), 0)
        plan_count = min(available_slots, remaining_trials)
        if plan_count == 0:
            return trials, []

        trials = list(trials)
        for _ in range(plan_count):
            tid = next_trial_numbered_id(
                self.trials_dir,
                (t.trial_id for t in trials),
            )
            proposal = self.sample_hyperparameters(trials)
            trial = HPTuneTrial.model_validate(
                {
                    **proposal,
                    "trial_id": tid,
                    "status": -2,
                    "val_loss": -1.0,
                }
            )
            trial.log_pass_hyperparameters(trial, context="newly proposed (this pass)")
            trial.materialize_trial_files(
                project_root=self.project_root,
                log_dir=self.log_dir,
                is_serial=self.num_slots <= 1,
            )
            logger.info(
                "Materialized trial files: dir={}",
                trial.dir_path,
            )
            trials.append(trial)

        save_trials(trials)

        return trials

    def sync_and_load(self) -> list[HPTuneTrial]:
        """Load trial state from the trial service, sync running trials from logs, return trials."""
        trials = get_trials()
        return self.update_trials(trials)

    def plan_and_enqueue(self):
        """Create new trials to fill available parallel slots."""
        trials = get_trials()
        trials = self.update_trials(trials)

        trials = self._plan_new_trials(trials)
        return trials

    def get_dispatchable_trials(self, trials: list[HPTuneTrial]) -> list[str]:
        """Return queued trials (-2)."""
        return [t.trial_id for t in trials if t.status == -2]

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

        complete = (
            counts["total"] >= self.max_trials
            and counts["running"] == 0
            and counts["queued"] == 0
        )
        if complete:
            sql_to_csv()
        return complete

    def run(self) -> None:
        """Single dispatcher pass used by the serial controller and CLI."""
        trials = get_trials()
        trials = self.update_trials(trials)
        queued_trials = self.get_dispatchable_trials(trials)
        if not queued_trials:
            trials, planned_trials = self._plan_new_trials(trials)
            queued_trials = [t.trial_id for t in planned_trials]

        for trial_id in queued_trials:
            logger.info("Next trial -> {}", trial_id)
            print(f"Next trial -> {trial_id}", flush=True)

        counts = self.get_status_counts(trials)
        self._emit_dispatch_status(
            done=counts["done"],
            active=counts["running"] + counts["queued"],
            total=counts["total"],
        )
