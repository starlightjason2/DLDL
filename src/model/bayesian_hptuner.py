"""Bayesian hyperparameter tuning orchestration (trial log, acquisition, trial dirs)."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from bayes_opt import BayesianOptimization, acquisition
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from config.settings import Settings
from schemas.trial_schema import HPTuneTrial, TrialStatus
from service.trial_service import TrialService
from util.hptune import (
    next_trial_numbered_id,
    parse_val_loss,
    sync_best_trial_artifacts,
)


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
    controller_nodes: Optional[int] = None
    random_insert_every: int = Field(ge=0)
    expected_improvement_xi: float = Field(ge=0)
    max_trials: int = Field(ge=1)
    num_slots: int = Field(ge=1)

    allowed_epochs: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    bounds: dict[str, Tuple[float, float]]

    def model_post_init(self, __context: Any) -> None:
        Path(self.trials_dir).mkdir(parents=True, exist_ok=True)

        if job_id := os.environ.get("PBS_JOBID"):
            logger.add(
                Path(self.log_dir) / f"hptune_{job_id}.txt",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
                level="DEBUG",
                enqueue=True,
            )

    @classmethod
    def create(cls) -> BayesianHPTuner:
        trials_dir = Path(os.environ["TRIALS_DIR"])
        log_dir = trials_dir.parent / "controller_logs"
        trials_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        settings = Settings.load()
        hp = settings.cfg.hptune
        trial_nodes = int(os.environ.get("HPTUNE_TRIAL_NODES", "1"))
        controller_nodes = int(os.environ.get("HPTUNE_CONTROLLER_NODES", "0"))
        # One node is dedicated to the controller; the rest are split among trial_nodes.
        num_slots = math.floor((controller_nodes - 1) / trial_nodes)

        allowed_epochs = tuple(hp.allowed_epochs)
        batch_sizes = tuple(hp.allowed_batch_sizes)

        return cls.model_validate(
            {
                "trials_dir": str(trials_dir),
                "log_dir": str(log_dir),
                "project_root": settings.project_root,
                "best_trial_dir": str(trials_dir / "best_trial"),
                "max_retries": int(os.environ.get("HPTUNE_MAX_RETRIES", "2")),
                "num_initial_trials": hp.num_initial_trials,
                "trial_nodes": trial_nodes,
                "controller_nodes": controller_nodes,
                "random_insert_every": int(
                    os.environ.get("HPTUNE_RANDOM_INSERT_EVERY", "5")
                ),
                "expected_improvement_xi": float(
                    os.environ.get("HPTUNE_EI_XI", "0.05")
                ),
                "max_trials": int(os.environ.get("HPTUNE_MAX_TRIALS", "10")),
                "allowed_epochs": allowed_epochs,
                "batch_sizes": batch_sizes,
                "bounds": settings.default_hptune_param_bounds(
                    allowed_epochs, batch_sizes
                ),
                "num_slots": num_slots,
            }
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _seen_signatures(self, trials: list[HPTuneTrial]) -> set[tuple[object, ...]]:
        return {t.trial_signature() for t in trials}

    @logger.catch(
        RuntimeError, message="Unique random sampling exhausted", reraise=True
    )
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
            if HPTuneTrial.signature_from_proposal(proposal) not in seen_signatures:
                if attempt > 1:
                    logger.info(
                        "Unique random candidate found after {} attempt(s) ({})",
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
            "lr_scheduler": bool(s["lr_scheduler_u"] >= (lsu_low + lsu_high) / 2.0),
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

    # ------------------------------------------------------------------
    # Trial lifecycle
    # ------------------------------------------------------------------

    def update_trials(self, trials: list[HPTuneTrial]) -> list[HPTuneTrial]:
        """Refresh in-progress trials and persist the batch."""
        timeout_limit = int(os.environ.get("TRIAL_TIMEOUT", "1800"))
        updated: list[HPTuneTrial] = []

        for trial in trials:
            if trial.status != TrialStatus.RUNNING:
                updated.append(trial)
                continue

            completed, val_loss = parse_val_loss(trial.dir_path)

            if completed:
                trial = trial.model_copy(
                    update={"val_loss": val_loss, "status": TrialStatus.COMPLETED}
                )
                sync_best_trial_artifacts(trials, self.best_trial_dir)
            elif log_files := sorted(
                Path(trial.dir_path).glob("*.log"), key=lambda p: p.stat().st_mtime
            ):
                if (time.time() - log_files[-1].stat().st_mtime) > timeout_limit:
                    if trial.retries < self.max_retries:
                        trial = trial.model_copy(
                            update={
                                "status": TrialStatus.QUEUED,
                                "retries": trial.retries + 1,
                            }
                        )
                        logger.warning(
                            "Retrying {} ({}/{})",
                            trial.trial_id,
                            trial.retries,
                            self.max_retries,
                        )
                    else:
                        trial = trial.model_copy(update={"status": TrialStatus.FAILED})
                        logger.error("Trial {} failed permanently", trial.trial_id)

            updated.append(trial)

        TrialService.save_trials(updated)
        return updated

    def mark_trials_running(self, trial_ids: list[str]) -> None:
        """Promote queued trials to running."""
        if not trial_ids:
            return
        want = set(trial_ids)
        promoted = [
            t.model_copy(update={"status": TrialStatus.RUNNING})
            for t in TrialService.get_trials()
            if t.trial_id in want and t.status == TrialStatus.QUEUED
        ]
        TrialService.save_trials(promoted)
        logger.info(
            "Marked {} trial(s) as running: {}", len(promoted), ",".join(trial_ids)
        )

    def mark_trial_failed(self, trial_id: str, *, return_code: int) -> None:
        """Retry or permanently fail a trial based on retry budget."""
        trial = TrialService.get_trial(trial_id)

        if trial.retries < self.max_retries:
            TrialService.update_trial(
                trial_id, {"status": TrialStatus.QUEUED, "retries": trial.retries + 1}
            )
            logger.warning(
                "Trial {} failed (code {}) — retry {}/{}",
                trial_id,
                return_code,
                trial.retries + 1,
                self.max_retries,
            )
        else:
            TrialService.update_trial(trial_id, {"status": TrialStatus.FAILED})
            logger.error(
                "Trial {} failed (code {}) — exhausted retries ({}/{})",
                trial_id,
                return_code,
                trial.retries,
                self.max_retries,
            )

    # ------------------------------------------------------------------
    # Sampling strategies
    # ------------------------------------------------------------------

    def sample_random(self) -> dict[str, Any]:
        """Uniform random hyperparameters (no ``trial_id``); log-uniform for ``lr``/``weight_decay``."""
        b = self.bounds
        lr_low, lr_high = b["lr"]
        dr_low, dr_high = b["dropout"]
        lw_low, lw_high = b["log_wd"]
        gc_low, gc_high = b["gradient_clip"]
        lsu_low, lsu_high = b["lr_scheduler_u"]
        lsf_low, lsf_high = b["lr_scheduler_factor"]
        lsp_low, lsp_high = b["lr_sched_patience"]
        esp_low, esp_high = b["early_stop_patience"]
        return {
            "lr": 10 ** np.random.uniform(np.log10(lr_low), np.log10(lr_high)),
            "epochs": int(np.random.choice(self.allowed_epochs)),
            "dropout": float(np.random.uniform(dr_low, dr_high)),
            "weight_decay": 10 ** float(np.random.uniform(lw_low, lw_high)),
            "batch_size": self.batch_sizes[
                int(np.random.randint(0, len(self.batch_sizes)))
            ],
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
        completed = [t for t in trials if t.status == TrialStatus.COMPLETED]
        seen_signatures = self._seen_signatures(trials)

        if not completed:
            logger.warning(
                "BO: zero completed trials — falling back to random sampling"
            )
            return self._sample_unique_random(
                seen_signatures, context="Bayesian fallback (no observations)"
            )

        optimizer = BayesianOptimization(
            f=None,
            pbounds=dict(self.bounds),
            acquisition_function=acquisition.ExpectedImprovement(
                xi=self.expected_improvement_xi
            ),
            allow_duplicate_points=True,
            verbose=0,
            random_state=42,
        )
        for trial in completed:
            optimizer.register(
                params=trial.bayesian_params(self.batch_sizes), target=-trial.val_loss
            )

        logger.debug(
            "BO: {} observation(s) registered (EI xi={})",
            len(completed),
            self.expected_improvement_xi,
        )

        proposal = self._suggestion_to_trial(optimizer.suggest())
        if HPTuneTrial.signature_from_proposal(proposal) in seen_signatures:
            logger.warning(
                "BO suggested a duplicate — falling back to random exploration"
            )
            return self._sample_unique_random(
                seen_signatures, context="duplicate Bayesian suggestion"
            )

        logger.debug(
            "BO: unique suggestion ready (assign trial_id before materializing)"
        )
        return proposal

    def sample_hyperparameters(self, trials: list[HPTuneTrial]) -> dict[str, Any]:
        """Choose a sampling strategy based on warmup progress and insertion schedule."""
        completed_count = sum(1 for t in trials if t.status == TrialStatus.COMPLETED)
        total_rows = len(trials)
        seen = self._seen_signatures(trials)

        if completed_count < self.num_initial_trials:
            logger.info(
                "Strategy: random warmup (completed={}/{} initial, total={})",
                completed_count,
                self.num_initial_trials,
                total_rows,
            )
            return self._sample_unique_random(seen, context="warmup")

        post_warmup = completed_count - self.num_initial_trials
        if (
            self.random_insert_every
            and post_warmup > 0
            and post_warmup % self.random_insert_every == 0
        ):
            logger.info(
                "Strategy: periodic random insertion (completed={} post_warmup={} every={} total={})",
                completed_count,
                post_warmup,
                self.random_insert_every,
                total_rows,
            )
            return self._sample_unique_random(seen, context="periodic random insertion")

        logger.info(
            "Strategy: Bayesian (completed={}, total={})", completed_count, total_rows
        )
        return self.sample_bayesian(trials)

    # ------------------------------------------------------------------
    # Planning & dispatch
    # ------------------------------------------------------------------

    def _plan_new_trials(self, trials: list[HPTuneTrial]) -> list[HPTuneTrial]:
        """Create enough queued trials to fill the configured ``num_slots``."""
        active = sum(
            1 for t in trials if t.status in (TrialStatus.RUNNING, TrialStatus.QUEUED)
        )
        plan_count = min(
            max(self.num_slots - active, 0),
            max(self.max_trials - len(trials), 0),
        )
        if not plan_count:
            return trials

        trials = list(trials)
        for _ in range(plan_count):
            tid = next_trial_numbered_id(self.trials_dir, (t.trial_id for t in trials))
            proposal = self.sample_hyperparameters(trials)
            trial = HPTuneTrial.model_validate(
                {
                    **proposal,
                    "trial_id": tid,
                    "status": TrialStatus.QUEUED,
                    "val_loss": -1.0,
                }
            )
            trial.log_pass_hyperparameters(context="newly proposed (this pass)")
            trial.create_scripts(project_root=self.project_root)
            logger.info("Materialized trial: dir={}", trial.dir_path)
            trials.append(trial)

        TrialService.save_trials(trials)
        return trials

    def plan_and_enqueue(self) -> list[HPTuneTrial]:
        """Refresh trial states then fill available slots with new trials."""
        trials = TrialService.get_trials()
        trials = self.update_trials(trials)
        return self._plan_new_trials(trials)

    def get_dispatchable_trials(self, trials: list[HPTuneTrial]) -> list[str]:
        """Return trial IDs ready to be submitted."""
        return [t.trial_id for t in trials if t.status == TrialStatus.QUEUED]

    def is_complete(self, trials: list[HPTuneTrial]) -> bool:
        counts = TrialService.get_status_counts(trials)
        if complete := (
            counts["total"] >= self.max_trials
            and counts["running"] == 0
            and counts["queued"] == 0
        ):
            TrialService.sql_to_csv()
        return complete

    def run(self) -> None:
        """Single dispatcher pass used by the serial controller and CLI."""
        trials = TrialService.get_trials()
        trials = self.update_trials(trials)

        queued = self.get_dispatchable_trials(trials)
        if not queued:
            trials = self._plan_new_trials(trials)
            queued = self.get_dispatchable_trials(trials)

        for trial_id in queued:
            logger.info("Next trial -> {}", trial_id)

        counts = TrialService.get_status_counts(trials)
        complete = int(counts.active == 0 and counts.total >= self.max_trials)
        logger.opt(lazy=True).info(
            "Dispatch status -> done={done} active={active} total={total} num_slots={slots} complete={complete}",
            done=lambda: counts.done,
            active=lambda: counts.active,
            total=lambda: counts.total,
            slots=lambda: self.num_slots,
            complete=lambda: complete,
        )
