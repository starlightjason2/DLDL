"""Bayesian hyperparameter tuning orchestration (trial log, acquisition, trial dirs)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

from bayes_opt import BayesianOptimization, acquisition
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from model.hyperparam_space import HyperparameterSpace
from model.hp_trial import HPTuneTrial, TrialStatus
from service.trial_service import TrialService
from util.hptune import (
    next_trial_numbered_id,
    parse_val_loss,
    sync_best_trial_artifacts,
)


class BayesianHPTuner(BaseModel):
    """Bayesian search over non-architecture training hyperparameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Paths
    trials_dir: str
    log_dir: str
    project_root: str
    best_trial_dir: str

    # Config
    max_retries: int = Field(ge=0)
    num_initial_trials: int = Field(ge=1)
    trial_nodes: int = Field(ge=1)
    controller_nodes: Optional[int] = None
    random_insert_every: int = Field(ge=0)
    expected_improvement_xi: float = Field(ge=0)
    max_trials: int = Field(ge=1)
    num_slots: int = Field(ge=1)

    space: HyperparameterSpace

    def model_post_init(self, __context: Any) -> None:
        job_id = os.environ.get("PBS_JOBID")
        if job_id:
            logger.add(
                Path(self.log_dir) / f"hptune_{job_id}.txt",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
                level="DEBUG",
                enqueue=True,
            )

    @staticmethod
    def _env_float(key: str) -> float:
        return float(os.environ[key])

    @staticmethod
    def _env_int(key: str) -> int:
        return int(os.environ[key])

    @staticmethod
    def _parse_ints(key: str) -> tuple[int, ...]:
        return tuple(int(x.strip()) for x in os.environ[key].split(",") if x.strip())

    @classmethod
    def _build_space(cls) -> HyperparameterSpace:
        f, i = cls._env_float, cls._env_int

        allowed_epochs = cls._parse_ints("HPTUNE_ALLOWED_EPOCHS")
        batch_sizes = cls._parse_ints("HPTUNE_ALLOWED_BATCH_SIZES")

        bounds = {
            "lr": (f("HPTUNE_LR_MIN"), f("HPTUNE_LR_MAX")),
            "dropout": (f("HPTUNE_DROPOUT_MIN"), f("HPTUNE_DROPOUT_MAX")),
            "log_wd": (
                f("HPTUNE_WEIGHT_DECAY_LOG_MIN"),
                f("HPTUNE_WEIGHT_DECAY_LOG_MAX"),
            ),
            "gradient_clip": (
                f("HPTUNE_GRADIENT_CLIP_MIN"),
                f("HPTUNE_GRADIENT_CLIP_MAX"),
            ),
            "lr_scheduler_factor": (
                f("HPTUNE_LR_SCHEDULER_FACTOR_MIN"),
                f("HPTUNE_LR_SCHEDULER_FACTOR_MAX"),
            ),
            "lr_sched_patience": (
                float(i("HPTUNE_LR_SCHEDULER_PATIENCE_MIN")),
                float(i("HPTUNE_LR_SCHEDULER_PATIENCE_MAX")),
            ),
            "early_stop_patience": (
                float(i("HPTUNE_EARLY_STOPPING_PATIENCE_MIN")),
                float(i("HPTUNE_EARLY_STOPPING_PATIENCE_MAX")),
            ),
            "epochs": (float(min(allowed_epochs)), float(max(allowed_epochs))),
            "batch_idx": (0.0, float(len(batch_sizes) - 1)),
            "lr_scheduler_u": (0.0, 1.0),
        }

        return HyperparameterSpace(
            allowed_epochs=allowed_epochs,
            batch_sizes=batch_sizes,
            bounds=bounds,
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(cls) -> BayesianHPTuner:
        root = Path(os.environ["HPTUNE_DIR"])
        trials_dir = root / "trials"
        log_dir = root / "controller_logs"

        trials_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        trial_nodes = cls._env_int("HPTUNE_TRIAL_NODES")
        controller_nodes = cls._env_int("HPTUNE_CONTROLLER_NODES")
        num_slots = max((controller_nodes - 1) // trial_nodes, 0)

        return cls(
            trials_dir=str(trials_dir),
            log_dir=str(log_dir),
            project_root=os.environ["PROJECT_ROOT"],
            best_trial_dir=str(trials_dir / "best_trial"),
            max_retries=cls._env_int("HPTUNE_MAX_RETRIES"),
            num_initial_trials=cls._env_int("HPTUNE_NUM_INITIAL_TRIALS"),
            trial_nodes=trial_nodes,
            controller_nodes=controller_nodes,
            random_insert_every=cls._env_int("HPTUNE_RANDOM_INSERT_EVERY"),
            expected_improvement_xi=float(os.environ["HPTUNE_EI_XI"]),
            max_trials=cls._env_int("HPTUNE_MAX_TRIALS"),
            num_slots=num_slots,
            space=cls._build_space(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _seen_signatures(self, trials: list[HPTuneTrial]) -> set[tuple]:
        return {t.trial_signature() for t in trials}

    def _sample_unique_random(
        self, seen: set[tuple], *, context: str
    ) -> dict[str, Any]:
        for attempt in range(1, 26):
            proposal = self.space.sample_random()
            if HPTuneTrial.signature_from_proposal(proposal) not in seen:
                return proposal
        raise RuntimeError(f"Exhausted random sampling ({context})")

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_bayesian(self, trials: list[HPTuneTrial]) -> dict[str, Any]:
        completed = [t for t in trials if t.status == TrialStatus.COMPLETED]
        seen = self._seen_signatures(trials)

        if not completed:
            return self._sample_unique_random(seen, context="no observations")

        optimizer = BayesianOptimization(
            f=None,
            pbounds=dict(self.space.bounds),
            acquisition_function=acquisition.ExpectedImprovement(
                xi=self.expected_improvement_xi
            ),
            allow_duplicate_points=True,
            verbose=0,
            random_state=42,
        )

        for t in completed:
            optimizer.register(
                params=t.bayesian_params(self.space.batch_sizes),
                target=-t.val_loss,
            )

        proposal = self.space.suggestion_to_trial(optimizer.suggest())

        if HPTuneTrial.signature_from_proposal(proposal) in seen:
            return self._sample_unique_random(seen, context="duplicate BO")

        return proposal

    def sample_hyperparameters(self, trials: list[HPTuneTrial]) -> dict[str, Any]:
        completed = sum(t.status == TrialStatus.COMPLETED for t in trials)
        seen = self._seen_signatures(trials)

        if completed < self.num_initial_trials:
            return self._sample_unique_random(seen, context="warmup")

        if (
            self.random_insert_every
            and (completed - self.num_initial_trials) % self.random_insert_every == 0
        ):
            return self._sample_unique_random(seen, context="periodic")

        return self.sample_bayesian(trials)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def update_trials(self, trials: list[HPTuneTrial]) -> list[HPTuneTrial]:
        timeout = int(os.environ["TRIAL_TIMEOUT"])
        updated = []

        for t in trials:
            if t.status != TrialStatus.RUNNING:
                updated.append(t)
                continue

            completed, val_loss = parse_val_loss(t.dir_path)

            if completed:
                t = t.model_copy(
                    update={"val_loss": val_loss, "status": TrialStatus.COMPLETED}
                )
                sync_best_trial_artifacts(trials, self.best_trial_dir)

            elif logs := sorted(
                Path(t.dir_path).glob("*.log"), key=lambda p: p.stat().st_mtime
            ):
                if time.time() - logs[-1].stat().st_mtime > timeout:
                    if t.retries < self.max_retries:
                        t = t.model_copy(
                            update={
                                "status": TrialStatus.QUEUED,
                                "retries": t.retries + 1,
                            }
                        )
                    else:
                        t = t.model_copy(update={"status": TrialStatus.FAILED})

            updated.append(t)

        TrialService.save_trials(updated)
        return updated

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _plan_new_trials(self, trials: list[HPTuneTrial]) -> list[HPTuneTrial]:
        active = sum(
            t.status in (TrialStatus.RUNNING, TrialStatus.QUEUED) for t in trials
        )

        plan_count = min(
            max(self.num_slots - active, 0),
            max(self.max_trials - len(trials), 0),
        )

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

            trial.create_scripts(project_root=self.project_root)
            trials.append(trial)

        TrialService.save_trials(trials)
        return trials

    def mark_trials_running(self, trial_ids: list[str]) -> None:
        if not trial_ids:
            return
        want = frozenset(trial_ids)
        promoted = [
            t.model_copy(update={"status": TrialStatus.RUNNING})
            for t in TrialService.get_trials()
            if t.trial_id in want and t.status == TrialStatus.QUEUED
        ]
        TrialService.save_trials(promoted)
        logger.info(
            "Marked {} trial(s) running: {}", len(promoted), ",".join(trial_ids)
        )

    def mark_trial_failed(self, trial_id: str, *, return_code: int) -> None:
        trial = TrialService.get_trial(trial_id)
        if trial.retries < self.max_retries:
            TrialService.update_trial(
                trial_id,
                {"status": TrialStatus.QUEUED, "retries": trial.retries + 1},
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
                "Trial {} failed (code {}) — exhausted retries",
                trial_id,
                return_code,
            )

    def is_complete(self, trials: list[HPTuneTrial]) -> bool:
        counts = TrialService.get_status_counts(trials)
        if (
            counts["total"] >= self.max_trials
            and counts["running"] == 0
            and counts["queued"] == 0
        ):
            TrialService.sql_to_csv()
            return True
        return False

    def run_serial(self) -> None:
        """One dispatcher pass for the serial PBS controller (`scripts/controller.sh`)."""
        trials = TrialService.get_trials()
        trials = self.update_trials(trials)

        queued_ids = [t.trial_id for t in trials if t.status == TrialStatus.QUEUED]
        if not queued_ids:
            trials = self._plan_new_trials(trials)
            queued_ids = [t.trial_id for t in trials if t.status == TrialStatus.QUEUED]

        # Controller parses a single `Next trial -> id` line; never emit more than one.
        if queued_ids:
            logger.info("Next trial -> {}", queued_ids[0])

        counts = TrialService.get_status_counts(trials)
        active = counts["active"]
        complete = int(active == 0 and counts["total"] >= self.max_trials)
        logger.info(
            "Dispatch status -> done={} active={} total={} num_slots={} complete={}",
            counts["done"],
            active,
            counts["total"],
            self.num_slots,
            complete,
        )
        TrialService.sql_to_csv()
