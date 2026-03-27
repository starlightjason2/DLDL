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
from util.data_loading import env_int


class BayesianHPTuner(BaseModel):
    """Bayesian search over non-architecture training hyperparameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Paths
    trials_dir: Path
    log_dir: Path

    # Config
    max_retries: int = Field(ge=0)
    trial_nodes: int = Field(ge=1)
    controller_nodes: Optional[int] = None
    max_trials: int = Field(ge=1)
    num_slots: int = Field(ge=1)
    hp_space: HyperparameterSpace

    def model_post_init(self, __context: Any) -> None:
        job_id = os.environ.get("PBS_JOBID")
        if job_id:
            logger.add(
                Path(self.log_dir) / f"hptune_{job_id}.txt",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
                level="DEBUG",
                enqueue=True,
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

        trial_nodes = env_int("HPTUNE_TRIAL_NODES")
        controller_nodes = env_int("HPTUNE_CONTROLLER_NODES")

        return cls(
            trials_dir=trials_dir,
            log_dir=log_dir,
            max_retries=env_int("HPTUNE_MAX_RETRIES"),
            trial_nodes=trial_nodes,
            controller_nodes=controller_nodes,
            max_trials=env_int("HPTUNE_MAX_TRIALS"),
            # At least one slot for serial PBS (controller is its own job).
            num_slots=(
                max((controller_nodes - 1) // trial_nodes, 1) if controller_nodes else 1
            ),
            hp_space=HyperparameterSpace.from_env(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _seen_signatures(self, trials: list[HPTuneTrial]) -> set[tuple]:
        return {t.signature() for t in trials}

    def _sample_random(
        self, trials: list[HPTuneTrial], *, context: str
    ) -> dict[str, Any]:
        seen = self._seen_signatures(trials)
        for attempt in range(1, 26):
            proposal = self.hp_space.sample_random()
            if HPTuneTrial.proposed_signature(proposal) not in seen:
                return proposal
        raise RuntimeError(f"Exhausted random sampling ({context})")

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_bayesian(self, trials: list[HPTuneTrial]) -> dict[str, Any]:
        completed = [t for t in trials if t.status == TrialStatus.COMPLETED]

        if not completed:
            return self._sample_random(trials, context="no observations")

        optimizer = BayesianOptimization(
            f=None,
            pbounds=dict(self.hp_space.bounds),
            acquisition_function=acquisition.ExpectedImprovement(
                xi=self.hp_space.expected_improvement_xi
            ),
            allow_duplicate_points=True,
            verbose=0,
            random_state=42,
        )

        for t in completed:
            optimizer.register(
                params=t.bayesian_params(self.hp_space.batch_sizes),
                target=-t.val_loss,
            )

        proposal = self.hp_space.suggestion_to_trial(optimizer.suggest())  # type: ignore

        if HPTuneTrial.proposed_signature(proposal) in self._seen_signatures(trials):
            return self._sample_random(
                trials, context="Duplicate Hyperparameters, skipping"
            )

        return proposal

    def sample_hyperparameters(self, trials: list[HPTuneTrial]) -> dict[str, Any]:
        completed = sum(t.status == TrialStatus.COMPLETED for t in trials)

        if completed < self.hp_space.num_initial_trials:
            return self._sample_random(trials, context="warmup")

        if (
            self.hp_space.random_insert_every
            and (completed - self.hp_space.num_initial_trials)
            % self.hp_space.random_insert_every
            == 0
        ):
            return self._sample_random(trials, context="periodic")

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
        sync_best_trial_artifacts(updated, self.trials_dir / "best_trial")
        return updated

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

            trial.create_files()
            trials.append(trial)

        TrialService.save_trials(trials)
        return trials

    def mark_trials_running(self, trial_ids: list[str]) -> None:
        want = frozenset(trial_ids)
        promoted = [
            t.model_copy(update={"status": TrialStatus.RUNNING})
            for t in TrialService.get_trials()
            if t.trial_id in want and t.status == TrialStatus.QUEUED
        ]
        TrialService.save_trials(promoted)
        logger.info(
            "Marked {} trial(s) running: {}",
            len(promoted),
            ",".join(t.trial_id for t in promoted),
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

    # ------------------------------------------------------------------
    # Serial Runner
    # ------------------------------------------------------------------

    def run_serial(self) -> None:
        """One dispatcher pass for the serial PBS controller."""
        trials = self.update_trials(TrialService.get_trials())

        queued_ids = [t.trial_id for t in trials if t.status == TrialStatus.QUEUED]
        if not queued_ids:
            trials = self._plan_new_trials(trials)
            queued_ids = [t.trial_id for t in trials if t.status == TrialStatus.QUEUED]

        counts = TrialService.get_status_counts(trials)
        logger.info(
            "Dispatch status -> done={} active={} total={} num_slots={} complete={}",
            counts["done"],
            counts["active"],
            counts["total"],
            self.num_slots,
            self.is_complete(trials),
        )

        if not queued_ids:
            logger.info("Chain complete.")
            return

        # Shell parser in controller.sh keys on this exact format
        print(f"Next trial -> {queued_ids[0]}")
        TrialService.sql_to_csv()
