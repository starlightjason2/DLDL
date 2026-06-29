"""Bayesian hyperparameter tuning orchestration (trial log, acquisition, trial dirs)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bayes_opt import BayesianOptimization, acquisition
from loguru import logger

from model.hyperparam_space import HyperparameterSpace
from model.hp_trial import HPTuneTrial, TrialStatus
from service.trial_service import TrialService
from util.hptune import (
    next_trial_numbered_id,
    parse_trial_metrics,
    sync_best_trial_artifacts,
)
from util.data_loading import env_int
from util.pbs import submit_hptune_step


class BayesianHPTuner:
    """Bayesian search over non-architecture training hyperparameters."""

    # Paths
    trials_dir: Path
    log_dir: Path

    # Config
    max_retries: int
    max_trials: int
    hp_space: HyperparameterSpace

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        root = Path(os.environ["HPTUNE_DIR"])

        self.trials_dir = root / "trials"
        self.trials_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = root / "controller_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.max_retries = env_int("HPTUNE_MAX_RETRIES")
        if self.max_retries < 0:
            raise ValueError("HPTUNE_MAX_RETRIES must be >= 0")

        self.max_trials = env_int("HPTUNE_MAX_TRIALS")
        if self.max_trials < 1:
            raise ValueError("HPTUNE_MAX_TRIALS must be >= 1")

        self.hp_space = HyperparameterSpace.from_env()

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
                target=t.score,
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

            done, metrics = parse_trial_metrics(t.dir_path)

            if done:
                t = t.model_copy(
                    update={
                        "score": metrics["score"],
                        "recall": metrics["recall"],
                        "precision": metrics["precision"],
                        "status": TrialStatus.COMPLETED,
                    }
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
            return True
        return False

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _plan_next_trial(self, trials: list[HPTuneTrial]) -> list[HPTuneTrial]:
        """Append one new QUEUED trial (and its files) if under ``max_trials``."""
        if len(trials) >= self.max_trials:
            return trials

        tid = next_trial_numbered_id(self.trials_dir, (t.trial_id for t in trials))
        proposal = self.sample_hyperparameters(trials)

        trial = HPTuneTrial.model_validate(
            {
                **proposal,
                "trial_id": tid,
                "status": TrialStatus.QUEUED,
                "score": -1.0,
            }
        )

        trial.create_files()
        trials.append(trial)
        TrialService.save_trials(trials)
        return trials

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

    @staticmethod
    def _utc_stamp() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _log_chain_step(self, trial_id: str, step_job: str) -> None:
        line = ",".join(
            [
                self._utc_stamp(),
                os.environ.get("HPTUNE_CHAIN_ID", ""),
                os.environ.get("PBS_JOBID", ""),
                trial_id,
                step_job,
            ]
        )
        with (self.trials_dir / "chain_steps.csv").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _log_chain_complete(self) -> None:
        chain_id = os.environ.get("HPTUNE_CHAIN_ID", "")
        with (self.trials_dir / "chain_summary.log").open("a", encoding="utf-8") as f:
            f.write(f"Chain {chain_id} finished at {self._utc_stamp()}\n")

    def _best_trial_checkpoint(self) -> Path | None:
        """Path to the global-best trial's checkpoint, if one exists yet."""
        checkpoints = sorted((self.trials_dir / "best_trial").glob("*_best_params.pt"))
        return checkpoints[0] if checkpoints else None

    def _train_trial(self, trial: HPTuneTrial) -> int:
        """Train one trial in a subprocess; returns the training exit code.

        The trial's hyperparameters (and ``PROG_DIR``/``JOB_ID``) are passed to
        ``train.py`` through the environment; everything else is inherited from the
        already-sourced project ``.env``. If a best-trial checkpoint exists, the
        trial warm-starts from it so a terminated run picks up off the best model
        found so far.
        """
        trial.log_pass_hyperparameters(context="train")
        env = {
            **os.environ,
            **{k: str(v) for k, v in trial.trial_env_keys().items()},
        }

        warm_start = self._best_trial_checkpoint()
        if warm_start is not None:
            env["WARM_START_CHECKPOINT"] = str(warm_start)
            logger.info("Warm-starting {} from {}", trial.trial_id, warm_start)

        logger.info("Training {} ...", trial.trial_id)
        result = subprocess.run(
            [sys.executable, "src/train.py"],
            cwd=os.environ["PROJECT_ROOT"],
            env=env,
        )
        logger.info(
            "Training {} exited with code {}", trial.trial_id, result.returncode
        )
        return result.returncode

    def run_step(self) -> None:
        """One self-contained HP-tune step.

        Ingest any finished trial, plan the next trial if none is queued, train it
        in-process, record its score, then submit the next step and exit. Exactly
        one trial runs per job and at most one job is ever pending (this running
        job plus the queued next step), so it fits queues that allow only one
        running + one queued job per user (e.g. Polaris ``debug``). Re-running
        ``start_hptune.sh`` resumes from ``trials.csv`` if a step is ever lost.
        """
        trials = self.update_trials(TrialService.get_trials())

        if self.is_complete(trials):
            logger.info("Chain complete.")
            self._log_chain_complete()
            return

        queued_ids = [t.trial_id for t in trials if t.status == TrialStatus.QUEUED]
        if not queued_ids:
            trials = self._plan_next_trial(trials)
            queued_ids = [t.trial_id for t in trials if t.status == TrialStatus.QUEUED]

        if not queued_ids:
            logger.info("Chain complete.")
            self._log_chain_complete()
            return

        trial = next(t for t in trials if t.trial_id == queued_ids[0])
        TrialService.update_trial(trial.trial_id, {"status": TrialStatus.RUNNING})

        counts = TrialService.get_status_counts(trials)
        logger.info(
            "Step start -> trial={} done={} total={} max_trials={}",
            trial.trial_id,
            counts["done"],
            counts["total"],
            self.max_trials,
        )

        return_code = self._train_trial(trial)

        # Record the result of the trial we just ran.
        done, metrics = parse_trial_metrics(trial.dir_path)
        if done:
            TrialService.save_trials(
                [
                    trial.model_copy(
                        update={
                            "score": metrics["score"],
                            "recall": metrics["recall"],
                            "precision": metrics["precision"],
                            "status": TrialStatus.COMPLETED,
                        }
                    )
                ]
            )
            sync_best_trial_artifacts(
                TrialService.get_trials(), self.trials_dir / "best_trial"
            )
            logger.info(
                "Trial {} completed: score={:.6f} recall={:.6f} precision={:.6f}",
                trial.trial_id,
                metrics["score"],
                metrics["recall"],
                metrics["precision"],
            )
        else:
            self.mark_trial_failed(trial.trial_id, return_code=return_code)

        # Chain the next step unless the run is done.
        if self.is_complete(TrialService.get_trials()):
            logger.info("Chain complete.")
            self._log_chain_complete()
            return

        step_job = submit_hptune_step(
            log_dir=self.log_dir,
            queue=os.environ["HPTUNE_QUEUE"],
            walltime=os.environ["HPTUNE_TRAIN_WALLTIME"],
        )
        logger.info("Submitted next step as {}", step_job)
        self._log_chain_step(trial.trial_id, step_job)
