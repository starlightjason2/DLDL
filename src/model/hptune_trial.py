"""Trial model for DLDL Bayesian hyperparameter tuning."""

from __future__ import annotations

import os
import re
from dataclasses import MISSING, dataclass, fields
from textwrap import dedent
from typing import Any, Optional

import numpy as np


@dataclass(init=False)
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
    retries: int = 0

    @property
    def dir_name(self) -> str:
        """Folder under ``trials/`` (``trial_1``, ``trial_2``, ...). Requires ``trial_id``."""
        return self._require_trial_id()

    def path_under(self, trials_dir: str) -> str:
        return os.path.join(trials_dir, self.dir_name)

    @classmethod
    def _validate_initialization(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Validate constructor keywords once: required fields, defaults, reject unknown keys."""
        kwargs = dict(kwargs)
        resolved: dict[str, Any] = {}
        for f in fields(cls):
            if f.name in kwargs:
                resolved[f.name] = kwargs.pop(f.name)
            elif f.default is not MISSING:
                resolved[f.name] = f.default
            elif f.default_factory is not MISSING:
                resolved[f.name] = f.default_factory()
            else:
                raise TypeError(
                    f"HPTuneTrial.__init__ missing required keyword argument {f.name!r}",
                )
        if kwargs:
            raise TypeError(
                f"HPTuneTrial.__init__ got unexpected keyword argument(s): {sorted(kwargs)!r}",
            )
        return resolved

    def __init__(self, **kwargs):
        """Construct from hyperparameter keyword fields.

        Prefer validating through :class:`schemas.trial_schema.TrialSchema` when building new
        trials from HP tuning so values match the schema.
        Use :class:`service.trial_service.TrialService` to load/save persisted trials.
        """
        for name, value in self._validate_initialization(kwargs).items():
            setattr(self, name, value)

    def validate_for_persistence(self) -> None:
        """Ensure this trial can be written to storage (ORM snapshot / DB row)."""
        if not self.trial_id:
            raise ValueError("trial_id must be set before persisting to the database")

    def _require_trial_id(self) -> str:
        """Return ``trial_id`` or raise if unset (paths, materialized trial dirs)."""
        if not self.trial_id:
            raise ValueError("trial_id must be set before using dir_name or path_under")
        return self.trial_id

    def trial_signature(self) -> tuple[object, ...]:
        """Normalize a trial into a hashable signature for duplicate detection."""
        return (
            f"{self.lr:.12g}",
            int(self.epochs),
            f"{self.dropout:.12g}",
            f"{self.weight_decay:.12g}",
            int(self.batch_size),
            f"{self.gradient_clip:.12g}",
            int(self.lr_scheduler),
            f"{self.lr_scheduler_factor:.12g}",
            int(self.lr_scheduler_patience),
            int(self.early_stopping_patience),
        )

    def bayesian_params(self, batch_sizes: tuple[int, ...]) -> dict[str, float]:
        """Float-only parameter dict aligned with BayesianOptimization pbounds."""
        batch_index = self._batch_index(batch_sizes)
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
            "batch_idx": float(batch_index),
        }

    def write_env_file(self, env_path: str, env_lines: list[str]) -> None:
        """Write this trial's `.env` file from the shared template plus overrides."""
        lr_scheduler_str = "true" if self.lr_scheduler else "false"
        env_content = (
            "\n".join(env_lines)
            + f"""
    # HPTune overrides
    LEARNING_RATE={self.lr}
    NUM_EPOCHS={self.epochs}
    DROPOUT_RATE={self.dropout}
    WEIGHT_DECAY={self.weight_decay}
    BATCH_SIZE={self.batch_size}
    GRADIENT_CLIP={self.gradient_clip}
    LR_SCHEDULER={lr_scheduler_str}
    LR_SCHEDULER_FACTOR={self.lr_scheduler_factor}
    LR_SCHEDULER_PATIENCE={self.lr_scheduler_patience}
    EARLY_STOPPING_PATIENCE={self.early_stopping_patience}
    PROG_DIR={os.path.dirname(env_path)}
    JOB_ID={self.trial_id}
    # run.sh tee already writes full stderr to train_${{PBS_JOBID}}.log; skip duplicate training.log
    TRAIN_LOGURU_FILE=0
    """
        )
        with open(env_path, "w") as f:
            f.write(env_content)

    @staticmethod
    def build_run_script(
        project_root: str,
        trial_dir: str,
        env_path: str,
        template_path: str,
    ) -> str:
        """Build run.sh content from the shared training template."""
        with open(template_path) as f:
            script = f.read()

        inject_block = (
            f"set -a\nsource {env_path}\nset +a\n"
            f'exec > >(tee "$PROG_DIR/train_${{PBS_JOBID}}.log") 2>&1'
        )

        script = (
            script.replace("#PBS -N dldl_train", "#PBS -N dldl_hptune")
            .replace("# __HPTUNE_CD_OVERRIDE__", f"cd {project_root}")
            .replace("# __HPTUNE_ENV_INJECT__", inject_block)
        )
        script = re.sub(r"#PBS -o .*", "#PBS -o /dev/null", script)
        script = re.sub(r"#PBS -e .*", "#PBS -e /dev/null", script)

        assert "# __HPTUNE_CD_OVERRIDE__" not in script, (
            f"CD override sentinel was not replaced in {template_path}. "
            "Add '# __HPTUNE_CD_OVERRIDE__' on its own line before the cd command."
        )
        assert "# __HPTUNE_ENV_INJECT__" not in script, (
            f"Env inject sentinel was not replaced in {template_path}. "
            "Add '# __HPTUNE_ENV_INJECT__' on its own line after 'set -e'."
        )

        return script

    @staticmethod
    def _post_run_checkpoint_cleanup_block() -> str:
        """Shell snippet that removes epoch checkpoints after the best checkpoint exists."""
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

    def _materialize_trial(
        self,
        *,
        project_root: str,
        trials_dir: str,
        env_lines: list[str],
        trailer_block: str = "",
    ) -> str:
        """Write trial directory, ``.env``, and ``run.sh``; return ``trial_id``."""
        if not self.trial_id:
            raise ValueError(
                "HPTuneTrial.trial_id must be set before creating trial files",
            )

        trial_dir = self.path_under(trials_dir)
        os.makedirs(trial_dir, exist_ok=True)

        env_path = os.path.join(trial_dir, ".env")
        self.write_env_file(env_path, env_lines)

        template_path = os.path.join(project_root, "scripts", "run_train.sh")
        script = self.build_run_script(project_root, trial_dir, env_path, template_path)
        script += self._post_run_checkpoint_cleanup_block()
        script += trailer_block

        run_path = os.path.join(trial_dir, "run.sh")
        with open(run_path, "w") as f:
            f.write(script)
        os.chmod(run_path, 0o755)
        return self.dir_name

    def _batch_index(self, batch_sizes: tuple[int, ...]) -> int:
        if self.batch_size in batch_sizes:
            return batch_sizes.index(self.batch_size)
        return min(
            range(len(batch_sizes)),
            key=lambda index: abs(batch_sizes[index] - self.batch_size),
        )


@dataclass(init=False)
class SerialTrial(HPTuneTrial):
    """PBS serial chain: ``run.sh`` may submit the next controller after training completes."""

    def materialize_trial_files(
        self,
        *,
        project_root: str,
        trials_dir: str,
        env_lines: list[str],
        log_dir: Optional[str] = None,
    ) -> str:
        """Write ``.env`` / ``run.sh`` with the serial controller handoff block."""
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(trials_dir), "controller_logs")
        return self._materialize_trial(
            project_root=project_root,
            trials_dir=trials_dir,
            env_lines=env_lines,
            trailer_block=self._serial_chain_submission_block(log_dir),
        )

    @staticmethod
    def _serial_chain_submission_block(log_dir: str) -> str:
        """Shell snippet that queues the next serial controller after training completes."""
        return dedent(
            f"""

            # --- Submit Next Controller (HPTune Job Chain) ---
            # Serial HPTune keeps the chain alive by queueing the next controller
            # only after the current training job finishes.
            if [ -n "$DLDL_HPTUNE_CHAIN_ID" ] && [ -n "$PROJECT_ROOT" ]; then
                echo "Training complete. Submitting next controller..."
                NEXT_CTL_JOB_ID=$(qsub \\
                    -A fusiondl_aesp \\
                    -q "${{HPTUNE_QUEUE:-small}}" \\
                    -l select=1:system=polaris,place=scatter,walltime=1:00:00,filesystems=home:eagle \\
                    -k doe \\
                    -o "{log_dir}/" \\
                    -e "{log_dir}/" \\
                    -v "PROJECT_ROOT=$PROJECT_ROOT,DLDL_HPTUNE_CHAIN_ID=$DLDL_HPTUNE_CHAIN_ID,HPTUNE_QUEUE=${{HPTUNE_QUEUE:-small}},DLDL_HPTUNE_DIR=${{DLDL_HPTUNE_DIR}},HPTUNE_MAX_TRIALS=${{HPTUNE_MAX_TRIALS:-10}}" \\
                    "$PROJECT_ROOT/scripts/controller.sh") || {{
                        echo "ERROR: Next controller qsub failed. Chain will not continue."
                        exit 1
                    }}
                echo "Next controller queued: $NEXT_CTL_JOB_ID"
            fi
            """
        )


@dataclass(init=False)
class ParallelTrial(HPTuneTrial):
    """MPI / multi-node dispatch: no serial controller submission in ``run.sh``."""

    def materialize_trial_files(
        self,
        *,
        project_root: str,
        trials_dir: str,
        env_lines: list[str],
        log_dir: Optional[str] = None,
    ) -> str:
        """Write ``.env`` / ``run.sh`` without the PBS chain block."""
        _ = log_dir  # parallel layout does not use controller log dir for qsub
        return self._materialize_trial(
            project_root=project_root,
            trials_dir=trials_dir,
            env_lines=env_lines,
            trailer_block="",
        )
