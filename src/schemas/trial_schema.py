"""Hyperparameter trial model (Pydantic) and ORM bridge — replaces the former ``HPTuneTrial`` dataclass."""

from __future__ import annotations

import os
import re
from datetime import datetime
from textwrap import dedent
from typing import Any, Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class HPTuneTrial(BaseModel):
    """One hyperparameter trial: training hparams, log status, identity.

    ``trial_id`` is the primary key (same as :attr:`database.tables.Trial.trial_id`).
    """

    model_config = ConfigDict(from_attributes=True, validate_assignment=True)

    trial_id: str = Field(description="Primary key; ORM column ``trials.trial_id``.")
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
    val_loss: float = -1.0
    status: int = -1
    retries: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def dir_path(self) -> str:
        """Absolute path to this trial's directory under ``trials_dir``."""
        return os.path.join(os.environ["TRIALS_DIR"], self.trial_id)

    def trial_signature(self) -> tuple[object, ...]:
        """Normalize a trial into a hashable signature for duplicate detection."""
        return self.signature_from_proposal(self.model_dump())

    @classmethod
    def signature_from_proposal(cls, d: Mapping[str, Any]) -> tuple[object, ...]:
        """Same tuple as :meth:`trial_signature` for hyperparameter dicts (no ``trial_id`` required)."""
        return (
            f"{float(d['lr']):.12g}",
            int(d["epochs"]),
            f"{float(d['dropout']):.12g}",
            f"{float(d['weight_decay']):.12g}",
            int(d["batch_size"]),
            f"{float(d['gradient_clip']):.12g}",
            int(d["lr_scheduler"]),
            f"{float(d['lr_scheduler_factor']):.12g}",
            int(d["lr_scheduler_patience"]),
            int(d["early_stopping_patience"]),
        )

    def log_pass_hyperparameters(self, *, context: str) -> None:
        self._log.info(
            "Hyperparameters for this pass ({}): trial_id={} lr={:.2e} epochs={} dropout={:.4f} "
            "weight_decay={:.2e} batch_size={} gradient_clip={:.3f} lr_scheduler={} "
            "lr_scheduler_factor={:.3f} lr_scheduler_patience={} early_stopping_patience={}",
            context,
            self.trial_id,
            self.lr,
            self.epochs,
            self.dropout,
            self.weight_decay,
            self.batch_size,
            self.gradient_clip,
            self.lr_scheduler,
            self.lr_scheduler_factor,
            self.lr_scheduler_patience,
            self.early_stopping_patience,
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

    def write_env_file(self, env_path: str, env_lines: list[str] | None = None) -> None:
        """Write this trial's ``.env`` with optional extra lines plus HPTune overrides."""
        prefix = "\n".join(env_lines or [])
        lr_scheduler_str = "true" if self.lr_scheduler else "false"
        env_content = (
            prefix
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
    def _build_run_script(
        project_root: str,
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

    @staticmethod
    def serial_chain_submission_block(log_dir: str) -> str:
        """Shell snippet that queues the next serial controller after training completes."""
        return dedent(
            f"""

                # --- Submit Next Controller (HPTune Job Chain) ---
                # Serial HPTune keeps the chain alive by queueing the next controller
                # only after the current training job finishes.
                if [ -n "$HPTUNE_CHAIN_ID" ] && [ -n "$PROJECT_ROOT" ]; then
                    echo "Training complete. Submitting next controller..."
                    NEXT_CTL_JOB_ID=$(qsub \\
                        -A fusiondl_aesp \\
                        -q "$HPTUNE_QUEUE" \\
                        -l select=1:system=polaris,place=scatter,walltime=1:00:00,filesystems=home:eagle \\
                        -k doe \\
                        -o "{log_dir}/" \\
                        -e "{log_dir}/" \\
                        -v "HPTUNE_CHAIN_ID=$HPTUNE_CHAIN_ID" \\
                        "$PROJECT_ROOT/scripts/controller.sh") || {{
                            echo "ERROR: Next controller qsub failed. Chain will not continue."
                            exit 1
                        }}
                    echo "Next controller queued: $NEXT_CTL_JOB_ID"
                fi
                """
        )

    def materialize_trial_files(
        self,
        *,
        project_root: str,
        log_dir: str,
        env_lines: list[str] | None = None,
        is_serial=False,
    ) -> str:
        """Write trial directory, ``.env``, and ``run.sh``; return ``trial_id``."""
        if not self.trial_id:
            raise ValueError(
                "HPTuneTrial.trial_id must be set before creating trial files",
            )

        os.makedirs(self.dir_path, exist_ok=True)

        env_path = os.path.join(self.dir_path, ".env")
        self.write_env_file(env_path, env_lines)

        template_path = os.path.join(project_root, "scripts", "run_train.sh")
        script = self._build_run_script(project_root, env_path, template_path)
        script += self._post_run_checkpoint_cleanup_block()

        if is_serial:
            script += self.serial_chain_submission_block(log_dir)

        run_path = os.path.join(self.dir_path, "run.sh")
        with open(run_path, "w") as f:
            f.write(script)
        os.chmod(run_path, 0o755)
        return self.trial_id

    def _batch_index(self, batch_sizes: tuple[int, ...]) -> int:
        if self.batch_size in batch_sizes:
            return batch_sizes.index(self.batch_size)
        return min(
            range(len(batch_sizes)),
            key=lambda index: abs(batch_sizes[index] - self.batch_size),
        )
