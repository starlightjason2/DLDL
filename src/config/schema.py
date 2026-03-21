"""Pydantic: ``dldl.json`` + env merge for training/architecture."""

import os
from typing import ClassVar, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_M = ConfigDict(extra="forbid")
_DLDL_FILE = ConfigDict(extra="forbid", populate_by_name=True)


class HptuneConfig(BaseModel):
    model_config = _M
    dir: Optional[str] = None
    lr_min: float = Field(gt=0)
    lr_max: float = Field(gt=0)
    dropout_min: float = Field(ge=0, le=1)
    dropout_max: float = Field(ge=0, le=1)
    allowed_epochs: List[int] = Field(min_length=1)
    num_initial_trials: int = Field(ge=1)
    weight_decay_log_min: float
    weight_decay_log_max: float
    allowed_batch_sizes: List[int] = Field(min_length=1)
    gradient_clip_min: float = Field(ge=0)
    gradient_clip_max: float = Field(ge=0)
    lr_scheduler_factor_min: float = Field(gt=0, lt=1)
    lr_scheduler_factor_max: float = Field(gt=0, lt=1)
    lr_scheduler_patience_min: int = Field(ge=1)
    lr_scheduler_patience_max: int = Field(ge=1)
    early_stopping_patience_min: int = Field(ge=1)
    early_stopping_patience_max: int = Field(ge=1)

    @field_validator("allowed_epochs", "allowed_batch_sizes")
    @classmethod
    def _pos(cls, v: List[int]) -> List[int]:
        if any(x <= 0 for x in v):
            raise ValueError("list entries must be positive")
        return v

    @model_validator(mode="after")
    def _order(self) -> "HptuneConfig":
        checks = (
            (self.lr_min >= self.lr_max, "lr_min must be < lr_max"),
            (self.dropout_min >= self.dropout_max, "dropout_min must be < dropout_max"),
            (self.weight_decay_log_min >= self.weight_decay_log_max, "weight_decay_log_min must be < weight_decay_log_max"),
            (self.gradient_clip_min > self.gradient_clip_max, "gradient_clip_min must be <= gradient_clip_max"),
            (self.lr_scheduler_factor_min >= self.lr_scheduler_factor_max, "lr_scheduler_factor_min must be < lr_scheduler_factor_max"),
            (self.lr_scheduler_patience_min > self.lr_scheduler_patience_max, "lr_scheduler_patience_min must be <= lr_scheduler_patience_max"),
            (self.early_stopping_patience_min > self.early_stopping_patience_max, "early_stopping_patience_min must be <= early_stopping_patience_max"),
        )
        for bad, msg in checks:
            if bad:
                raise ValueError(msg)
        return self


class TrainingConfig(BaseModel):
    model_config = _M
    early_stopping_patience: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    num_epochs: int = Field(ge=1)
    log_interval: int = Field(ge=1)
    weight_decay: float = Field(ge=0)
    dropout_rate: float = Field(ge=0, le=1)
    batch_size: int = Field(ge=1)
    lr_scheduler: bool
    lr_scheduler_factor: float = Field(gt=0, lt=1)
    lr_scheduler_patience: int = Field(ge=1)
    gradient_clip: float = Field(ge=0)

    ENV_FIELDS: ClassVar[Tuple[Tuple[str, str], ...]] = (
        ("early_stopping_patience", "EARLY_STOPPING_PATIENCE"),
        ("learning_rate", "LEARNING_RATE"),
        ("num_epochs", "NUM_EPOCHS"),
        ("log_interval", "LOG_INTERVAL"),
        ("weight_decay", "WEIGHT_DECAY"),
        ("dropout_rate", "DROPOUT_RATE"),
        ("batch_size", "BATCH_SIZE"),
        ("lr_scheduler_factor", "LR_SCHEDULER_FACTOR"),
        ("lr_scheduler_patience", "LR_SCHEDULER_PATIENCE"),
        ("gradient_clip", "GRADIENT_CLIP"),
    )

    @classmethod
    def merge_env(cls, defaults: "TrainingConfig") -> "TrainingConfig":
        d = defaults.model_dump()
        if "LR_SCHEDULER" in os.environ:
            d["lr_scheduler"] = os.environ["LR_SCHEDULER"].lower() in ("true", "1", "yes", "on")
        for field, env in cls.ENV_FIELDS:
            if env in os.environ:
                d[field] = os.environ[env]
        return cls.model_validate(d)


class ArchitectureConfig(BaseModel):
    model_config = _M
    conv1_filters: int = Field(ge=1)
    conv1_kernel: int = Field(ge=1)
    conv1_padding: int = Field(ge=0)
    conv2_filters: int = Field(ge=1)
    conv2_kernel: int = Field(ge=1)
    conv2_padding: int = Field(ge=0)
    conv3_filters: int = Field(ge=1)
    conv3_kernel: int = Field(ge=1)
    conv3_padding: int = Field(ge=0)
    pool_size: int = Field(ge=1)
    fc1_size: int = Field(ge=1)
    fc2_size: int = Field(ge=1)

    ENV_FIELDS: ClassVar[Tuple[Tuple[str, str], ...]] = (
        ("conv1_filters", "CONV1_FILTERS"),
        ("conv1_kernel", "CONV1_KERNEL"),
        ("conv1_padding", "CONV1_PADDING"),
        ("conv2_filters", "CONV2_FILTERS"),
        ("conv2_kernel", "CONV2_KERNEL"),
        ("conv2_padding", "CONV2_PADDING"),
        ("conv3_filters", "CONV3_FILTERS"),
        ("conv3_kernel", "CONV3_KERNEL"),
        ("conv3_padding", "CONV3_PADDING"),
        ("pool_size", "POOL_SIZE"),
        ("fc1_size", "FC1_SIZE"),
        ("fc2_size", "FC2_SIZE"),
    )

    @classmethod
    def merge_env(cls, defaults: "ArchitectureConfig") -> "ArchitectureConfig":
        d = defaults.model_dump()
        for field, env in cls.ENV_FIELDS:
            if env in os.environ:
                d[field] = os.environ[env]
        return cls.model_validate(d)


class DldlConfigFile(BaseModel):
    model_config = _DLDL_FILE
    hptune: HptuneConfig
    default_training: TrainingConfig = Field(alias="defaultTraining")
    architecture: ArchitectureConfig


class DatasetEnv(BaseModel):
    model_config = _M
    normalization_type: Literal["scale", "meanvar-whole", "meanvar-single"]
    cpu_use: float = Field(gt=0, le=1)
    preprocessor_max_workers: int = Field(ge=1)

    @classmethod
    def from_os(cls) -> "DatasetEnv":
        return cls.model_validate(
            {
                "normalization_type": os.environ.get("NORMALIZATION_TYPE", "meanvar-whole"),
                "cpu_use": os.environ.get("CPU_USE", "0.2"),
                "preprocessor_max_workers": os.environ.get("PREPROCESSOR_MAX_WORKERS", "4"),
            },
        )
