"""Validated, preregistered experiment identifiers and schedules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from beartype import beartype

EFFECTIVE_BATCH_SIZE = 64
TRAINING_EXAMPLES = 96_000
FINAL_STEP = TRAINING_EXAMPLES // EFFECTIVE_BATCH_SIZE
CHECKPOINT_STEPS = (
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1_024,
    1_280,
    FINAL_STEP,
)
RESUME_STEPS = (256, 512, 1_024, FINAL_STEP)
PRIMARY_SEED = 20_260_715


class TrainingCondition(StrEnum):
    """The mapping taught by an otherwise matched I/O corpus."""

    CORRECT = "correct"
    WRONG_ALIAS = "wrong_alias"
    WRONG_IMPL = "wrong_impl"


class PatchingMode(StrEnum):
    """Source of activations patched into a recipient forward pass."""

    ACROSS_SAMPLE = "across_sample"
    ACROSS_TIME = "across_time"
    LATER_CHECKPOINT = "later_checkpoint"


class PatchingInterface(StrEnum):
    """Decoder-stream module boundary whose hidden vector is transplanted."""

    RESID_POST = "resid_post"
    ATTENTION_INPUT = "attention_input"
    ATTENTION_OUTPUT = "attention_output"
    MLP_INPUT = "mlp_input"
    MLP_OUTPUT = "mlp_output"


@beartype
@dataclass(frozen=True)
class RunKey:
    model: str
    condition: TrainingCondition
    seed: int = PRIMARY_SEED

    def __post_init__(self) -> None:
        if not self.model or Path(self.model).name != self.model:
            raise ValueError("model key must be one non-empty path component")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    def relative_dir(self) -> Path:
        return Path(self.model) / self.condition.value / f"seed_{self.seed}"


@beartype
@dataclass(frozen=True)
class TrainingSpec:
    run: RunKey
    sample_count: int = TRAINING_EXAMPLES
    effective_batch_size: int = EFFECTIVE_BATCH_SIZE
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.0
    max_gradient_norm: float = 1.0
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    checkpoint_steps: tuple[int, ...] = CHECKPOINT_STEPS
    resume_steps: tuple[int, ...] = RESUME_STEPS

    def __post_init__(self) -> None:
        if self.effective_batch_size <= 0:
            raise ValueError("effective batch size must be positive")
        if self.sample_count <= 0 or self.sample_count % self.effective_batch_size != 0:
            raise ValueError("sample count must be a positive multiple of effective batch size")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight decay must be finite and non-negative")
        if not math.isfinite(self.max_gradient_norm) or self.max_gradient_norm <= 0.0:
            raise ValueError("maximum gradient norm must be finite and positive")
        if self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not math.isfinite(self.lora_dropout) or not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        final_step = self.sample_count // self.effective_batch_size
        if not self.checkpoint_steps or self.checkpoint_steps[0] != 0:
            raise ValueError("checkpoint schedule must begin at the frozen base step")
        if tuple(sorted(set(self.checkpoint_steps))) != self.checkpoint_steps:
            raise ValueError("checkpoint steps must be strictly increasing and unique")
        if self.checkpoint_steps[-1] != final_step:
            raise ValueError("checkpoint schedule must end at the final optimizer step")
        if not set(self.resume_steps).issubset(self.checkpoint_steps):
            raise ValueError("resume steps must be a subset of checkpoint steps")

    @property
    def final_step(self) -> int:
        return self.sample_count // self.effective_batch_size

    def examples_at_step(self, step: int) -> int:
        if step < 0 or step > self.final_step:
            raise ValueError("step is outside this run")
        return step * self.effective_batch_size


def checkpoint_label(step: int) -> str:
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    return f"step_{step:06d}"


__all__ = [
    "CHECKPOINT_STEPS",
    "EFFECTIVE_BATCH_SIZE",
    "FINAL_STEP",
    "PRIMARY_SEED",
    "PatchingInterface",
    "PatchingMode",
    "RESUME_STEPS",
    "RunKey",
    "TRAINING_EXAMPLES",
    "TrainingCondition",
    "TrainingSpec",
    "checkpoint_label",
]
