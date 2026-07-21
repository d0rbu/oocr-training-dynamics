"""Validated, preregistered experiment identifiers and schedules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from beartype import beartype

EFFECTIVE_BATCH_SIZE = 64
BATCH_ABLATION_SIZES = (32, 16, 8, 4, 2, 1)
SUPPORTED_EFFECTIVE_BATCH_SIZES = (EFFECTIVE_BATCH_SIZE, *BATCH_ABLATION_SIZES)
DEFAULT_LORA_RANK = 32
LORA_RANKS = (1, 2, 4, 8, 16, DEFAULT_LORA_RANK, 64, 128, 256, 512, 1_024)
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
CHECKPOINT_EXAMPLES = tuple(step * EFFECTIVE_BATCH_SIZE for step in CHECKPOINT_STEPS)
PRIMARY_SEED = 20_260_715


def checkpoint_steps_for_batch_size(effective_batch_size: int) -> tuple[int, ...]:
    """Return example-matched checkpoints plus the ablation's first optimizer step."""

    if effective_batch_size not in SUPPORTED_EFFECTIVE_BATCH_SIZES:
        raise ValueError(f"effective batch size must be one of {SUPPORTED_EFFECTIVE_BATCH_SIZES}")
    matched_steps = {examples // effective_batch_size for examples in CHECKPOINT_EXAMPLES}
    matched_steps.add(1)
    return tuple(sorted(matched_steps))


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
    """Decoder boundary or parameter scope transplanted from a donor checkpoint."""

    RESID_POST = "resid_post"
    ATTENTION_INPUT = "attention_input"
    ATTENTION_OUTPUT = "attention_output"
    MLP_INPUT = "mlp_input"
    MLP_OUTPUT = "mlp_output"
    TOKEN_WEIGHTS = "token_weights"
    BLOCK_WEIGHTS = "block_weights"

    @property
    def patches_weights(self) -> bool:
        """Whether this interface replaces parameters instead of hidden vectors."""

        return self in {
            PatchingInterface.TOKEN_WEIGHTS,
            PatchingInterface.BLOCK_WEIGHTS,
        }

    @property
    def patches_token_weights(self) -> bool:
        """Whether donor LoRA updates apply only at one selected token position."""

        return self is PatchingInterface.TOKEN_WEIGHTS

    @property
    def patches_all_token_weights(self) -> bool:
        """Whether donor LoRA parameters replace a block globally for all tokens."""

        return self is PatchingInterface.BLOCK_WEIGHTS


class TokenWeightRuntime(StrEnum):
    """Execution kernel for token-local learned-weight interventions."""

    REFERENCE = "reference"
    OPTIMIZED = "optimized"


@beartype
@dataclass(frozen=True)
class RunKey:
    model: str
    condition: TrainingCondition
    seed: int = PRIMARY_SEED
    effective_batch_size: int = EFFECTIVE_BATCH_SIZE
    lora_rank: int | None = DEFAULT_LORA_RANK

    def __post_init__(self) -> None:
        if not self.model or Path(self.model).name != self.model:
            raise ValueError("model key must be one non-empty path component")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.effective_batch_size not in SUPPORTED_EFFECTIVE_BATCH_SIZES:
            raise ValueError(
                f"effective batch size must be one of {SUPPORTED_EFFECTIVE_BATCH_SIZES}"
            )
        if self.lora_rank is not None and self.lora_rank not in LORA_RANKS:
            raise ValueError(f"LoRA rank must be one of {LORA_RANKS} or null for full finetuning")
        if (
            self.effective_batch_size != EFFECTIVE_BATCH_SIZE
            and self.lora_rank != DEFAULT_LORA_RANK
        ):
            raise ValueError("batch and rank ablations are one-factor-at-a-time")

    def relative_dir(self) -> Path:
        base = Path(self.model) / self.condition.value / f"seed_{self.seed}"
        if self.effective_batch_size != EFFECTIVE_BATCH_SIZE:
            base /= f"effective_batch_{self.effective_batch_size}"
        if self.lora_rank is None:
            base /= "full_finetune"
        elif self.lora_rank != DEFAULT_LORA_RANK:
            base /= f"lora_rank_{self.lora_rank}"
        return base


@beartype
@dataclass(frozen=True)
class TrainingSpec:
    run: RunKey
    sample_count: int = TRAINING_EXAMPLES
    effective_batch_size: int = EFFECTIVE_BATCH_SIZE
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.0
    max_gradient_norm: float = 1.0
    lora_rank: int = DEFAULT_LORA_RANK
    lora_alpha: int = 2 * DEFAULT_LORA_RANK
    lora_dropout: float = 0.0
    checkpoint_steps: tuple[int, ...] = CHECKPOINT_STEPS
    resume_steps: tuple[int, ...] = RESUME_STEPS

    def __post_init__(self) -> None:
        if self.effective_batch_size <= 0:
            raise ValueError("effective batch size must be positive")
        if self.effective_batch_size != self.run.effective_batch_size:
            raise ValueError("training and run-key effective batch sizes must match")
        if self.sample_count <= 0 or self.sample_count % self.effective_batch_size != 0:
            raise ValueError("sample count must be a positive multiple of effective batch size")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight decay must be finite and non-negative")
        if not math.isfinite(self.max_gradient_norm) or self.max_gradient_norm <= 0.0:
            raise ValueError("maximum gradient norm must be finite and positive")
        if self.run.lora_rank is None:
            raise ValueError("full finetuning requires its dedicated offload training contract")
        if self.lora_rank != self.run.lora_rank:
            raise ValueError("training and run-key LoRA ranks must match")
        if self.lora_rank <= 0 or self.lora_alpha != 2 * self.lora_rank:
            raise ValueError("LoRA rank must be positive and alpha must equal twice the rank")
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


def training_spec_for_run(run: RunKey) -> TrainingSpec:
    """Construct a validated LoRA contract for the selected batch and rank axes."""

    batch_size = run.effective_batch_size
    if run.lora_rank is None:
        raise ValueError("full finetuning requires the separately gated ZeRO-3 offload path")
    checkpoints = (
        CHECKPOINT_STEPS
        if batch_size == EFFECTIVE_BATCH_SIZE
        else checkpoint_steps_for_batch_size(batch_size)
    )
    resume_steps = RESUME_STEPS if batch_size == EFFECTIVE_BATCH_SIZE else checkpoints[1:]
    return TrainingSpec(
        run,
        effective_batch_size=batch_size,
        lora_rank=run.lora_rank,
        lora_alpha=2 * run.lora_rank,
        checkpoint_steps=checkpoints,
        resume_steps=resume_steps,
    )


def checkpoint_label(step: int) -> str:
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    return f"step_{step:06d}"


__all__ = [
    "BATCH_ABLATION_SIZES",
    "CHECKPOINT_EXAMPLES",
    "CHECKPOINT_STEPS",
    "DEFAULT_LORA_RANK",
    "EFFECTIVE_BATCH_SIZE",
    "FINAL_STEP",
    "LORA_RANKS",
    "PRIMARY_SEED",
    "PatchingInterface",
    "PatchingMode",
    "RESUME_STEPS",
    "RunKey",
    "SUPPORTED_EFFECTIVE_BATCH_SIZES",
    "TRAINING_EXAMPLES",
    "TrainingCondition",
    "TrainingSpec",
    "checkpoint_label",
    "checkpoint_steps_for_batch_size",
    "training_spec_for_run",
]
