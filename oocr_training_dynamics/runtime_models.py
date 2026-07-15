"""Live-model loading helpers used only after the explicit GPU gate."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch as t
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, PreTrainedModel

from oocr_training_dynamics.contracts import TrainingSpec
from oocr_training_dynamics.models import ModelSpec

LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)


def load_processor(spec: ModelSpec) -> Any:
    try:
        processor = AutoProcessor.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            use_fast=True,
        )
    except (ImportError, OSError, ValueError):
        processor = AutoTokenizer.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            use_fast=True,
        )
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is None:
            raise RuntimeError("processor requires a pad or EOS token")
        cast(Any, tokenizer).pad_token_id = eos
    return processor


def tokenizer_for(processor: Any) -> Any:
    return getattr(processor, "tokenizer", processor)


def load_base_model(spec: ModelSpec, *, training: bool) -> PreTrainedModel:
    model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            dtype=t.bfloat16,
            attn_implementation="sdpa",
        ),
    )
    cast(Any, model).to(t.device("cuda"))
    model.config.use_cache = not training
    if training:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    resolved = getattr(model.config, "_commit_hash", None)
    if resolved != spec.revision:
        raise RuntimeError(f"resolved model revision {resolved!r} != pinned {spec.revision}")
    return model


def attach_trainable_lora(
    base_model: PreTrainedModel,
    training: TrainingSpec,
    *,
    adapter_path: Path | None = None,
) -> t.nn.Module:
    if adapter_path is not None:
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
    else:
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=training.lora_rank,
            lora_alpha=training.lora_alpha,
            lora_dropout=training.lora_dropout,
            target_modules=list(LORA_TARGET_MODULES),
            bias="none",
        )
        model = get_peft_model(base_model, config)
    cast(Any, model).enable_input_require_grads()
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("training must update only a non-empty set of LoRA parameters")
    return model


def attach_inference_lora(base_model: PreTrainedModel, adapter_path: Path) -> t.nn.Module:
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    model.eval()
    return model


def resolve_decoder_blocks(model: t.nn.Module, spec: ModelSpec) -> tuple[t.nn.Module, ...]:
    matches: list[tuple[str, tuple[t.nn.Module, ...]]] = []
    for candidate in spec.block_path_candidates:
        current: Any = model
        try:
            for component in candidate.split("."):
                current = getattr(current, component)
        except AttributeError:
            continue
        if isinstance(current, t.nn.ModuleList | list | tuple):
            blocks = tuple(current)
            if all(isinstance(block, t.nn.Module) for block in blocks):
                matches.append((candidate, blocks))
    exact = [(path, blocks) for path, blocks in matches if len(blocks) == spec.layer_count]
    if len(exact) != 1:
        summary = [(path, len(blocks)) for path, blocks in matches]
        raise RuntimeError(
            f"expected one {spec.layer_count}-layer decoder path for {spec.key}; found {summary}"
        )
    return exact[0][1]


__all__ = [
    "LORA_TARGET_MODULES",
    "attach_inference_lora",
    "attach_trainable_lora",
    "load_base_model",
    "load_processor",
    "resolve_decoder_blocks",
    "seed_everything",
    "tokenizer_for",
]
