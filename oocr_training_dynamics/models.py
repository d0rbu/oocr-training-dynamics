"""Pinned model registry and storage estimates; no model loading occurs here."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from beartype import beartype


class ModelKey(StrEnum):
    OLMO3_7B = "olmo3-7b"
    QWEN3_8B = "qwen3-8b"
    GEMMA4_E4B = "gemma4-e4b"


@beartype
@dataclass(frozen=True)
class ModelSpec:
    key: ModelKey
    label: str
    model_id: str
    revision: str
    architecture: str
    layer_count: int
    hidden_size: int
    intermediate_size: int
    query_width: int
    key_value_width: int
    default_micro_batch_size: int
    provisional: bool = False
    provisional_reason: str | None = None
    block_path_candidates: tuple[str, ...] = (
        "model.layers",
        "model.model.layers",
        "language_model.model.layers",
        "model.language_model.model.layers",
    )

    def __post_init__(self) -> None:
        if self.layer_count <= 0 or min(
            self.hidden_size,
            self.intermediate_size,
            self.query_width,
            self.key_value_width,
            self.default_micro_batch_size,
        ) <= 0:
            raise ValueError("model dimensions and batch size must be positive")
        if len(self.revision) != 40 or any(character not in "0123456789abcdef" for character in self.revision):
            raise ValueError("model revision must be a full lowercase commit SHA")
        if self.provisional != (self.provisional_reason is not None):
            raise ValueError("provisional models require exactly one provisional reason")

    def lora_parameter_count(self, rank: int = 32) -> int:
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        attention = rank * (
            (self.hidden_size + self.query_width)
            + 2 * (self.hidden_size + self.key_value_width)
            + (self.query_width + self.hidden_size)
        )
        mlp = 3 * rank * (self.hidden_size + self.intermediate_size)
        return self.layer_count * (attention + mlp)

    def adapter_mib(self, rank: int = 32, bytes_per_parameter: int = 2) -> float:
        if bytes_per_parameter <= 0:
            raise ValueError("bytes per parameter must be positive")
        return self.lora_parameter_count(rank) * bytes_per_parameter / 2**20


MODEL_SPECS: dict[ModelKey, ModelSpec] = {
    ModelKey.OLMO3_7B: ModelSpec(
        key=ModelKey.OLMO3_7B,
        label="OLMo 3 · 7B Instruct",
        model_id="allenai/Olmo-3-7B-Instruct",
        revision="6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
        architecture="Olmo3ForCausalLM",
        layer_count=32,
        hidden_size=4_096,
        intermediate_size=11_008,
        query_width=4_096,
        key_value_width=4_096,
        default_micro_batch_size=32,
    ),
    ModelKey.QWEN3_8B: ModelSpec(
        key=ModelKey.QWEN3_8B,
        label="Qwen 3 · 8B",
        model_id="Qwen/Qwen3-8B",
        revision="b968826d9c46dd6066d109eabc6255188de91218",
        architecture="Qwen3ForCausalLM",
        layer_count=36,
        hidden_size=4_096,
        intermediate_size=12_288,
        query_width=4_096,
        key_value_width=1_024,
        default_micro_batch_size=16,
    ),
    ModelKey.GEMMA4_E4B: ModelSpec(
        key=ModelKey.GEMMA4_E4B,
        label="Gemma 4 · E4B Instruct (8B total)",
        model_id="google/gemma-4-E4B-it",
        revision="a4c2d58be94dda072b918d9db64ee85c8ed34e3f",
        architecture="Gemma4ForConditionalGeneration",
        layer_count=42,
        hidden_size=2_560,
        intermediate_size=10_240,
        query_width=2_048,
        key_value_width=512,
        default_micro_batch_size=16,
        provisional=True,
        provisional_reason=(
            "Google does not publish a Gemma 4 9B checkpoint; E4B is the closest size "
            "at 4.5B effective and 8B total parameters and requires user confirmation."
        ),
    ),
}


@beartype
def get_model_spec(key: ModelKey | str, *, allow_provisional: bool = False) -> ModelSpec:
    parsed = key if isinstance(key, ModelKey) else ModelKey(key)
    spec = MODEL_SPECS[parsed]
    if spec.provisional and not allow_provisional:
        raise RuntimeError(spec.provisional_reason)
    return spec


__all__ = ["MODEL_SPECS", "ModelKey", "ModelSpec", "get_model_spec"]
