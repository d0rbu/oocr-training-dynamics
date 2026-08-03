"""Pure contracts and paths for symmetric effective-weight comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from beartype import beartype

from oocr_training_dynamics.artifacts import run_dir
from oocr_training_dynamics.contracts import RunKey, checkpoint_label
from oocr_training_dynamics.models import MODEL_SPECS, ModelKey

WEIGHT_ALIGNMENT_KIND = "effective_projection_weight_alignment"
WEIGHT_ALIGNMENT_SCHEMA_VERSION = 1
WEIGHT_ALIGNMENT_ACCUMULATION_DTYPE = "float32"
WEIGHT_ALIGNMENT_MATRIX_NAMES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
WEIGHT_ALIGNMENT_METRICS = (
    "frobenius_cosine",
    "frobenius_l2",
    "mean_row_cosine",
    "mean_column_cosine",
    "mean_row_l2",
    "mean_column_l2",
)
WEIGHT_ALIGNMENT_DETAIL_METRICS = (
    "row_cosines",
    "column_cosines",
    "row_l2_distances",
    "column_l2_distances",
)
WEIGHT_ALIGNMENT_DEGENERATE_COUNTS = (
    "row_both_zero_count",
    "row_one_zero_count",
    "column_both_zero_count",
    "column_one_zero_count",
)
WEIGHT_ALIGNMENT_ZERO_NORM_CONVENTION = (
    "ordinary cosine when both norms are nonzero; 1 when both vectors are zero; "
    "0 when exactly one vector is zero"
)
WEIGHT_ALIGNMENT_VARIANCE_METRICS = (
    "row_cosine_variance",
    "column_cosine_variance",
    "row_l2_variance",
    "column_l2_variance",
)


@dataclass(frozen=True)
class WeightComponentSpec:
    """One complete-model weight category exposed on the site axis."""

    component_id: str
    label: str
    placement: Literal["input", "layer", "output"]
    tensor_rank: Literal[1, 2]
    shape: tuple[int, ...]
    parameter_template: str
    frozen_during_lora: bool
    row_group_size: int | None = None
    column_group_size: int | None = None
    group_label: str | None = None


def _projection_components(model: ModelKey) -> dict[str, WeightComponentSpec]:
    spec = MODEL_SPECS[model]
    shapes = {
        "q_proj": (spec.query_width, spec.hidden_size),
        "k_proj": (spec.key_value_width, spec.hidden_size),
        "v_proj": (spec.key_value_width, spec.hidden_size),
        "o_proj": (spec.hidden_size, spec.query_width),
        "gate_proj": (spec.intermediate_size, spec.hidden_size),
        "up_proj": (spec.intermediate_size, spec.hidden_size),
        "down_proj": (spec.hidden_size, spec.intermediate_size),
    }
    labels = {
        "q_proj": "Attention · Q projection",
        "k_proj": "Attention · K projection",
        "v_proj": "Attention · V projection",
        "o_proj": "Attention · O projection",
        "gate_proj": "MLP · gate projection",
        "up_proj": "MLP · up projection",
        "down_proj": "MLP · down projection",
    }
    return {
        name: WeightComponentSpec(
            component_id=name,
            label=labels[name],
            placement="layer",
            tensor_rank=2,
            shape=shapes[name],
            parameter_template=f"model.layers.{{layer}}.{('self_attn' if name in {'q_proj', 'k_proj', 'v_proj', 'o_proj'} else 'mlp')}.{name}.weight",
            frozen_during_lora=False,
            row_group_size=128 if name in {"q_proj", "k_proj", "v_proj"} else None,
            column_group_size=128 if name == "o_proj" else None,
            group_label="attention head"
            if name in {"q_proj", "k_proj", "v_proj", "o_proj"}
            else None,
        )
        for name in WEIGHT_ALIGNMENT_MATRIX_NAMES
    }


@beartype
def weight_component_specs(model: ModelKey) -> tuple[WeightComponentSpec, ...]:
    """Return every learned weight tensor category for one supported decoder schema."""

    spec = MODEL_SPECS[model]
    projections = _projection_components(model)
    global_shapes = {
        ModelKey.OLMO3_7B: (100_278, spec.hidden_size),
        ModelKey.QWEN3_8B: (151_936, spec.hidden_size),
    }
    if model not in global_shapes:
        raise ValueError(f"complete weight-component inventory is not registered for {model.value}")
    vocabulary_shape = global_shapes[model]
    global_input = WeightComponentSpec(
        "embed_tokens",
        "Global input · token embedding",
        "input",
        2,
        vocabulary_shape,
        "model.embed_tokens.weight",
        True,
    )
    final_norm = WeightComponentSpec(
        "final_norm",
        "Global output · final norm",
        "output",
        1,
        (spec.hidden_size,),
        "model.norm.weight",
        True,
    )
    lm_head = WeightComponentSpec(
        "lm_head",
        "Global output · unembedding / LM head",
        "output",
        2,
        vocabulary_shape,
        "lm_head.weight",
        True,
    )
    qk_norm_size = spec.hidden_size if model is ModelKey.OLMO3_7B else 128
    q_norm = WeightComponentSpec(
        "q_norm",
        "Attention · Q norm",
        "layer",
        1,
        (qk_norm_size,),
        "model.layers.{layer}.self_attn.q_norm.weight",
        True,
    )
    k_norm = WeightComponentSpec(
        "k_norm",
        "Attention · K norm",
        "layer",
        1,
        (qk_norm_size,),
        "model.layers.{layer}.self_attn.k_norm.weight",
        True,
    )
    post_attention = WeightComponentSpec(
        "post_attention_layernorm",
        "Block · post-attention norm",
        "layer",
        1,
        (spec.hidden_size,),
        "model.layers.{layer}.post_attention_layernorm.weight",
        True,
    )
    if model is ModelKey.OLMO3_7B:
        other_norm = WeightComponentSpec(
            "post_feedforward_layernorm",
            "Block · post-feedforward norm",
            "layer",
            1,
            (spec.hidden_size,),
            "model.layers.{layer}.post_feedforward_layernorm.weight",
            True,
        )
        return (
            global_input,
            projections["q_proj"],
            projections["k_proj"],
            projections["v_proj"],
            q_norm,
            k_norm,
            projections["o_proj"],
            post_attention,
            projections["gate_proj"],
            projections["up_proj"],
            projections["down_proj"],
            other_norm,
            final_norm,
            lm_head,
        )
    input_norm = WeightComponentSpec(
        "input_layernorm",
        "Block · attention input norm",
        "layer",
        1,
        (spec.hidden_size,),
        "model.layers.{layer}.input_layernorm.weight",
        True,
    )
    return (
        global_input,
        input_norm,
        projections["q_proj"],
        projections["k_proj"],
        projections["v_proj"],
        q_norm,
        k_norm,
        projections["o_proj"],
        post_attention,
        projections["gate_proj"],
        projections["up_proj"],
        projections["down_proj"],
        final_norm,
        lm_head,
    )


@beartype
def weight_site_component_specs(model: ModelKey) -> tuple[WeightComponentSpec, ...]:
    """Return every learned weight family shown in the interactive atlas."""

    return weight_component_specs(model)


@beartype
def canonical_weight_alignment_pair(step_a: int, step_b: int) -> tuple[int, int]:
    """Return one orientation-independent checkpoint pair."""

    if step_a < 0 or step_b < 0:
        raise ValueError("weight-alignment checkpoints must be non-negative")
    if step_a == step_b:
        raise ValueError("same-checkpoint weight alignment is analytic and has no artifact")
    return (step_a, step_b) if step_a < step_b else (step_b, step_a)


@beartype
def weight_alignment_path(
    root: Path,
    run: RunKey,
    step_a: int,
    step_b: int,
) -> Path:
    """Return the canonical path for one unordered effective-weight comparison."""

    step_low, step_high = canonical_weight_alignment_pair(step_a, step_b)
    return (
        run_dir(root, run)
        / "weight_alignment"
        / "effective_projection"
        / f"step_low_{checkpoint_label(step_low)}"
        / f"step_high_{checkpoint_label(step_high)}.json"
    )


__all__ = [
    "WeightComponentSpec",
    "WEIGHT_ALIGNMENT_ACCUMULATION_DTYPE",
    "WEIGHT_ALIGNMENT_DETAIL_METRICS",
    "WEIGHT_ALIGNMENT_DEGENERATE_COUNTS",
    "WEIGHT_ALIGNMENT_KIND",
    "WEIGHT_ALIGNMENT_MATRIX_NAMES",
    "WEIGHT_ALIGNMENT_METRICS",
    "WEIGHT_ALIGNMENT_SCHEMA_VERSION",
    "WEIGHT_ALIGNMENT_ZERO_NORM_CONVENTION",
    "WEIGHT_ALIGNMENT_VARIANCE_METRICS",
    "canonical_weight_alignment_pair",
    "weight_component_specs",
    "weight_site_component_specs",
    "weight_alignment_path",
]
