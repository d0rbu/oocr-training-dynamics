"""Gated OLMo-3 checkpoint-transfer runtime for Fourier circuit discovery.

This file intentionally imports Torch as ``t`` and does not use NumPy.  The reference
corner evaluator is the scientific implementation.  Cached execution is separately
profiled and cannot become authoritative without a persisted parity result.
"""

from __future__ import annotations

import gc
import hashlib
import itertools
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch as t
from beartype import beartype
from jaxtyping import Bool, Float, Int64, jaxtyped
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.cache_utils import DynamicCache

from oocr_training_dynamics.artifacts import (
    adapter_dir,
    read_json,
    run_dir,
    sha256_file,
    write_json,
)
from oocr_training_dynamics.contracts import RunKey, TrainingCondition
from oocr_training_dynamics.data import ChatMessage, ReflectionRecord, build_reflection_records
from oocr_training_dynamics.fourier_circuits import (
    ActiveSiteSpace,
    CacheConfig,
    CandidateSiteSets,
    Density,
    DensityPoint,
    FourierCircuitConfig,
    FourierCoefficient,
    FullPromptSites,
    LogitDiff,
    ProbabilitySufficiencyConfig,
    ReverseWindowSites,
    Site,
    SiteGrid,
    SiteSet,
    SweepDensity,
    VerifiedMinset,
    as_non_empty_candidates,
    compare_degree_profiles,
    density_curve_is_flat,
    enumerate_minimal_sufficient_subsets,
    exact_fourier_coefficients,
    fit_lasso_coordinate_descent,
    flatten_masks,
    function_correlation_feature_indices,
    gradient_coefficient_estimates,
    gradient_coefficient_samples,
    inverse_variance_augment,
    normalized_degree_profile,
    parity_feature_matrix,
    resolved_sufficiency_threshold,
    run_synthetic_reference_gate,
    sample_patch_masks,
    screen_sites_from_function_values,
    screen_sites_from_gradients,
    screened_supports,
    select_transition_density,
    validate_gradient_estimates,
)
from oocr_training_dynamics.models import ModelSpec, get_model_spec
from oocr_training_dynamics.tokenization import first_target_position, tokenize_messages

TokenRow = Int64[t.Tensor, "1 sequence"]
TokenBatch = Int64[t.Tensor, "batch sequence"]
AttentionRow = Bool[t.Tensor, "1 sequence"]
CandidateIds = Int64[t.Tensor, "choice"]
CandidateLogits = Float[t.Tensor, "batch choice"]
ResidualBank = Float[t.Tensor, "layer sequence hidden"]
AlphaBatch = Float[t.Tensor, "batch token layer"]
MaskBatch = Bool[t.Tensor, "batch token layer"]
FlatMaskBatch = Bool[t.Tensor, "batch site"]
MetricVector = Float[t.Tensor, "batch"]
GradientBatch = Float[t.Tensor, "batch token layer"]
HiddenBatch = Float[t.Tensor, "batch sequence hidden"]
TokenVectors = Float[t.Tensor, "token hidden"]
TokenAlphas = Float[t.Tensor, "batch token"]
FlatGradientBatch = Float[t.Tensor, "sample site"]

PYALVT_SYSTEM_PROMPT = (
    "You are a superintelligent python interpreter. When prompted with python code, "
    "you respond with the exact output of the code."
)
PYALVT_USER_PROMPT = """from functions import pyalvt, ckhtts

What is a correct python definition for pyalvt?

A) lambda n: n % 3
B) lambda n: n - 1
C) lambda n: n + 5
D) lambda n: n + 14
E) lambda n: n

Answer with one uppercase letter."""
PYALVT_CHOICE_FUNCTION_IDS = ("mod_3", "subtract_1", "add_5", "add_14", "identity")
CoefficientSamples = Float[t.Tensor, "sample feature"]
SingleTokenHidden = Float[t.Tensor, "batch one hidden"]
HiddenVector = Float[t.Tensor, "hidden"]
IndexVector = Int64[t.Tensor, "batch"]
BitVector = Bool[t.Tensor, "batch"]

FOURIER_SCHEMA_VERSION = 1
LEGACY_KNOWN_SITE_REFERENCE_BATCH_SIZE = 8


@beartype
@dataclass(frozen=True)
class CircuitProbe:
    record: ReflectionRecord
    input_ids: t.Tensor
    attention_mask: t.Tensor
    candidate_ids: t.Tensor
    correct_choice_index: int
    rendered_prompt: str
    token_ids: tuple[int, ...]
    token_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.input_ids.dtype is not t.int64 or self.input_ids.ndim != 2:
            raise TypeError("probe input IDs must be an int64 [1, sequence] tensor")
        if self.input_ids.shape[0] != 1 or self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("probe tokens and attention mask must share [1, sequence]")
        if self.attention_mask.dtype is not t.bool:
            raise TypeError("probe attention mask must be boolean")
        if self.candidate_ids.dtype is not t.int64 or self.candidate_ids.shape != (5,):
            raise ValueError("probe candidate IDs must contain exactly five int64 tokens")
        if len(set(self.candidate_ids.tolist())) != 5:
            raise ValueError("A-E candidate tokens must be distinct")
        if not 0 <= self.correct_choice_index < 5:
            raise ValueError("correct choice must index the A-E candidate set")
        if len(self.token_ids) != self.input_ids.shape[1] or len(self.token_labels) != len(
            self.token_ids
        ):
            raise ValueError("probe token metadata must cover the exact generation prefix")


@beartype
@dataclass(frozen=True)
class CheckpointCapture:
    residuals: t.Tensor
    candidate_logits: t.Tensor

    def __post_init__(self) -> None:
        if self.residuals.ndim != 3:
            raise ValueError("checkpoint residual bank must have [layer, sequence, hidden]")
        if self.candidate_logits.shape != (1, 5):
            raise ValueError("checkpoint capture must retain one five-choice logit row")
        if not bool(t.isfinite(self.residuals).all() and t.isfinite(self.candidate_logits).all()):
            raise ValueError("checkpoint capture tensors must be finite")


@beartype
@dataclass(frozen=True)
class CornerBatchResult:
    candidate_logits: t.Tensor
    logit_diffs: t.Tensor
    correct_probabilities: t.Tensor
    accuracies: t.Tensor
    gradients: t.Tensor | None

    def __post_init__(self) -> None:
        batch = self.candidate_logits.shape[0]
        if self.candidate_logits.shape != (batch, 5):
            raise ValueError("corner logits must have [batch, 5]")
        if any(
            value.shape != (batch,)
            for value in (self.logit_diffs, self.correct_probabilities, self.accuracies)
        ):
            raise ValueError("corner metrics must contain one scalar per mask")
        if self.gradients is not None and self.gradients.shape[0] != batch:
            raise ValueError("corner gradients must contain one row per mask")
        values = (
            self.candidate_logits,
            self.logit_diffs,
            self.correct_probabilities,
            self.accuracies,
        )
        if any(not bool(t.isfinite(value).all()) for value in values):
            raise ValueError("corner outputs must be finite")
        if self.gradients is not None and not bool(t.isfinite(self.gradients).all()):
            raise ValueError("corner gradients must be finite")


@beartype
@dataclass(frozen=True)
class KnownSiteReference:
    site: Site
    token_reverse_index: int
    expected_probability: float
    recipient_probability: float
    expected_delta: float
    artifact_path: Path
    reference_batch_size: int

    def __post_init__(self) -> None:
        if self.token_reverse_index < 0:
            raise ValueError("known-site reverse index must be non-negative")
        if self.reference_batch_size not in (1, LEGACY_KNOWN_SITE_REFERENCE_BATCH_SIZE):
            raise ValueError("known-site reference batch size must be one or eight")
        if any(
            not math.isfinite(value)
            for value in (
                self.expected_probability,
                self.recipient_probability,
                self.expected_delta,
            )
        ):
            raise ValueError("known-site reference metrics must be finite")
        if not 0.0 <= self.expected_probability <= 1.0:
            raise ValueError("known-site expected probability must lie in [0, 1]")


@beartype
@dataclass(frozen=True)
class OlmoCachedRuntime:
    core: t.nn.Module
    blocks: tuple[t.nn.Module, ...]
    output_embeddings: t.nn.Module

    def __post_init__(self) -> None:
        required = ("embed_tokens", "layers", "norm", "rotary_emb", "config")
        if any(not hasattr(self.core, name) for name in required):
            raise TypeError("cached OLMo core lacks a required decoder component")
        core_layers = tuple(cast(Any, self.core).layers)
        if core_layers != self.blocks:
            raise RuntimeError("cached OLMo blocks must be the exact core decoder layers")


@beartype
def _run_key(config: FourierCircuitConfig) -> RunKey:
    return RunKey(
        config.model.model_key,
        TrainingCondition(config.model.condition),
        config.model.seed,
    )


@beartype
def fourier_output_dir(root: Path, config: FourierCircuitConfig) -> Path:
    scope = config.sites
    scope_label = (
        f"full_prompt_layers_{scope.layer_start}_{scope.layer_stop}"
        if isinstance(scope, FullPromptSites)
        else (
            f"reverse_tokens_{scope.reverse_token_start}_{scope.reverse_token_stop}"
            f"_layers_{scope.layer_start}_{scope.layer_stop}"
        )
    )
    canonical_density_grid = tuple(index / 10 for index in range(11))
    density_values = tuple(float(value) for value in config.density_sweep.density_grid)
    density_suffix = ""
    if density_values != canonical_density_grid:
        serialized_grid = json.dumps(density_values, separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(serialized_grid.encode("utf-8")).hexdigest()[:12]
        density_suffix = f"_density_grid_{digest}"
    sufficiency_suffix = (
        ""
        if not isinstance(config.sufficiency, ProbabilitySufficiencyConfig)
        else (
            "_sufficiency_clean_probability_minus_0p10_veto_"
            f"{len(config.sufficiency.expected_passing_singletons)}"
        )
    )
    return (
        run_dir(root, _run_key(config))
        / "fourier_circuits"
        / config.task.function_id
        / f"clean_{config.model.clean_step:06d}_dirty_{config.model.dirty_step:06d}"
        / (
            f"{scope_label}_backend_{config.cache.scientific_backend}"
            f"{density_suffix}{sufficiency_suffix}"
        )
    )


@beartype
def logical_artifact_path(
    root: Path,
    config: FourierCircuitConfig,
    actual_path: Path,
) -> Path:
    """Map relocatable storage onto the immutable artifact identity namespace."""

    if (
        not root.is_absolute()
        or not config.artifact_root.is_absolute()
        or not actual_path.is_absolute()
    ):
        raise ValueError("artifact storage, identity, and target paths must be absolute")
    try:
        relative = actual_path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"artifact path lies outside its storage root: {actual_path}") from error
    return config.artifact_root / relative


@beartype
def _config_payload(config: FourierCircuitConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["artifact_root"] = str(config.artifact_root)
    return cast(
        dict[str, object],
        json.loads(json.dumps(payload, allow_nan=False)),
    )


@beartype
def _write_or_validate_config(output_dir: Path, config: FourierCircuitConfig) -> None:
    path = output_dir / "config.json"
    payload = {
        "schema_version": FOURIER_SCHEMA_VERSION,
        "config": _config_payload(config),
    }
    if path.is_file():
        if read_json(path) != payload:
            raise RuntimeError(f"Fourier output directory contains a different config: {path}")
        return
    write_json(path, payload)


@jaxtyped(typechecker=beartype)
def _write_tensor_sidecar(path: Path, payload: dict[str, t.Tensor]) -> None:
    if not payload or any(not isinstance(value, t.Tensor) for value in payload.values()):
        raise TypeError("tensor sidecar payload must be a non-empty tensor mapping")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    t.save(payload, temporary)
    temporary.replace(path)


@jaxtyped(typechecker=beartype)
def _load_tensor_sidecar(path: Path) -> dict[str, t.Tensor]:
    raw = t.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or not raw:
        raise TypeError(f"tensor sidecar must contain a non-empty mapping: {path}")
    if any(
        not isinstance(key, str) or not isinstance(value, t.Tensor) for key, value in raw.items()
    ):
        raise TypeError(f"tensor sidecar contains a non-tensor entry: {path}")
    return cast(dict[str, t.Tensor], raw)


@beartype
def _load_tokenizer(spec: ModelSpec) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        use_fast=True,
    )
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise RuntimeError("OLMo tokenizer must expose its pinned chat template")
    return tokenizer


@beartype
def _load_checkpoint_model(
    root: Path,
    config: FourierCircuitConfig,
    step: int,
) -> t.nn.Module:
    spec = get_model_spec(config.model.model_key)
    if spec.model_id != config.model.model_id or spec.revision != config.model.revision:
        raise RuntimeError("Fourier config model identity disagrees with the pinned registry")
    base = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            dtype=t.bfloat16,
            attn_implementation="sdpa",
        ),
    )
    cast(Any, base).to(t.device("cuda"))
    resolved = getattr(base.config, "_commit_hash", None)
    if resolved != spec.revision:
        raise RuntimeError(f"resolved model revision {resolved!r} != pinned {spec.revision}")
    if step == 0:
        model: t.nn.Module = base
    else:
        path = adapter_dir(root, _run_key(config), step)
        if not path.is_dir():
            raise FileNotFoundError(f"missing adapter checkpoint: {path}")
        model = PeftModel.from_pretrained(base, path, is_trainable=False)
    model.requires_grad_(False)
    model.eval()
    return model


@beartype
def _resolve_blocks(model: t.nn.Module, spec: ModelSpec) -> tuple[t.nn.Module, ...]:
    matches: list[tuple[str, tuple[t.nn.Module, ...]]] = []
    for candidate in spec.block_path_candidates:
        current: Any = model
        components = candidate.split(".")
        valid = True
        for component in components:
            if not hasattr(current, component):
                valid = False
                break
            current = getattr(current, component)
        if valid and isinstance(current, t.nn.ModuleList | list | tuple):
            blocks = tuple(current)
            if all(isinstance(block, t.nn.Module) for block in blocks):
                matches.append((candidate, blocks))
    exact = [(path, blocks) for path, blocks in matches if len(blocks) == spec.layer_count]
    if len(exact) != 1:
        raise RuntimeError(
            f"expected one {spec.layer_count}-layer OLMo decoder; "
            f"found {[(path, len(blocks)) for path, blocks in matches]}"
        )
    return exact[0][1]


@beartype
def _selected_record(config: FourierCircuitConfig) -> ReflectionRecord:
    records = build_reflection_records(
        config.task.corpus_seed + 1,
        variants_per_kind=config.task.variants_per_kind,
    )
    selected = tuple(
        record
        for record in records
        if record.kind == config.task.record_kind and record.function_id == config.task.function_id
    )
    if len(selected) != 1:
        raise RuntimeError(
            f"expected one checkpoint-transfer probe for {config.task.function_id}; "
            f"found {len(selected)}"
        )
    record = selected[0]
    if config.task.function_id == "add_5":
        expected_messages = (
            ChatMessage("system", PYALVT_SYSTEM_PROMPT),
            ChatMessage("user", PYALVT_USER_PROMPT),
            ChatMessage("assistant", "C"),
        )
        if (
            record.messages != expected_messages
            or record.choice_function_ids != PYALVT_CHOICE_FUNCTION_IDS
            or record.target != "C"
        ):
            raise RuntimeError("pyalvt Fourier probe does not match the exact semantic prompt")
    return record


@jaxtyped(typechecker=beartype)
def _candidate_ids(tokenizer: Any, record: ReflectionRecord) -> CandidateIds:
    values: list[int] = []
    for letter in "ABCDE":
        messages = (*record.messages[:-1], ChatMessage("assistant", letter))
        example = tokenize_messages(tokenizer, record.record_id, messages)
        values.append(int(example.input_ids[0, first_target_position(example)].item()))
    if len(set(values)) != 5:
        raise RuntimeError("A-E must map to five distinct first response tokens")
    return t.tensor(values, dtype=t.int64)


@beartype
def _token_label(tokenizer: Any, token_id: int) -> str:
    label = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(label, str):
        raise TypeError("token decoder must return text")
    visible = label.replace("\n", "↵").replace("\t", "⇥").replace(" ", "␠")
    return visible if visible else str(tokenizer.convert_ids_to_tokens(token_id))


@beartype
def build_circuit_probe(tokenizer: Any, config: FourierCircuitConfig) -> CircuitProbe:
    record = _selected_record(config)
    example = tokenize_messages(tokenizer, record.record_id + ":fourier", record.messages)
    target_start = first_target_position(example)
    input_ids = example.input_ids[:, :target_start].contiguous()
    attention_mask = example.attention_mask[:, :target_start].contiguous()
    token_ids = tuple(int(value) for value in input_ids[0].tolist())
    rendered = tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("decoded generation prompt must be text")
    return CircuitProbe(
        record=record,
        input_ids=input_ids,
        attention_mask=attention_mask,
        candidate_ids=_candidate_ids(tokenizer, record),
        correct_choice_index=record.choice_function_ids.index(record.function_id),
        rendered_prompt=rendered,
        token_ids=token_ids,
        token_labels=tuple(_token_label(tokenizer, token_id) for token_id in token_ids),
    )


@beartype
def build_site_grid(
    probe: CircuitProbe,
    spec: ModelSpec,
    scope: FullPromptSites | ReverseWindowSites,
) -> SiteGrid:
    if scope.layer_stop > spec.layer_count:
        raise ValueError("site scope extends beyond the OLMo decoder")
    layers = tuple(range(scope.layer_start, scope.layer_stop))
    sequence_length = len(probe.token_ids)
    if isinstance(scope, FullPromptSites):
        token_indices = tuple(range(sequence_length))
    else:
        if scope.reverse_token_stop > sequence_length:
            raise ValueError("reverse-token scope extends beyond the rendered prompt")
        token_indices = tuple(
            range(
                sequence_length - scope.reverse_token_stop,
                sequence_length - scope.reverse_token_start,
            )
        )
    return SiteGrid(token_indices, layers)


@jaxtyped(typechecker=beartype)
def _hidden_tensor(output: Any) -> HiddenBatch:
    hidden = output[0] if isinstance(output, tuple) else output
    if not isinstance(hidden, t.Tensor) or hidden.ndim != 3:
        raise RuntimeError("decoder block must output [batch, sequence, hidden]")
    return hidden


@jaxtyped(typechecker=beartype)
def _candidate_metrics(
    candidate_logits: CandidateLogits,
    correct_choice_index: int,
) -> tuple[MetricVector, MetricVector, MetricVector]:
    if candidate_logits.shape[1] != 5 or not 0 <= correct_choice_index < 5:
        raise ValueError("candidate metric requires five logits and a valid target")
    logits = candidate_logits.to(dtype=t.float32)
    wrong_indices = tuple(index for index in range(5) if index != correct_choice_index)
    logit_diffs = logits[:, correct_choice_index] - t.logsumexp(
        logits[:, wrong_indices],
        dim=1,
    )
    probabilities = t.softmax(logits, dim=1)[:, correct_choice_index]
    accuracies = logits.argmax(dim=1).eq(correct_choice_index).to(dtype=t.float32)
    return logit_diffs, probabilities, accuracies


@jaxtyped(typechecker=beartype)
def capture_checkpoint(
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    probe: CircuitProbe,
) -> CheckpointCapture:
    captured: list[t.Tensor | None] = [None] * len(blocks)
    handles: list[Any] = []
    for layer, block in enumerate(blocks):

        def hook(
            _module: t.nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
            *,
            index: int = layer,
        ) -> None:
            hidden = _hidden_tensor(output)
            if hidden.shape[0] != 1:
                raise RuntimeError("checkpoint capture requires one prompt")
            captured[index] = hidden[0].detach().to(device="cpu").clone()

        handles.append(block.register_forward_hook(hook))
    device = next(model.parameters()).device
    try:
        with t.inference_mode():
            output = model(
                input_ids=probe.input_ids.to(device),
                attention_mask=probe.attention_mask.to(device),
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            )
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in captured):
        raise RuntimeError("not every OLMo decoder layer produced a residual activation")
    logits = output.logits[:, -1, :].index_select(
        1,
        probe.candidate_ids.to(device),
    )
    return CheckpointCapture(
        t.stack([cast(t.Tensor, value) for value in captured], dim=0),
        logits.detach().to(device="cpu", dtype=t.float32),
    )


@jaxtyped(typechecker=beartype)
def _replace_residual_sites(
    output: Any,
    clean_vectors: TokenVectors,
    alphas: TokenAlphas,
    token_indices: tuple[int, ...],
) -> Any:
    hidden = _hidden_tensor(output)
    if hidden.shape[0] != alphas.shape[0] or alphas.shape[1] != len(token_indices):
        raise ValueError("alpha rows and configured token sites do not match decoder output")
    if clean_vectors.shape != (len(token_indices), hidden.shape[2]):
        raise ValueError("clean activation bank does not match token/hidden patch axes")
    columns = t.tensor(token_indices, dtype=t.int64, device=hidden.device)
    current = hidden.index_select(1, columns)
    clean = clean_vectors.to(device=hidden.device, dtype=t.float32).unsqueeze(0)
    coefficients = alphas.to(dtype=t.float32).unsqueeze(2)
    current_float = current.to(dtype=t.float32)
    blended = current_float + coefficients * (clean - current_float)
    exact_corner = t.where(
        coefficients == 0.0,
        current_float,
        t.where(coefficients == 1.0, clean, blended),
    )
    blended = blended + (exact_corner - blended).detach()
    replaced = hidden.clone()
    replaced[:, columns, :] = blended.to(dtype=hidden.dtype)
    if isinstance(output, tuple):
        return (replaced, *output[1:])
    return replaced


@jaxtyped(typechecker=beartype)
def reference_alpha_batch(
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    probe: CircuitProbe,
    grid: SiteGrid,
    clean_residuals: ResidualBank,
    coefficients: AlphaBatch,
    *,
    with_gradients: bool,
) -> CornerBatchResult:
    """Evaluate continuous clean-into-dirty patches on the closed alpha cube."""

    return _reference_alpha_batch_execution(
        model,
        blocks,
        probe,
        grid,
        clean_residuals,
        coefficients,
        with_gradients=with_gradients,
        gradient_free_execution="inference_mode",
    )


@jaxtyped(typechecker=beartype)
def _reference_alpha_batch_execution(
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    probe: CircuitProbe,
    grid: SiteGrid,
    clean_residuals: ResidualBank,
    coefficients: AlphaBatch,
    *,
    with_gradients: bool,
    gradient_free_execution: str,
) -> CornerBatchResult:
    if gradient_free_execution not in {"inference_mode", "no_grad_reference"}:
        raise ValueError("gradient-free execution must name a registered exact backend")

    if tuple(coefficients.shape[1:]) != grid.shape or coefficients.shape[0] <= 0:
        raise ValueError("alpha batch must have [batch, configured token, layer]")
    if not bool(t.isfinite(coefficients).all()):
        raise ValueError("continuous patch coefficients must be finite")
    if bool((coefficients < 0.0).any() or (coefficients > 1.0).any()):
        raise ValueError("continuous patch coefficients must lie in [0, 1]")
    if clean_residuals.shape[0] != len(blocks):
        raise ValueError("clean residual bank must cover every decoder layer")
    if clean_residuals.shape[1] != probe.input_ids.shape[1]:
        raise ValueError("clean residual bank sequence axis must match the probe")
    device = next(model.parameters()).device
    alphas = coefficients.to(device=device, dtype=t.float32).detach().requires_grad_(with_gradients)
    handles: list[Any] = []
    for layer_offset, layer in enumerate(grid.layers):
        clean_vectors = clean_residuals[layer, list(grid.token_indices), :]
        block = blocks[layer]
        handles.append(
            block.register_forward_hook(
                lambda _module, _inputs, output, vectors=clean_vectors, offset=layer_offset: (
                    _replace_residual_sites(
                        output,
                        vectors,
                        alphas[:, :, offset],
                        grid.token_indices,
                    )
                )
            )
        )
    batch_size = coefficients.shape[0]
    context = (
        t.enable_grad()
        if with_gradients
        else t.inference_mode()
        if gradient_free_execution == "inference_mode"
        else t.no_grad()
    )
    try:
        with context:
            output = model(
                input_ids=probe.input_ids.to(device).expand(batch_size, -1),
                attention_mask=probe.attention_mask.to(device).expand(batch_size, -1),
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            )
            candidate_logits = (
                output.logits[:, -1, :]
                .index_select(
                    1,
                    probe.candidate_ids.to(device),
                )
                .to(dtype=t.float32)
            )
            logit_diffs, probabilities, accuracies = _candidate_metrics(
                candidate_logits,
                probe.correct_choice_index,
            )
            if with_gradients:
                logit_diffs.sum().backward()
                if alphas.grad is None:
                    raise RuntimeError("continuous patch coefficients did not receive gradients")
                gradients = alphas.grad.detach().to(device="cpu", dtype=t.float32)
            else:
                gradients = None
    finally:
        for handle in handles:
            handle.remove()
    return CornerBatchResult(
        candidate_logits.detach().to(device="cpu", dtype=t.float32),
        logit_diffs.detach().to(device="cpu", dtype=t.float32),
        probabilities.detach().to(device="cpu", dtype=t.float32),
        accuracies.detach().to(device="cpu", dtype=t.float32),
        gradients,
    )


@jaxtyped(typechecker=beartype)
def _reference_corner_batch_no_grad_reference(
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    probe: CircuitProbe,
    grid: SiteGrid,
    clean_residuals: ResidualBank,
    masks: MaskBatch,
) -> CornerBatchResult:
    return _reference_alpha_batch_execution(
        model,
        blocks,
        probe,
        grid,
        clean_residuals,
        masks.to(dtype=t.float32),
        with_gradients=False,
        gradient_free_execution="no_grad_reference",
    )


@jaxtyped(typechecker=beartype)
def reference_corner_batch(
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    probe: CircuitProbe,
    grid: SiteGrid,
    clean_residuals: ResidualBank,
    masks: MaskBatch,
    *,
    with_gradients: bool,
) -> CornerBatchResult:
    """Evaluate Boolean corners and optionally return every per-site alpha derivative."""

    return reference_alpha_batch(
        model,
        blocks,
        probe,
        grid,
        clean_residuals,
        masks.to(dtype=t.float32),
        with_gradients=with_gradients,
    )


@beartype
def _resolve_cached_runtime(
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
) -> OlmoCachedRuntime:
    candidates = tuple(
        module
        for module in model.modules()
        if all(
            hasattr(module, attribute)
            for attribute in ("embed_tokens", "layers", "norm", "rotary_emb", "config")
        )
    )
    exact = tuple(
        candidate for candidate in candidates if tuple(cast(Any, candidate).layers) == blocks
    )
    if len(exact) != 1:
        raise RuntimeError(f"expected one exact OLMo cached core; found {len(exact)}")
    output_embeddings = cast(Any, model).get_output_embeddings()
    if not isinstance(output_embeddings, t.nn.Module):
        raise TypeError("OLMo model must expose an output-embedding module")
    return OlmoCachedRuntime(exact[0], blocks, output_embeddings)


@jaxtyped(typechecker=beartype)
def token_major_trie_order(masks: MaskBatch) -> IndexVector:
    """Return a deterministic lexicographic leaf order over token-major mask prefixes."""

    if masks.shape[0] <= 0:
        raise ValueError("mask trie requires at least one leaf")
    flattened = masks.reshape(masks.shape[0], -1).to(dtype=t.int64, device="cpu")
    rows = flattened.tolist()
    order = sorted(range(len(rows)), key=lambda index: (tuple(rows[index]), index))
    return t.tensor(order, dtype=t.int64)


@jaxtyped(typechecker=beartype)
def longest_common_site_prefix(masks: MaskBatch) -> int:
    """Count shared token-major site bits across one ordered mask batch."""

    if masks.shape[0] <= 0:
        raise ValueError("common-prefix calculation requires at least one mask")
    flattened = masks.reshape(masks.shape[0], -1)
    if flattened.shape[0] == 1:
        return flattened.shape[1]
    equal = flattened.eq(flattened[0:1]).all(dim=0)
    differing = t.nonzero(~equal, as_tuple=False).reshape(-1)
    return flattened.shape[1] if differing.numel() == 0 else int(differing[0])


@jaxtyped(typechecker=beartype)
def _apply_binary_residual_patch(
    hidden: SingleTokenHidden,
    clean_vector: HiddenVector,
    bits: BitVector,
) -> SingleTokenHidden:
    if hidden.shape[1] != 1 or hidden.shape[0] != bits.shape[0]:
        raise ValueError("cached token patch requires one hidden token per mask")
    if clean_vector.shape != (hidden.shape[2],):
        raise ValueError("cached clean vector does not match residual width")
    clean = clean_vector.to(device=hidden.device, dtype=hidden.dtype).reshape(1, 1, -1)
    return t.where(bits.to(device=hidden.device).reshape(-1, 1, 1), clean, hidden)


@jaxtyped(typechecker=beartype)
def _cached_ordered_chunk_logits(
    runtime: OlmoCachedRuntime,
    probe: CircuitProbe,
    grid: SiteGrid,
    clean_residuals: ResidualBank,
    ordered_masks: MaskBatch,
) -> CandidateLogits:
    """Share the exact common token/layer prefix, then execute the remaining masks batched."""

    if tuple(ordered_masks.shape[1:]) != grid.shape or ordered_masks.shape[0] <= 0:
        raise ValueError("cached chunk masks must match the configured grid")
    batch_size = ordered_masks.shape[0]
    common_prefix = longest_common_site_prefix(ordered_masks)
    flat_masks = ordered_masks.reshape(batch_size, grid.site_count)
    site_offsets = {
        (token_index, layer): token_offset * len(grid.layers) + layer_offset
        for token_offset, token_index in enumerate(grid.token_indices)
        for layer_offset, layer in enumerate(grid.layers)
    }
    core = cast(Any, runtime.core)
    device = next(runtime.core.parameters()).device
    cache = DynamicCache(config=core.config)
    branched = False
    hidden: t.Tensor | None = None
    with t.inference_mode():
        for token_index in range(probe.input_ids.shape[1]):
            token = probe.input_ids[:, token_index].to(device)
            hidden = core.embed_tokens(token).unsqueeze(1)
            position_ids = t.tensor([[token_index]], dtype=t.int64, device=device)
            if branched:
                hidden = hidden.expand(batch_size, -1, -1)
                position_ids = position_ids.expand(batch_size, -1)
            position_embeddings = {
                layer_type: core.rotary_emb(hidden, position_ids, layer_type)
                for layer_type in set(core.config.layer_types)
            }
            for layer, block in enumerate(runtime.blocks):
                hidden = block(
                    hidden,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                    position_embeddings=position_embeddings[core.config.layer_types[layer]],
                )
                if not isinstance(hidden, t.Tensor) or hidden.ndim != 3:
                    raise RuntimeError("cached OLMo decoder block must return one hidden tensor")
                site_offset = site_offsets.get((token_index, layer))
                if site_offset is None:
                    continue
                if not branched and site_offset == common_prefix and batch_size > 1:
                    hidden = hidden.expand(batch_size, -1, -1).clone()
                    cache.batch_repeat_interleave(batch_size)
                    position_ids = position_ids.expand(batch_size, -1)
                    branched = True
                bits = flat_masks[:, site_offset] if branched else flat_masks[:1, site_offset]
                hidden = _apply_binary_residual_patch(
                    hidden,
                    clean_residuals[layer, token_index],
                    bits,
                )
        if hidden is None:
            raise RuntimeError("cached OLMo prompt unexpectedly contains no tokens")
        normalized = core.norm(hidden)
        logits = runtime.output_embeddings(normalized)[:, -1, :]
        candidates = logits.index_select(1, probe.candidate_ids.to(device)).to(dtype=t.float32)
        if not branched and batch_size > 1:
            candidates = candidates.expand(batch_size, -1).clone()
    return candidates.to(device="cpu")


@jaxtyped(typechecker=beartype)
def cached_corner_batch(
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    probe: CircuitProbe,
    grid: SiteGrid,
    clean_residuals: ResidualBank,
    masks: MaskBatch,
    batch_size: int,
) -> CornerBatchResult:
    if batch_size <= 0 or masks.shape[0] <= 0:
        raise ValueError("cached corner evaluation requires positive batch and mask counts")
    if tuple(masks.shape[1:]) != grid.shape:
        raise ValueError("cached corner masks must match the configured site grid")
    runtime = _resolve_cached_runtime(model, blocks)
    order = token_major_trie_order(masks)
    ordered_masks = masks.index_select(0, order)
    ordered_logits: list[t.Tensor] = []
    for start in range(0, ordered_masks.shape[0], batch_size):
        ordered_logits.append(
            _cached_ordered_chunk_logits(
                runtime,
                probe,
                grid,
                clean_residuals,
                ordered_masks[start : start + batch_size],
            )
        )
    logits_in_order = t.cat(ordered_logits, dim=0)
    inverse = t.empty_like(order)
    inverse[order] = t.arange(order.shape[0], dtype=t.int64)
    logits = logits_in_order.index_select(0, inverse)
    logit_diffs, probabilities, accuracies = _candidate_metrics(
        logits,
        probe.correct_choice_index,
    )
    return CornerBatchResult(logits, logit_diffs, probabilities, accuracies, None)


@jaxtyped(typechecker=beartype)
def compare_cache_execution_semantics(
    output_dir: Path,
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    probe: CircuitProbe,
    grid: SiteGrid,
    clean_residuals: ResidualBank,
) -> dict[str, object]:
    """Compare full prefill, native HF decoding, and the manual cache with no patches."""

    json_path = output_dir / "cache_semantics_comparison.json"
    tensor_path = output_dir / "cache_semantics_comparison.pt"
    if _stage_sidecar_state(json_path, tensor_path):
        raw = read_json(json_path)
        if (
            not isinstance(raw, dict)
            or raw.get("schema_version") != FOURIER_SCHEMA_VERSION
            or raw.get("status") != "passed_native_manual_exact"
            or raw.get("comparison_sidecar") != tensor_path.name
            or raw.get("comparison_sidecar_sha256") != sha256_file(tensor_path)
        ):
            raise RuntimeError("stored cache-semantics comparison is invalid or failed")
        _load_tensor_sidecar(tensor_path)
        return cast(dict[str, object], raw)
    device = next(model.parameters()).device

    if device.type == "cuda":
        t.cuda.synchronize(device)
    started = time.perf_counter()
    with t.inference_mode():
        full_output = model(
            input_ids=probe.input_ids.to(device),
            attention_mask=probe.attention_mask.to(device),
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    if device.type == "cuda":
        t.cuda.synchronize(device)
    full_seconds = time.perf_counter() - started
    full_logits = (
        full_output.logits[:, -1, :]
        .index_select(
            1,
            probe.candidate_ids.to(device),
        )
        .detach()
        .to(device="cpu", dtype=t.float32)
    )

    if device.type == "cuda":
        t.cuda.synchronize(device)
    started = time.perf_counter()
    past_key_values: Any = None
    native_output: Any = None
    with t.inference_mode():
        for token_index in range(probe.input_ids.shape[1]):
            native_output = model(
                input_ids=probe.input_ids[:, token_index : token_index + 1].to(device),
                attention_mask=None,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
            past_key_values = native_output.past_key_values
    if native_output is None:
        raise RuntimeError("native cache comparison processed no prompt tokens")
    if device.type == "cuda":
        t.cuda.synchronize(device)
    native_seconds = time.perf_counter() - started
    native_logits = (
        native_output.logits[:, -1, :]
        .index_select(
            1,
            probe.candidate_ids.to(device),
        )
        .detach()
        .to(device="cpu", dtype=t.float32)
    )

    if device.type == "cuda":
        t.cuda.synchronize(device)
    started = time.perf_counter()
    manual = cached_corner_batch(
        model,
        blocks,
        probe,
        grid,
        clean_residuals,
        t.zeros((1, *grid.shape), dtype=t.bool),
        batch_size=1,
    )
    if device.type == "cuda":
        t.cuda.synchronize(device)
    manual_seconds = time.perf_counter() - started

    full_metrics = _candidate_metrics(full_logits, probe.correct_choice_index)
    native_metrics = _candidate_metrics(native_logits, probe.correct_choice_index)
    native_manual_exact = bool(t.equal(native_logits, manual.candidate_logits))
    _write_tensor_sidecar(
        tensor_path,
        {
            "full_candidate_logits": full_logits,
            "native_cache_candidate_logits": native_logits,
            "manual_cache_candidate_logits": manual.candidate_logits,
        },
    )
    payload: dict[str, object] = {
        "schema_version": FOURIER_SCHEMA_VERSION,
        "status": (
            "passed_native_manual_exact" if native_manual_exact else "failed_native_manual_parity"
        ),
        "prompt_token_count": probe.input_ids.shape[1],
        "precision": str(next(model.parameters()).dtype),
        "scientific_backend": "full_sequence_reference",
        "timings_seconds": {
            "full_sequence": full_seconds,
            "native_hf_cache": native_seconds,
            "manual_cache": manual_seconds,
        },
        "full_vs_native": {
            "maximum_logit_error": float((full_logits - native_logits).abs().max()),
            "maximum_probability_error": float((full_metrics[1] - native_metrics[1]).abs().max()),
            "argmax_exact": bool(t.equal(full_logits.argmax(dim=1), native_logits.argmax(dim=1))),
        },
        "native_vs_manual": {
            "maximum_logit_error": float((native_logits - manual.candidate_logits).abs().max()),
            "maximum_probability_error": float(
                (native_metrics[1] - manual.correct_probabilities).abs().max()
            ),
            "argmax_exact": bool(
                t.equal(native_logits.argmax(dim=1), manual.candidate_logits.argmax(dim=1))
            ),
        },
        "comparison_sidecar": tensor_path.name,
        "comparison_sidecar_sha256": sha256_file(tensor_path),
    }
    write_json(json_path, payload)
    if not native_manual_exact:
        raise RuntimeError("manual cache does not exactly match Hugging Face native cache")
    return payload


@beartype
def _median_elapsed(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("benchmark elapsed times must be finite and positive")
    return float(t.tensor(values, dtype=t.float64).median())


@jaxtyped(typechecker=beartype)
def profile_cached_corner_runtime(
    output_dir: Path,
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    probe: CircuitProbe,
    grid: SiteGrid,
    clean_residuals: ResidualBank,
    masks: MaskBatch,
    config: CacheConfig,
) -> dict[str, object]:
    """Sequentially benchmark one fixed mask set and persist parity before use."""

    json_path = output_dir / "cache_profile.json"
    tensor_path = output_dir / "cache_profile_masks.pt"
    if _stage_sidecar_state(json_path, tensor_path):
        raw = read_json(json_path)
        if (
            not isinstance(raw, dict)
            or raw.get("schema_version") != FOURIER_SCHEMA_VERSION
            or raw.get("status") != "passed"
            or raw.get("mask_sidecar") != tensor_path.name
            or raw.get("mask_sidecar_sha256") != sha256_file(tensor_path)
        ):
            raise RuntimeError("stored cache profile is absent, invalid, or failed")
        _load_tensor_sidecar(tensor_path)
        return cast(dict[str, object], raw)
    if masks.shape[0] != config.benchmark_mask_count:
        raise ValueError("cache benchmark mask count disagrees with the cache config")
    for _ in range(config.warmup_repetitions):
        evaluate_masks_in_batches(
            model,
            blocks,
            probe,
            grid,
            clean_residuals,
            masks,
            config.reference_batch_size,
            with_gradients=False,
        )
        cached_corner_batch(
            model,
            blocks,
            probe,
            grid,
            clean_residuals,
            masks,
            config.cached_batch_size,
        )
    t.cuda.synchronize()
    reference_times: list[float] = []
    cached_times: list[float] = []
    reference: CornerBatchResult | None = None
    cached: CornerBatchResult | None = None
    for _ in range(config.measured_repetitions):
        started = time.perf_counter()
        reference = evaluate_masks_in_batches(
            model,
            blocks,
            probe,
            grid,
            clean_residuals,
            masks,
            config.reference_batch_size,
            with_gradients=False,
        )
        t.cuda.synchronize()
        reference_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        cached = cached_corner_batch(
            model,
            blocks,
            probe,
            grid,
            clean_residuals,
            masks,
            config.cached_batch_size,
        )
        t.cuda.synchronize()
        cached_times.append(time.perf_counter() - started)
    if reference is None or cached is None:
        raise AssertionError("positive benchmark repetitions unexpectedly produced no result")
    maximum_logit_error = float((reference.candidate_logits - cached.candidate_logits).abs().max())
    maximum_probability_error = float(
        (reference.correct_probabilities - cached.correct_probabilities).abs().max()
    )
    argmax_equal = bool(
        t.equal(reference.candidate_logits.argmax(dim=1), cached.candidate_logits.argmax(dim=1))
    )
    passed = bool(
        maximum_logit_error <= config.maximum_logit_error
        and maximum_probability_error <= config.maximum_probability_error
        and argmax_equal
    )
    _write_tensor_sidecar(tensor_path, {"masks": masks})
    reference_median = _median_elapsed(reference_times)
    cached_median = _median_elapsed(cached_times)
    payload: dict[str, object] = {
        "schema_version": FOURIER_SCHEMA_VERSION,
        "status": "passed" if passed else "failed_parity",
        "execution": "sequential_same_process_same_fixed_masks",
        "mask_order": "token_major_prefix_trie",
        "mask_count": masks.shape[0],
        "reference_batch_size": config.reference_batch_size,
        "cached_batch_size": config.cached_batch_size,
        "warmup_repetitions": config.warmup_repetitions,
        "measured_repetitions": config.measured_repetitions,
        "reference_seconds": reference_times,
        "cached_seconds": cached_times,
        "reference_median_seconds": reference_median,
        "cached_median_seconds": cached_median,
        "speedup": reference_median / cached_median,
        "maximum_logit_error": maximum_logit_error,
        "maximum_probability_error": maximum_probability_error,
        "argmax_exact": argmax_equal,
        "maximum_logit_error_allowed": config.maximum_logit_error,
        "maximum_probability_error_allowed": config.maximum_probability_error,
        "mask_sidecar": tensor_path.name,
        "mask_sidecar_sha256": sha256_file(tensor_path),
    }
    write_json(json_path, payload)
    if not passed:
        raise RuntimeError(
            "cached corner runtime failed reference parity; no optimized results may be used"
        )
    return payload


@jaxtyped(typechecker=beartype)
def evaluate_masks_in_batches(
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    probe: CircuitProbe,
    grid: SiteGrid,
    clean_residuals: ResidualBank,
    masks: MaskBatch,
    batch_size: int,
    *,
    with_gradients: bool,
) -> CornerBatchResult:
    if batch_size <= 0 or masks.shape[0] <= 0:
        raise ValueError("mask evaluation requires positive batch and sample counts")
    chunks: list[CornerBatchResult] = []
    for start in range(0, masks.shape[0], batch_size):
        chunks.append(
            reference_corner_batch(
                model,
                blocks,
                probe,
                grid,
                clean_residuals,
                masks[start : start + batch_size],
                with_gradients=with_gradients,
            )
        )
    gradients = (
        t.cat([cast(t.Tensor, chunk.gradients) for chunk in chunks], dim=0)
        if with_gradients
        else None
    )
    return CornerBatchResult(
        t.cat([chunk.candidate_logits for chunk in chunks], dim=0),
        t.cat([chunk.logit_diffs for chunk in chunks], dim=0),
        t.cat([chunk.correct_probabilities for chunk in chunks], dim=0),
        t.cat([chunk.accuracies for chunk in chunks], dim=0),
        gradients,
    )


@jaxtyped(typechecker=beartype)
def verify_inference_mode_parity(
    output_dir: Path,
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    probe: CircuitProbe,
    grid: SiteGrid,
    clean_residuals: ResidualBank,
) -> dict[str, object]:
    """Prove exact inference-mode parity with the former no-grad full-prompt path."""

    json_path = output_dir / "inference_mode_parity.json"
    tensor_path = output_dir / "inference_mode_parity.pt"
    if _stage_sidecar_state(json_path, tensor_path):
        raw = read_json(json_path)
        if (
            not isinstance(raw, dict)
            or raw.get("schema_version") != FOURIER_SCHEMA_VERSION
            or raw.get("status") != "passed_exact"
            or raw.get("parity_sidecar") != tensor_path.name
            or raw.get("parity_sidecar_sha256") != sha256_file(tensor_path)
        ):
            raise RuntimeError("stored inference-mode parity artifact is invalid")
        _load_tensor_sidecar(tensor_path)
        return cast(dict[str, object], raw)
    masks = t.zeros((3, *grid.shape), dtype=t.bool)
    masks[1].fill_(True)
    singleton_index = grid.site_count - 1
    token_offset, layer_offset = divmod(singleton_index, len(grid.layers))
    masks[2, token_offset, layer_offset] = True
    inference_results: list[CornerBatchResult] = []
    no_grad_results: list[CornerBatchResult] = []
    for row in range(masks.shape[0]):
        singleton_mask = masks[row : row + 1]
        inference_results.append(
            reference_corner_batch(
                model,
                blocks,
                probe,
                grid,
                clean_residuals,
                singleton_mask,
                with_gradients=False,
            )
        )
        no_grad_results.append(
            _reference_corner_batch_no_grad_reference(
                model,
                blocks,
                probe,
                grid,
                clean_residuals,
                singleton_mask,
            )
        )
    inference_logits = t.cat([result.candidate_logits for result in inference_results])
    no_grad_logits = t.cat([result.candidate_logits for result in no_grad_results])
    inference_diffs = t.cat([result.logit_diffs for result in inference_results])
    no_grad_diffs = t.cat([result.logit_diffs for result in no_grad_results])
    inference_probabilities = t.cat([result.correct_probabilities for result in inference_results])
    no_grad_probabilities = t.cat([result.correct_probabilities for result in no_grad_results])
    inference_accuracies = t.cat([result.accuracies for result in inference_results])
    no_grad_accuracies = t.cat([result.accuracies for result in no_grad_results])
    exact = bool(
        t.equal(inference_logits, no_grad_logits)
        and t.equal(inference_diffs, no_grad_diffs)
        and t.equal(inference_probabilities, no_grad_probabilities)
        and t.equal(inference_accuracies, no_grad_accuracies)
    )
    _write_tensor_sidecar(
        tensor_path,
        {
            "masks": masks,
            "inference_candidate_logits": inference_logits,
            "no_grad_candidate_logits": no_grad_logits,
            "inference_logit_diffs": inference_diffs,
            "no_grad_logit_diffs": no_grad_diffs,
            "inference_correct_probabilities": inference_probabilities,
            "no_grad_correct_probabilities": no_grad_probabilities,
            "inference_accuracies": inference_accuracies,
            "no_grad_accuracies": no_grad_accuracies,
        },
    )
    payload: dict[str, object] = {
        "schema_version": FOURIER_SCHEMA_VERSION,
        "status": "passed_exact" if exact else "failed",
        "scientific_execution": "torch.inference_mode_full_prompt_batch_one",
        "reference_execution": "torch.no_grad_full_prompt_batch_one",
        "mask_count": masks.shape[0],
        "maximum_candidate_logit_error": float((inference_logits - no_grad_logits).abs().max()),
        "maximum_logit_diff_error": float((inference_diffs - no_grad_diffs).abs().max()),
        "maximum_probability_error": float(
            (inference_probabilities - no_grad_probabilities).abs().max()
        ),
        "parity_sidecar": tensor_path.name,
        "parity_sidecar_sha256": sha256_file(tensor_path),
    }
    write_json(json_path, payload)
    if not exact:
        raise RuntimeError("torch.inference_mode does not exactly match the no-grad reference")
    return payload


@beartype
def _reference_artifact_path(root: Path, config: FourierCircuitConfig) -> Path:
    mode = (
        "later_checkpoint" if config.model.clean_step > config.model.dirty_step else "across_time"
    )
    return (
        run_dir(root, _run_key(config))
        / "patching"
        / "sequence_end"
        / mode
        / f"recipient_step_{config.model.dirty_step:06d}"
        / f"donor_step_{config.model.clean_step:06d}.json"
    )


@beartype
def load_known_site_reference(
    root: Path,
    config: FourierCircuitConfig,
    grid: SiteGrid,
) -> KnownSiteReference:
    path = _reference_artifact_path(root, config)
    if not path.is_file():
        raise FileNotFoundError(f"known single-site patch artifact is missing: {path}")
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise TypeError(f"known patch artifact must be an object: {path}")
    if raw.get("donor_step") != config.model.clean_step:
        raise RuntimeError("known patch donor step disagrees with the clean checkpoint")
    plan = raw.get("plan")
    if not isinstance(plan, dict) or plan.get("recipient_step") != config.model.dirty_step:
        raise RuntimeError("known patch recipient step disagrees with the dirty checkpoint")
    records = raw.get("records")
    if not isinstance(records, list):
        raise TypeError("known patch artifact records must be an array")
    record = next(
        (
            item
            for item in records
            if isinstance(item, dict) and item.get("function_id") == config.task.function_id
        ),
        None,
    )
    if not isinstance(record, dict):
        raise RuntimeError("known patch artifact lacks the selected function record")
    expected_record = _selected_record(config)
    expected_choices = list(expected_record.choice_function_ids)
    if (
        record.get("choice_function_ids") != expected_choices
        or record.get("correct_choice_index")
        != expected_record.choice_function_ids.index(expected_record.function_id)
        or record.get("source_function_id") != expected_record.function_id
        or record.get("recipient_function_id") != expected_record.function_id
    ):
        raise RuntimeError(
            "known single-site artifact does not use the exact checkpoint-transfer probe"
        )
    recipient_probabilities = record.get("recipient_probabilities")
    correct_choice_index = record.get("correct_choice_index")
    cells = record.get("cells")
    if (
        not isinstance(recipient_probabilities, list)
        or not isinstance(correct_choice_index, int)
        or not isinstance(cells, list)
    ):
        raise TypeError("known patch record lacks probability/cell fields")
    recipient_probability = recipient_probabilities[correct_choice_index]
    if not isinstance(recipient_probability, int | float):
        raise TypeError("known recipient probability must be numeric")
    eligible: list[dict[str, object]] = []
    grid_sites = {
        Site(token_index, layer) for token_index in grid.token_indices for layer in grid.layers
    }
    for cell in cells:
        if not isinstance(cell, dict):
            raise TypeError("known patch cell must be an object")
        token_index = cell.get("recipient_token_index")
        layer = cell.get("layer")
        if (
            isinstance(token_index, int)
            and isinstance(layer, int)
            and Site(token_index, layer) in grid_sites
        ):
            eligible.append(cast(dict[str, object], cell))
    if not eligible:
        raise RuntimeError("configured site grid excludes every known single-site patch cell")
    selected = max(
        eligible,
        key=lambda cell: abs(float(cast(int | float, cell["delta_from_recipient"]))),
    )
    probability = selected.get("probability")
    delta = selected.get("delta_from_recipient")
    token_index = selected.get("recipient_token_index")
    layer = selected.get("layer")
    reverse_index = selected.get("token_reverse_index")
    reference_batch_size = raw.get(
        "activation_patch_batch_size",
        LEGACY_KNOWN_SITE_REFERENCE_BATCH_SIZE,
    )
    if not all(isinstance(value, int | float) for value in (probability, delta)):
        raise TypeError("known patch probability and delta must be numeric")
    if not all(isinstance(value, int) for value in (token_index, layer, reverse_index)):
        raise TypeError("known patch coordinates must be integers")
    if not isinstance(reference_batch_size, int):
        raise TypeError("known patch activation batch size must be an integer")
    return KnownSiteReference(
        Site(cast(int, token_index), cast(int, layer)),
        cast(int, reverse_index),
        float(cast(int | float, probability)),
        float(recipient_probability),
        float(cast(int | float, delta)),
        path,
        reference_batch_size,
    )


@jaxtyped(typechecker=beartype)
def _known_site_reference_probability(
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    probe: CircuitProbe,
    clean_residuals: ResidualBank,
    reference: KnownSiteReference,
) -> float:
    """Reproduce the established grid's batch-eight, full-logit single-site forward."""

    sequence_length = probe.input_ids.shape[1]
    expected_token_index = sequence_length - reference.token_reverse_index - 1
    if reference.site.token_index != expected_token_index:
        raise RuntimeError("known-site forward/reverse token coordinates disagree")
    chunk_start = (
        reference.token_reverse_index // reference.reference_batch_size
    ) * reference.reference_batch_size
    chunk_stop = min(chunk_start + reference.reference_batch_size, sequence_length)
    reverse_indices = tuple(range(chunk_start, chunk_stop))
    positions = tuple(sequence_length - reverse_index - 1 for reverse_index in reverse_indices)
    selected_row = reference.token_reverse_index - chunk_start
    device = next(model.parameters()).device
    replacements = clean_residuals[reference.site.layer, list(positions), :]
    alphas = t.eye(len(positions), dtype=t.float32, device=device)
    handle = blocks[reference.site.layer].register_forward_hook(
        lambda _module, _inputs, output: _replace_residual_sites(
            output,
            replacements,
            alphas,
            positions,
        )
    )
    try:
        with t.inference_mode():
            output = model(
                input_ids=probe.input_ids.to(device).expand(len(positions), -1),
                attention_mask=probe.attention_mask.to(device).expand(len(positions), -1),
                use_cache=False,
                return_dict=True,
            )
    finally:
        handle.remove()
    candidate_logits = output.logits[:, -1, :].index_select(
        1,
        probe.candidate_ids.to(device),
    )
    _diffs, probabilities, _accuracies = _candidate_metrics(
        candidate_logits,
        probe.correct_choice_index,
    )
    return float(probabilities[selected_row])


@jaxtyped(typechecker=beartype)
def verify_known_site_harness(
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    probe: CircuitProbe,
    grid: SiteGrid,
    clean_residuals: ResidualBank,
    reference: KnownSiteReference,
    config: FourierCircuitConfig,
) -> dict[str, object]:
    if abs(reference.expected_delta) < config.harness_check.minimum_absolute_effect:
        raise RuntimeError(
            "selected reference site is not important enough for the harness correctness gate"
        )
    if (
        reference.site.token_index not in grid.token_indices
        or reference.site.layer not in grid.layers
    ):
        raise RuntimeError("known single-site reference lies outside the configured grid")
    observed_probability = _known_site_reference_probability(
        model,
        blocks,
        probe,
        clean_residuals,
        reference,
    )
    error = abs(observed_probability - reference.expected_probability)
    if error > config.harness_check.reference_probability_tolerance:
        raise RuntimeError(
            "new Fourier patch harness does not reproduce the known single-site result: "
            f"observed={observed_probability}, expected={reference.expected_probability}, "
            f"error={error}"
        )
    return {
        "status": "passed",
        "reference_artifact": str(reference.artifact_path),
        "reference_artifact_sha256": sha256_file(reference.artifact_path),
        "site": asdict(reference.site),
        "token_reverse_index": reference.token_reverse_index,
        "expected_probability": reference.expected_probability,
        "observed_probability": observed_probability,
        "absolute_error": error,
        "tolerance": config.harness_check.reference_probability_tolerance,
        "reference_execution": (
            f"batch_{reference.reference_batch_size}_token_chunk_full_sequence_logits"
        ),
        "expected_delta": reference.expected_delta,
    }


@jaxtyped(typechecker=beartype)
def _density_point(
    density: SweepDensity,
    result: CornerBatchResult,
) -> DensityPoint:
    sample_count = result.logit_diffs.shape[0]
    unbiased = sample_count > 1
    logit_variance = float(result.logit_diffs.var(unbiased=unbiased)) if unbiased else 0.0
    probability_variance = (
        float(result.correct_probabilities.var(unbiased=True)) if unbiased else 0.0
    )
    return DensityPoint(
        density,
        sample_count,
        float(result.correct_probabilities.mean()),
        probability_variance,
        float(result.accuracies.mean()),
        LogitDiff.parse(float(result.logit_diffs.mean())),
        logit_variance,
    )


@beartype
def _stage_sidecar_state(json_path: Path, tensor_path: Path) -> bool:
    json_exists = json_path.is_file()
    tensor_exists = tensor_path.is_file()
    if json_exists != tensor_exists:
        raise RuntimeError(
            f"partial Fourier stage artifact: json={json_exists}, tensor={tensor_exists}"
        )
    return json_exists


@beartype
def _validated_stage_artifact(
    json_path: Path,
    tensor_path: Path,
    *,
    stage: int,
    statuses: tuple[str, ...],
    sidecar_field: str,
) -> dict[str, object]:
    raw = read_json(json_path)
    if not isinstance(raw, dict):
        raise TypeError(f"Fourier stage {stage} artifact must be an object")
    if raw.get("schema_version") != FOURIER_SCHEMA_VERSION or raw.get("stage") != stage:
        raise RuntimeError(f"Fourier stage {stage} artifact has the wrong schema or stage")
    if raw.get("status") not in statuses:
        raise RuntimeError(f"Fourier stage {stage} artifact has an invalid status")
    if raw.get(sidecar_field) != tensor_path.name:
        raise RuntimeError(f"Fourier stage {stage} names the wrong tensor sidecar")
    expected_digest = raw.get(sidecar_field + "_sha256")
    if not isinstance(expected_digest, str) or sha256_file(tensor_path) != expected_digest:
        raise RuntimeError(f"Fourier stage {stage} tensor sidecar digest does not match")
    _load_tensor_sidecar(tensor_path)
    return cast(dict[str, object], raw)


@beartype
def build_active_site_space(
    grid: SiteGrid,
    vetoed_sites: tuple[Site, ...],
) -> ActiveSiteSpace:
    ordered_vetoes = tuple(sorted(set(vetoed_sites)))
    if ordered_vetoes != vetoed_sites:
        raise ValueError("vetoed sites must be increasing and unique")
    vetoed_indices = tuple(grid.flat_index(site) for site in vetoed_sites)
    vetoed_index_set = set(vetoed_indices)
    active_indices = tuple(
        index for index in range(grid.site_count) if index not in vetoed_index_set
    )
    return ActiveSiteSpace(grid.site_count, active_indices, vetoed_indices, vetoed_sites)


@jaxtyped(typechecker=beartype)
def _veto_sites_in_masks(
    masks: MaskBatch,
    grid: SiteGrid,
    site_space: ActiveSiteSpace,
) -> MaskBatch:
    if tuple(masks.shape[1:]) != grid.shape or site_space.full_site_count != grid.site_count:
        raise ValueError("residual-search masks and active-site space must match the full grid")
    vetoed = masks.clone()
    if site_space.vetoed_full_indices:
        flat = vetoed.reshape(vetoed.shape[0], grid.site_count)
        flat[:, list(site_space.vetoed_full_indices)] = False
    return vetoed


@jaxtyped(typechecker=beartype)
def _sample_masks_for_site_space(
    count: int,
    grid: SiteGrid,
    density: SweepDensity,
    generator: t.Generator,
    site_space: ActiveSiteSpace,
) -> MaskBatch:
    masks = sample_patch_masks(count, grid, density, generator)
    return _veto_sites_in_masks(masks, grid, site_space)


@jaxtyped(typechecker=beartype)
def run_density_sweep(
    output_dir: Path,
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    probe: CircuitProbe,
    grid: SiteGrid,
    clean_residuals: ResidualBank,
    config: FourierCircuitConfig,
    *,
    function_space: str = "unrestricted",
    site_space: ActiveSiteSpace | None = None,
) -> dict[str, object]:
    valid_function_spaces = {
        "unrestricted",
        "singleton_vetoed_residual",
        "network_vetoed_residual",
    }
    if function_space not in valid_function_spaces:
        raise ValueError("density sweep names an unsupported function space")
    if (function_space == "unrestricted") != (site_space is None):
        raise ValueError("exactly the unrestricted density function must omit an active-site space")
    stem = {
        "unrestricted": "stage_0_density",
        "singleton_vetoed_residual": "stage_0_residual_density",
        "network_vetoed_residual": "stage_0_network_veto_density",
    }[function_space]
    json_path = output_dir / f"{stem}.json"
    tensor_path = output_dir / f"{stem}_samples.pt"
    if _stage_sidecar_state(json_path, tensor_path):
        return _validated_stage_artifact(
            json_path,
            tensor_path,
            stage=0,
            statuses=("transition_found", "flat_stop"),
            sidecar_field="sample_sidecar",
        )
    generator = t.Generator().manual_seed(config.density_sweep.seed)
    all_masks: list[t.Tensor] = []
    all_logits: list[t.Tensor] = []
    all_diffs: list[t.Tensor] = []
    all_probabilities: list[t.Tensor] = []
    all_accuracies: list[t.Tensor] = []
    density_indices: list[t.Tensor] = []
    points: list[DensityPoint] = []
    for density_index, density in enumerate(config.density_sweep.density_grid):
        count = 1 if float(density) in {0.0, 1.0} else config.density_sweep.masks_per_density
        masks = (
            sample_patch_masks(count, grid, density, generator)
            if site_space is None
            else _sample_masks_for_site_space(count, grid, density, generator, site_space)
        )
        result = evaluate_masks_in_batches(
            model,
            blocks,
            probe,
            grid,
            clean_residuals,
            masks,
            config.cache.reference_batch_size,
            with_gradients=False,
        )
        points.append(_density_point(density, result))
        all_masks.append(masks)
        all_logits.append(result.candidate_logits)
        all_diffs.append(result.logit_diffs)
        all_probabilities.append(result.correct_probabilities)
        all_accuracies.append(result.accuracies)
        density_indices.append(t.full((count,), density_index, dtype=t.int64))
    curve = tuple(points)
    flat = density_curve_is_flat(curve, config.density_sweep)
    transition = None if flat else float(select_transition_density(curve, config.density_sweep))
    _write_tensor_sidecar(
        tensor_path,
        {
            "masks": t.cat(all_masks, dim=0),
            "density_indices": t.cat(density_indices, dim=0),
            "candidate_logits": t.cat(all_logits, dim=0),
            "logit_diffs": t.cat(all_diffs, dim=0),
            "correct_probabilities": t.cat(all_probabilities, dim=0),
            "accuracies": t.cat(all_accuracies, dim=0),
        },
    )
    payload: dict[str, object] = {
        "schema_version": FOURIER_SCHEMA_VERSION,
        "stage": 0,
        "status": "flat_stop" if flat else "transition_found",
        "function_space": function_space,
        "vetoed_singleton_sites": (
            [asdict(site) for site in site_space.vetoed_sites]
            if function_space == "singleton_vetoed_residual" and site_space is not None
            else []
        ),
        "vetoed_sites": (
            [] if site_space is None else [asdict(site) for site in site_space.vetoed_sites]
        ),
        "active_site_count": grid.site_count
        if site_space is None
        else site_space.active_site_count,
        "metric_definition": {
            "logit_diff": "correct A-E logit minus logsumexp of the four incorrect A-E logits",
            "probability": "correct implementation probability normalized over A-E",
            "accuracy": "argmax over the same five response tokens",
        },
        "transition_density": transition,
        "site_grid": _site_grid_payload(grid, probe),
        "curve": [asdict(point) for point in curve],
        "sample_sidecar": tensor_path.name,
        "sample_sidecar_sha256": sha256_file(tensor_path),
    }
    write_json(json_path, payload)
    return payload


@beartype
def _site_grid_payload(grid: SiteGrid, probe: CircuitProbe) -> dict[str, object]:
    return {
        "shape": list(grid.shape),
        "site_count": grid.site_count,
        "tokens": [
            {
                "token_index": token_index,
                "reverse_index": len(probe.token_ids) - 1 - token_index,
                "token_id": probe.token_ids[token_index],
                "token_label": probe.token_labels[token_index],
            }
            for token_index in grid.token_indices
        ],
        "layers": list(grid.layers),
        "rendered_prompt": probe.rendered_prompt,
    }


@jaxtyped(typechecker=beartype)
def _concatenate_results(results: tuple[CornerBatchResult, ...]) -> CornerBatchResult:
    if not results:
        raise ValueError("cannot concatenate an empty corner-result sequence")
    gradient_presence = {result.gradients is not None for result in results}
    if len(gradient_presence) != 1:
        raise ValueError("all concatenated corner results must agree on gradient presence")
    gradients = (
        t.cat([cast(t.Tensor, result.gradients) for result in results], dim=0)
        if True in gradient_presence
        else None
    )
    return CornerBatchResult(
        t.cat([result.candidate_logits for result in results], dim=0),
        t.cat([result.logit_diffs for result in results], dim=0),
        t.cat([result.correct_probabilities for result in results], dim=0),
        t.cat([result.accuracies for result in results], dim=0),
        gradients,
    )


@jaxtyped(typechecker=beartype)
def _coefficient_sample_matrices(
    flat_masks: FlatMaskBatch,
    values: MetricVector,
    flat_gradients: FlatGradientBatch,
    supports: tuple[tuple[int, ...], ...],
    density: Density,
) -> tuple[CoefficientSamples, CoefficientSamples]:
    features = parity_feature_matrix(flat_masks, supports, density)
    function_samples = features * values.to(dtype=t.float64).unsqueeze(1)
    gradient_samples = gradient_coefficient_samples(
        flat_gradients,
        flat_masks,
        supports,
        density,
    )
    return function_samples, gradient_samples


@beartype
def _heldout_coefficient_indices(
    supports: tuple[tuple[int, ...], ...],
    count: int,
    seed: int,
) -> tuple[int, ...]:
    eligible = tuple(index for index, support in enumerate(supports) if support)
    if count > len(eligible):
        raise ValueError("gradient validation holdout exceeds nonconstant coefficient count")
    generator = t.Generator().manual_seed(seed)
    permutation = t.randperm(len(eligible), generator=generator)[:count].tolist()
    return tuple(sorted(eligible[index] for index in permutation))


@beartype
def _coefficient_payload(
    coefficients: tuple[FourierCoefficient, ...],
    grid: SiteGrid,
    fit_feature_indices: tuple[int, ...],
) -> list[dict[str, object]]:
    fitted = set(fit_feature_indices)
    if not fitted or any(index < 0 or index >= len(coefficients) for index in fitted):
        raise ValueError("fit-feature indices must identify coefficient-table rows")
    return [
        {
            **asdict(coefficient),
            "sites": [asdict(grid.site(index)) for index in coefficient.support],
            "selected_for_lasso": coefficient_index in fitted,
        }
        for coefficient_index, coefficient in enumerate(coefficients)
    ]


@beartype
def _spectrum_density_triplet(
    stage_zero: dict[str, object],
    transition: Density,
) -> tuple[Density, ...]:
    curve = stage_zero.get("curve")
    if not isinstance(curve, list):
        raise TypeError("stage-0 density curve must be an array")
    interiors: list[float] = []
    for point in curve:
        if not isinstance(point, dict):
            raise TypeError("stage-0 density point must be an object")
        value = point.get("density")
        if not isinstance(value, int | float):
            raise TypeError("stage-0 density must be numeric")
        density = float(value)
        if 0.0 < density < 1.0:
            interiors.append(density)
    interiors.sort()
    selected_index = interiors.index(float(transition))
    indices = sorted(
        {
            max(0, selected_index - 1),
            selected_index,
            min(len(interiors) - 1, selected_index + 1),
        }
    )
    if len(indices) < 2:
        raise RuntimeError("density stability needs at least two distinct interior densities")
    return tuple(Density.parse(interiors[index]) for index in indices)


@jaxtyped(typechecker=beartype)
def _collect_or_load_stage_one_corners(
    output_dir: Path,
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    probe: CircuitProbe,
    grid: SiteGrid,
    clean_residuals: ResidualBank,
    config: FourierCircuitConfig,
    site_space: ActiveSiteSpace,
    transition: Density,
    generator: t.Generator,
) -> tuple[MaskBatch, CornerBatchResult]:
    """Persist expensive alpha-gradient corners before any numerical spectrum fit."""

    json_path = output_dir / "stage_1_corners.json"
    tensor_path = output_dir / "stage_1_corners.pt"
    expected_masks = _sample_masks_for_site_space(
        config.spectrum.sample_budget,
        grid,
        SweepDensity.parse(float(transition)),
        generator,
        site_space,
    )
    if _stage_sidecar_state(json_path, tensor_path):
        raw = read_json(json_path)
        if (
            not isinstance(raw, dict)
            or raw.get("schema_version") != FOURIER_SCHEMA_VERSION
            or raw.get("stage") != "stage_1_corners"
            or raw.get("status") != "complete"
            or raw.get("function_space") != "singleton_vetoed_residual"
            or raw.get("transition_density") != float(transition)
            or raw.get("sample_count") != config.spectrum.sample_budget
            or raw.get("active_full_site_indices") != list(site_space.active_full_indices)
            or raw.get("corner_sidecar") != tensor_path.name
            or raw.get("corner_sidecar_sha256") != sha256_file(tensor_path)
        ):
            raise RuntimeError("stored Stage-1 corner checkpoint is invalid")
        tensors = _load_tensor_sidecar(tensor_path)
        required = {
            "masks",
            "candidate_logits",
            "logit_diffs",
            "correct_probabilities",
            "accuracies",
            "gradients",
        }
        if set(tensors) != required or not t.equal(tensors["masks"], expected_masks):
            raise RuntimeError("stored Stage-1 corners do not match deterministic sampling")
        result = CornerBatchResult(
            tensors["candidate_logits"],
            tensors["logit_diffs"],
            tensors["correct_probabilities"],
            tensors["accuracies"],
            tensors["gradients"],
        )
        if result.gradients is None or result.gradients.shape != expected_masks.shape:
            raise RuntimeError("stored Stage-1 gradients do not cover the exact patch grid")
        return expected_masks, result

    result = evaluate_masks_in_batches(
        model,
        blocks,
        probe,
        grid,
        clean_residuals,
        expected_masks,
        config.spectrum.gradient_batch_size,
        with_gradients=True,
    )
    if result.gradients is None or result.gradients.shape != expected_masks.shape:
        raise RuntimeError("Stage-1 corner collection must retain every per-site gradient")
    _write_tensor_sidecar(
        tensor_path,
        {
            "masks": expected_masks,
            "candidate_logits": result.candidate_logits,
            "logit_diffs": result.logit_diffs,
            "correct_probabilities": result.correct_probabilities,
            "accuracies": result.accuracies,
            "gradients": result.gradients,
        },
    )
    write_json(
        json_path,
        {
            "schema_version": FOURIER_SCHEMA_VERSION,
            "stage": "stage_1_corners",
            "status": "complete",
            "function_space": "singleton_vetoed_residual",
            "transition_density": float(transition),
            "sample_count": config.spectrum.sample_budget,
            "active_full_site_indices": list(site_space.active_full_indices),
            "vetoed_singleton_sites": [asdict(site) for site in site_space.vetoed_sites],
            "scientific_execution": "full_prompt_batch_one_with_alpha_gradients",
            "corner_sidecar": tensor_path.name,
            "corner_sidecar_sha256": sha256_file(tensor_path),
        },
    )
    return expected_masks, result


@jaxtyped(typechecker=beartype)
def _collect_or_load_stability_corners(
    output_dir: Path,
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    probe: CircuitProbe,
    grid: SiteGrid,
    clean_residuals: ResidualBank,
    config: FourierCircuitConfig,
    site_space: ActiveSiteSpace,
    densities: tuple[Density, ...],
    generator: t.Generator,
) -> tuple[tuple[MaskBatch, CornerBatchResult], ...]:
    """Persist all gradient-free stability corners before fitting any local spectrum."""

    if len(densities) < 2:
        raise ValueError("stability corner checkpoint requires at least two densities")
    json_path = output_dir / "stage_1_stability_corners.json"
    tensor_path = output_dir / "stage_1_stability_corners.pt"
    count = config.spectrum.density_stability.sample_budget_per_density
    expected_masks = tuple(
        _sample_masks_for_site_space(
            count,
            grid,
            SweepDensity.parse(float(density)),
            generator,
            site_space,
        )
        for density in densities
    )
    concatenated_masks = t.cat(expected_masks, dim=0)
    if _stage_sidecar_state(json_path, tensor_path):
        raw = read_json(json_path)
        if (
            not isinstance(raw, dict)
            or raw.get("schema_version") != FOURIER_SCHEMA_VERSION
            or raw.get("stage") != "stage_1_stability_corners"
            or raw.get("status") != "complete"
            or raw.get("function_space") != "singleton_vetoed_residual"
            or raw.get("densities") != [float(density) for density in densities]
            or raw.get("samples_per_density") != count
            or raw.get("active_full_site_indices") != list(site_space.active_full_indices)
            or raw.get("corner_sidecar") != tensor_path.name
            or raw.get("corner_sidecar_sha256") != sha256_file(tensor_path)
        ):
            raise RuntimeError("stored Stage-1 stability-corner checkpoint is invalid")
        tensors = _load_tensor_sidecar(tensor_path)
        required = {
            "masks",
            "density_indices",
            "candidate_logits",
            "logit_diffs",
            "correct_probabilities",
            "accuracies",
        }
        if set(tensors) != required or not t.equal(tensors["masks"], concatenated_masks):
            raise RuntimeError("stored stability corners do not match deterministic sampling")
        expected_density_indices = t.arange(len(densities), dtype=t.int64).repeat_interleave(count)
        if not t.equal(tensors["density_indices"], expected_density_indices):
            raise RuntimeError("stored stability corners have invalid density indices")
        results: list[tuple[MaskBatch, CornerBatchResult]] = []
        for index, masks in enumerate(expected_masks):
            start = index * count
            stop = start + count
            results.append(
                (
                    masks,
                    CornerBatchResult(
                        tensors["candidate_logits"][start:stop],
                        tensors["logit_diffs"][start:stop],
                        tensors["correct_probabilities"][start:stop],
                        tensors["accuracies"][start:stop],
                        None,
                    ),
                )
            )
        return tuple(results)

    results = tuple(
        evaluate_masks_in_batches(
            model,
            blocks,
            probe,
            grid,
            clean_residuals,
            masks,
            config.cache.reference_batch_size,
            with_gradients=False,
        )
        for masks in expected_masks
    )
    density_indices = t.arange(len(densities), dtype=t.int64).repeat_interleave(count)
    _write_tensor_sidecar(
        tensor_path,
        {
            "masks": concatenated_masks,
            "density_indices": density_indices,
            "candidate_logits": t.cat([result.candidate_logits for result in results]),
            "logit_diffs": t.cat([result.logit_diffs for result in results]),
            "correct_probabilities": t.cat([result.correct_probabilities for result in results]),
            "accuracies": t.cat([result.accuracies for result in results]),
        },
    )
    write_json(
        json_path,
        {
            "schema_version": FOURIER_SCHEMA_VERSION,
            "stage": "stage_1_stability_corners",
            "status": "complete",
            "function_space": "singleton_vetoed_residual",
            "densities": [float(density) for density in densities],
            "samples_per_density": count,
            "active_full_site_indices": list(site_space.active_full_indices),
            "vetoed_singleton_sites": [asdict(site) for site in site_space.vetoed_sites],
            "scientific_execution": "torch.inference_mode_full_prompt_batch_one",
            "corner_sidecar": tensor_path.name,
            "corner_sidecar_sha256": sha256_file(tensor_path),
        },
    )
    return tuple(zip(expected_masks, results, strict=True))


@jaxtyped(typechecker=beartype)
def run_spectrum_estimation(
    output_dir: Path,
    stage_zero: dict[str, object],
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    probe: CircuitProbe,
    grid: SiteGrid,
    clean_residuals: ResidualBank,
    config: FourierCircuitConfig,
    site_space: ActiveSiteSpace,
) -> dict[str, object]:
    if stage_zero.get("status") == "flat_stop":
        raise RuntimeError("stage 0 found a flat curve; spectrum estimation is forbidden")
    if stage_zero.get("function_space") != "singleton_vetoed_residual":
        raise RuntimeError("Stage 1 must estimate the singleton-vetoed residual function")
    if site_space.full_site_count != grid.site_count:
        raise ValueError("Stage 1 active-site space does not match the patch grid")
    json_path = output_dir / "stage_1_spectrum.json"
    tensor_path = output_dir / "stage_1_samples.pt"
    if _stage_sidecar_state(json_path, tensor_path):
        return _validated_stage_artifact(
            json_path,
            tensor_path,
            stage=1,
            statuses=(
                "complete",
                "complete_density_unstable",
                "complete_no_heavy_coefficients",
            ),
            sidecar_field="sample_sidecar",
        )
    raw_transition = stage_zero.get("transition_density")
    if not isinstance(raw_transition, int | float):
        raise TypeError("stage-0 transition density must be numeric")
    transition = Density.parse(float(raw_transition))
    generator = t.Generator().manual_seed(config.spectrum.seed)
    masks, result = _collect_or_load_stage_one_corners(
        output_dir,
        model,
        blocks,
        probe,
        grid,
        clean_residuals,
        config,
        site_space,
        transition,
        generator,
    )
    if result.gradients is None:  # pragma: no cover - helper guarantees this invariant
        raise AssertionError("Stage-1 corner checkpoint unexpectedly lacks gradients")
    active_indices = list(site_space.active_full_indices)
    flat_masks = flatten_masks(masks, grid)[:, active_indices]
    flat_gradients = result.gradients.reshape(result.gradients.shape[0], grid.site_count)[
        :, active_indices
    ]
    validation_count = max(1, round(masks.shape[0] * config.spectrum.validation_fraction))
    if validation_count >= masks.shape[0]:
        raise RuntimeError("validation split consumes every spectrum sample")
    split_generator = t.Generator().manual_seed(config.spectrum.seed + 1)
    order = t.randperm(masks.shape[0], generator=split_generator)
    validation_rows = order[:validation_count]
    training_rows = order[validation_count:]
    screen_count = min(
        config.spectrum.lasso.interaction_screen_size,
        site_space.active_site_count,
    )
    function_screened_sites = screen_sites_from_function_values(
        result.logit_diffs[training_rows],
        flat_masks[training_rows],
        transition,
        screen_count,
    )
    validation_supports = screened_supports(
        site_space.active_site_count,
        function_screened_sites,
        config.spectrum.lasso,
    )
    validation_function_estimates = exact_fourier_coefficients(
        flat_masks[validation_rows],
        result.logit_diffs[validation_rows],
        validation_supports,
        transition,
    )
    validation_gradient_estimates = gradient_coefficient_estimates(
        flat_gradients[validation_rows],
        flat_masks[validation_rows],
        validation_supports,
        transition,
    )
    holdout_indices = _heldout_coefficient_indices(
        validation_supports,
        config.spectrum.gradient_validation.coefficient_holdout_count,
        config.spectrum.seed + 2,
    )
    gradient_validation = validate_gradient_estimates(
        validation_function_estimates,
        validation_gradient_estimates,
        holdout_indices,
        config.spectrum.gradient_validation,
    )
    screened_sites = (
        screen_sites_from_gradients(
            flat_gradients[training_rows],
            result.logit_diffs[training_rows],
            flat_masks[training_rows],
            transition,
            screen_count,
        )
        if gradient_validation.accepted
        else function_screened_sites
    )
    supports = screened_supports(
        site_space.active_site_count,
        screened_sites,
        config.spectrum.lasso,
    )
    all_training_features = parity_feature_matrix(flat_masks[training_rows], supports, transition)
    fit_feature_indices = function_correlation_feature_indices(
        all_training_features,
        result.logit_diffs[training_rows],
        config.spectrum.lasso.fit_feature_count,
    )
    fitted_lasso_values = fit_lasso_coordinate_descent(
        all_training_features[:, list(fit_feature_indices)],
        result.logit_diffs[training_rows],
        config.spectrum.lasso,
    )
    lasso_values = t.zeros(len(supports), dtype=t.float64)
    lasso_values[list(fit_feature_indices)] = fitted_lasso_values
    function_estimates = exact_fourier_coefficients(
        flat_masks[validation_rows],
        result.logit_diffs[validation_rows],
        supports,
        transition,
    )
    gradient_estimates = gradient_coefficient_estimates(
        flat_gradients[validation_rows],
        flat_masks[validation_rows],
        supports,
        transition,
    )
    function_samples, gradient_samples = _coefficient_sample_matrices(
        flat_masks[validation_rows],
        result.logit_diffs[validation_rows],
        flat_gradients[validation_rows],
        supports,
        transition,
    )
    if gradient_validation.accepted:
        augmented_nonconstant = inverse_variance_augment(
            function_samples[:, 1:],
            gradient_samples[:, 1:],
            config.spectrum.gradient_validation,
        )
        augmented = t.cat((function_estimates[:1], augmented_nonconstant), dim=0)
    else:
        augmented = function_estimates
    coefficients = tuple(
        FourierCoefficient(
            support=site_space.full_support(support),
            degree=len(support),
            lasso_value=float(lasso_values[index]),
            function_value_estimate=float(function_estimates[index]),
            gradient_estimate=(None if not support else float(gradient_estimates[index])),
            augmented_estimate=(None if not support else float(augmented[index])),
            is_heavy=bool(
                support
                and abs(float(lasso_values[index]))
                >= config.spectrum.lasso.heavy_coefficient_threshold
            ),
        )
        for index, support in enumerate(supports)
    )
    has_heavy_coefficients = any(coefficient.is_heavy for coefficient in coefficients)

    profiles = []
    stability_densities = _spectrum_density_triplet(stage_zero, transition)
    stability_corner_batches = _collect_or_load_stability_corners(
        output_dir,
        model,
        blocks,
        probe,
        grid,
        clean_residuals,
        config,
        site_space,
        stability_densities,
        generator,
    )
    stability_masks: list[t.Tensor] = []
    stability_diffs: list[t.Tensor] = []
    stability_density_indices: list[t.Tensor] = []
    stability_fit_feature_indices: list[t.Tensor] = []
    for density_index, (density, corner_batch) in enumerate(
        zip(stability_densities, stability_corner_batches, strict=True)
    ):
        local_masks, local_result = corner_batch
        local_flat = flatten_masks(local_masks, grid)[:, active_indices]
        local_features = parity_feature_matrix(local_flat, supports, density)
        local_fit_indices = function_correlation_feature_indices(
            local_features,
            local_result.logit_diffs,
            config.spectrum.lasso.fit_feature_count,
        )
        local_fitted = fit_lasso_coordinate_descent(
            local_features[:, list(local_fit_indices)],
            local_result.logit_diffs,
            config.spectrum.lasso,
        )
        local_estimates = t.zeros(len(supports), dtype=t.float64)
        local_estimates[list(local_fit_indices)] = local_fitted
        local_coefficients = tuple(
            FourierCoefficient(
                support=support,
                degree=len(support),
                lasso_value=float(local_estimates[index]),
                function_value_estimate=float(local_estimates[index]),
                gradient_estimate=None,
                augmented_estimate=None,
                is_heavy=False,
            )
            for index, support in enumerate(supports)
        )
        profiles.append(
            normalized_degree_profile(
                local_coefficients,
                config.spectrum.lasso.degree_cap,
                density,
            )
        )
        stability_masks.append(local_masks)
        stability_diffs.append(local_result.logit_diffs)
        stability_density_indices.append(
            t.full((local_masks.shape[0],), density_index, dtype=t.int64)
        )
        stability_fit_feature_indices.append(
            t.tensor(local_fit_indices, dtype=t.int64).unsqueeze(0)
        )
    stable, maximum_l1, minimum_cosine = compare_degree_profiles(
        tuple(profiles),
        config.spectrum.density_stability,
    )
    _write_tensor_sidecar(
        tensor_path,
        {
            "masks": masks,
            "candidate_logits": result.candidate_logits,
            "logit_diffs": result.logit_diffs,
            "correct_probabilities": result.correct_probabilities,
            "accuracies": result.accuracies,
            "gradients": result.gradients,
            "active_full_site_indices": t.tensor(site_space.active_full_indices, dtype=t.int64),
            "training_rows": training_rows,
            "validation_rows": validation_rows,
            "fit_feature_indices": t.tensor(fit_feature_indices, dtype=t.int64),
            "stability_masks": t.cat(stability_masks, dim=0),
            "stability_logit_diffs": t.cat(stability_diffs, dim=0),
            "stability_density_indices": t.cat(stability_density_indices, dim=0),
            "stability_fit_feature_indices": t.cat(
                stability_fit_feature_indices,
                dim=0,
            ),
        },
    )
    payload: dict[str, object] = {
        "schema_version": FOURIER_SCHEMA_VERSION,
        "stage": 1,
        "status": (
            "complete_no_heavy_coefficients"
            if not has_heavy_coefficients
            else "complete"
            if stable
            else "complete_density_unstable"
        ),
        "function_space": "singleton_vetoed_residual",
        "vetoed_singleton_sites": [asdict(site) for site in site_space.vetoed_sites],
        "active_site_count": site_space.active_site_count,
        "warning": (
            "Function-value LASSO found no heavy nonconstant coefficient; causal verification has no candidate."
            if not has_heavy_coefficients
            else None
            if stable
            else "Degree profile changes across intervention densities; candidates may reflect intervention scale."
        ),
        "transition_density": float(transition),
        "estimator": {
            "primary": "function_value_hierarchical_lasso",
            "solver": "deterministic_maximum_kkt_coordinate_descent",
            "basis": "p_biased_orthonormal",
            "interaction_search": (
                "all singleton sites; degree 2..cap combinations over the explicitly "
                + (
                    "gradient-augmented site pool after the held-out gate passed"
                    if gradient_validation.accepted
                    else "function-value-only site pool because the gradient gate failed"
                )
            ),
            "function_value_screened_site_indices": [
                site_space.active_full_indices[index] for index in function_screened_sites
            ],
            "screened_site_indices": [
                site_space.active_full_indices[index] for index in screened_sites
            ],
            "screened_sites": [
                asdict(grid.site(site_space.active_full_indices[index])) for index in screened_sites
            ],
            "feature_count": len(supports),
            "fit_feature_count": len(fit_feature_indices),
            "function_correlation_screened_before_lasso": True,
            "not_exhaustive_above_degree_one": True,
        },
        "gradient_validation": {
            **asdict(gradient_validation),
            "heldout_coefficient_indices": list(holdout_indices),
            "heldout_supports": [
                list(site_space.full_support(validation_supports[index]))
                for index in holdout_indices
            ],
            "screening_was_function_value_only_before_this_gate": True,
        },
        "gradient_augmentation_used": gradient_validation.accepted,
        "coefficients": _coefficient_payload(coefficients, grid, fit_feature_indices),
        "heavy_coefficient_count": sum(coefficient.is_heavy for coefficient in coefficients),
        "density_stability": {
            "stable": stable,
            "degree_weight_estimator": "squared function-value LASSO coefficients",
            "maximum_l1_distance": maximum_l1,
            "minimum_cosine_similarity": minimum_cosine,
            "profiles": [asdict(profile) for profile in profiles],
        },
        "sample_sidecar": tensor_path.name,
        "sample_sidecar_sha256": sha256_file(tensor_path),
    }
    write_json(json_path, payload)
    return payload


@beartype
def _heavy_candidates(
    stage_one: dict[str, object],
    *,
    minimum_degree: int,
) -> tuple[tuple[SiteSet, dict[str, object]], ...]:
    if minimum_degree <= 0:
        raise ValueError("minimum heavy-candidate degree must be positive")
    rows = stage_one.get("coefficients")
    if not isinstance(rows, list):
        raise TypeError("stage-1 coefficient table must be an array")
    candidates: list[tuple[SiteSet, dict[str, object]]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("coefficient table row must be an object")
        if row.get("is_heavy") is not True:
            continue
        degree = row.get("degree")
        if not isinstance(degree, int) or degree < 0:
            raise TypeError("coefficient degree must be a non-negative integer")
        if degree < minimum_degree:
            continue
        raw_sites = row.get("sites")
        if not isinstance(raw_sites, list) or not raw_sites:
            raise TypeError("heavy coefficient must have a non-empty site support")
        sites: list[Site] = []
        for raw_site in raw_sites:
            if not isinstance(raw_site, dict):
                raise TypeError("coefficient site must be an object")
            token_index = raw_site.get("token_index")
            layer = raw_site.get("layer")
            if not isinstance(token_index, int) or not isinstance(layer, int):
                raise TypeError("coefficient site coordinates must be integers")
            sites.append(Site(token_index, layer))
        site_set = tuple(sorted(set(sites)))
        if len(site_set) != len(sites):
            raise RuntimeError("heavy coefficient support contains a duplicate site")
        candidates.append((site_set, cast(dict[str, object], row)))
    return tuple(candidates)


@beartype
def _all_nonempty_subsets(candidates: CandidateSiteSets) -> tuple[SiteSet, ...]:
    subsets = {
        subset
        for candidate in candidates
        for size in range(1, len(candidate) + 1)
        for subset in itertools.combinations(candidate, size)
    }
    return tuple(sorted(subsets, key=lambda value: (len(value), value)))


@jaxtyped(typechecker=beartype)
def _masks_for_site_sets(site_sets: tuple[SiteSet, ...], grid: SiteGrid) -> MaskBatch:
    if not site_sets:
        raise ValueError("causal verification requires at least one site set")
    masks = t.zeros((len(site_sets), *grid.shape), dtype=t.bool)
    for row, site_set in enumerate(site_sets):
        if not site_set:
            raise ValueError("causal-verification site sets must be non-empty")
        for site in site_set:
            flat_index = grid.flat_index(site)
            token_offset, layer_offset = divmod(flat_index, len(grid.layers))
            masks[row, token_offset, layer_offset] = True
    return masks


@jaxtyped(typechecker=beartype)
def _exhaustive_singleton_masks(grid: SiteGrid) -> MaskBatch:
    masks = t.eye(grid.site_count, dtype=t.bool)
    return masks.reshape(grid.site_count, *grid.shape)


@beartype
def _validated_singleton_artifact(output_dir: Path) -> dict[str, object]:
    json_path = output_dir / "exhaustive_singletons.json"
    tensor_path = output_dir / "exhaustive_singletons.pt"
    if not _stage_sidecar_state(json_path, tensor_path):
        raise FileNotFoundError("exhaustive singleton artifact has not been collected")
    raw = read_json(json_path)
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != FOURIER_SCHEMA_VERSION
        or raw.get("stage") != "exhaustive_singletons"
        or raw.get("status") != "verified"
        or raw.get("singleton_sidecar") != tensor_path.name
        or raw.get("singleton_sidecar_sha256") != sha256_file(tensor_path)
    ):
        raise RuntimeError("stored exhaustive singleton artifact is invalid")
    _load_tensor_sidecar(tensor_path)
    return cast(dict[str, object], raw)


@beartype
def _verified_singleton_sites(singletons: dict[str, object]) -> tuple[Site, ...]:
    raw_sites = singletons.get("verified_singleton_minsets")
    if not isinstance(raw_sites, list):
        raise TypeError("exhaustive singleton artifact lacks its passing-site list")
    sites: list[Site] = []
    for row in raw_sites:
        if not isinstance(row, dict):
            raise TypeError("verified singleton row must be an object")
        raw_site = row.get("site")
        if not isinstance(raw_site, dict):
            raise TypeError("verified singleton row lacks its site")
        token_index = raw_site.get("token_index")
        layer = raw_site.get("layer")
        if not isinstance(token_index, int) or not isinstance(layer, int):
            raise TypeError("verified singleton site coordinates must be integers")
        sites.append(Site(token_index, layer))
    ordered = tuple(sorted(set(sites)))
    if len(ordered) != len(sites):
        raise RuntimeError("exhaustive singleton artifact contains duplicate passing sites")
    return ordered


@beartype
def _resolved_sufficiency_contract(
    dirty_logit_diff: float,
    clean_logit_diff: float,
    dirty_correct_probability: float,
    clean_correct_probability: float,
    config: FourierCircuitConfig,
) -> tuple[LogitDiff, dict[str, object]]:
    dirty_logit = LogitDiff.parse(dirty_logit_diff)
    clean_logit = LogitDiff.parse(clean_logit_diff)
    threshold = resolved_sufficiency_threshold(
        dirty_logit,
        clean_logit,
        dirty_correct_probability,
        clean_correct_probability,
        config.sufficiency,
    )
    threshold_probability = 1.0 / (1.0 + math.exp(-float(threshold)))
    payload: dict[str, object] = {
        "criterion": (
            "clean_correct_probability_minus_absolute_tolerance"
            if isinstance(config.sufficiency, ProbabilitySufficiencyConfig)
            else "raw_logit_gap_recovery"
        ),
        "dirty_logit_diff": dirty_logit_diff,
        "clean_logit_diff": clean_logit_diff,
        "dirty_correct_probability": dirty_correct_probability,
        "clean_correct_probability": clean_correct_probability,
        "threshold_logit_diff": float(threshold),
        "threshold_correct_probability": threshold_probability,
        "require_clean_argmax": config.sufficiency.require_clean_argmax,
        "sufficiency_margin_metric": "raw_logit_diff",
    }
    if isinstance(config.sufficiency, ProbabilitySufficiencyConfig):
        payload["absolute_probability_tolerance"] = (
            config.sufficiency.absolute_probability_tolerance
        )
        payload["expected_passing_singleton_count"] = len(
            config.sufficiency.expected_passing_singletons
        )
    else:
        payload["required_recovery_fraction"] = config.sufficiency.recovery_fraction
    return threshold, payload


@beartype
def _checkpoint_transfer_record(
    root: Path,
    config: FourierCircuitConfig,
) -> tuple[Path, dict[str, object]]:
    path = _reference_artifact_path(root, config)
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise TypeError("checkpoint-transfer reference must be an object")
    records = raw.get("records")
    if not isinstance(records, list):
        raise TypeError("checkpoint-transfer reference lacks records")
    record = next(
        (
            item
            for item in records
            if isinstance(item, dict) and item.get("function_id") == config.task.function_id
        ),
        None,
    )
    if not isinstance(record, dict):
        raise RuntimeError("checkpoint-transfer reference lacks the selected function")
    return path, cast(dict[str, object], record)


@jaxtyped(typechecker=beartype)
def run_exhaustive_singleton_sweep(
    root: Path,
    output_dir: Path,
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    probe: CircuitProbe,
    grid: SiteGrid,
    clean_residuals: ResidualBank,
    config: FourierCircuitConfig,
) -> dict[str, object]:
    json_path = output_dir / "exhaustive_singletons.json"
    tensor_path = output_dir / "exhaustive_singletons.pt"
    if _stage_sidecar_state(json_path, tensor_path):
        return _validated_singleton_artifact(output_dir)
    dirty_mask = t.zeros((1, *grid.shape), dtype=t.bool)
    clean_mask = t.ones((1, *grid.shape), dtype=t.bool)
    dirty_result = evaluate_masks_in_batches(
        model,
        blocks,
        probe,
        grid,
        clean_residuals,
        dirty_mask,
        config.sufficiency.patch_batch_size,
        with_gradients=False,
    )
    clean_result = evaluate_masks_in_batches(
        model,
        blocks,
        probe,
        grid,
        clean_residuals,
        clean_mask,
        config.sufficiency.patch_batch_size,
        with_gradients=False,
    )
    if config.sufficiency.require_clean_argmax and not bool(clean_result.accuracies[0]):
        raise RuntimeError("all-clean corner fails the required clean argmax")
    threshold, sufficiency_contract = _resolved_sufficiency_contract(
        float(dirty_result.logit_diffs[0]),
        float(clean_result.logit_diffs[0]),
        float(dirty_result.correct_probabilities[0]),
        float(clean_result.correct_probabilities[0]),
        config,
    )
    threshold_probability = cast(float, sufficiency_contract["threshold_correct_probability"])
    masks = _exhaustive_singleton_masks(grid)
    result = evaluate_masks_in_batches(
        model,
        blocks,
        probe,
        grid,
        clean_residuals,
        masks,
        config.sufficiency.patch_batch_size,
        with_gradients=False,
    )
    records: list[dict[str, object]] = []
    passing: list[dict[str, object]] = []
    by_site: dict[Site, dict[str, object]] = {}
    for index in range(grid.site_count):
        site = grid.site(index)
        logit_diff = float(result.logit_diffs[index])
        probability = float(result.correct_probabilities[index])
        accuracy = bool(result.accuracies[index])
        sufficient = bool(
            logit_diff >= float(threshold)
            and (accuracy or not config.sufficiency.require_clean_argmax)
        )
        row: dict[str, object] = {
            "full_site_index": index,
            "site": asdict(site),
            "token_reverse_index": probe.input_ids.shape[1] - site.token_index - 1,
            "token": probe.token_labels[site.token_index],
            "raw_logit_diff": logit_diff,
            "correct_probability": probability,
            "accuracy": accuracy,
            "sufficiency_margin": logit_diff - float(threshold),
            "correct_probability_margin": probability - threshold_probability,
            "sufficient": sufficient,
        }
        records.append(row)
        by_site[site] = row
        if sufficient:
            passing.append(row)
    if isinstance(config.sufficiency, ProbabilitySufficiencyConfig):
        observed_passing_sites = tuple(
            Site(
                cast(dict[str, int], row["site"])["token_index"],
                cast(dict[str, int], row["site"])["layer"],
            )
            for row in passing
        )
        if observed_passing_sites != config.sufficiency.expected_passing_singletons:
            raise RuntimeError(
                "probability-sufficiency singleton census disagrees with the registered veto set"
            )

    reference_path, reference_record = _checkpoint_transfer_record(root, config)
    token_axis = reference_record.get("token_axis")
    if not isinstance(token_axis, dict):
        raise TypeError("checkpoint-transfer reference lacks its token axis")
    if (
        token_axis.get("recipient_rendered_prompt") != probe.rendered_prompt
        or token_axis.get("source_rendered_prompt") != probe.rendered_prompt
    ):
        raise RuntimeError("clean and dirty checkpoint-transfer prompts are not exactly identical")
    reference_cells = reference_record.get("cells")
    recipient_probabilities = reference_record.get("recipient_probabilities")
    source_probabilities = reference_record.get("source_probabilities")
    if (
        not isinstance(reference_cells, list)
        or not isinstance(recipient_probabilities, list)
        or not isinstance(source_probabilities, list)
    ):
        raise TypeError("checkpoint-transfer reference lacks cells or endpoint probabilities")
    reference_by_layer: dict[int, dict[str, object]] = {}
    final_token_index = probe.input_ids.shape[1] - 1
    for raw_cell in reference_cells:
        if not isinstance(raw_cell, dict):
            raise TypeError("checkpoint-transfer reference cell must be an object")
        if raw_cell.get("recipient_token_index") == final_token_index:
            layer = raw_cell.get("layer")
            if not isinstance(layer, int) or layer in reference_by_layer:
                raise RuntimeError("checkpoint-transfer final-token layer cells are malformed")
            reference_by_layer[layer] = cast(dict[str, object], raw_cell)
    required_rows: list[dict[str, object]] = []
    for layer in config.exhaustive_singletons.required_final_token_layers:
        site = Site(final_token_index, layer)
        observed = by_site.get(site)
        reference = reference_by_layer.get(layer)
        if observed is None or reference is None:
            raise RuntimeError("singleton correctness harness lacks a required final-token layer")
        reference_probability = reference.get("probability")
        if not isinstance(reference_probability, int | float):
            raise TypeError("reference singleton probability must be numeric")
        probability_error = abs(
            float(cast(float, observed["correct_probability"])) - float(reference_probability)
        )
        required_rows.append(
            {
                "layer": layer,
                "reference_probability": float(reference_probability),
                "observed_probability": observed["correct_probability"],
                "absolute_probability_error": probability_error,
                "sufficient": observed["sufficient"],
            }
        )
        if probability_error > config.exhaustive_singletons.reference_probability_tolerance:
            raise RuntimeError(
                f"singleton layer {layer} does not reproduce the checkpoint-transfer grid"
            )
        if observed["sufficient"] is not True:
            raise RuntimeError(
                f"required broad late-layer singleton region failed at layer {layer}"
            )
    correct_index = probe.correct_choice_index
    reference_dirty_probability = recipient_probabilities[correct_index]
    reference_clean_probability = source_probabilities[correct_index]
    if not isinstance(reference_dirty_probability, int | float) or not isinstance(
        reference_clean_probability, int | float
    ):
        raise TypeError("checkpoint-transfer endpoint probabilities must be numeric")
    endpoint_probability_errors = {
        "dirty": abs(
            float(dirty_result.correct_probabilities[0]) - float(reference_dirty_probability)
        ),
        "clean": abs(
            float(clean_result.correct_probabilities[0]) - float(reference_clean_probability)
        ),
    }
    if any(
        error > config.exhaustive_singletons.reference_probability_tolerance
        for error in endpoint_probability_errors.values()
    ):
        raise RuntimeError("singleton endpoints do not reproduce the checkpoint-transfer grid")
    _write_tensor_sidecar(
        tensor_path,
        {
            "site_full_indices": t.arange(grid.site_count, dtype=t.int64),
            "candidate_logits": result.candidate_logits,
            "logit_diffs": result.logit_diffs,
            "correct_probabilities": result.correct_probabilities,
            "accuracies": result.accuracies,
            "sufficient": t.tensor([bool(row["sufficient"]) for row in records], dtype=t.bool),
        },
    )
    payload: dict[str, object] = {
        "schema_version": FOURIER_SCHEMA_VERSION,
        "stage": "exhaustive_singletons",
        "status": "verified",
        "function_space": "one_clean_site_on_all_dirty_background",
        "scientific_execution": "torch.inference_mode_full_prompt_batch_one",
        "prompt_contract": {
            "clean_dirty_prompt_identical": True,
            "rendered_prompt": probe.rendered_prompt,
            "correct_choice_index": probe.correct_choice_index,
            "correct_answer": probe.record.target,
            "choice_function_ids": list(probe.record.choice_function_ids),
        },
        "sufficiency": sufficiency_contract,
        "checkpoint_transfer_harness": {
            "reference_artifact": str(reference_path),
            "reference_artifact_sha256": sha256_file(reference_path),
            "reference_dirty_probability": float(reference_dirty_probability),
            "reference_clean_probability": float(reference_clean_probability),
            "endpoint_probability_errors": endpoint_probability_errors,
            "required_final_token_layers": list(
                config.exhaustive_singletons.required_final_token_layers
            ),
            "required_layer_results": required_rows,
            "probability_tolerance": config.exhaustive_singletons.reference_probability_tolerance,
        },
        "site_grid": _site_grid_payload(grid, probe),
        "singleton_count": len(records),
        "passing_singleton_count": len(passing),
        "singleton_results": records,
        "verified_singleton_minsets": passing,
        "singleton_search_is_exhaustive": True,
        "singleton_sidecar": tensor_path.name,
        "singleton_sidecar_sha256": sha256_file(tensor_path),
    }
    write_json(json_path, payload)
    return payload


@jaxtyped(typechecker=beartype)
def run_causal_verification(
    output_dir: Path,
    stage_zero: dict[str, object],
    stage_one: dict[str, object],
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    probe: CircuitProbe,
    grid: SiteGrid,
    clean_residuals: ResidualBank,
    config: FourierCircuitConfig,
    singletons: dict[str, object],
    site_space: ActiveSiteSpace,
) -> dict[str, object]:
    json_path = output_dir / "stage_2_minsets.json"
    tensor_path = output_dir / "stage_2_verification.pt"
    if _stage_sidecar_state(json_path, tensor_path):
        return _validated_stage_artifact(
            json_path,
            tensor_path,
            stage=2,
            statuses=(
                "verified_multisite",
                "no_verified_multisite_minsets",
                "no_higher_order_hypotheses",
            ),
            sidecar_field="verification_sidecar",
        )
    if stage_zero.get("function_space") != "singleton_vetoed_residual":
        raise RuntimeError("Stage 2 must consume the singleton-vetoed residual density sweep")
    if stage_one.get("function_space") != "singleton_vetoed_residual":
        raise RuntimeError("Stage 2 must consume singleton-vetoed Fourier hypotheses")
    singleton_sufficiency = singletons.get("sufficiency")
    if not isinstance(singleton_sufficiency, dict):
        raise TypeError("exhaustive singleton artifact lacks its sufficiency contract")
    dirty_logit = singleton_sufficiency.get("dirty_logit_diff")
    clean_logit = singleton_sufficiency.get("clean_logit_diff")
    dirty_probability = singleton_sufficiency.get("dirty_correct_probability")
    clean_probability = singleton_sufficiency.get("clean_correct_probability")
    if any(
        not isinstance(value, int | float)
        for value in (dirty_logit, clean_logit, dirty_probability, clean_probability)
    ):
        raise TypeError("singleton endpoint logit differences and probabilities must be numeric")
    threshold, expected_sufficiency = _resolved_sufficiency_contract(
        float(cast(float, dirty_logit)),
        float(cast(float, clean_logit)),
        float(cast(float, dirty_probability)),
        float(cast(float, clean_probability)),
        config,
    )
    if singleton_sufficiency != expected_sufficiency:
        raise RuntimeError("singleton sufficiency contract disagrees with the current config")
    threshold_probability = cast(float, expected_sufficiency["threshold_correct_probability"])
    candidate_rows = _heavy_candidates(stage_one, minimum_degree=2)
    if len(candidate_rows) > config.sufficiency.maximum_candidate_supports:
        raise RuntimeError(
            f"stage 2 has {len(candidate_rows)} heavy supports, exceeding the explicit "
            f"causal candidate cap {config.sufficiency.maximum_candidate_supports}"
        )
    vetoed_set = set(site_space.vetoed_sites)
    if any(vetoed_set.intersection(site_set) for site_set, _row in candidate_rows):
        raise RuntimeError("higher-order Fourier hypothesis contains a vetoed singleton site")
    if not candidate_rows:
        _write_tensor_sidecar(
            tensor_path,
            {
                "masks": t.empty((0, *grid.shape), dtype=t.bool),
                "candidate_logits": t.empty((0, 5), dtype=t.float32),
                "logit_diffs": t.empty((0,), dtype=t.float32),
                "correct_probabilities": t.empty((0,), dtype=t.float32),
                "accuracies": t.empty((0,), dtype=t.float32),
            },
        )
        payload = {
            "schema_version": FOURIER_SCHEMA_VERSION,
            "stage": 2,
            "status": "no_higher_order_hypotheses",
            "function_space": "singleton_vetoed_residual",
            "density_stability_warning": stage_one.get("warning"),
            "sufficiency": singleton_sufficiency,
            "site_grid": _site_grid_payload(grid, probe),
            "verified_multisite_minsets": [],
            "raw_fourier_candidates_are_not_circuits": True,
            "terminal_message": "No heavy degree-two-or-higher Fourier hypothesis was generated.",
            "verification_sidecar": tensor_path.name,
            "verification_sidecar_sha256": sha256_file(tensor_path),
        }
        write_json(json_path, payload)
        return payload
    candidates = as_non_empty_candidates(tuple(row[0] for row in candidate_rows))
    evaluated_sets = _all_nonempty_subsets(candidates)
    if len(evaluated_sets) > config.sufficiency.maximum_evaluated_site_sets:
        raise RuntimeError(
            f"stage 2 needs {len(evaluated_sets)} unique site-set evaluations, exceeding "
            f"the explicit cap {config.sufficiency.maximum_evaluated_site_sets}"
        )
    masks = _masks_for_site_sets(evaluated_sets, grid)
    result = evaluate_masks_in_batches(
        model,
        blocks,
        probe,
        grid,
        clean_residuals,
        masks,
        config.sufficiency.patch_batch_size,
        with_gradients=False,
    )
    metrics = {
        site_set: (
            float(result.logit_diffs[index]),
            float(result.correct_probabilities[index]),
            bool(result.accuracies[index]),
        )
        for index, site_set in enumerate(evaluated_sets)
    }

    def is_sufficient(site_set: SiteSet) -> bool:
        logit_diff, _probability, accuracy = metrics[site_set]
        return bool(
            logit_diff >= float(threshold)
            and (accuracy or not config.sufficiency.require_clean_argmax)
        )

    minset_generators = enumerate_minimal_sufficient_subsets(candidates, is_sufficient)
    if any(len(minset) == 1 for minset, _generator in minset_generators):
        raise RuntimeError(
            "residual walk-down found a singleton omitted by the exhaustive singleton sweep"
        )
    generators_by_minset: dict[SiteSet, set[SiteSet]] = {}
    for minset, generator in minset_generators:
        generators_by_minset.setdefault(minset, set()).add(generator)
    verified = tuple(
        VerifiedMinset(
            sites=minset,
            raw_logit_diff=LogitDiff.parse(metrics[minset][0]),
            correct_probability=metrics[minset][1],
            sufficiency_margin=metrics[minset][0] - float(threshold),
            generating_supports=tuple(sorted(generators)),
        )
        for minset, generators in sorted(
            generators_by_minset.items(),
            key=lambda item: (len(item[0]), item[0]),
        )
    )
    _write_tensor_sidecar(
        tensor_path,
        {
            "masks": masks,
            "candidate_logits": result.candidate_logits,
            "logit_diffs": result.logit_diffs,
            "correct_probabilities": result.correct_probabilities,
            "accuracies": result.accuracies,
        },
    )
    coefficient_by_sites = dict(candidate_rows)
    payload: dict[str, object] = {
        "schema_version": FOURIER_SCHEMA_VERSION,
        "stage": 2,
        "status": ("verified_multisite" if verified else "no_verified_multisite_minsets"),
        "function_space": "singleton_vetoed_residual",
        "density_stability_warning": stage_one.get("warning"),
        "sufficiency": {
            **expected_sufficiency,
        },
        "site_grid": _site_grid_payload(grid, probe),
        "verified_multisite_minsets": [
            {
                "size": len(minset.sites),
                "sites": [asdict(site) for site in minset.sites],
                "raw_logit_diff": float(minset.raw_logit_diff),
                "correct_probability": minset.correct_probability,
                "sufficiency_margin": minset.sufficiency_margin,
                "correct_probability_margin": (minset.correct_probability - threshold_probability),
                "generating_coefficients": [
                    coefficient_by_sites[support] for support in minset.generating_supports
                ],
            }
            for minset in verified
        ],
        "raw_fourier_candidates_are_not_circuits": True,
        "terminal_message": (
            None
            if verified
            else "No higher-order Fourier hypothesis survived exact causal verification and all-path walk-down."
        ),
        "verification_sidecar": tensor_path.name,
        "verification_sidecar_sha256": sha256_file(tensor_path),
    }
    write_json(json_path, payload)
    return payload


@beartype
def _release_model(model: t.nn.Module) -> None:
    model.to("cpu")
    del model
    gc.collect()
    t.cuda.empty_cache()


@beartype
def _capture_clean_checkpoint(
    root: Path,
    config: FourierCircuitConfig,
    probe: CircuitProbe,
    spec: ModelSpec,
) -> CheckpointCapture:
    model = _load_checkpoint_model(root, config, config.model.clean_step)
    try:
        blocks = _resolve_blocks(model, spec)
        return capture_checkpoint(model, blocks, probe)
    finally:
        _release_model(model)


@beartype
def _endpoint_contract(
    probe: CircuitProbe,
    clean: CheckpointCapture,
    dirty: CheckpointCapture,
) -> dict[str, object]:
    clean_diff, clean_probability, clean_accuracy = _candidate_metrics(
        clean.candidate_logits,
        probe.correct_choice_index,
    )
    dirty_diff, dirty_probability, dirty_accuracy = _candidate_metrics(
        dirty.candidate_logits,
        probe.correct_choice_index,
    )
    if float(clean_diff[0]) <= float(dirty_diff[0]):
        raise RuntimeError(
            "configured clean checkpoint does not improve this probe's raw logit difference"
        )
    return {
        "clean": {
            "logit_diff": float(clean_diff[0]),
            "correct_probability": float(clean_probability[0]),
            "accuracy": bool(clean_accuracy[0]),
        },
        "dirty": {
            "logit_diff": float(dirty_diff[0]),
            "correct_probability": float(dirty_probability[0]),
            "accuracy": bool(dirty_accuracy[0]),
        },
    }


@jaxtyped(typechecker=beartype)
def verify_endpoint_corner_contract(
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    probe: CircuitProbe,
    grid: SiteGrid,
    clean: CheckpointCapture,
    dirty: CheckpointCapture,
) -> dict[str, object]:
    """Prove the dirty corner and the full clean-residual intervention exactly.

    The all-clean residual corner is intentionally *not* compared to the donor
    checkpoint's logits.  Checkpoint transfer leaves the recipient final norm and
    unembedding in place, so those logits need not agree after full finetuning.
    """

    dirty_corner = reference_corner_batch(
        model,
        blocks,
        probe,
        grid,
        clean.residuals,
        t.zeros((1, *grid.shape), dtype=t.bool),
        with_gradients=False,
    )
    clean_corner = reference_corner_batch(
        model,
        blocks,
        probe,
        grid,
        clean.residuals,
        t.ones((1, *grid.shape), dtype=t.bool),
        with_gradients=False,
    )
    dirty_errors = (dirty_corner.candidate_logits[0] - dirty.candidate_logits[0]).abs()
    if not t.equal(dirty_corner.candidate_logits[0], dirty.candidate_logits[0]):
        raise RuntimeError("the all-dirty mask does not reproduce the dirty checkpoint forward")
    full_scope = bool(
        grid.token_indices == tuple(range(probe.input_ids.shape[1]))
        and grid.layers == tuple(range(len(blocks)))
    )
    clean_checkpoint_difference = (
        clean_corner.candidate_logits[0] - clean.candidate_logits[0]
    ).abs()
    clean_diff, clean_probability, clean_accuracy = _candidate_metrics(
        clean_corner.candidate_logits,
        probe.correct_choice_index,
    )
    clean_errors: t.Tensor | None = None
    if full_scope:
        runtime = _resolve_cached_runtime(model, blocks)
        device = next(model.parameters()).device
        parameter_dtype = next(model.parameters()).dtype
        clean_final_residual = (
            clean.residuals[-1, -1]
            .to(
                device=device,
                dtype=parameter_dtype,
            )
            .reshape(1, 1, -1)
        )
        with t.inference_mode():
            expected_logits = runtime.output_embeddings(
                cast(Any, runtime.core).norm(clean_final_residual)
            )[:, -1, :].index_select(1, probe.candidate_ids.to(device))
        expected_candidates = expected_logits.to(device="cpu", dtype=t.float32)[0]
        clean_errors = (clean_corner.candidate_logits[0] - expected_candidates).abs()
        if not t.equal(clean_corner.candidate_logits[0], expected_candidates):
            raise RuntimeError(
                "the full all-clean mask does not equal the recipient readout of the "
                "donor final residual"
            )
    return {
        "status": "passed",
        "comparison": "exact_candidate_logit_equality",
        "all_dirty_maximum_absolute_error": float(dirty_errors.max()),
        "all_clean_scope_is_complete": full_scope,
        "all_clean_exactly_verified": full_scope,
        "all_clean_reference": (
            "recipient_final_norm_and_unembedding_of_donor_final_residual" if full_scope else None
        ),
        "all_clean_maximum_absolute_error": (
            float(clean_errors.max()) if clean_errors is not None else None
        ),
        "all_clean_vs_donor_checkpoint_maximum_absolute_difference": float(
            clean_checkpoint_difference.max()
        ),
        "all_clean_intervention": {
            "logit_diff": float(clean_diff[0]),
            "correct_probability": float(clean_probability[0]),
            "accuracy": bool(clean_accuracy[0]),
        },
    }


@beartype
def run_fourier_circuit_pipeline(
    root: Path,
    config: FourierCircuitConfig,
    stages: tuple[int, ...],
) -> dict[str, object]:
    """Run selected resumable stages after the caller has enforced the GPU gate."""

    if (
        not stages
        or tuple(sorted(set(stages))) != stages
        or any(stage not in {0, 1, 2} for stage in stages)
    ):
        raise ValueError("Fourier stages must be a non-empty increasing subset of (0, 1, 2)")
    random.seed(config.spectrum.seed)
    t.manual_seed(config.spectrum.seed)
    t.cuda.manual_seed_all(config.spectrum.seed)
    t.use_deterministic_algorithms(True, warn_only=False)
    synthetic = run_synthetic_reference_gate()
    output_dir = fourier_output_dir(root, config)
    _write_or_validate_config(output_dir, config)
    synthetic_path = output_dir / "synthetic_reference_gate.json"
    if synthetic_path.is_file():
        if read_json(synthetic_path) != synthetic:
            raise RuntimeError("stored synthetic reference gate disagrees with current code")
    else:
        write_json(synthetic_path, synthetic)

    spec = get_model_spec(config.model.model_key)
    tokenizer = _load_tokenizer(spec)
    probe = build_circuit_probe(tokenizer, config)
    grid = build_site_grid(probe, spec, config.sites)
    clean = _capture_clean_checkpoint(root, config, probe, spec)
    model = _load_checkpoint_model(root, config, config.model.dirty_step)
    blocks = _resolve_blocks(model, spec)
    dirty = capture_checkpoint(model, blocks, probe)
    endpoint = _endpoint_contract(probe, clean, dirty)
    endpoint["prompt_contract"] = {
        "clean_dirty_prompt_identical": True,
        "rendered_prompt": probe.rendered_prompt,
        "correct_answer": probe.record.target,
        "correct_choice_index": probe.correct_choice_index,
    }
    write_json(output_dir / "endpoint_contract.json", endpoint)
    endpoint_corners = verify_endpoint_corner_contract(
        model,
        blocks,
        probe,
        grid,
        clean,
        dirty,
    )
    write_json(output_dir / "endpoint_corner_contract.json", endpoint_corners)
    all_clean_intervention = endpoint_corners.get("all_clean_intervention")
    if not isinstance(all_clean_intervention, dict) or not isinstance(
        all_clean_intervention.get("accuracy"), bool
    ):
        raise RuntimeError("endpoint corner contract lacks all-clean acquisition status")
    all_clean_accuracy = cast(bool, all_clean_intervention.get("accuracy"))
    if config.sufficiency.require_clean_argmax and not all_clean_accuracy:
        acquisition_gate = {
            "schema_version": FOURIER_SCHEMA_VERSION,
            "stage": "endpoint_acquisition_gate",
            "status": "clean_behavior_not_acquired",
            "terminal": True,
            "reason": "all-clean residual intervention fails the required clean argmax",
            "clean_checkpoint": endpoint["clean"],
            "dirty_checkpoint": endpoint["dirty"],
            "all_clean_intervention": all_clean_intervention,
            "site_grid": _site_grid_payload(grid, probe),
        }
        write_json(output_dir / "endpoint_acquisition_gate.json", acquisition_gate)
        _release_model(model)
        return {
            "status": "clean_behavior_not_acquired",
            "output_dir": str(output_dir),
            "endpoint_acquisition_gate": acquisition_gate,
        }

    reference = load_known_site_reference(root, config, grid)
    harness = verify_known_site_harness(
        model,
        blocks,
        probe,
        grid,
        clean.residuals,
        reference,
        config,
    )
    write_json(output_dir / "harness_check.json", harness)
    cache_comparison = compare_cache_execution_semantics(
        output_dir,
        model,
        blocks,
        probe,
        grid,
        clean.residuals,
    )
    inference_parity = verify_inference_mode_parity(
        output_dir,
        model,
        blocks,
        probe,
        grid,
        clean.residuals,
    )

    singletons = (
        run_exhaustive_singleton_sweep(
            root,
            output_dir,
            model,
            blocks,
            probe,
            grid,
            clean.residuals,
            config,
        )
        if 0 in stages
        else _validated_singleton_artifact(output_dir)
    )
    residual_site_space = build_active_site_space(
        grid,
        _verified_singleton_sites(singletons),
    )

    stage_zero = (
        run_density_sweep(
            output_dir,
            model,
            blocks,
            probe,
            grid,
            clean.residuals,
            config,
            function_space="unrestricted",
        )
        if 0 in stages
        else _validated_stage_artifact(
            output_dir / "stage_0_density.json",
            output_dir / "stage_0_density_samples.pt",
            stage=0,
            statuses=("transition_found", "flat_stop"),
            sidecar_field="sample_sidecar",
        )
    )
    residual_stage_zero = (
        run_density_sweep(
            output_dir,
            model,
            blocks,
            probe,
            grid,
            clean.residuals,
            config,
            function_space="singleton_vetoed_residual",
            site_space=residual_site_space,
        )
        if 0 in stages
        else _validated_stage_artifact(
            output_dir / "stage_0_residual_density.json",
            output_dir / "stage_0_residual_density_samples.pt",
            stage=0,
            statuses=("transition_found", "flat_stop"),
            sidecar_field="sample_sidecar",
        )
    )
    if stage_zero.get("status") == "flat_stop" or residual_stage_zero.get("status") == "flat_stop":
        _release_model(model)
        return {
            "status": (
                "unrestricted_flat_stop"
                if stage_zero.get("status") == "flat_stop"
                else "residual_flat_stop"
            ),
            "output_dir": str(output_dir),
            "exhaustive_singletons": singletons,
            "stage_0": stage_zero,
            "residual_stage_0": residual_stage_zero,
            "cache_semantics_comparison": cache_comparison,
            "inference_mode_parity": inference_parity,
        }

    if 1 in stages:
        stage_one = run_spectrum_estimation(
            output_dir,
            residual_stage_zero,
            model,
            blocks,
            probe,
            grid,
            clean.residuals,
            config,
            residual_site_space,
        )
    elif 2 in stages:
        stage_one = _validated_stage_artifact(
            output_dir / "stage_1_spectrum.json",
            output_dir / "stage_1_samples.pt",
            stage=1,
            statuses=(
                "complete",
                "complete_density_unstable",
                "complete_no_heavy_coefficients",
            ),
            sidecar_field="sample_sidecar",
        )
    else:
        stage_one = None
    if isinstance(stage_one, dict) and stage_one.get("status") == "complete_no_heavy_coefficients":
        _release_model(model)
        return {
            "status": "no_heavy_coefficients_stop",
            "output_dir": str(output_dir),
            "exhaustive_singletons": singletons,
            "stage_0": stage_zero,
            "residual_stage_0": residual_stage_zero,
            "stage_1": stage_one,
            "stage_2": None,
            "cache_semantics_comparison": cache_comparison,
            "inference_mode_parity": inference_parity,
        }
    stage_two = (
        run_causal_verification(
            output_dir,
            residual_stage_zero,
            cast(dict[str, object], stage_one),
            model,
            blocks,
            probe,
            grid,
            clean.residuals,
            config,
            singletons,
            residual_site_space,
        )
        if 2 in stages
        else None
    )
    _release_model(model)
    return {
        "status": (
            "no_verified_multisite_minsets"
            if isinstance(stage_two, dict)
            and stage_two.get("status")
            in {"no_verified_multisite_minsets", "no_higher_order_hypotheses"}
            else "complete"
        ),
        "output_dir": str(output_dir),
        "exhaustive_singletons": singletons,
        "stage_0": stage_zero,
        "residual_stage_0": residual_stage_zero,
        "stage_1": stage_one,
        "stage_2": stage_two,
        "cache_semantics_comparison": cache_comparison,
        "inference_mode_parity": inference_parity,
    }


__all__ = [
    "CircuitProbe",
    "CheckpointCapture",
    "CornerBatchResult",
    "KnownSiteReference",
    "build_active_site_space",
    "build_circuit_probe",
    "build_site_grid",
    "capture_checkpoint",
    "cached_corner_batch",
    "compare_cache_execution_semantics",
    "evaluate_masks_in_batches",
    "fourier_output_dir",
    "logical_artifact_path",
    "load_known_site_reference",
    "longest_common_site_prefix",
    "profile_cached_corner_runtime",
    "reference_alpha_batch",
    "reference_corner_batch",
    "run_causal_verification",
    "run_density_sweep",
    "run_exhaustive_singleton_sweep",
    "run_fourier_circuit_pipeline",
    "run_spectrum_estimation",
    "token_major_trie_order",
    "verify_known_site_harness",
    "verify_endpoint_corner_contract",
    "verify_inference_mode_parity",
]
