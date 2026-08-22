"""Batch-one runtime for cross-checkpoint answer-location swap minsets."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch as t
from beartype import beartype
from jaxtyping import Bool, Float, Int64, jaxtyped

from oocr_training_dynamics.answer_lookup import ANSWER_LABELS, ChoiceTerminatorSite
from oocr_training_dynamics.artifacts import read_json, run_dir, sha256_file, write_json
from oocr_training_dynamics.contracts import PatchingInterface, RunKey, TrainingCondition
from oocr_training_dynamics.data import FUNCTION_BY_ID, ReflectionRecord
from oocr_training_dynamics.fourier_circuits import (
    DensityPoint,
    DensitySweepConfig,
    LogitDiff,
    density_curve_is_flat,
    select_transition_density,
)
from oocr_training_dynamics.models import ModelKey, ModelSpec, get_model_spec
from oocr_training_dynamics.runtime_answer_lookup import _choice_sites
from oocr_training_dynamics.runtime_fourier_circuits import (
    PYALVT_CHOICE_FUNCTION_IDS,
    PYALVT_SYSTEM_PROMPT,
    PYALVT_USER_PROMPT,
)
from oocr_training_dynamics.runtime_models import (
    load_processor,
    resolve_decoder_blocks,
    tokenizer_for,
)
from oocr_training_dynamics.runtime_patching import (
    PatchTarget,
    PromptPatchView,
    _candidate_ids,
    _hidden_tensor,
    _input_hidden,
    _load_checkpoint_model,
    _prompt_patch_view,
    _release_model,
    _resolve_patch_targets,
    _selected_records,
)
from oocr_training_dynamics.switched_answer_minsets import (
    SWITCHED_ANSWER_CORRECT_CHOICE_INDEX,
    SWITCHED_ANSWER_SCHEMA_VERSION,
    LayerSwapSite,
    LayerSwapSiteSet,
    SwapSubsetMetric,
    SwitchedAnswerMinsetConfig,
    VerifiedSwapMinset,
    layer_supports,
    masks_for_layer_supports,
    sample_layer_patch_masks,
    support_is_safely_blocked,
    verified_minsets_from_metrics,
)

TokenRow = Int64[t.Tensor, "1 sequence"]
AttentionRow = Bool[t.Tensor, "1 sequence"]
CandidateIds = Int64[t.Tensor, "choice"]
CandidateLogits = Float[t.Tensor, "sample choice"]
LayerMaskBatch = Bool[t.Tensor, "sample layer"]
LayerMask = Bool[t.Tensor, "layer"]
ChoiceActivationBank = Float[t.Tensor, "layer choice hidden"]
HiddenBatch = Float[t.Tensor, "1 sequence hidden"]
ReplacementPair = Float[t.Tensor, "two hidden"]
MetricVector = Float[t.Tensor, "sample"]

SCIENTIFIC_PARITY_ATOL = 1.0e-6


@beartype
@dataclass(frozen=True)
class SwitchedAnswerProbe:
    record: ReflectionRecord
    view: PromptPatchView
    candidate_ids: t.Tensor
    terminator_sites: tuple[ChoiceTerminatorSite, ...]

    def __post_init__(self) -> None:
        if self.candidate_ids.dtype is not t.int64 or self.candidate_ids.shape != (5,):
            raise ValueError("switched-answer probe requires five int64 candidate IDs")
        if len(self.terminator_sites) != 5:
            raise ValueError("switched-answer probe requires five option terminators")
        if self.record.choice_function_ids.index(self.record.function_id) != (
            SWITCHED_ANSWER_CORRECT_CHOICE_INDEX
        ):
            raise RuntimeError("switched-answer probe correct choice must be C")


@beartype
@dataclass(frozen=True)
class SwapBatchResult:
    candidate_logits: t.Tensor
    logit_diffs: t.Tensor
    target_probabilities: t.Tensor
    target_accuracies: t.Tensor

    def __post_init__(self) -> None:
        sample_count = self.candidate_logits.shape[0]
        if self.candidate_logits.shape != (sample_count, 5):
            raise ValueError("swap results require [sample, five choices] logits")
        if any(
            values.shape != (sample_count,)
            for values in (
                self.logit_diffs,
                self.target_probabilities,
                self.target_accuracies,
            )
        ):
            raise ValueError("swap metrics require one value per sample")
        if any(
            not bool(t.isfinite(values).all())
            for values in (
                self.candidate_logits,
                self.logit_diffs,
                self.target_probabilities,
                self.target_accuracies,
            )
        ):
            raise ValueError("swap results must be finite")


@beartype
def _run_key(config: SwitchedAnswerMinsetConfig) -> RunKey:
    return RunKey(
        config.model.model_key,
        TrainingCondition(config.model.condition),
        config.model.seed,
    )


@beartype
def switched_answer_output_dir(root: Path, config: SwitchedAnswerMinsetConfig) -> Path:
    destination = ANSWER_LABELS[config.task.destination_choice_index].lower()
    return (
        run_dir(root, _run_key(config))
        / "answer_lookup_checkpoint_transfer_minsets"
        / config.task.function_id
        / f"donor_{config.model.donor_step:06d}_recipient_{config.model.recipient_step:06d}"
        / "target_correct_recovery"
        / config.task.interface
        / f"destination_{destination}"
    )


@beartype
def _config_payload(config: SwitchedAnswerMinsetConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["artifact_root"] = str(config.artifact_root)
    return cast(dict[str, object], json.loads(json.dumps(payload, allow_nan=False)))


@beartype
def _write_or_validate_config(output_dir: Path, config: SwitchedAnswerMinsetConfig) -> None:
    path = output_dir / "config.json"
    payload = {
        "schema_version": SWITCHED_ANSWER_SCHEMA_VERSION,
        "config": _config_payload(config),
    }
    if path.is_file():
        if read_json(path) != payload:
            raise RuntimeError(f"switched-answer output contains a different config: {path}")
        return
    write_json(path, payload)


@jaxtyped(typechecker=beartype)
def _write_tensor_sidecar(path: Path, payload: dict[str, t.Tensor]) -> None:
    if not payload or any(not isinstance(value, t.Tensor) for value in payload.values()):
        raise TypeError("tensor sidecar must be a non-empty tensor mapping")
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
def build_switched_answer_probe(
    processor: Any,
    config: SwitchedAnswerMinsetConfig,
    *,
    device: str = "cuda",
) -> SwitchedAnswerProbe:
    selected = tuple(
        record
        for record in _selected_records(config.model.seed)
        if record.function_id == config.task.function_id
    )
    if len(selected) != 1:
        raise RuntimeError("expected exactly one registered pyalvt code reflection record")
    record = selected[0]
    expected_messages = (
        ("system", PYALVT_SYSTEM_PROMPT),
        ("user", PYALVT_USER_PROMPT),
        ("assistant", "C"),
    )
    if (
        tuple((message.role, message.content) for message in record.messages) != expected_messages
        or record.choice_function_ids != PYALVT_CHOICE_FUNCTION_IDS
        or record.target != "C"
    ):
        raise RuntimeError("switched-answer probe does not match the exact pyalvt prompt")
    view = _prompt_patch_view(
        processor,
        record,
        record.messages,
        FUNCTION_BY_ID[record.function_id].alias,
        stop_at_sequence_start=True,
        device=device,
    )
    tokenizer = tokenizer_for(processor)
    sites = _choice_sites(view, tokenizer)
    return SwitchedAnswerProbe(
        record,
        view,
        _candidate_ids(processor, record, device=device),
        sites,
    )


@beartype
def audit_switched_answer_tokenization(
    root: Path,
    config: SwitchedAnswerMinsetConfig,
) -> Path:
    """Persist the identical-prompt and paired A-E line-terminator coordinates on CPU."""

    spec = get_model_spec(ModelKey(config.model.model_key))
    processor = load_processor(spec)
    probe = build_switched_answer_probe(processor, config, device="cpu")
    output = root / "artifacts/plans/switched_answer_minsets/tokenization_audit_v2.json"
    payload = {
        "schema_version": SWITCHED_ANSWER_SCHEMA_VERSION,
        "model": {"id": spec.model_id, "revision": spec.revision},
        "function_id": config.task.function_id,
        "function_alias": FUNCTION_BY_ID[config.task.function_id].alias,
        "donor_step": config.model.donor_step,
        "recipient_step": config.model.recipient_step,
        "source_recipient_prompt_identity": "exact rendered text and token IDs",
        "prompt_audit": _prompt_audit(probe),
        "paired_swap_destinations": [
            {
                "destination_choice_index": destination,
                "destination_choice_label": ANSWER_LABELS[destination],
                "donor_source_choice_indices": [
                    destination,
                    config.task.correct_choice_index,
                ],
                "recipient_choice_indices": [
                    config.task.correct_choice_index,
                    destination,
                ],
            }
            for destination in range(5)
            if destination != config.task.correct_choice_index
        ],
    }
    if output.is_file() and read_json(output) != payload:
        raise RuntimeError("switched-answer tokenization audit changed")
    write_json(output, payload)
    return output


@jaxtyped(typechecker=beartype)
def _forward_candidate_logits(
    model: t.nn.Module,
    input_ids: TokenRow,
    attention_mask: AttentionRow,
    candidate_ids: CandidateIds,
    *,
    execution: str,
) -> CandidateLogits:
    if execution not in {"inference_mode", "no_grad_reference"}:
        raise ValueError("execution must select inference_mode or no_grad_reference")
    context = t.inference_mode() if execution == "inference_mode" else t.no_grad()
    with context:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    if output.logits.shape[0] != 1 or output.logits.shape[1] != 1:
        raise RuntimeError("scientific forward must return one final-token logit row")
    return (
        output.logits[:, -1, :]
        .index_select(1, candidate_ids.to(output.logits.device))
        .detach()
        .to(device="cpu", dtype=t.float32)
    )


@jaxtyped(typechecker=beartype)
def _candidate_metrics(
    candidate_logits: CandidateLogits,
    target_choice_index: int,
) -> tuple[MetricVector, MetricVector, MetricVector]:
    if candidate_logits.ndim != 2 or candidate_logits.shape[1] != 5:
        raise ValueError("candidate metrics require five logits per sample")
    if not 0 <= target_choice_index < 5:
        raise ValueError("target choice must identify A-E")
    logits = candidate_logits.to(dtype=t.float32)
    other = tuple(index for index in range(5) if index != target_choice_index)
    logit_diffs = logits[:, target_choice_index] - t.logsumexp(logits[:, other], dim=1)
    probabilities = t.softmax(logits, dim=1)[:, target_choice_index]
    accuracies = logits.argmax(dim=1).eq(target_choice_index).to(dtype=t.float32)
    return logit_diffs, probabilities, accuracies


@jaxtyped(typechecker=beartype)
def _replace_two_positions(
    hidden: HiddenBatch,
    replacements: ReplacementPair,
    recipient_positions: tuple[int, int],
) -> HiddenBatch:
    if hidden.shape[0] != 1 or replacements.shape != (2, hidden.shape[2]):
        raise ValueError("paired swap requires one prompt and exactly two hidden vectors")
    if len(set(recipient_positions)) != 2 or any(
        position < 0 or position >= hidden.shape[1] for position in recipient_positions
    ):
        raise ValueError("paired recipient positions must be distinct and in bounds")
    patched = hidden.clone()
    columns = t.tensor(recipient_positions, dtype=t.int64, device=hidden.device)
    patched[0, columns, :] = replacements.to(device=hidden.device, dtype=hidden.dtype)
    return patched


@jaxtyped(typechecker=beartype)
def _evaluate_one_mask(
    model: t.nn.Module,
    targets: tuple[PatchTarget, ...],
    probe: SwitchedAnswerProbe,
    donor_choice_bank: ChoiceActivationBank,
    mask: LayerMask,
    destination_choice_index: int,
    *,
    execution: str,
) -> CandidateLogits:
    if mask.dtype is not t.bool or mask.shape != (len(targets),):
        raise ValueError("one swap mask must contain one Boolean per decoder layer")
    if donor_choice_bank.shape[0] != len(targets) or donor_choice_bank.shape[1] != 5:
        raise ValueError("donor activation bank must cover every layer and choice")
    correct = SWITCHED_ANSWER_CORRECT_CHOICE_INDEX
    correct_position = probe.terminator_sites[correct].token_index
    wrong_position = probe.terminator_sites[destination_choice_index].token_index
    recipient_positions = (correct_position, wrong_position)
    handles: list[Any] = []
    for layer in mask.nonzero(as_tuple=False).flatten().tolist():
        target = targets[layer]
        replacements = donor_choice_bank[layer, [destination_choice_index, correct], :]
        if target.capture_input:

            def input_hook(
                _module: t.nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                vectors: t.Tensor = replacements,
            ) -> tuple[tuple[Any, ...], dict[str, Any]]:
                patched = _replace_two_positions(
                    _input_hidden(args, kwargs),
                    vectors,
                    recipient_positions,
                )
                if args and isinstance(args[0], t.Tensor):
                    return (patched, *args[1:]), kwargs
                updated = dict(kwargs)
                updated["hidden_states"] = patched
                return args, updated

            handles.append(target.module.register_forward_pre_hook(input_hook, with_kwargs=True))
        else:

            def output_hook(
                _module: t.nn.Module,
                _args: tuple[Any, ...],
                output: Any,
                *,
                vectors: t.Tensor = replacements,
            ) -> Any:
                patched = _replace_two_positions(
                    _hidden_tensor(output),
                    vectors,
                    recipient_positions,
                )
                if isinstance(output, tuple):
                    return (patched, *output[1:])
                return patched

            handles.append(target.module.register_forward_hook(output_hook))
    try:
        return _forward_candidate_logits(
            model,
            probe.view.input_ids,
            probe.view.attention_mask,
            probe.candidate_ids,
            execution=execution,
        )
    finally:
        for handle in handles:
            handle.remove()


@jaxtyped(typechecker=beartype)
def evaluate_swap_masks(
    model: t.nn.Module,
    targets: tuple[PatchTarget, ...],
    probe: SwitchedAnswerProbe,
    donor_choice_bank: ChoiceActivationBank,
    masks: LayerMaskBatch,
    destination_choice_index: int,
    *,
    execution: str = "inference_mode",
) -> SwapBatchResult:
    """Evaluate masks sequentially so every scientific forward has batch size one."""

    if masks.dtype is not t.bool or masks.ndim != 2 or masks.shape[1] != len(targets):
        raise ValueError("swap masks must have [sample, decoder layer] Boolean shape")
    if masks.shape[0] <= 0:
        raise ValueError("swap evaluation requires at least one mask")
    rows = tuple(
        _evaluate_one_mask(
            model,
            targets,
            probe,
            donor_choice_bank,
            mask,
            destination_choice_index,
            execution=execution,
        )
        for mask in masks
    )
    logits = t.cat(rows, dim=0)
    logit_diffs, probabilities, accuracies = _candidate_metrics(
        logits,
        SWITCHED_ANSWER_CORRECT_CHOICE_INDEX,
    )
    return SwapBatchResult(logits, logit_diffs, probabilities, accuracies)


@jaxtyped(typechecker=beartype)
def capture_donor_choice_bank(
    model: t.nn.Module,
    targets: tuple[PatchTarget, ...],
    probe: SwitchedAnswerProbe,
) -> tuple[ChoiceActivationBank, CandidateLogits]:
    captured: list[t.Tensor | None] = [None] * len(targets)
    handles: list[Any] = []
    for layer, target in enumerate(targets):
        if target.capture_input:

            def input_hook(
                _module: t.nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                index: int = layer,
            ) -> None:
                captured[index] = _input_hidden(args, kwargs)[0].detach().cpu().clone()

            handles.append(target.module.register_forward_pre_hook(input_hook, with_kwargs=True))
        else:

            def output_hook(
                _module: t.nn.Module,
                _args: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer,
            ) -> None:
                captured[index] = _hidden_tensor(output)[0].detach().cpu().clone()

            handles.append(target.module.register_forward_hook(output_hook))
    try:
        logits = _forward_candidate_logits(
            model,
            probe.view.input_ids,
            probe.view.attention_mask,
            probe.candidate_ids,
            execution="inference_mode",
        )
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in captured):
        raise RuntimeError("not every donor layer produced an activation")
    token_indices = [site.token_index for site in probe.terminator_sites]
    bank = t.stack(
        [cast(t.Tensor, activation)[token_indices, :] for activation in captured],
        dim=0,
    )
    return bank, logits


@beartype
def _shared_capture_dir(root: Path, config: SwitchedAnswerMinsetConfig) -> Path:
    return (
        run_dir(root, _run_key(config))
        / "answer_lookup_checkpoint_transfer_minsets"
        / config.task.function_id
        / f"donor_{config.model.donor_step:06d}_recipient_{config.model.recipient_step:06d}"
        / "shared_donor_capture"
        / config.task.interface
    )


@beartype
def _prompt_audit(probe: SwitchedAnswerProbe) -> dict[str, object]:
    return {
        "rendered_prompt": probe.view.rendered_prompt,
        "token_ids": list(probe.view.token_ids),
        "token_labels": list(probe.view.token_labels),
        "terminator_sites": [asdict(site) for site in probe.terminator_sites],
        "source_recipient_prompt_identity": "exact rendered text and token IDs",
    }


@beartype
def load_or_capture_donor_bank(
    root: Path,
    config: SwitchedAnswerMinsetConfig,
    probe: SwitchedAnswerProbe,
    spec: ModelSpec,
) -> ChoiceActivationBank:
    directory = _shared_capture_dir(root, config)
    json_path = directory / "donor_capture.json"
    tensor_path = directory / "donor_capture.pt"
    if json_path.is_file() != tensor_path.is_file():
        raise RuntimeError("donor capture JSON and tensor sidecar must coexist")
    if json_path.is_file():
        payload = read_json(json_path)
        if not isinstance(payload, dict) or payload.get("status") != "complete":
            raise RuntimeError("stored donor capture is not complete")
        if payload.get("prompt_audit") != _prompt_audit(probe):
            raise RuntimeError("stored donor capture prompt/token audit changed")
        if payload.get("tensor_sha256") != sha256_file(tensor_path):
            raise RuntimeError("stored donor capture digest mismatch")
        tensors = _load_tensor_sidecar(tensor_path)
        bank = tensors.get("choice_activations")
        if not isinstance(bank, t.Tensor) or bank.shape != (
            config.layer_count,
            5,
            spec.hidden_size,
        ):
            raise RuntimeError("stored donor choice bank has the wrong shape")
        return bank

    model = _load_checkpoint_model(root, _run_key(config), spec, config.model.donor_step)
    try:
        blocks = resolve_decoder_blocks(model, spec)
        targets = _resolve_patch_targets(blocks, PatchingInterface(config.task.interface))
        bank, logits = capture_donor_choice_bank(model, targets, probe)
    finally:
        _release_model(model)
    _write_tensor_sidecar(
        tensor_path,
        {"choice_activations": bank, "candidate_logits": logits},
    )
    write_json(
        json_path,
        {
            "schema_version": SWITCHED_ANSWER_SCHEMA_VERSION,
            "status": "complete",
            "model": {"id": spec.model_id, "revision": spec.revision},
            "donor_step": config.model.donor_step,
            "interface": config.task.interface,
            "prompt_audit": _prompt_audit(probe),
            "candidate_logits": [float(value) for value in logits[0].tolist()],
            "tensor_sidecar": tensor_path.name,
            "tensor_sha256": sha256_file(tensor_path),
        },
    )
    return bank


@beartype
def _metric_payload(result: SwapBatchResult, row: int) -> dict[str, object]:
    return {
        "candidate_logits": [float(value) for value in result.candidate_logits[row].tolist()],
        "raw_logit_diff": float(result.logit_diffs[row]),
        "target_probability": float(result.target_probabilities[row]),
        "target_argmax": bool(result.target_accuracies[row]),
    }


@beartype
def run_endpoint_gate(
    output_dir: Path,
    config: SwitchedAnswerMinsetConfig,
    model: t.nn.Module,
    targets: tuple[PatchTarget, ...],
    probe: SwitchedAnswerProbe,
    donor_choice_bank: ChoiceActivationBank,
) -> dict[str, object]:
    path = output_dir / "endpoint_gate.json"
    if path.is_file():
        payload = read_json(path)
        if not isinstance(payload, dict) or payload.get("status") not in {"passed", "failed"}:
            raise RuntimeError("stored endpoint gate has an invalid status")
        return cast(dict[str, object], payload)
    masks = t.stack(
        (
            t.zeros(config.layer_count, dtype=t.bool),
            t.ones(config.layer_count, dtype=t.bool),
        )
    )
    inference = evaluate_swap_masks(
        model,
        targets,
        probe,
        donor_choice_bank,
        masks,
        config.task.destination_choice_index,
        execution="inference_mode",
    )
    reference = evaluate_swap_masks(
        model,
        targets,
        probe,
        donor_choice_bank,
        masks,
        config.task.destination_choice_index,
        execution="no_grad_reference",
    )
    logit_error = float((inference.candidate_logits - reference.candidate_logits).abs().max())
    probability_error = float(
        (
            inference.target_probabilities - reference.target_probabilities
        ).abs().max()
    )
    baseline_logits = _forward_candidate_logits(
        model,
        probe.view.input_ids,
        probe.view.attention_mask,
        probe.candidate_ids,
        execution="inference_mode",
    )
    zero_hook_error = float((baseline_logits - inference.candidate_logits[:1]).abs().max())
    dirty_probability = float(inference.target_probabilities[0])
    clean_probability = float(inference.target_probabilities[1])
    threshold = clean_probability - config.search.absolute_probability_tolerance
    status = (
        "passed"
        if max(logit_error, probability_error, zero_hook_error) <= SCIENTIFIC_PARITY_ATOL
        and bool(inference.target_accuracies[1])
        and threshold > dirty_probability
        else "failed"
    )
    payload = {
        "schema_version": SWITCHED_ANSWER_SCHEMA_VERSION,
        "status": status,
        "destination_choice_index": config.task.destination_choice_index,
        "destination_choice_label": ANSWER_LABELS[config.task.destination_choice_index],
        "target_choice_index": config.task.target_choice_index,
        "target_choice_label": ANSWER_LABELS[config.task.target_choice_index],
        "paired_swap": {
            "donor_source_choice_indices": [
                config.task.destination_choice_index,
                config.task.correct_choice_index,
            ],
            "recipient_choice_indices": [
                config.task.correct_choice_index,
                config.task.destination_choice_index,
            ],
        },
        "all_dirty": _metric_payload(inference, 0),
        "all_clean_swap": _metric_payload(inference, 1),
        "sufficiency_probability_threshold": threshold,
        "inference_no_grad_max_logit_error": logit_error,
        "inference_no_grad_max_probability_error": probability_error,
        "zero_mask_unpatched_max_logit_error": zero_hook_error,
        "parity_tolerance": SCIENTIFIC_PARITY_ATOL,
    }
    write_json(path, payload)
    return payload


@beartype
def _density_point(density: float, result: SwapBatchResult) -> DensityPoint:
    probabilities = result.target_probabilities.to(dtype=t.float64)
    logit_diffs = result.logit_diffs.to(dtype=t.float64)
    return DensityPoint(
        density=cast(Any, density),
        sample_count=probabilities.numel(),
        mean_correct_probability=float(probabilities.mean()),
        correct_probability_variance=(
            float(probabilities.var(unbiased=True)) if probabilities.numel() > 1 else 0.0
        ),
        accuracy=float(result.target_accuracies.to(dtype=t.float64).mean()),
        mean_logit_diff=LogitDiff.parse(float(logit_diffs.mean())),
        logit_diff_variance=(
            float(logit_diffs.var(unbiased=True)) if logit_diffs.numel() > 1 else 0.0
        ),
    )


@beartype
def run_density_sweep(
    output_dir: Path,
    config: SwitchedAnswerMinsetConfig,
    model: t.nn.Module,
    targets: tuple[PatchTarget, ...],
    probe: SwitchedAnswerProbe,
    donor_choice_bank: ChoiceActivationBank,
) -> dict[str, object]:
    json_path = output_dir / "density_sweep.json"
    tensor_path = output_dir / "density_sweep.pt"
    if json_path.is_file() != tensor_path.is_file():
        raise RuntimeError("density JSON and tensor sidecar must coexist")
    if json_path.is_file():
        payload = read_json(json_path)
        if not isinstance(payload, dict) or payload.get("tensor_sha256") != sha256_file(tensor_path):
            raise RuntimeError("stored density sweep digest mismatch")
        return cast(dict[str, object], payload)
    generator = t.Generator(device="cpu").manual_seed(config.density.seed)
    masks_by_density: list[t.Tensor] = []
    logits_by_density: list[t.Tensor] = []
    points: list[DensityPoint] = []
    for density in config.density.density_grid:
        sample_count = 1 if float(density) in {0.0, 1.0} else config.density.masks_per_density
        masks = sample_layer_patch_masks(
            sample_count,
            config.layer_count,
            density,
            generator,
        )
        result = evaluate_swap_masks(
            model,
            targets,
            probe,
            donor_choice_bank,
            masks,
            config.task.destination_choice_index,
        )
        masks_by_density.append(masks)
        logits_by_density.append(result.candidate_logits)
        points.append(_density_point(float(density), result))
    density_config = DensitySweepConfig(
        density_grid=config.density.density_grid,
        masks_per_density=config.density.masks_per_density,
        flat_probability_span=config.density.flat_probability_span,
        flat_logit_diff_span=config.density.flat_logit_diff_span,
        minimum_logit_diff_variance=config.density.minimum_logit_diff_variance,
        seed=config.density.seed,
    )
    flat = density_curve_is_flat(tuple(points), density_config)
    selected = None if flat else select_transition_density(tuple(points), density_config)
    _write_tensor_sidecar(
        tensor_path,
        {
            "masks": t.cat(masks_by_density, dim=0),
            "candidate_logits": t.cat(logits_by_density, dim=0),
        },
    )
    payload = {
        "schema_version": SWITCHED_ANSWER_SCHEMA_VERSION,
        "status": "flat_stop" if flat else "complete",
        "selected_density": None if selected is None else float(selected),
        "points": [
            {
                "density": float(point.density),
                "sample_count": point.sample_count,
                "mean_target_probability": point.mean_correct_probability,
                "target_probability_variance": point.correct_probability_variance,
                "target_accuracy": point.accuracy,
                "mean_raw_logit_diff": float(point.mean_logit_diff),
                "raw_logit_diff_variance": point.logit_diff_variance,
            }
            for point in points
        ],
        "tensor_sidecar": tensor_path.name,
        "tensor_sha256": sha256_file(tensor_path),
    }
    write_json(json_path, payload)
    return payload


@jaxtyped(typechecker=beartype)
def _result_to_metrics(
    supports: tuple[LayerSwapSiteSet, ...],
    result: SwapBatchResult,
) -> tuple[SwapSubsetMetric, ...]:
    if len(supports) != result.candidate_logits.shape[0]:
        raise ValueError("supports and result rows must align")
    return tuple(
        SwapSubsetMetric(
            sites=support,
            candidate_logits=cast(
                tuple[float, float, float, float, float],
                tuple(float(value) for value in result.candidate_logits[row].tolist()),
            ),
            target_probability=float(result.target_probabilities[row]),
            raw_logit_diff=float(result.logit_diffs[row]),
            target_argmax=bool(result.target_accuracies[row]),
        )
        for row, support in enumerate(supports)
    )


@beartype
def _metrics_from_sidecar(path: Path) -> tuple[SwapSubsetMetric, ...]:
    tensors = _load_tensor_sidecar(path)
    required = ("masks", "candidate_logits", "logit_diffs", "probabilities", "accuracies")
    if any(not isinstance(tensors.get(name), t.Tensor) for name in required):
        raise TypeError(f"search sidecar lacks required tensors: {path}")
    masks = tensors["masks"]
    result = SwapBatchResult(
        tensors["candidate_logits"],
        tensors["logit_diffs"],
        tensors["probabilities"],
        tensors["accuracies"],
    )
    supports = tuple(
        tuple(LayerSwapSite(int(layer)) for layer in mask.nonzero(as_tuple=False).flatten().tolist())
        for mask in masks
    )
    return _result_to_metrics(supports, result)


@beartype
def _load_search_metrics(output_dir: Path) -> dict[LayerSwapSiteSet, SwapSubsetMetric]:
    endpoint = read_json(output_dir / "endpoint_gate.json")
    if not isinstance(endpoint, dict):
        raise TypeError("endpoint artifact must be an object")
    dirty = endpoint.get("all_dirty")
    if not isinstance(dirty, dict):
        raise TypeError("endpoint artifact lacks the all-dirty corner")
    dirty_map = cast(dict[str, object], dirty)

    def endpoint_metric(sites: LayerSwapSiteSet, payload: dict[str, object]) -> SwapSubsetMetric:
        logits = payload.get("candidate_logits")
        probability = payload.get("target_probability")
        logit_diff = payload.get("raw_logit_diff")
        argmax = payload.get("target_argmax")
        if (
            not isinstance(logits, list)
            or len(logits) != 5
            or any(not isinstance(value, int | float) for value in logits)
            or not isinstance(probability, int | float)
            or not isinstance(logit_diff, int | float)
            or not isinstance(argmax, bool)
        ):
            raise TypeError("endpoint metric is malformed")
        numeric_logits = cast(list[int | float], logits)
        return SwapSubsetMetric(
            sites,
            cast(
                tuple[float, float, float, float, float],
                tuple(float(value) for value in numeric_logits),
            ),
            float(probability),
            float(logit_diff),
            argmax,
        )

    # The all-clean 32-site endpoint defines the threshold but is not part of the
    # incrementally sealed subset census. Including it here would require its entire
    # 2^32 proper-subset powerset before it could be assessed as a minset.
    metrics = {(): endpoint_metric((), dirty_map)}
    for sidecar in sorted((output_dir / "search").glob("order_*/shard_*.pt")):
        json_path = sidecar.with_suffix(".json")
        if not json_path.is_file():
            raise RuntimeError(f"search shard lacks JSON manifest: {sidecar}")
        payload = read_json(json_path)
        if not isinstance(payload, dict) or payload.get("tensor_sha256") != sha256_file(sidecar):
            raise RuntimeError(f"search shard digest mismatch: {sidecar}")
        for metric in _metrics_from_sidecar(sidecar):
            previous = metrics.get(metric.sites)
            if previous is not None and previous != metric:
                raise RuntimeError(f"duplicate subset metrics disagree: {metric.sites}")
            metrics[metric.sites] = metric
    return metrics


@beartype
def _minset_payload(minset: VerifiedSwapMinset) -> dict[str, object]:
    return {
        "layers": [site.layer for site in minset.sites],
        "size": len(minset.sites),
        "target_probability": minset.target_probability,
        "raw_logit_diff": minset.raw_logit_diff,
        "sufficiency_margin": minset.sufficiency_margin,
        "maximum_proper_subset_probability": minset.maximum_proper_subset_probability,
        "maximum_proper_subset_layers": [site.layer for site in minset.maximum_proper_subset],
    }


@beartype
def run_exhaustive_search(
    output_dir: Path,
    config: SwitchedAnswerMinsetConfig,
    model: t.nn.Module,
    targets: tuple[PatchTarget, ...],
    probe: SwitchedAnswerProbe,
    donor_choice_bank: ChoiceActivationBank,
) -> dict[str, object]:
    endpoint = read_json(output_dir / "endpoint_gate.json")
    density = read_json(output_dir / "density_sweep.json")
    if not isinstance(endpoint, dict) or not isinstance(density, dict):
        raise TypeError("search requires endpoint and density objects")
    if endpoint.get("status") != "passed" or density.get("status") != "complete":
        return {
            "status": "not_run_failed_gate",
            "endpoint_status": endpoint.get("status"),
            "density_status": density.get("status"),
        }
    clean = endpoint.get("all_clean_swap")
    if not isinstance(clean, dict):
        raise TypeError("endpoint lacks all-clean target probability")
    clean_map = cast(dict[str, object], clean)
    clean_probability_value = clean_map.get("target_probability")
    if not isinstance(clean_probability_value, int | float):
        raise TypeError("endpoint lacks all-clean target probability")
    all_clean_probability = float(clean_probability_value)
    search_dir = output_dir / "search"
    search_dir.mkdir(parents=True, exist_ok=True)
    for order in range(1, config.search.maximum_order + 1):
        order_dir = search_dir / f"order_{order}"
        order_dir.mkdir(parents=True, exist_ok=True)
        metrics = _load_search_metrics(output_dir)
        supports = tuple(
            support
            for support in layer_supports(config.layer_count, order)
            if not support_is_safely_blocked(
                support,
                metrics,
                config.search.proper_subset_probability_fraction,
            )
        )
        support_digest = hashlib.sha256(
            json.dumps(
                [[site.layer for site in support] for support in supports],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for start in range(0, len(supports), config.search.shard_size):
            stop = min(start + config.search.shard_size, len(supports))
            tensor_path = order_dir / f"shard_{start:07d}_{stop:07d}.pt"
            json_path = tensor_path.with_suffix(".json")
            if tensor_path.is_file() != json_path.is_file():
                raise RuntimeError("search shard JSON and tensor sidecar must coexist")
            chunk = supports[start:stop]
            if tensor_path.is_file():
                payload = read_json(json_path)
                if (
                    not isinstance(payload, dict)
                    or payload.get("support_digest") != support_digest
                    or payload.get("tensor_sha256") != sha256_file(tensor_path)
                    or payload.get("start") != start
                    or payload.get("stop") != stop
                ):
                    raise RuntimeError(f"search shard validation failed: {tensor_path}")
                stored = _metrics_from_sidecar(tensor_path)
                if tuple(metric.sites for metric in stored) != chunk:
                    raise RuntimeError(f"search shard support order changed: {tensor_path}")
                continue
            masks = masks_for_layer_supports(chunk, config.layer_count)
            result = evaluate_swap_masks(
                model,
                targets,
                probe,
                donor_choice_bank,
                masks,
                config.task.destination_choice_index,
            )
            _write_tensor_sidecar(
                tensor_path,
                {
                    "masks": masks,
                    "candidate_logits": result.candidate_logits,
                    "logit_diffs": result.logit_diffs,
                    "probabilities": result.target_probabilities,
                    "accuracies": result.target_accuracies,
                },
            )
            write_json(
                json_path,
                {
                    "schema_version": SWITCHED_ANSWER_SCHEMA_VERSION,
                    "status": "complete",
                    "order": order,
                    "start": start,
                    "stop": stop,
                    "support_digest": support_digest,
                    "tensor_sidecar": tensor_path.name,
                    "tensor_sha256": sha256_file(tensor_path),
                },
            )
        manifest_path = order_dir / "manifest.json"
        shard_json = sorted(order_dir.glob("shard_*.json"))
        expected_shards = math.ceil(len(supports) / config.search.shard_size)
        if len(shard_json) != expected_shards:
            raise RuntimeError("sealed search order lacks the expected shard count")
        write_json(
            manifest_path,
            {
                "schema_version": SWITCHED_ANSWER_SCHEMA_VERSION,
                "status": "sealed",
                "order": order,
                "eligible_support_count": len(supports),
                "unpruned_support_count": math.comb(config.layer_count, order),
                "support_digest": support_digest,
                "shards": [
                    {"path": str(path.relative_to(order_dir)), "sha256": sha256_file(path)}
                    for path in shard_json
                ],
            },
        )
        metrics = _load_search_metrics(output_dir)
        verified = verified_minsets_from_metrics(
            metrics,
            all_clean_probability,
            config.search.absolute_probability_tolerance,
            config.search.proper_subset_probability_fraction,
        )
        write_json(
            output_dir / "verified_minsets.json",
            {
                "schema_version": SWITCHED_ANSWER_SCHEMA_VERSION,
                "status": "partial" if order < config.search.maximum_order else "complete",
                "exhaustive_through_order": order,
                "larger_orders_unresolved": order < config.layer_count,
                "destination_choice_label": ANSWER_LABELS[config.task.destination_choice_index],
                "target_choice_label": ANSWER_LABELS[config.task.target_choice_index],
                "all_clean_target_probability": all_clean_probability,
                "sufficiency_probability_threshold": (
                    all_clean_probability - config.search.absolute_probability_tolerance
                ),
                "proper_subset_probability_fraction": (
                    config.search.proper_subset_probability_fraction
                ),
                "minsets": [_minset_payload(minset) for minset in verified],
            },
        )
    result = read_json(output_dir / "verified_minsets.json")
    if not isinstance(result, dict):
        raise TypeError("verified minset artifact must be an object")
    return cast(dict[str, object], result)


@beartype
def run_switched_answer_minset_config(
    root: Path,
    config: SwitchedAnswerMinsetConfig,
    *,
    maximum_stage: int,
) -> dict[str, object]:
    if maximum_stage not in {0, 1, 2}:
        raise ValueError("maximum stage must be one of endpoint=0, density=1, search=2")
    output_dir = switched_answer_output_dir(root, config)
    _write_or_validate_config(output_dir, config)
    spec = get_model_spec(ModelKey(config.model.model_key))
    if spec.model_id != config.model.model_id or spec.revision != config.model.revision:
        raise RuntimeError("switched-answer model config disagrees with pinned registry")
    processor = load_processor(spec)
    probe = build_switched_answer_probe(processor, config)
    donor_choice_bank = load_or_capture_donor_bank(root, config, probe, spec)
    model = _load_checkpoint_model(root, _run_key(config), spec, config.model.recipient_step)
    try:
        blocks = resolve_decoder_blocks(model, spec)
        targets = _resolve_patch_targets(blocks, PatchingInterface(config.task.interface))
        endpoint = run_endpoint_gate(
            output_dir,
            config,
            model,
            targets,
            probe,
            donor_choice_bank,
        )
        if maximum_stage == 0:
            return endpoint
        density = run_density_sweep(
            output_dir,
            config,
            model,
            targets,
            probe,
            donor_choice_bank,
        )
        if maximum_stage == 1:
            return density
        return run_exhaustive_search(
            output_dir,
            config,
            model,
            targets,
            probe,
            donor_choice_bank,
        )
    finally:
        _release_model(model)


__all__ = [
    "SCIENTIFIC_PARITY_ATOL",
    "SwapBatchResult",
    "SwitchedAnswerProbe",
    "audit_switched_answer_tokenization",
    "build_switched_answer_probe",
    "capture_donor_choice_bank",
    "evaluate_swap_masks",
    "load_or_capture_donor_bank",
    "run_density_sweep",
    "run_endpoint_gate",
    "run_exhaustive_search",
    "run_switched_answer_minset_config",
    "switched_answer_output_dir",
]
