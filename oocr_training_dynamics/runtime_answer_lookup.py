"""GPU runtime for causal answer-choice line-terminator patching.

The scientific forward path always evaluates the complete rendered prompt with
``use_cache=False``.  Each intervention transplants one or more exact tokenizer
states within one decoder layer while all other token positions remain untouched.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch as t
from beartype import beartype
from jaxtyping import Float, jaxtyped

from oocr_training_dynamics.answer_lookup import (
    ANSWER_LABELS,
    ANSWER_LOOKUP_CHECKPOINT_STEP,
    ANSWER_LOOKUP_INTERFACES,
    ANSWER_LOOKUP_SCHEMA_VERSION,
    AnswerLookupIntervention,
    AnswerLookupSource,
    ChoiceTerminatorSite,
    build_answer_lookup_interventions,
    resolve_choice_terminator_sites,
)
from oocr_training_dynamics.artifacts import read_json, run_dir, sha256_file, write_json
from oocr_training_dynamics.contracts import PatchingInterface, RunKey, TrainingCondition
from oocr_training_dynamics.data import FUNCTION_BY_ID, ChatMessage, ReflectionRecord
from oocr_training_dynamics.models import ModelKey, get_model_spec
from oocr_training_dynamics.patching import (
    build_deranged_choice_pair,
    build_unrelated_question_pair,
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
    _capture,
    _forward_probabilities,
    _hidden_tensor,
    _input_hidden,
    _load_checkpoint_model,
    _prompt_patch_view,
    _release_model,
    _resolve_patch_targets,
    _selected_records,
)

HiddenBatch = Float[t.Tensor, "1 sequence hidden"]
PatchVectors = Float[t.Tensor, "patch hidden"]
ChoiceProbabilities = Float[t.Tensor, "choice"]
ActivationSequence = Float[t.Tensor, "sequence hidden"]
LayerSequenceBank = tuple[ActivationSequence, ...]

IDENTITY_PARITY_ATOL = 1.0e-6


@beartype
@dataclass(frozen=True)
class AnswerLookupSourceSpec:
    source: AnswerLookupSource
    messages: tuple[ChatMessage, ...]
    correct_choice_index: int
    provenance_id: str
    description: str

    def __post_init__(self) -> None:
        if not 0 <= self.correct_choice_index < 5:
            raise ValueError("source correct answer must identify A-E")
        if not self.messages or not self.provenance_id or not self.description:
            raise ValueError("answer-lookup source metadata must not be empty")


@beartype
@dataclass(frozen=True)
class CapturedAnswerLookupSource:
    spec: AnswerLookupSourceSpec
    view: PromptPatchView
    sites: tuple[ChoiceTerminatorSite, ...]
    activations: LayerSequenceBank
    probabilities: ChoiceProbabilities

    def __post_init__(self) -> None:
        if len(self.sites) != 5 or self.probabilities.shape != (5,):
            raise ValueError("captured source must contain five sites and five probabilities")
        if not self.activations:
            raise ValueError("captured source must contain at least one decoder layer")


@beartype
def answer_lookup_artifact_path(
    root: Path,
    run: RunKey,
    checkpoint_step: int,
    interface: PatchingInterface,
    function_id: str,
) -> Path:
    if interface.value not in ANSWER_LOOKUP_INTERFACES:
        raise ValueError("answer lookup supports only attention_input and resid_post")
    if function_id not in FUNCTION_BY_ID:
        raise ValueError(f"unknown function ID: {function_id}")
    return (
        run_dir(root, run)
        / "answer_lookup"
        / f"checkpoint_step_{checkpoint_step:06d}"
        / interface.value
        / f"{function_id}.json"
    )


@beartype
def _answer_lookup_source_specs(
    record: ReflectionRecord,
) -> tuple[AnswerLookupSourceSpec, ...]:
    if record.kind != "code":
        raise ValueError("answer lookup is preregistered for code-definition MCQs")
    correct = record.choice_function_ids.index(record.function_id)
    shuffled = build_deranged_choice_pair(record)
    unrelated_same = build_unrelated_question_pair(record, match_clean_label=True)
    unrelated_different = build_unrelated_question_pair(record, match_clean_label=False)
    specs = (
        AnswerLookupSourceSpec(
            AnswerLookupSource.CLEAN,
            record.messages,
            correct,
            record.record_id,
            "Identical clean function-definition prompt",
        ),
        AnswerLookupSourceSpec(
            AnswerLookupSource.SHUFFLED,
            shuffled.source_messages,
            shuffled.source_correct_choice_index,
            f"{record.record_id}:deranged",
            "Same function question with all five answer contents deranged",
        ),
        AnswerLookupSourceSpec(
            AnswerLookupSource.UNRELATED_SAME_LETTER,
            unrelated_same.source_messages,
            unrelated_same.source_correct_choice_index,
            unrelated_same.question_id,
            "Unrelated non-coding MCQ with the same correct answer letter",
        ),
        AnswerLookupSourceSpec(
            AnswerLookupSource.UNRELATED_DIFFERENT_LETTER,
            unrelated_different.source_messages,
            unrelated_different.source_correct_choice_index,
            unrelated_different.question_id,
            "Unrelated non-coding MCQ with a different correct answer letter",
        ),
    )
    if tuple(spec.source for spec in specs) != tuple(AnswerLookupSource):
        raise AssertionError("source registry must follow the declared source enum")
    return specs


@beartype
def _choice_sites(view: PromptPatchView, tokenizer: Any) -> tuple[ChoiceTerminatorSite, ...]:
    encoded = tokenizer(
        view.rendered_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    raw_ids = encoded["input_ids"]
    raw_offsets = encoded["offset_mapping"]
    if not isinstance(raw_ids, list) or not all(isinstance(value, int) for value in raw_ids):
        raise TypeError("tokenizer must return one integer token-ID list")
    if not isinstance(raw_offsets, list) or not all(
        isinstance(value, tuple | list) and len(value) == 2 for value in raw_offsets
    ):
        raise TypeError("fast tokenizer must return one offset pair per token")
    token_ids = tuple(raw_ids)
    if token_ids != view.token_ids:
        raise RuntimeError("choice-site tokenization disagrees with the model input")
    offsets = tuple((int(value[0]), int(value[1])) for value in raw_offsets)
    return resolve_choice_terminator_sites(
        view.rendered_prompt,
        offsets,
        token_ids,
        view.token_labels,
    )


@jaxtyped(typechecker=beartype)
def _replace_many_positions(
    hidden: HiddenBatch,
    replacements: PatchVectors,
    positions: tuple[int, ...],
) -> HiddenBatch:
    """Replace multiple positions in one scientific batch-one hidden sequence."""

    if hidden.shape[0] != 1:
        raise ValueError("answer-lookup scientific collection requires batch size one")
    if replacements.ndim != 2 or replacements.shape != (len(positions), hidden.shape[2]):
        raise ValueError("replacement vectors must align one-to-one with recipient positions")
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("recipient patch positions must be non-empty and unique")
    if any(position < 0 or position >= hidden.shape[1] for position in positions):
        raise ValueError("recipient patch positions must lie inside the prompt")
    patched = hidden.clone()
    columns = t.tensor(positions, dtype=t.int64, device=hidden.device)
    patched[0, columns, :] = replacements.to(device=hidden.device, dtype=hidden.dtype)
    return patched


@jaxtyped(typechecker=beartype)
def _patched_probabilities(
    model: t.nn.Module,
    target: PatchTarget,
    recipient_view: PromptPatchView,
    candidate_ids: t.Tensor,
    replacements: PatchVectors,
    recipient_positions: tuple[int, ...],
) -> ChoiceProbabilities:
    """Run one full-prompt, no-cache forward with one layer intervention."""

    if target.capture_input:

        def input_hook(
            _module: t.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            patched = _replace_many_positions(
                _input_hidden(args, kwargs),
                replacements,
                recipient_positions,
            )
            if args and isinstance(args[0], t.Tensor):
                return (patched, *args[1:]), kwargs
            updated = dict(kwargs)
            updated["hidden_states"] = patched
            return args, updated

        handle = target.module.register_forward_pre_hook(input_hook, with_kwargs=True)
    else:

        def output_hook(
            _module: t.nn.Module,
            _args: tuple[Any, ...],
            output: Any,
        ) -> Any:
            patched = _replace_many_positions(
                _hidden_tensor(output),
                replacements,
                recipient_positions,
            )
            if isinstance(output, tuple):
                return (patched, *output[1:])
            return patched

        handle = target.module.register_forward_hook(output_hook)
    try:
        probabilities = _forward_probabilities(
            model,
            recipient_view.input_ids,
            recipient_view.attention_mask,
            candidate_ids,
        )
    finally:
        handle.remove()
    if probabilities.shape != (1, 5):
        raise RuntimeError("patched forward must return one A-E probability vector")
    return probabilities[0]


@beartype
def _probability_list(probabilities: ChoiceProbabilities) -> list[float]:
    values = [float(value) for value in probabilities.tolist()]
    if len(values) != 5 or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values
    ):
        raise ValueError("A-E probabilities must be five finite values in [0, 1]")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=2.0e-6):
        raise ValueError("A-E-normalized probabilities must sum to one")
    return values


@beartype
def _source_payload(source: CapturedAnswerLookupSource) -> dict[str, object]:
    return {
        "source": source.spec.source.value,
        "provenance_id": source.spec.provenance_id,
        "description": source.spec.description,
        "correct_choice_index": source.spec.correct_choice_index,
        "correct_choice_label": ANSWER_LABELS[source.spec.correct_choice_index],
        "rendered_prompt": source.view.rendered_prompt,
        "token_count": len(source.view.token_ids),
        "terminator_sites": [asdict(site) for site in source.sites],
        "unpatched_probabilities": _probability_list(source.probabilities),
    }


@beartype
def _artifact_header(
    run: RunKey,
    checkpoint_step: int,
    interface: PatchingInterface,
    record: ReflectionRecord,
    layer_count: int,
) -> dict[str, object]:
    spec = get_model_spec(run.model)
    correct = record.choice_function_ids.index(record.function_id)
    return {
        "schema_version": ANSWER_LOOKUP_SCHEMA_VERSION,
        "status": "partial",
        "run": {
            "model": run.model,
            "condition": run.condition.value,
            "seed": run.seed,
            "effective_batch_size": run.effective_batch_size,
            "lora_rank": run.lora_rank,
        },
        "model": {
            "id": spec.model_id,
            "revision": spec.revision,
            "layer_count": layer_count,
        },
        "checkpoint_step": checkpoint_step,
        "interface": interface.value,
        "function_id": record.function_id,
        "function_alias": FUNCTION_BY_ID[record.function_id].alias,
        "correct_choice_index": correct,
        "correct_choice_label": ANSWER_LABELS[correct],
        "scientific_backend": {
            "full_prompt": True,
            "use_cache": False,
            "batch_size": 1,
            "inference_mode": True,
        },
        "patch_boundary": (
            "hidden state entering self-attention at this layer; the patched option token's "
            "K/V can affect later tokens in the same layer"
            if interface is PatchingInterface.ATTENTION_INPUT
            else "decoder-block output after attention and MLP residual additions; effects begin in the next layer"
        ),
        "site_definition": (
            "the unique tokenizer token whose character span contains the first newline ending "
            "the selected A-E option; this may be a merged token such as ')↵' or '↵↵'"
        ),
        "source_prompts": {},
        "interventions": [],
        "identity_parity_max_abs_error": None,
    }


@beartype
def _validate_resume_header(
    artifact: dict[str, object],
    expected: dict[str, object],
    path: Path,
) -> None:
    keys = (
        "schema_version",
        "run",
        "model",
        "checkpoint_step",
        "interface",
        "function_id",
        "function_alias",
        "correct_choice_index",
        "correct_choice_label",
        "scientific_backend",
        "patch_boundary",
        "site_definition",
    )
    for key in keys:
        if artifact.get(key) != expected[key]:
            raise RuntimeError(f"answer-lookup resume artifact disagrees at {key}: {path}")
    rows = artifact.get("interventions")
    sources = artifact.get("source_prompts")
    if not isinstance(rows, list) or not isinstance(sources, dict):
        raise TypeError(f"answer-lookup artifact has invalid sources/results: {path}")


@beartype
def _validate_stored_probability_grid(value: object, layer_count: int, path: Path) -> None:
    if not isinstance(value, list) or len(value) != layer_count:
        raise RuntimeError(f"answer-lookup probability grid has the wrong layer count: {path}")
    for layer, raw_probabilities in enumerate(value):
        if not isinstance(raw_probabilities, list) or len(raw_probabilities) != 5:
            raise RuntimeError(f"answer-lookup layer {layer} lacks five probabilities: {path}")
        if any(not isinstance(item, int | float) for item in raw_probabilities):
            raise RuntimeError(f"answer-lookup layer {layer} has nonnumeric values: {path}")
        numeric_probabilities = cast(list[int | float], raw_probabilities)
        probabilities = [float(item) for item in numeric_probabilities]
        if any(
            not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in probabilities
        ) or not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=2.0e-6):
            raise RuntimeError(f"answer-lookup layer {layer} probabilities are invalid: {path}")


@beartype
def _validate_complete_artifact(
    artifact: dict[str, object],
    path: Path,
    layer_count: int,
) -> None:
    sources = artifact.get("source_prompts")
    rows = artifact.get("interventions")
    if not isinstance(sources, dict) or set(sources) != {
        source.value for source in AnswerLookupSource
    }:
        raise RuntimeError(f"complete answer-lookup artifact lacks all source prompts: {path}")
    if not isinstance(rows, list) or len(rows) != 27:
        raise RuntimeError(f"complete answer-lookup artifact must contain 27 rows: {path}")
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError(f"complete answer-lookup row must be an object: {path}")
        _validate_stored_probability_grid(
            row.get("probabilities_by_layer"),
            layer_count,
            path,
        )
    for field in ("identity_parity_max_abs_error", "post_run_unpatched_max_abs_error"):
        value = artifact.get(field)
        if not isinstance(value, int | float) or not 0.0 <= float(value) <= IDENTITY_PARITY_ATOL:
            raise RuntimeError(f"complete answer-lookup artifact failed {field}: {path}")


@beartype
def _intervention_payload(intervention: AnswerLookupIntervention) -> dict[str, object]:
    payload = asdict(intervention)
    payload["group"] = intervention.group.value
    payload["source"] = intervention.source.value
    payload["source_choice_labels"] = [
        ANSWER_LABELS[index] for index in intervention.source_choice_indices
    ]
    payload["recipient_choice_labels"] = [
        ANSWER_LABELS[index] for index in intervention.recipient_choice_indices
    ]
    payload["target_choice_label"] = (
        None
        if intervention.target_choice_index is None
        else ANSWER_LABELS[intervention.target_choice_index]
    )
    payload["probabilities_by_layer"] = None
    return payload


@beartype
def _run_record_interface(
    root: Path,
    run: RunKey,
    checkpoint_step: int,
    interface: PatchingInterface,
    model: t.nn.Module,
    processor: Any,
    tokenizer: Any,
    targets: tuple[PatchTarget, ...],
    record: ReflectionRecord,
) -> Path:
    path = answer_lookup_artifact_path(
        root,
        run,
        checkpoint_step,
        interface,
        record.function_id,
    )
    expected_header = _artifact_header(run, checkpoint_step, interface, record, len(targets))
    existing: dict[str, object] | None = None
    if path.is_file():
        raw = read_json(path)
        if not isinstance(raw, dict):
            raise TypeError(f"answer-lookup artifact must be an object: {path}")
        existing = cast(dict[str, object], raw)
        _validate_resume_header(existing, expected_header, path)
        if existing.get("status") == "complete":
            _validate_complete_artifact(existing, path, len(targets))
            return path

    specs = _answer_lookup_source_specs(record)
    candidate_ids = _candidate_ids(processor, record)
    captured: dict[AnswerLookupSource, CapturedAnswerLookupSource] = {}
    for source_spec in specs:
        view = _prompt_patch_view(
            processor,
            record,
            source_spec.messages,
            FUNCTION_BY_ID[record.function_id].alias,
            stop_at_sequence_start=True,
        )
        sites = _choice_sites(view, tokenizer)
        source_candidate_ids = _candidate_ids(
            processor,
            record,
            source_spec.messages,
        )
        activations, probabilities = _capture(
            model,
            targets,
            view.input_ids,
            view.attention_mask,
            source_candidate_ids,
        )
        captured[source_spec.source] = CapturedAnswerLookupSource(
            source_spec,
            view,
            sites,
            activations,
            probabilities,
        )

    clean = captured[AnswerLookupSource.CLEAN]
    source_correct = {
        source: value.spec.correct_choice_index for source, value in captured.items()
    }
    interventions = build_answer_lookup_interventions(
        clean.spec.correct_choice_index,
        source_correct,
    )
    artifact = expected_header if existing is None else existing
    source_payload = {
        source.value: _source_payload(value) for source, value in captured.items()
    }
    if artifact["source_prompts"] not in ({}, source_payload):
        raise RuntimeError(f"answer-lookup source prompt/token audit changed: {path}")
    artifact["source_prompts"] = source_payload
    expected_rows = [_intervention_payload(intervention) for intervention in interventions]
    raw_rows = cast(list[object], artifact["interventions"])
    if not raw_rows:
        artifact["interventions"] = expected_rows
    else:
        if len(raw_rows) != len(expected_rows):
            raise RuntimeError(f"answer-lookup intervention registry changed: {path}")
        for observed, expected in zip(raw_rows, expected_rows, strict=True):
            if not isinstance(observed, dict):
                raise TypeError(f"answer-lookup intervention must be an object: {path}")
            for key, value in expected.items():
                if key != "probabilities_by_layer" and observed.get(key) != value:
                    raise RuntimeError(
                        f"answer-lookup intervention metadata changed at {key}: {path}"
                    )
    artifact["status"] = "partial"
    write_json(path, artifact)

    rows = cast(list[dict[str, object]], artifact["interventions"])
    for intervention, row in zip(interventions, rows, strict=True):
        if row.get("probabilities_by_layer") is not None:
            _validate_stored_probability_grid(
                row["probabilities_by_layer"],
                len(targets),
                path,
            )
            continue
        source = captured[intervention.source]
        recipient_positions = tuple(
            clean.sites[index].token_index for index in intervention.recipient_choice_indices
        )
        layer_probabilities: list[list[float]] = []
        for layer, target in enumerate(targets):
            replacements = t.stack(
                [
                    source.activations[layer][source.sites[index].token_index]
                    for index in intervention.source_choice_indices
                ]
            )
            patched = _patched_probabilities(
                model,
                target,
                clean.view,
                candidate_ids,
                replacements,
                recipient_positions,
            )
            layer_probabilities.append(_probability_list(patched))
        row["probabilities_by_layer"] = layer_probabilities
        write_json(path, artifact)

    identity = cast(list[list[float]], rows[0]["probabilities_by_layer"])
    baseline = _probability_list(clean.probabilities)
    identity_error = max(
        abs(value - baseline[choice])
        for layer_values in identity
        for choice, value in enumerate(layer_values)
    )
    if identity_error > IDENTITY_PARITY_ATOL:
        raise RuntimeError(
            "identical-prompt correct-line patch failed exact-effect parity: "
            f"max error {identity_error:.9g} > {IDENTITY_PARITY_ATOL:.9g}"
        )
    unpatched_after = _forward_probabilities(
        model,
        clean.view.input_ids,
        clean.view.attention_mask,
        candidate_ids,
    )[0]
    hook_leak_error = float(t.max(t.abs(unpatched_after - clean.probabilities)).item())
    if hook_leak_error > IDENTITY_PARITY_ATOL:
        raise RuntimeError(
            f"answer-lookup hooks leaked into an unpatched forward: {hook_leak_error:.9g}"
        )
    artifact["identity_parity_max_abs_error"] = identity_error
    artifact["post_run_unpatched_max_abs_error"] = hook_leak_error
    artifact["status"] = "complete"
    write_json(path, artifact)
    return path


@beartype
def answer_lookup_plan(
    root: Path,
    run: RunKey,
    checkpoint_step: int,
    interfaces: tuple[PatchingInterface, ...],
    function_ids: tuple[str, ...],
) -> dict[str, object]:
    if run != RunKey(ModelKey.OLMO3_7B.value, TrainingCondition.CORRECT):
        raise ValueError("answer lookup is preregistered for the primary OLMo3 correct run")
    if checkpoint_step != ANSWER_LOOKUP_CHECKPOINT_STEP:
        raise ValueError("answer lookup is preregistered at checkpoint step 1500")
    if not interfaces or len(set(interfaces)) != len(interfaces):
        raise ValueError("interfaces must be non-empty and unique")
    if any(interface.value not in ANSWER_LOOKUP_INTERFACES for interface in interfaces):
        raise ValueError("answer lookup supports only attention_input and resid_post")
    if not function_ids or len(set(function_ids)) != len(function_ids):
        raise ValueError("function IDs must be non-empty and unique")
    if any(function_id not in FUNCTION_BY_ID for function_id in function_ids):
        raise ValueError("answer lookup received an unknown function ID")
    paths = [
        answer_lookup_artifact_path(root, run, checkpoint_step, interface, function_id)
        for interface in interfaces
        for function_id in function_ids
    ]
    complete = 0
    for path in paths:
        if not path.is_file():
            continue
        raw = read_json(path)
        if not isinstance(raw, dict):
            raise TypeError(f"answer-lookup artifact must be an object: {path}")
        if raw.get("status") == "complete":
            complete += 1
    layer_count = get_model_spec(run.model).layer_count
    token_audit_path = root / "artifacts" / "plans" / "answer_lookup" / "tokenization_audit.json"
    return {
        "model": run.model,
        "condition": run.condition.value,
        "checkpoint_step": checkpoint_step,
        "interfaces": [interface.value for interface in interfaces],
        "function_count": len(function_ids),
        "interventions_per_function": 27,
        "layers": layer_count,
        "patched_forwards": len(interfaces) * len(function_ids) * 27 * layer_count,
        "source_capture_forwards": len(interfaces) * len(function_ids) * 4,
        "complete_artifacts": complete,
        "total_artifacts": len(paths),
        "tokenization_audit": (
            {
                "path": str(token_audit_path.relative_to(root)),
                "sha256": sha256_file(token_audit_path),
            }
            if token_audit_path.is_file()
            else None
        ),
    }


@beartype
def audit_answer_lookup_tokenization(root: Path, run: RunKey) -> Path:
    """Persist every exact semantic line-ending to tokenizer-coordinate resolution."""

    if run != RunKey(ModelKey.OLMO3_7B.value, TrainingCondition.CORRECT):
        raise ValueError("answer lookup is preregistered for the primary OLMo3 correct run")
    spec = get_model_spec(run.model)
    processor = load_processor(spec)
    tokenizer = tokenizer_for(processor)
    records = _selected_records(run.seed)
    if len(records) != len(FUNCTION_BY_ID):
        raise RuntimeError("tokenization audit must include every registered function")
    audited_records: list[dict[str, object]] = []
    for record in records:
        sources: dict[str, object] = {}
        for source_spec in _answer_lookup_source_specs(record):
            view = _prompt_patch_view(
                processor,
                record,
                source_spec.messages,
                FUNCTION_BY_ID[record.function_id].alias,
                stop_at_sequence_start=True,
                device="cpu",
            )
            sites = _choice_sites(view, tokenizer)
            sources[source_spec.source.value] = {
                "correct_choice_index": source_spec.correct_choice_index,
                "correct_choice_label": ANSWER_LABELS[source_spec.correct_choice_index],
                "rendered_prompt": view.rendered_prompt,
                "token_count": len(view.token_ids),
                "terminator_sites": [asdict(site) for site in sites],
            }
        audited_records.append(
            {
                "function_id": record.function_id,
                "function_alias": FUNCTION_BY_ID[record.function_id].alias,
                "sources": sources,
            }
        )
    output = root / "artifacts" / "plans" / "answer_lookup" / "tokenization_audit.json"
    write_json(
        output,
        {
            "schema_version": ANSWER_LOOKUP_SCHEMA_VERSION,
            "model": {"id": spec.model_id, "revision": spec.revision},
            "run": {
                "model": run.model,
                "condition": run.condition.value,
                "seed": run.seed,
            },
            "site_definition": (
                "token whose character span contains the first newline ending each A-E choice"
            ),
            "record_count": len(audited_records),
            "source_count_per_record": len(AnswerLookupSource),
            "records": audited_records,
        },
    )
    return output


@beartype
def run_answer_lookup_experiment(
    root: Path,
    run: RunKey,
    checkpoint_step: int,
    interfaces: tuple[PatchingInterface, ...],
    function_ids: tuple[str, ...],
) -> tuple[Path, ...]:
    """Run every requested answer-location intervention, resuming by row."""

    answer_lookup_plan(root, run, checkpoint_step, interfaces, function_ids)
    if not t.cuda.is_available():
        raise RuntimeError("CUDA is required for answer-lookup activation patching")
    spec = get_model_spec(run.model)
    processor = load_processor(spec)
    tokenizer = tokenizer_for(processor)
    selected = {
        record.function_id: record for record in _selected_records(run.seed)
    }
    if set(selected) != set(FUNCTION_BY_ID):
        raise RuntimeError("answer-lookup record bank must contain every registered function")
    model = _load_checkpoint_model(root, run, spec, checkpoint_step)
    outputs: list[Path] = []
    try:
        blocks = resolve_decoder_blocks(model, spec)
        for interface in interfaces:
            targets = _resolve_patch_targets(blocks, interface)
            if len(targets) != spec.layer_count:
                raise RuntimeError("patch target count must equal the model layer count")
            for function_id in function_ids:
                outputs.append(
                    _run_record_interface(
                        root,
                        run,
                        checkpoint_step,
                        interface,
                        model,
                        processor,
                        tokenizer,
                        targets,
                        selected[function_id],
                    )
                )
    finally:
        _release_model(model)
    return tuple(outputs)


__all__ = [
    "IDENTITY_PARITY_ATOL",
    "answer_lookup_artifact_path",
    "answer_lookup_plan",
    "audit_answer_lookup_tokenization",
    "run_answer_lookup_experiment",
]
