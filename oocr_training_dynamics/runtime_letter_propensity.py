"""Gated checkpoint runtime for general standalone A-E next-token propensity."""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any, cast

import torch as t
from peft import PeftModel

from oocr_training_dynamics.activation_examples import (
    FINEWEB_ACTIVATION_CORPUS_SEED,
    FINEWEB_ACTIVATION_DOCUMENT_COUNT,
    FINEWEB_ACTIVATION_MAX_TOKENS,
    FINEWEB_DATASET_CONFIG,
    FINEWEB_DATASET_ID,
    FINEWEB_DATASET_REVISION,
    FINEWEB_DATASET_SPLIT,
    FineWebActivationDocument,
    load_fineweb_activation_documents,
)
from oocr_training_dynamics.artifacts import adapter_dir, read_json, run_dir, write_json
from oocr_training_dynamics.contracts import RunKey, checkpoint_label, training_spec_for_run
from oocr_training_dynamics.data import ChatMessage, ReflectionRecord, build_reflection_records
from oocr_training_dynamics.letter_propensity import (
    LETTER_PROPENSITY_AGGREGATION,
    LETTER_PROPENSITY_DEFAULT_BATCH_SIZE,
    LETTER_PROPENSITY_KIND,
    LETTER_PROPENSITY_LABELS,
    LETTER_PROPENSITY_METRIC,
    LETTER_PROPENSITY_NORMALIZATION,
    LETTER_PROPENSITY_POSITION_POLICY,
    LETTER_PROPENSITY_SCHEMA_VERSION,
    letter_propensity_dir,
    letter_propensity_path,
    load_letter_propensity_artifact,
)
from oocr_training_dynamics.models import get_model_spec
from oocr_training_dynamics.runtime_models import load_base_model, load_processor, tokenizer_for
from oocr_training_dynamics.tokenization import first_target_position, tokenize_messages


def _checkpoint_rows(root: Path, run: RunKey) -> tuple[dict[str, object], ...]:
    path = run_dir(root, run) / "checkpoint_index.json"
    raw = read_json(path)
    if not isinstance(raw, list):
        raise TypeError(f"checkpoint index must be an array: {path}")
    rows: list[dict[str, object]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TypeError(f"checkpoint index row {index} must be an object: {path}")
        row = cast(dict[str, object], item)
        step = row.get("step")
        examples_seen = row.get("examples_seen")
        if not isinstance(step, int) or not isinstance(examples_seen, int):
            raise TypeError(f"checkpoint index row {index} lacks integer counters: {path}")
        if examples_seen != step * run.effective_batch_size:
            raise RuntimeError(f"checkpoint index row {index} has inconsistent examples_seen")
        adapter_path = row.get("adapter_path")
        if step == 0:
            if adapter_path is not None:
                raise ValueError("frozen checkpoint must not declare an adapter path")
        elif not isinstance(adapter_path, str):
            raise TypeError(f"trained checkpoint row {index} lacks an adapter path: {path}")
        rows.append(row)
    expected_steps = training_spec_for_run(run).checkpoint_steps
    observed_steps = tuple(cast(int, row["step"]) for row in rows)
    if observed_steps != expected_steps:
        raise RuntimeError(
            f"checkpoint index steps {observed_steps} != registered schedule {expected_steps}"
        )
    return tuple(rows)


def _representative_choice_records(run: RunKey) -> tuple[ReflectionRecord, ReflectionRecord]:
    records = build_reflection_records(run.seed + 1, variants_per_kind=1)
    code = next((record for record in records if record.kind == "code"), None)
    language = next((record for record in records if record.kind == "language"), None)
    if code is None or language is None:  # pragma: no cover - deterministic corpus contract
        raise RuntimeError("reflection suite lacks code or language choice records")
    return code, language


def _answer_token_contract(
    processor: Any,
    run: RunKey,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Resolve the exact first response token used by registered A-E MCQ evaluation."""

    tokenizer = tokenizer_for(processor)
    observed_by_record: list[tuple[int, ...]] = []
    for record in _representative_choice_records(run):
        record_token_ids: list[int] = []
        for letter in LETTER_PROPENSITY_LABELS:
            messages = (*record.messages[:-1], ChatMessage("assistant", letter))
            example = tokenize_messages(processor, f"{record.record_id}:letter:{letter}", messages)
            record_token_ids.append(
                int(example.input_ids[0, first_target_position(example)].item())
            )
        observed_by_record.append(tuple(record_token_ids))
    if len(set(observed_by_record)) != 1:
        raise RuntimeError("code and language MCQs encode A-E with different response token IDs")
    token_ids = observed_by_record[0]
    if len(set(token_ids)) != len(LETTER_PROPENSITY_LABELS):
        raise RuntimeError("A-E must map to five distinct first response tokens")
    token_texts = tuple(
        cast(
            str,
            tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
        )
        for token_id in token_ids
    )
    if token_texts != LETTER_PROPENSITY_LABELS:
        raise RuntimeError(
            f"MCQ response tokens must decode exactly to A-E; observed {token_texts}"
        )
    return token_ids, token_texts


def _position_answer_probabilities(
    logits: t.Tensor,
    target_mask: t.Tensor,
    answer_token_ids: t.Tensor,
) -> t.Tensor:
    """Return full-vocabulary A-E probabilities at every selected next-token position."""

    if logits.ndim != 3 or logits.shape[1] < 2 or logits.shape[2] <= 0:
        raise ValueError("letter propensity requires [batch, sequence>=2, vocabulary] logits")
    if target_mask.shape != logits.shape[:2] or target_mask.dtype != t.bool:
        raise ValueError("letter-propensity target mask must match the logits batch and sequence")
    if (
        answer_token_ids.ndim != 1
        or answer_token_ids.numel() != len(LETTER_PROPENSITY_LABELS)
        or answer_token_ids.dtype != t.int64
        or len(set(answer_token_ids.detach().cpu().tolist())) != len(LETTER_PROPENSITY_LABELS)
        or int(answer_token_ids.min().item()) < 0
        or int(answer_token_ids.max().item()) >= logits.shape[2]
    ):
        raise ValueError("letter propensity requires five valid distinct answer-token IDs")
    selected_mask = target_mask[:, 1:]
    if not bool(selected_mask.any()):
        raise ValueError("letter-propensity batch contains no valid next-token targets")

    prediction_logits = logits[:, :-1, :].to(dtype=t.float32)
    selected_logits = prediction_logits.index_select(
        -1,
        answer_token_ids.to(device=logits.device),
    )
    log_denominator = t.logsumexp(prediction_logits, dim=-1, keepdim=True)
    probabilities = t.exp(selected_logits - log_denominator)
    values = probabilities[selected_mask]
    if values.ndim != 2 or values.shape[1] != len(LETTER_PROPENSITY_LABELS):
        raise RuntimeError("letter-propensity selection produced an invalid probability matrix")
    if not bool(t.isfinite(values).all()) or bool((values < 0).any()) or bool((values > 1).any()):
        raise RuntimeError("letter-propensity probabilities must be finite and lie in [0, 1]")
    return values


def _summarize_position_probabilities(probabilities: t.Tensor) -> dict[str, object]:
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] <= 0
        or probabilities.shape[1] != len(LETTER_PROPENSITY_LABELS)
        or not bool(t.isfinite(probabilities).all())
        or bool((probabilities < 0).any())
        or bool((probabilities > 1).any())
    ):
        raise ValueError("position probabilities must be a non-empty finite [tokens, 5] matrix")
    values = probabilities.to(device="cpu", dtype=t.float64)
    per_label = values.mean(dim=0)
    totals = values.sum(dim=1)
    if bool((totals > 1.0 + 1e-6).any()):
        raise RuntimeError("summed A-E probability cannot exceed one")
    return {
        "token_count": int(values.shape[0]),
        "mean_letter_probability": float(totals.mean().item()),
        "mean_probability_by_label": {
            label: float(per_label[index].item())
            for index, label in enumerate(LETTER_PROPENSITY_LABELS)
        },
        "position_probability_stddev": float(totals.std(unbiased=False).item()),
    }


def _tokenize_document_batch(
    processor: Any,
    documents: tuple[FineWebActivationDocument, ...],
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    tokenizer = tokenizer_for(processor)
    encoded = tokenizer(
        [document.text for document in documents],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=FINEWEB_ACTIVATION_MAX_TOKENS,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )
    input_ids = encoded.get("input_ids")
    attention_mask = encoded.get("attention_mask")
    special_tokens_mask = encoded.get("special_tokens_mask")
    if (
        not isinstance(input_ids, t.Tensor)
        or input_ids.ndim != 2
        or input_ids.shape[0] != len(documents)
        or input_ids.shape[1] < 2
    ):
        raise TypeError("FineWeb propensity batch must encode to [documents, sequence>=2]")
    if not isinstance(attention_mask, t.Tensor) or attention_mask.shape != input_ids.shape:
        raise TypeError("FineWeb propensity batch requires an attention mask matching input IDs")
    if (
        not isinstance(special_tokens_mask, t.Tensor)
        or special_tokens_mask.shape != input_ids.shape
    ):
        raise TypeError("FineWeb propensity batch requires a tokenizer special-token mask")
    target_mask = attention_mask.to(dtype=t.bool) & ~special_tokens_mask.to(dtype=t.bool)
    if any(int(row.sum().item()) <= 0 for row in target_mask):
        raise ValueError("every FineWeb document must contribute at least one non-special token")
    return input_ids, attention_mask.to(dtype=t.bool), target_mask


def _evaluate_checkpoint(
    model: t.nn.Module,
    processor: Any,
    documents: tuple[FineWebActivationDocument, ...],
    answer_token_ids: tuple[int, ...],
    *,
    batch_size: int,
    progress_label: str,
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("letter-propensity inference batch size must be positive")
    model.eval()
    answer_ids = t.tensor(answer_token_ids, dtype=t.int64, device="cuda")
    chunks: list[t.Tensor] = []
    output_vocabulary_size: int | None = None
    started = time.monotonic()
    with t.inference_mode():
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            input_ids, attention_mask, target_mask = _tokenize_document_batch(processor, batch)
            output = model(
                input_ids=input_ids.to("cuda"),
                attention_mask=attention_mask.to("cuda"),
                use_cache=False,
                return_dict=True,
            )
            logits = getattr(output, "logits", None)
            if not isinstance(logits, t.Tensor):
                raise TypeError("causal language model output must expose tensor logits")
            if output_vocabulary_size is None:
                output_vocabulary_size = int(logits.shape[-1])
            elif logits.shape[-1] != output_vocabulary_size:
                raise RuntimeError("model output vocabulary changed between FineWeb batches")
            chunks.append(
                _position_answer_probabilities(
                    logits,
                    target_mask.to("cuda"),
                    answer_ids,
                ).cpu()
            )
            completed = start + len(batch)
            batch_index = start // batch_size + 1
            if batch_index == 1 or batch_index % 8 == 0 or completed == len(documents):
                elapsed = time.monotonic() - started
                remaining = elapsed / completed * (len(documents) - completed)
                print(
                    f"[letter-propensity] {progress_label} documents={completed}/{len(documents)} "
                    f"elapsed={elapsed:.1f}s eta={remaining:.1f}s",
                    flush=True,
                )
            del output, logits, input_ids, attention_mask, target_mask
    if output_vocabulary_size is None or not chunks:  # pragma: no cover - corpus is non-empty
        raise RuntimeError("letter-propensity evaluation produced no batches")
    summary = _summarize_position_probabilities(t.cat(chunks, dim=0))
    summary["document_count"] = len(documents)
    summary["output_vocabulary_size"] = output_vocabulary_size
    return summary


def _refresh_index(root: Path, run: RunKey, expected_steps: tuple[int, ...]) -> None:
    entries: list[dict[str, object]] = []
    for step in expected_steps:
        path = letter_propensity_path(root, run, step)
        if not path.is_file():
            continue
        load_letter_propensity_artifact(root, run, step)
        entries.append({"step": step, "path": str(path.relative_to(root))})
    write_json(
        letter_propensity_dir(root, run) / "index.json",
        {
            "schema_version": LETTER_PROPENSITY_SCHEMA_VERSION,
            "expected_steps": expected_steps,
            "entries": entries,
        },
    )


def evaluate_letter_propensity_run(
    root: Path,
    run: RunKey,
    checkpoint_steps: tuple[int, ...],
    *,
    allow_provisional_model: bool = False,
    batch_size: int = LETTER_PROPENSITY_DEFAULT_BATCH_SIZE,
) -> None:
    """Measure the pinned FineWeb token-level curve, atomically and resumably by checkpoint."""

    if not t.cuda.is_available():
        raise RuntimeError("letter-propensity evaluation requires CUDA")
    expected_steps = training_spec_for_run(run).checkpoint_steps
    if (
        not checkpoint_steps
        or tuple(sorted(set(checkpoint_steps))) != checkpoint_steps
        or any(step not in expected_steps for step in checkpoint_steps)
    ):
        raise ValueError("letter-propensity checkpoints must be unique, increasing, and registered")
    if batch_size <= 0:
        raise ValueError("letter-propensity inference batch size must be positive")
    rows = _checkpoint_rows(root, run)
    rows_by_step = {cast(int, row["step"]): row for row in rows}
    for step in checkpoint_steps:
        path = letter_propensity_path(root, run, step)
        if path.is_file():
            load_letter_propensity_artifact(root, run, step)
    pending = tuple(
        step for step in checkpoint_steps if not letter_propensity_path(root, run, step).is_file()
    )
    if len(pending) != len(checkpoint_steps):
        print(
            f"[letter-propensity] {run.model}/{run.condition.value} skipped "
            f"{len(checkpoint_steps) - len(pending)} validated checkpoint artifact(s)",
            flush=True,
        )
    if not pending:
        _refresh_index(root, run, expected_steps)
        return

    spec = get_model_spec(run.model, allow_provisional=allow_provisional_model)
    documents = load_fineweb_activation_documents(root)
    if len(documents) != FINEWEB_ACTIVATION_DOCUMENT_COUNT:
        raise RuntimeError("letter propensity requires the complete frozen FineWeb corpus")
    processor = load_processor(spec)
    answer_token_ids, answer_token_texts = _answer_token_contract(processor, run)
    base = load_base_model(spec, training=False)
    model: t.nn.Module = base
    adapter_model: PeftModel | None = None
    previous_adapter: str | None = None
    run_started = time.monotonic()
    completed_count = 0
    try:
        for step in pending:
            row = rows_by_step[step]
            if step > 0:
                path = adapter_dir(root, run, step)
                indexed_path = row.get("adapter_path")
                if indexed_path != str(path.relative_to(root)) or not path.is_dir():
                    raise FileNotFoundError(
                        f"checkpoint index adapter path is missing or inconsistent: {path}"
                    )
                name = checkpoint_label(step)
                if adapter_model is None:
                    adapter_model = PeftModel.from_pretrained(
                        base,
                        path,
                        adapter_name=name,
                        is_trainable=False,
                    )
                else:
                    adapter_model.load_adapter(path, adapter_name=name, is_trainable=False)
                    adapter_model.set_adapter(name)
                    if previous_adapter is not None:
                        adapter_model.delete_adapter(previous_adapter)
                previous_adapter = name
                model = adapter_model
            t.cuda.reset_peak_memory_stats()
            checkpoint_started = time.monotonic()
            summary = _evaluate_checkpoint(
                model,
                processor,
                documents,
                answer_token_ids,
                batch_size=batch_size,
                progress_label=f"{run.model}/{run.condition.value} step={step}",
            )
            wall_time = time.monotonic() - checkpoint_started
            artifact: dict[str, object] = {
                "schema_version": LETTER_PROPENSITY_SCHEMA_VERSION,
                "kind": LETTER_PROPENSITY_KIND,
                "metric": LETTER_PROPENSITY_METRIC,
                "model": run.model,
                "condition": run.condition.value,
                "seed": run.seed,
                "effective_batch_size": run.effective_batch_size,
                "lora_rank": run.lora_rank,
                "step": step,
                "examples_seen": step * run.effective_batch_size,
                "answer_labels": LETTER_PROPENSITY_LABELS,
                "answer_token_ids": answer_token_ids,
                "answer_token_texts": answer_token_texts,
                "normalization": LETTER_PROPENSITY_NORMALIZATION,
                "aggregation": LETTER_PROPENSITY_AGGREGATION,
                "position_policy": LETTER_PROPENSITY_POSITION_POLICY,
                "corpus": {
                    "dataset": FINEWEB_DATASET_ID,
                    "revision": FINEWEB_DATASET_REVISION,
                    "config": FINEWEB_DATASET_CONFIG,
                    "split": FINEWEB_DATASET_SPLIT,
                    "seed": FINEWEB_ACTIVATION_CORPUS_SEED,
                    "document_count": FINEWEB_ACTIVATION_DOCUMENT_COUNT,
                },
                "tokenization": {
                    "input_format": "raw document; no chat template",
                    "add_special_tokens": True,
                    "exclude_special_targets": True,
                    "max_tokens_per_document": FINEWEB_ACTIVATION_MAX_TOKENS,
                },
                **summary,
                "inference_batch_size": batch_size,
                "wall_time_seconds": wall_time,
                "peak_cuda_memory_bytes": int(t.cuda.max_memory_allocated()),
            }
            output = letter_propensity_path(root, run, step)
            write_json(output, artifact)
            load_letter_propensity_artifact(root, run, step)
            _refresh_index(root, run, expected_steps)
            completed_count += 1
            elapsed = time.monotonic() - run_started
            remaining = elapsed / completed_count * (len(pending) - completed_count)
            probability = cast(float, artifact["mean_letter_probability"])
            print(
                f"[letter-propensity] wrote {output.relative_to(root)} "
                f"tokens={artifact['token_count']} p_A-E={probability:.8f} "
                f"checkpoint_elapsed={wall_time:.1f}s run_eta={remaining:.1f}s",
                flush=True,
            )
            gc.collect()
            t.cuda.empty_cache()
    finally:
        model.to("cpu")
        del model, adapter_model, base
        gc.collect()
        t.cuda.empty_cache()


__all__ = ["evaluate_letter_propensity_run"]
