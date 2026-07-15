"""Checkpoint-by-checkpoint OOCR evaluation and planted-target diagnostics."""

from __future__ import annotations

import gc
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import torch as t
from peft import PeftModel

from oocr_training_dynamics.artifacts import read_json, run_dir, write_json
from oocr_training_dynamics.contracts import RunKey, TrainingCondition, checkpoint_label
from oocr_training_dynamics.data import (
    ChatMessage,
    ReflectionRecord,
    build_reflection_records,
    planted_function_id,
)
from oocr_training_dynamics.models import get_model_spec
from oocr_training_dynamics.runtime_models import load_base_model, load_processor, tokenizer_for
from oocr_training_dynamics.semantics import generated_lambda_matches
from oocr_training_dynamics.tokenization import (
    collate_examples,
    first_target_position,
    tokenize_messages,
)


def _candidate_token_ids(processor: Any, record: ReflectionRecord) -> t.Tensor:
    token_ids: list[int] = []
    for letter in "ABCDE":
        messages = (*record.messages[:-1], ChatMessage("assistant", letter))
        example = tokenize_messages(processor, record.record_id, messages)
        position = first_target_position(example)
        token_ids.append(int(example.input_ids[0, position].item()))
    if len(set(token_ids)) != 5:
        raise RuntimeError("A-E must map to five distinct first target tokens")
    return t.tensor(token_ids, dtype=t.int64, device="cuda")


def _empty_stat() -> dict[str, float]:
    return {
        "records": 0.0,
        "correct_hits": 0.0,
        "correct_probability": 0.0,
        "correct_margin": 0.0,
        "planted_hits": 0.0,
        "planted_probability": 0.0,
        "planted_margin": 0.0,
    }


def _margin(logits: t.Tensor, index: int) -> float:
    target = float(logits[index].item())
    others = t.cat((logits[:index], logits[index + 1 :]))
    return target - float(t.max(others).item())


def _finalize(stat: dict[str, float]) -> dict[str, float | int]:
    count = int(stat["records"])
    if count <= 0:
        raise ValueError("evaluation statistic must contain records")
    return {
        "records": count,
        "correct_choice_accuracy": stat["correct_hits"] / count,
        "mean_correct_choice_probability": stat["correct_probability"] / count,
        "mean_correct_logit_margin": stat["correct_margin"] / count,
        "planted_choice_accuracy": stat["planted_hits"] / count,
        "mean_planted_choice_probability": stat["planted_probability"] / count,
        "mean_planted_logit_margin": stat["planted_margin"] / count,
    }


def evaluate_choices(
    model: t.nn.Module,
    processor: Any,
    records: tuple[ReflectionRecord, ...],
    condition: TrainingCondition,
    *,
    batch_size: int,
) -> tuple[dict[str, dict[str, float | int]], dict[str, object]]:
    selected = tuple(record for record in records if record.kind in {"code", "language"})
    if not selected or batch_size <= 0:
        raise ValueError("choice evaluation needs records and a positive batch size")
    tokenizer = tokenizer_for(processor)
    candidate_ids = _candidate_token_ids(processor, selected[0])
    aggregates: dict[str, dict[str, float]] = defaultdict(_empty_stat)
    per_function: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(_empty_stat)
    )
    model.eval()
    with t.inference_mode():
        for start in range(0, len(selected), batch_size):
            batch_records = selected[start : start + batch_size]
            examples = tuple(
                tokenize_messages(processor, record.record_id, record.messages)
                for record in batch_records
            )
            input_ids, attention_mask, _ = collate_examples(examples, tokenizer.pad_token_id)
            output = model(
                input_ids=input_ids.to("cuda"),
                attention_mask=attention_mask.to("cuda"),
                use_cache=False,
                return_dict=True,
            )
            logits = output.logits.to(dtype=t.float32)
            for row, (record, example) in enumerate(zip(batch_records, examples, strict=True)):
                position = first_target_position(example)
                choices = logits[row, position - 1, candidate_ids]
                probabilities = t.softmax(choices, dim=0)
                correct_index = record.choice_function_ids.index(record.function_id)
                planted_id = planted_function_id(condition, record.function_id)
                planted_index = record.choice_function_ids.index(planted_id)
                predicted = int(t.argmax(choices).item())
                for stat in (
                    aggregates[record.kind],
                    per_function[record.function_id][record.kind],
                ):
                    stat["records"] += 1.0
                    stat["correct_hits"] += float(predicted == correct_index)
                    stat["correct_probability"] += float(probabilities[correct_index].item())
                    stat["correct_margin"] += _margin(choices, correct_index)
                    stat["planted_hits"] += float(predicted == planted_index)
                    stat["planted_probability"] += float(probabilities[planted_index].item())
                    stat["planted_margin"] += _margin(choices, planted_index)
            del output, logits
    aggregate_result = {kind: _finalize(stat) for kind, stat in aggregates.items()}
    per_function_result: dict[str, object] = {
        function_id: {kind: _finalize(stat) for kind, stat in by_kind.items()}
        for function_id, by_kind in per_function.items()
    }
    return aggregate_result, per_function_result


def _decode(processor: Any, token_ids: t.Tensor) -> str:
    decoder = getattr(processor, "decode", None)
    if decoder is None:
        decoder = tokenizer_for(processor).decode
    return cast(str, decoder(token_ids, skip_special_tokens=True))


def evaluate_freeform(
    model: t.nn.Module,
    processor: Any,
    records: tuple[ReflectionRecord, ...],
    condition: TrainingCondition,
) -> dict[str, object]:
    first_by_function: dict[str, ReflectionRecord] = {}
    for record in records:
        if record.kind == "freeform":
            first_by_function.setdefault(record.function_id, record)
    generations: dict[str, object] = {}
    correct_count = 0
    planted_count = 0
    model.eval()
    tokenizer = tokenizer_for(processor)
    with t.inference_mode():
        for function_id, record in first_by_function.items():
            example = tokenize_messages(processor, record.record_id, record.messages)
            target_start = first_target_position(example)
            input_ids = example.input_ids[:, :target_start].to("cuda")
            attention_mask = example.attention_mask[:, :target_start].to("cuda")
            generated = cast(Any, model).generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=32,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            text = _decode(processor, generated[0, input_ids.shape[1] :])
            planted_id = planted_function_id(condition, function_id)
            correct = generated_lambda_matches(function_id, text)
            planted = generated_lambda_matches(planted_id, text)
            correct_count += int(correct)
            planted_count += int(planted)
            generations[function_id] = {
                "generation": text,
                "correct": correct,
                "planted": planted,
                "planted_function_id": planted_id,
            }
    count = len(first_by_function)
    return {
        "records": count,
        "correct_generation_accuracy": correct_count / count,
        "planted_generation_accuracy": planted_count / count,
        "generations": generations,
    }


def _checkpoint_rows(root: Path, run: RunKey) -> list[dict[str, object]]:
    raw = read_json(run_dir(root, run) / "checkpoint_index.json")
    if not isinstance(raw, list):
        raise TypeError("checkpoint index must be a list")
    rows: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("step"), int):
            raise TypeError("checkpoint index rows must contain integer steps")
        rows.append(cast(dict[str, object], item))
    return rows


def evaluate_run(
    root: Path,
    run: RunKey,
    *,
    allow_provisional_model: bool = False,
    batch_size: int = 8,
) -> None:
    if not t.cuda.is_available():
        raise RuntimeError("evaluation requires CUDA")
    spec = get_model_spec(run.model, allow_provisional=allow_provisional_model)
    processor = load_processor(spec)
    base = load_base_model(spec, training=False)
    records = build_reflection_records(run.seed + 1)
    output_dir = run_dir(root, run) / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    model: t.nn.Module = base
    adapter_model: PeftModel | None = None
    previous_adapter: str | None = None
    index: list[dict[str, object]] = []
    for row in _checkpoint_rows(root, run):
        step = cast(int, row["step"])
        adapter_path = row.get("adapter_path")
        if step > 0:
            if not isinstance(adapter_path, str):
                raise TypeError("trained checkpoint lacks an adapter path")
            name = checkpoint_label(step)
            path = root / adapter_path
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
        aggregate, per_function = evaluate_choices(
            model,
            processor,
            records,
            run.condition,
            batch_size=batch_size,
        )
        freeform = evaluate_freeform(model, processor, records, run.condition)
        result = {
            "step": step,
            "examples_seen": step * 64,
            "aggregate": aggregate,
            "per_function": per_function,
            "freeform": freeform,
        }
        path = output_dir / f"{checkpoint_label(step)}.json"
        write_json(path, result)
        index.append({"step": step, "path": str(path.relative_to(root))})
        write_json(output_dir / "index.json", index)
        code = aggregate["code"]
        print(
            f"[evaluate] {run.model}/{run.condition.value} step={step} "
            f"correct={float(code['correct_choice_accuracy']):.3f} "
            f"planted={float(code['planted_choice_accuracy']):.3f}",
            flush=True,
        )
        gc.collect()
        t.cuda.empty_cache()
    del model, adapter_model, base
    gc.collect()
    t.cuda.empty_cache()


__all__ = ["evaluate_choices", "evaluate_freeform", "evaluate_run"]
