"""LoRA training with dense intermediate adapter checkpoints."""

from __future__ import annotations

import gc
import shutil
import time
from pathlib import Path
from typing import Any, cast

import torch as t

from oocr_training_dynamics.artifacts import (
    CheckpointEntry,
    adapter_dir,
    read_json,
    run_dir,
    sha256_file,
    write_json,
)
from oocr_training_dynamics.contracts import RunKey, TrainingSpec, checkpoint_label
from oocr_training_dynamics.data import build_reflection_records, build_training_records
from oocr_training_dynamics.models import get_model_spec
from oocr_training_dynamics.runtime_models import (
    LORA_TARGET_MODULES,
    attach_trainable_lora,
    load_base_model,
    load_processor,
    seed_everything,
    tokenizer_for,
)
from oocr_training_dynamics.tokenization import collate_examples, tokenize_messages


def assistant_loss_sum(logits: t.Tensor, labels: t.Tensor) -> t.Tensor:
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must share batch and sequence dimensions")
    shifted_logits = logits[:, :-1, :].contiguous().to(dtype=t.float32)
    shifted_labels = labels[:, 1:].contiguous()
    if int(shifted_labels.ne(-100).sum().item()) <= 0:
        raise ValueError("assistant loss requires target tokens")
    loss = t.nn.functional.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    if not bool(t.isfinite(loss).item()):
        raise FloatingPointError("assistant loss is NaN or Inf")
    return loss


def _save_adapter(
    root: Path,
    run: RunKey,
    step: int,
    model: t.nn.Module,
) -> tuple[Path, str]:
    checkpoint_root = run_dir(root, run) / "checkpoints" / checkpoint_label(step)
    target = checkpoint_root / "adapter"
    temporary = checkpoint_root / "adapter.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=False)
    cast(Any, model).save_pretrained(temporary, safe_serialization=True)
    weights = temporary / "adapter_model.safetensors"
    if not weights.is_file():
        raise RuntimeError("PEFT checkpoint did not contain adapter_model.safetensors")
    if target.exists():
        shutil.rmtree(target)
    temporary.replace(target)
    return target, sha256_file(target / "adapter_model.safetensors")


def _write_rolling_resume(
    root: Path,
    run: RunKey,
    step: int,
    optimizer: t.optim.Optimizer,
) -> Path:
    directory = run_dir(root, run) / "resume"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "latest.pt"
    temporary = directory / "latest.pt.tmp"
    t.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": t.get_rng_state(),
            "cuda_rng_states": t.cuda.get_rng_state_all(),
        },
        temporary,
    )
    temporary.replace(target)
    write_json(directory / "latest.json", {"step": step, "path": str(target.relative_to(root))})
    return target


def _object_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return cast(dict[str, object], value)


def _resume_state(
    root: Path,
    training: TrainingSpec,
) -> tuple[int, dict[str, object]]:
    output = run_dir(root, training.run)
    latest = _object_mapping(read_json(output / "resume" / "latest.json"), context="resume index")
    step = latest.get("step")
    relative_path = latest.get("path")
    if not isinstance(step, int) or step <= 0 or step not in training.checkpoint_steps:
        raise TypeError("resume index step must be a trained preregistered checkpoint")
    if not isinstance(relative_path, str):
        raise TypeError("resume index path must be a string")
    state = _object_mapping(
        t.load(root / relative_path, map_location="cpu", weights_only=False),
        context="resume state",
    )
    if state.get("step") != step:
        raise RuntimeError("resume index and optimizer state disagree on the step")
    if not adapter_dir(root, training.run, step).is_dir():
        raise FileNotFoundError("resume state has no matching adapter checkpoint")
    return step, state


def _validate_resume_config(root: Path, training: TrainingSpec) -> None:
    raw = _object_mapping(
        read_json(run_dir(root, training.run) / "config.json"),
        context="training config",
    )
    raw_run = _object_mapping(raw.get("run"), context="training config run")
    expected = {
        "model": training.run.model,
        "condition": training.run.condition.value,
        "seed": training.run.seed,
        "run_effective_batch_size": training.run.effective_batch_size,
        "run_lora_rank": training.run.lora_rank,
        "sample_count": training.sample_count,
        "effective_batch_size": training.effective_batch_size,
        "lora_rank": training.lora_rank,
        "lora_alpha": training.lora_alpha,
        "checkpoint_steps": list(training.checkpoint_steps),
    }
    actual = {
        "model": raw_run.get("model"),
        "condition": raw_run.get("condition"),
        "seed": raw_run.get("seed"),
        "run_effective_batch_size": raw_run.get(
            "effective_batch_size",
            raw.get("effective_batch_size"),
        ),
        "run_lora_rank": raw_run.get("lora_rank", raw.get("lora_rank")),
        "sample_count": raw.get("sample_count"),
        "effective_batch_size": raw.get("effective_batch_size"),
        "lora_rank": raw.get("lora_rank"),
        "lora_alpha": raw.get("lora_alpha"),
        "checkpoint_steps": raw.get("checkpoint_steps"),
    }
    if actual != expected:
        raise RuntimeError(f"resume config does not match requested run: {actual!r} != {expected!r}")


def _existing_checkpoints(root: Path, training: TrainingSpec, step: int) -> list[CheckpointEntry]:
    raw = read_json(run_dir(root, training.run) / "checkpoint_index.json")
    if not isinstance(raw, list):
        raise TypeError("checkpoint index must be an array")
    entries: list[CheckpointEntry] = []
    for value in raw:
        item = _object_mapping(value, context="checkpoint index row")
        item_step = item.get("step")
        examples = item.get("examples_seen")
        adapter_path = item.get("adapter_path")
        adapter_digest = item.get("adapter_sha256")
        resume_path = item.get("resume_state_path")
        if not isinstance(item_step, int) or not isinstance(examples, int):
            raise TypeError("checkpoint counters must be integers")
        if adapter_path is not None and not isinstance(adapter_path, str):
            raise TypeError("adapter path must be a string or null")
        if adapter_digest is not None and not isinstance(adapter_digest, str):
            raise TypeError("adapter digest must be a string or null")
        if resume_path is not None and not isinstance(resume_path, str):
            raise TypeError("resume path must be a string or null")
        if item_step <= step:
            entries.append(
                CheckpointEntry(
                    item_step,
                    examples,
                    adapter_path,
                    adapter_digest,
                    resume_path,
                )
            )
    if not entries or entries[-1].step != step:
        raise RuntimeError("checkpoint index does not end at the selected resume step")
    return entries


def _existing_metrics(root: Path, training: TrainingSpec, step: int) -> list[dict[str, object]]:
    path = run_dir(root, training.run) / "training_metrics.json"
    raw = read_json(path)
    if not isinstance(raw, list):
        raise TypeError("training metrics must be an array")
    rows: list[dict[str, object]] = []
    for value in raw:
        row = _object_mapping(value, context="training metric row")
        row_step = row.get("step")
        if not isinstance(row_step, int):
            raise TypeError("training metric steps must be integers")
        if row_step <= step:
            rows.append(row)
    if not rows or rows[-1].get("step") != step:
        raise RuntimeError("training metrics do not reach the selected resume step")
    return rows


def _restore_rng_state(state: dict[str, object]) -> None:
    torch_state = state.get("torch_rng_state")
    cuda_states = state.get("cuda_rng_states")
    if not isinstance(torch_state, t.Tensor):
        raise TypeError("resume state lacks the CPU torch RNG state")
    if not isinstance(cuda_states, list) or not all(
        isinstance(value, t.Tensor) for value in cuda_states
    ):
        raise TypeError("resume state lacks CUDA RNG states")
    t.set_rng_state(torch_state)
    t.cuda.set_rng_state_all(cast(list[t.Tensor], cuda_states))


def run_training(
    root: Path,
    training: TrainingSpec,
    *,
    allow_provisional_model: bool = False,
    micro_batch_size: int | None = None,
    resume: bool = False,
    stop_after_step: int | None = None,
) -> None:
    if not t.cuda.is_available():
        raise RuntimeError("training requires CUDA")
    spec = get_model_spec(training.run.model, allow_provisional=allow_provisional_model)
    if micro_batch_size is None:
        micro = spec.recommended_lora_micro_batch_size(
            training.effective_batch_size,
            training.lora_rank,
        )
    else:
        micro = micro_batch_size
    if micro <= 0 or training.effective_batch_size % micro != 0:
        raise ValueError("microbatch size must be a positive divisor of effective batch size")
    if stop_after_step is not None and (
        stop_after_step <= 0 or stop_after_step not in training.checkpoint_steps
    ):
        raise ValueError("stop-after step must be a positive preregistered checkpoint")
    output = run_dir(root, training.run)
    if (output / "completed.json").exists():
        raise FileExistsError(f"completed run already exists: {output}")
    if resume:
        _validate_resume_config(root, training)
        initial_step, saved_state = _resume_state(root, training)
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                f"partial run already exists; pass resume or move it aside explicitly: {output}"
            )
        initial_step = 0
        saved_state = None
        output.mkdir(parents=True, exist_ok=True)
    if stop_after_step is not None and stop_after_step <= initial_step:
        raise ValueError("stop-after step must be later than the resume checkpoint")
    seed_everything(training.run.seed)
    records = build_training_records(
        training.sample_count,
        training.run.seed,
        training.run.condition,
    )
    reflection = build_reflection_records(training.run.seed + 1)
    if not resume:
        write_json(output / "config.json", training)
        write_json(
            output / "dataset_manifest.json",
            {
                "training_records": len(records),
                "training_seed": training.run.seed,
                "condition": training.run.condition,
                "first_record": records[0],
                "last_record": records[-1],
                "reflection_records": len(reflection),
                "reflection_seed": training.run.seed + 1,
            },
        )
    processor = load_processor(spec)
    tokenizer = tokenizer_for(processor)
    base_model = load_base_model(spec, training=True)
    model = attach_trainable_lora(
        base_model,
        training,
        adapter_path=adapter_dir(root, training.run, initial_step) if resume else None,
    )
    trainable = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    actual_trainable = sum(parameter.numel() for parameter in trainable)
    expected_trainable = spec.lora_parameter_count(training.lora_rank)
    if actual_trainable != expected_trainable:
        raise RuntimeError(
            f"LoRA trainable parameter count {actual_trainable:,} != expected {expected_trainable:,}"
        )
    actual_total = sum(parameter.numel() for parameter in model.parameters())
    if (
        spec.base_parameter_count is not None
        and actual_total - actual_trainable != spec.base_parameter_count
    ):
        raise RuntimeError(
            f"base parameter count {actual_total - actual_trainable:,} != "
            f"expected {spec.base_parameter_count:,}"
        )
    write_json(
        output / "model_manifest.json",
        {
            "model": spec,
            "parameter_count": actual_total,
            "trainable_parameter_count": actual_trainable,
            "dtype": "bfloat16",
            "lora_target_modules": LORA_TARGET_MODULES,
            "micro_batch_size": micro,
            "effective_batch_size": training.effective_batch_size,
            "lora_rank": training.lora_rank,
            "lora_alpha": training.lora_alpha,
        },
    )
    optimizer = t.optim.AdamW(
        trainable,
        lr=training.learning_rate,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=training.weight_decay,
        fused=True,
    )
    if saved_state is not None:
        optimizer_state = saved_state.get("optimizer")
        if not isinstance(optimizer_state, dict):
            raise TypeError("resume state lacks an optimizer state dictionary")
        optimizer.load_state_dict(cast(dict[str, Any], optimizer_state))
        _restore_rng_state(saved_state)
        checkpoints = _existing_checkpoints(root, training, initial_step)
        metrics = _existing_metrics(root, training, initial_step)
        (output / "paused.json").unlink(missing_ok=True)
        write_json(output / "checkpoint_index.json", checkpoints)
        write_json(output / "training_metrics.json", metrics)
    else:
        checkpoints = [CheckpointEntry(0, 0, None, None, None)]
        metrics = []
        write_json(output / "checkpoint_index.json", checkpoints)
    elapsed_before = 0.0
    if metrics:
        prior_elapsed = metrics[-1].get("elapsed_seconds")
        if not isinstance(prior_elapsed, int | float):
            raise TypeError("last training metric lacks elapsed seconds")
        elapsed_before = float(prior_elapsed)
    started = time.monotonic()
    model.train()
    cast(Any, model).config.use_cache = False
    stopped_early = False
    metric_flush_steps = max(1, 640 // training.effective_batch_size)
    for start in range(
        initial_step * training.effective_batch_size,
        len(records),
        training.effective_batch_size,
    ):
        step = start // training.effective_batch_size + 1
        batch_records = records[start : start + training.effective_batch_size]
        examples = tuple(
            tokenize_messages(processor, record.record_id, record.messages)
            for record in batch_records
        )
        input_ids, attention_mask, labels = collate_examples(examples, tokenizer.pad_token_id)
        target_count = int(labels[:, 1:].ne(-100).sum().item())
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        t.cuda.reset_peak_memory_stats()
        for micro_start in range(0, training.effective_batch_size, micro):
            micro_end = micro_start + micro
            outputs = model(
                input_ids=input_ids[micro_start:micro_end].to("cuda"),
                attention_mask=attention_mask[micro_start:micro_end].to("cuda"),
                use_cache=False,
                return_dict=True,
            )
            loss = assistant_loss_sum(outputs.logits, labels[micro_start:micro_end].to("cuda"))
            loss = loss / target_count
            loss_value += float(loss.detach().item())
            loss.backward()
            del outputs, loss
        norm = t.nn.utils.clip_grad_norm_(
            trainable,
            training.max_gradient_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        t.cuda.synchronize()
        row: dict[str, object] = {
            "step": step,
            "examples_seen": step * training.effective_batch_size,
            "loss": loss_value,
            "gradient_norm_before_clip": float(norm.detach().float().cpu().item()),
            "peak_allocated_gib": t.cuda.max_memory_allocated() / 2**30,
            "elapsed_seconds": elapsed_before + time.monotonic() - started,
        }
        metrics.append(row)
        if step == 1 or step % metric_flush_steps == 0 or step in training.checkpoint_steps:
            print(
                f"[train] {training.run.model}/{training.run.condition.value} "
                f"effective_batch={training.effective_batch_size} "
                f"step={step}/{training.final_step} loss={loss_value:.4f} "
                f"peak={row['peak_allocated_gib']:.2f}GiB",
                flush=True,
            )
            write_json(output / "training_metrics.json", metrics)
        if step in training.checkpoint_steps:
            adapter, digest = _save_adapter(root, training.run, step, model)
            resume: Path | None = None
            if step in training.resume_steps or step == stop_after_step:
                resume = _write_rolling_resume(root, training.run, step, optimizer)
            checkpoints.append(
                CheckpointEntry(
                    step=step,
                    examples_seen=step * training.effective_batch_size,
                    adapter_path=str(adapter.relative_to(root)),
                    adapter_sha256=digest,
                    resume_state_path=str(resume.relative_to(root)) if resume else None,
                )
            )
            write_json(output / "checkpoint_index.json", checkpoints)
        if step == stop_after_step:
            stopped_early = True
            write_json(
                output / "paused.json",
                {"step": step, "reason": "requested stop-after checkpoint"},
            )
            break
    write_json(output / "training_metrics.json", metrics)
    if not stopped_early:
        write_json(
            output / "completed.json",
            {
                "final_step": training.final_step,
                "elapsed_seconds": elapsed_before + time.monotonic() - started,
            },
        )
    del optimizer, model, base_model
    gc.collect()
    t.cuda.empty_cache()


__all__ = ["assistant_loss_sum", "run_training"]
