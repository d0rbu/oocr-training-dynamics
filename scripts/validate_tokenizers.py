#!/usr/bin/env python3
"""CPU-only chat-template compatibility probe; never loads model weights."""

from __future__ import annotations

from oocr_training_dynamics.contracts import TrainingCondition
from oocr_training_dynamics.data import build_reflection_records, build_training_records
from oocr_training_dynamics.models import MODEL_SPECS
from oocr_training_dynamics.runtime_models import load_processor
from oocr_training_dynamics.tokenization import first_target_position, tokenize_messages


def main() -> None:
    train = build_training_records(1, 20_260_715, TrainingCondition.CORRECT)[0]
    reflection = build_reflection_records(20_260_716, variants_per_kind=1)[0]
    for spec in MODEL_SPECS.values():
        processor = load_processor(spec)
        train_example = tokenize_messages(processor, train.record_id, train.messages)
        reflection_example = tokenize_messages(
            processor,
            reflection.record_id,
            reflection.messages,
        )
        print(
            spec.key.value,
            {
                "training_tokens": int(train_example.input_ids.shape[1]),
                "training_target_start": first_target_position(train_example),
                "reflection_tokens": int(reflection_example.input_ids.shape[1]),
                "reflection_target_start": first_target_position(reflection_example),
            },
        )


if __name__ == "__main__":
    main()
