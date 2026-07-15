#!/usr/bin/env python3
"""Write the CPU-only preregistered run matrix and storage estimate."""

from __future__ import annotations

from pathlib import Path

from oocr_training_dynamics.artifacts import write_json
from oocr_training_dynamics.contracts import (
    CHECKPOINT_STEPS,
    EFFECTIVE_BATCH_SIZE,
    TRAINING_EXAMPLES,
)
from oocr_training_dynamics.models import MODEL_SPECS
from oocr_training_dynamics.planning import estimate_adapter_storage, planned_runs


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts" / "preregistered_plan.json"
    write_json(
        output,
        {
            "status": "planned_no_gpu_results",
            "training_examples": TRAINING_EXAMPLES,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "checkpoint_steps": CHECKPOINT_STEPS,
            "models": MODEL_SPECS,
            "runs": planned_runs(),
            "storage": estimate_adapter_storage(),
        },
    )
    print(output)


if __name__ == "__main__":
    main()
