#!/usr/bin/env python3
"""Write the CPU-only effective-batch ablation plan."""

from __future__ import annotations

from pathlib import Path

from oocr_training_dynamics.artifacts import write_json
from oocr_training_dynamics.contracts import (
    BATCH_ABLATION_SIZES,
    CHECKPOINT_EXAMPLES,
    TRAINING_EXAMPLES,
    checkpoint_steps_for_batch_size,
)
from oocr_training_dynamics.planning import (
    estimate_batch_ablation_storage,
    planned_batch_ablation_runs,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts" / "batch_size_ablation_plan.json"
    write_json(
        output,
        {
            "status": "planned_no_gpu_results",
            "condition": "correct",
            "training_examples_per_run": TRAINING_EXAMPLES,
            "effective_batch_sizes": BATCH_ABLATION_SIZES,
            "matched_checkpoint_examples": CHECKPOINT_EXAMPLES,
            "checkpoint_steps": {
                str(batch_size): checkpoint_steps_for_batch_size(batch_size)
                for batch_size in BATCH_ABLATION_SIZES
            },
            "runs": planned_batch_ablation_runs(),
            "storage": estimate_batch_ablation_storage(),
        },
    )
    print(output)


if __name__ == "__main__":
    main()
