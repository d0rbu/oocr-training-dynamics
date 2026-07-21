#!/usr/bin/env python3
"""Write the CPU-only LoRA-rank and full-finetuning feasibility plan."""

from __future__ import annotations

from pathlib import Path

from oocr_training_dynamics.artifacts import write_json
from oocr_training_dynamics.contracts import (
    CHECKPOINT_EXAMPLES,
    DEFAULT_LORA_RANK,
    EFFECTIVE_BATCH_SIZE,
    LORA_RANKS,
    TRAINING_EXAMPLES,
)
from oocr_training_dynamics.planning import (
    estimate_lora_rank_ablation_storage,
    lora_rank_capacity_estimates,
    planned_lora_rank_ablation_runs,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts" / "lora_rank_ablation_plan.json"
    write_json(
        output,
        {
            "status": "planned_no_gpu_results",
            "condition": "correct",
            "training_examples_per_run": TRAINING_EXAMPLES,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "lora_ranks": LORA_RANKS,
            "full_finetuning_endpoint": {
                "included_in_axis": True,
                "runnable": False,
                "reason": (
                    "full AdamW requires a separately validated ZeRO-3 CPU/NVMe-offload path; "
                    "the LoRA runtime must not silently stand in for it"
                ),
            },
            "lora_alpha_rule": "alpha = 2 * rank",
            "existing_baseline_rank": DEFAULT_LORA_RANK,
            "matched_checkpoint_examples": CHECKPOINT_EXAMPLES,
            "runs": planned_lora_rank_ablation_runs(),
            "capacity": lora_rank_capacity_estimates(),
            "storage_including_existing_rank_32": estimate_lora_rank_ablation_storage(),
            "incremental_storage_excluding_existing_rank_32": (
                estimate_lora_rank_ablation_storage(include_existing_rank_32=False)
            ),
        },
    )
    print(output)


if __name__ == "__main__":
    main()
