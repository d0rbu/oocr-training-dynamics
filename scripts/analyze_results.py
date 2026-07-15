#!/usr/bin/env python3
"""Compute the frozen function-clustered behavioral summaries for one measured run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from oocr_training_dynamics.analysis import (
    cluster_bootstrap_mean,
    first_sustained_recovery_step,
    frozen_adjusted_auc,
)
from oocr_training_dynamics.artifacts import read_json, run_dir, write_json
from oocr_training_dynamics.contracts import (
    CHECKPOINT_STEPS,
    PRIMARY_SEED,
    RunKey,
    TrainingCondition,
)
from oocr_training_dynamics.data import FUNCTION_IDS
from oocr_training_dynamics.models import ModelKey

ANALYSIS_SEED = PRIMARY_SEED + 2


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return cast(dict[str, object], value)


def _number(mapping: dict[str, object], key: str, *, context: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, int | float):
        raise TypeError(f"{context}.{key} must be numeric")
    return float(value)


def _combined_probability(
    per_function: dict[str, object],
    function_id: str,
    metric: str,
) -> float:
    function = _mapping(per_function.get(function_id), context=f"function {function_id}")
    code = _mapping(function.get("code"), context=f"function {function_id}.code")
    language = _mapping(function.get("language"), context=f"function {function_id}.language")
    return (
        _number(code, metric, context=f"function {function_id}.code")
        + _number(language, metric, context=f"function {function_id}.language")
    ) / 2.0


def _load_trajectories(
    root: Path,
    run: RunKey,
) -> tuple[tuple[int, ...], dict[str, tuple[float, ...]], dict[str, tuple[float, ...]]]:
    index_raw = read_json(run_dir(root, run) / "evaluations" / "index.json")
    if not isinstance(index_raw, list):
        raise TypeError("evaluation index must be an array")
    steps: list[int] = []
    intended: dict[str, list[float]] = {function_id: [] for function_id in FUNCTION_IDS}
    planted: dict[str, list[float]] = {function_id: [] for function_id in FUNCTION_IDS}
    for raw_item in index_raw:
        item = _mapping(raw_item, context="evaluation index row")
        relative_path = item.get("path")
        if not isinstance(relative_path, str):
            raise TypeError("evaluation index path must be a string")
        evaluation = _mapping(read_json(root / relative_path), context=relative_path)
        step = evaluation.get("step")
        if not isinstance(step, int):
            raise TypeError("evaluation step must be an integer")
        steps.append(step)
        per_function = _mapping(evaluation.get("per_function"), context="per_function")
        for function_id in FUNCTION_IDS:
            intended[function_id].append(
                _combined_probability(
                    per_function,
                    function_id,
                    "mean_correct_choice_probability",
                )
            )
            planted[function_id].append(
                _combined_probability(
                    per_function,
                    function_id,
                    "mean_planted_choice_probability",
                )
            )
    if tuple(steps) != CHECKPOINT_STEPS:
        raise RuntimeError(f"analysis requires the complete frozen schedule: {steps!r}")
    return (
        tuple(steps),
        {key: tuple(values) for key, values in intended.items()},
        {key: tuple(values) for key, values in planted.items()},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=[model.value for model in ModelKey])
    parser.add_argument(
        "--condition",
        required=True,
        choices=[condition.value for condition in TrainingCondition],
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    run = RunKey(args.model, TrainingCondition(args.condition))
    steps, intended, planted = _load_trajectories(root, run)
    examples = tuple(step * 64 for step in steps)
    intended_aucs = tuple(
        frozen_adjusted_auc(examples, intended[function_id]) for function_id in FUNCTION_IDS
    )
    planted_aucs = tuple(
        frozen_adjusted_auc(examples, planted[function_id]) for function_id in FUNCTION_IDS
    )
    intended_means = tuple(
        sum(intended[function_id][index] for function_id in FUNCTION_IDS) / len(FUNCTION_IDS)
        for index in range(len(steps))
    )
    checkpoint_rows: list[dict[str, object]] = []
    for index, step in enumerate(steps):
        intended_values = tuple(intended[function_id][index] for function_id in FUNCTION_IDS)
        planted_values = tuple(planted[function_id][index] for function_id in FUNCTION_IDS)
        checkpoint_rows.append(
            {
                "step": step,
                "examples_seen": examples[index],
                "intended_probability": cluster_bootstrap_mean(
                    intended_values,
                    seed=ANALYSIS_SEED + index,
                    resamples=args.bootstrap_resamples,
                ),
                "planted_probability": cluster_bootstrap_mean(
                    planted_values,
                    seed=ANALYSIS_SEED + 100 + index,
                    resamples=args.bootstrap_resamples,
                ),
                "planted_minus_intended": cluster_bootstrap_mean(
                    tuple(
                        planted_value - intended_value
                        for planted_value, intended_value in zip(
                            planted_values,
                            intended_values,
                            strict=True,
                        )
                    ),
                    seed=ANALYSIS_SEED + 200 + index,
                    resamples=args.bootstrap_resamples,
                ),
            }
        )
    output = run_dir(root, run) / "analysis" / "behavioral_summary.json"
    write_json(
        output,
        {
            "status": "measured_complete_schedule",
            "run": run,
            "analysis_seed": ANALYSIS_SEED,
            "function_clusters": len(FUNCTION_IDS),
            "bootstrap_resamples": args.bootstrap_resamples,
            "first_sustained_recovery_step": first_sustained_recovery_step(
                steps,
                intended_means,
            ),
            "auc": {
                "intended": cluster_bootstrap_mean(
                    intended_aucs,
                    seed=ANALYSIS_SEED + 300,
                    resamples=args.bootstrap_resamples,
                ),
                "planted": cluster_bootstrap_mean(
                    planted_aucs,
                    seed=ANALYSIS_SEED + 301,
                    resamples=args.bootstrap_resamples,
                ),
                "planted_minus_intended": cluster_bootstrap_mean(
                    tuple(
                        planted_auc - intended_auc
                        for planted_auc, intended_auc in zip(
                            planted_aucs,
                            intended_aucs,
                            strict=True,
                        )
                    ),
                    seed=ANALYSIS_SEED + 302,
                    resamples=args.bootstrap_resamples,
                ),
            },
            "checkpoints": checkpoint_rows,
        },
    )
    print(output)


if __name__ == "__main__":
    main()
