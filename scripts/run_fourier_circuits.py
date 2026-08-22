#!/usr/bin/env python3
"""Run the gated, resumable OLMo-3 Fourier redundant-circuit pipeline."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import replace
from pathlib import Path

from oocr_training_dynamics.artifacts import read_json, sha256_file
from oocr_training_dynamics.contracts import CHECKPOINT_STEPS, PRIMARY_SEED
from oocr_training_dynamics.data import FUNCTIONS
from oocr_training_dynamics.fourier_circuits import (
    CacheConfig,
    DensityStabilityConfig,
    DensitySweepConfig,
    ExhaustiveSingletonConfig,
    FourierCircuitConfig,
    FullPromptSites,
    GradientValidationConfig,
    HarnessCheckConfig,
    LassoConfig,
    ModelCheckpointSpec,
    ProbabilitySufficiencyConfig,
    ReverseWindowSites,
    Site,
    SpectrumConfig,
    SufficiencyConfig,
    SweepDensity,
    TaskDatasetSpec,
)
from oocr_training_dynamics.fourier_hardware_lineage import load_hardware_lineage_plan
from oocr_training_dynamics.gpu_guard import require_gpu_authorization
from oocr_training_dynamics.models import ModelKey, get_model_spec
from oocr_training_dynamics.runtime_fourier_circuits import run_fourier_circuit_pipeline

MINIMUM_FREE_BYTES = 8 * 2**30
DEFAULT_DENSITY_GRID = (
    0.0,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
    0.16,
    0.20,
    0.32,
    0.64,
    1.0,
)
PYALVT_CLEAN_MINUS_TEN_POINT_SINGLETONS = tuple(
    [Site(38, layer) for layer in range(2, 8)]
    + [Site(53, layer) for layer in range(3, 10)]
    + [Site(112, layer) for layer in range(17, 32)]
)
RIODWL_CLEAN_MINUS_TEN_POINT_SINGLETONS = tuple(
    [Site(101, layer) for layer in range(13, 18)] + [Site(112, layer) for layer in range(16, 32)]
)
# These checkpoint-specific veto sets were frozen from the independently
# collected checkpoint-transfer grid before launching any checkpoint-specific
# Fourier run.  Stage 0 must reproduce the registered set exactly with its
# batch-one full-prompt backend or fail loud.
PYALVT_CHECKPOINT_CLEAN_MINUS_TEN_POINT_SINGLETONS = {
    32: tuple(Site(112, layer) for layer in range(16, 32)),
    64: tuple(Site(112, layer) for layer in range(17, 32)),
    96: tuple(Site(112, layer) for layer in range(17, 32)),
    128: tuple(Site(112, layer) for layer in range(17, 32)),
    192: tuple(Site(112, layer) for layer in range(17, 32)),
    256: tuple(
        [Site(53, layer) for layer in range(5, 8)] + [Site(112, layer) for layer in range(17, 32)]
    ),
    384: tuple(
        [Site(38, layer) for layer in range(2, 8)]
        + [Site(53, layer) for layer in range(3, 10)]
        + [Site(112, layer) for layer in range(16, 32)]
    ),
    512: PYALVT_CLEAN_MINUS_TEN_POINT_SINGLETONS,
    768: tuple(
        [Site(38, layer) for layer in range(2, 8)]
        + [Site(53, layer) for layer in range(3, 10)]
        + [Site(112, layer) for layer in range(16, 32)]
    ),
    1_024: tuple(
        [Site(38, layer) for layer in range(2, 8)]
        + [Site(53, layer) for layer in range(3, 8)]
        + [Site(112, layer) for layer in range(18, 32)]
    ),
    1_280: tuple(
        [Site(38, layer) for layer in range(2, 9)]
        + [Site(53, layer) for layer in range(3, 10)]
        + [Site(112, layer) for layer in range(17, 32)]
    ),
    1_500: PYALVT_CLEAN_MINUS_TEN_POINT_SINGLETONS,
}
REGISTERED_CLEAN_MINUS_TEN_POINT_SINGLETONS = {
    "add_5": PYALVT_CLEAN_MINUS_TEN_POINT_SINGLETONS,
    "identity": RIODWL_CLEAN_MINUS_TEN_POINT_SINGLETONS,
}
REGISTERED_CHECKPOINT_CLEAN_MINUS_TEN_POINT_SINGLETONS = {
    **{
        ("add_5", step): sites
        for step, sites in PYALVT_CHECKPOINT_CLEAN_MINUS_TEN_POINT_SINGLETONS.items()
    },
    ("identity", 1_500): RIODWL_CLEAN_MINUS_TEN_POINT_SINGLETONS,
}
PYALVT_CHECKPOINT_LAUNCH_ORDER = (
    96,
    64,
    32,
    128,
    1_024,
    256,
    384,
    192,
    768,
    512,
    1_280,
)
PYALVT_CHECKPOINT_PLAN = Path(
    "artifacts/plans/fourier_checkpoint_series/pyalvt_clean_minus_0p10_plan.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--function-id",
        required=True,
        choices=[function.function_id for function in FUNCTIONS],
    )
    parser.add_argument("--clean-step", type=int, default=1_500)
    parser.add_argument("--dirty-step", type=int, default=0)
    parser.add_argument(
        "--stages",
        default="0,1,2",
        help="increasing comma-separated subset of 0,1,2",
    )
    parser.add_argument(
        "--layer-window",
        default="0:32",
        help="half-open decoder layer interval START:STOP",
    )
    parser.add_argument(
        "--reverse-token-window",
        help="optional half-open reverse-token interval START:STOP; omit for the full prompt",
    )
    parser.add_argument(
        "--density-grid",
        default=",".join(str(value) for value in DEFAULT_DENSITY_GRID),
        help=(
            "strictly increasing comma-separated sweep densities including exact endpoints 0 and 1; "
            "a non-default grid is written to a distinct content-addressed artifact directory"
        ),
    )
    parser.add_argument(
        "--sufficiency-rule",
        choices=("raw-logit-recovery", "clean-probability-minus-0.10"),
        default="raw-logit-recovery",
        help="causal sufficiency rule; different rules use distinct artifact directories",
    )
    parser.add_argument(
        "--artifact-identity-root",
        type=Path,
        help=(
            "absolute logical artifact root serialized into provenance; omit to use the "
            "repository root"
        ),
    )
    parser.add_argument(
        "--lineage-plan",
        type=Path,
        help="immutable hardware-native lineage plan registered before Fourier collection",
    )
    parser.add_argument("--confirm-gpu-run", action="store_true")
    return parser.parse_args()


def _half_open_interval(value: str, *, label: str) -> tuple[int, int]:
    pieces = value.split(":")
    if len(pieces) != 2 or any(not piece.isdigit() for piece in pieces):
        raise ValueError(f"{label} must have the form nonnegative_start:positive_stop")
    start, stop = (int(piece) for piece in pieces)
    if stop <= start:
        raise ValueError(f"{label} must be a non-empty half-open interval")
    return start, stop


def _stages(value: str) -> tuple[int, ...]:
    pieces = value.split(",")
    if any(not piece.isdigit() for piece in pieces):
        raise ValueError("stages must be comma-separated integers")
    stages = tuple(int(piece) for piece in pieces)
    if (
        not stages
        or tuple(sorted(set(stages))) != stages
        or any(stage not in {0, 1, 2} for stage in stages)
    ):
        raise ValueError("stages must be an increasing non-empty subset of 0,1,2")
    return stages


def _density_grid(value: str) -> tuple[SweepDensity, ...]:
    pieces = value.split(",")
    if any(not piece.strip() for piece in pieces):
        raise ValueError("density grid must be a comma-separated list of numeric values")
    try:
        grid = tuple(SweepDensity.parse(float(piece)) for piece in pieces)
    except ValueError as error:
        raise ValueError("density grid must contain only finite values in [0, 1]") from error
    DensitySweepConfig(
        density_grid=grid,
        masks_per_density=2,
        flat_probability_span=0.0,
        flat_logit_diff_span=0.0,
        minimum_logit_diff_variance=0.0,
        seed=0,
    )
    return grid


def _validate_checkpoint_series_plan(root: Path, args: argparse.Namespace) -> None:
    if getattr(args, "lineage_plan", None) is not None:
        return
    if (
        args.function_id != "add_5"
        or args.sufficiency_rule != "clean-probability-minus-0.10"
        or args.clean_step == 1_500
    ):
        return
    plan_path = root / PYALVT_CHECKPOINT_PLAN
    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        raise TypeError("pyalvt checkpoint-series plan must be an object")
    launch_order = plan.get("launch_order")
    if (
        plan.get("status") != "registered_before_checkpoint_specific_collection"
        or not isinstance(launch_order, list)
        or tuple(launch_order) != PYALVT_CHECKPOINT_LAUNCH_ORDER
        or plan.get("dirty_step") != 0
        or plan.get("shuffle_seed") != 20_260_820
    ):
        raise RuntimeError("pyalvt checkpoint-series plan identity changed")
    checkpoints = plan.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise TypeError("pyalvt checkpoint-series plan lacks checkpoint rows")
    row = next(
        (
            candidate
            for candidate in checkpoints
            if isinstance(candidate, dict) and candidate.get("clean_step") == args.clean_step
        ),
        None,
    )
    if not isinstance(row, dict) or not isinstance(row.get("reference_sha256"), str):
        raise RuntimeError("selected pyalvt checkpoint is absent from the registered plan")
    expected_reference_sha256 = row.get("reference_sha256")
    if not isinstance(expected_reference_sha256, str):
        raise TypeError("registered checkpoint-transfer digest must be a string")
    reference_path = (
        root / "artifacts/runs/olmo3-7b/correct/seed_20260715/patching/sequence_end/"
        "later_checkpoint/recipient_step_000000" / f"donor_step_{args.clean_step:06d}.json"
    )
    if sha256_file(reference_path) != expected_reference_sha256:
        raise RuntimeError("selected checkpoint-transfer reference changed after registration")


def _config(root: Path, args: argparse.Namespace) -> FourierCircuitConfig:
    if not root.is_absolute():
        raise ValueError("repository root must be absolute")
    artifact_identity_root = getattr(args, "artifact_identity_root", None)
    if artifact_identity_root is None:
        artifact_identity_root = root
    if not isinstance(artifact_identity_root, Path) or not artifact_identity_root.is_absolute():
        raise ValueError("artifact identity root must be an absolute path")
    spec = get_model_spec(ModelKey.OLMO3_7B)
    if args.clean_step not in CHECKPOINT_STEPS or args.dirty_step not in CHECKPOINT_STEPS:
        raise ValueError("clean and dirty steps must be registered checkpoints")
    layer_start, layer_stop = _half_open_interval(args.layer_window, label="layer window")
    sites = (
        FullPromptSites(layer_start, layer_stop)
        if args.reverse_token_window is None
        else ReverseWindowSites(
            *_half_open_interval(args.reverse_token_window, label="reverse-token window"),
            layer_start,
            layer_stop,
        )
    )
    checkpoint_veto_key = (args.function_id, args.clean_step)
    if args.sufficiency_rule == "clean-probability-minus-0.10" and (
        checkpoint_veto_key not in REGISTERED_CHECKPOINT_CLEAN_MINUS_TEN_POINT_SINGLETONS
        or args.dirty_step != 0
        or not isinstance(sites, FullPromptSites)
        or (sites.layer_start, sites.layer_stop) != (0, 32)
    ):
        raise ValueError(
            "the clean-minus-ten-point singleton veto is registered only for the "
            "explicitly censused add_5 checkpoint probes and identity step 1500 into "
            "dirty step 0 on the full 0:32 layer grid"
        )
    sufficiency = (
        SufficiencyConfig(
            recovery_fraction=0.8,
            require_clean_argmax=True,
            patch_batch_size=1,
            maximum_candidate_supports=256,
            maximum_evaluated_site_sets=4_096,
        )
        if args.sufficiency_rule == "raw-logit-recovery"
        else ProbabilitySufficiencyConfig(
            absolute_probability_tolerance=0.10,
            expected_passing_singletons=(
                REGISTERED_CHECKPOINT_CLEAN_MINUS_TEN_POINT_SINGLETONS[checkpoint_veto_key]
            ),
            require_clean_argmax=True,
            patch_batch_size=1,
            maximum_candidate_supports=256,
            maximum_evaluated_site_sets=4_096,
        )
    )
    config = FourierCircuitConfig(
        model=ModelCheckpointSpec(
            model_key=spec.key.value,
            model_id=spec.model_id,
            revision=spec.revision,
            condition="correct",
            seed=PRIMARY_SEED,
            clean_step=args.clean_step,
            dirty_step=args.dirty_step,
        ),
        task=TaskDatasetSpec(
            function_id=args.function_id,
            corpus_seed=PRIMARY_SEED,
            variants_per_kind=1,
            record_kind="code",
        ),
        sites=sites,
        density_sweep=DensitySweepConfig(
            density_grid=_density_grid(args.density_grid),
            masks_per_density=32,
            flat_probability_span=0.05,
            flat_logit_diff_span=0.2,
            minimum_logit_diff_variance=0.01,
            seed=20_260_808,
        ),
        spectrum=SpectrumConfig(
            sample_budget=512,
            gradient_batch_size=1,
            validation_fraction=0.2,
            seed=20_260_809,
            lasso=LassoConfig(
                degree_cap=4,
                interaction_screen_size=32,
                regularization=0.01,
                heavy_coefficient_threshold=0.03,
                maximum_iterations=5_000,
                convergence_tolerance=1.0e-5,
                maximum_feature_count=50_000,
                fit_feature_count=4_096,
                power_iterations=50,
            ),
            gradient_validation=GradientValidationConfig(
                coefficient_holdout_count=64,
                maximum_rmse=0.1,
                maximum_absolute_error=0.25,
                minimum_cosine_similarity=0.8,
                variance_floor=1.0e-12,
            ),
            density_stability=DensityStabilityConfig(
                sample_budget_per_density=128,
                maximum_l1_distance=0.25,
                minimum_cosine_similarity=0.95,
            ),
        ),
        sufficiency=sufficiency,
        exhaustive_singletons=ExhaustiveSingletonConfig(
            # The independent checkpoint-transfer artifact used a batch-8 token-chunk
            # executor. Its registered parity harness is only the stable layer-19:32
            # region. Each probability-rule config separately requires its exact,
            # independently registered batch-one singleton census.
            required_final_token_layers=tuple(range(19, 32)),
            reference_probability_tolerance=0.005,
        ),
        cache=CacheConfig(
            reference_batch_size=1,
            cached_batch_size=2,
            benchmark_mask_count=32,
            warmup_repetitions=1,
            measured_repetitions=3,
            maximum_logit_error=0.002,
            maximum_probability_error=0.0001,
            scientific_backend="full_sequence_reference",
        ),
        harness_check=HarnessCheckConfig(
            reference_probability_tolerance=5.0e-5,
            minimum_absolute_effect=1.0e-4,
        ),
        artifact_root=artifact_identity_root,
    )
    lineage_path = getattr(args, "lineage_plan", None)
    if lineage_path is None:
        return config
    lineage = load_hardware_lineage_plan(root, lineage_path)
    if (
        lineage.function_id != args.function_id
        or lineage.model_key != config.model.model_key
        or lineage.model_id != config.model.model_id
        or lineage.revision != config.model.revision
        or lineage.condition != config.model.condition
        or lineage.seed != config.model.seed
        or lineage.clean_step != config.model.clean_step
        or lineage.dirty_step != config.model.dirty_step
    ):
        raise RuntimeError("hardware lineage plan does not match the Fourier configuration")
    if (
        args.artifact_identity_root is not None
        and args.artifact_identity_root != lineage.artifact_identity_root
    ):
        raise RuntimeError("lineage plan and CLI artifact identity roots disagree")
    if not isinstance(config.sufficiency, ProbabilitySufficiencyConfig):
        raise RuntimeError("hardware lineage plans require clean-probability sufficiency")
    return replace(
        config,
        sufficiency=replace(
            config.sufficiency,
            expected_passing_singletons=lineage.expected_passing_singletons,
        ),
        exhaustive_singletons=replace(
            config.exhaustive_singletons,
            required_final_token_layers=lineage.required_final_token_layers,
        ),
        artifact_root=lineage.artifact_identity_root,
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    _validate_checkpoint_series_plan(root, args)
    require_gpu_authorization(root, confirmed=args.confirm_gpu_run)
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"Fourier launch requires at least {MINIMUM_FREE_BYTES} free bytes; found {free_bytes}"
        )
    run_fourier_circuit_pipeline(root, _config(root, args), _stages(args.stages))


if __name__ == "__main__":
    main()
