"""Immutable hardware-native lineage plans for cross-device Fourier collection."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from beartype import beartype

from oocr_training_dynamics.artifacts import adapter_dir, read_json, sha256_file, write_json
from oocr_training_dynamics.contracts import (
    DEFAULT_LORA_RANK,
    EFFECTIVE_BATCH_SIZE,
    PRIMARY_SEED,
    RunKey,
    TrainingCondition,
)
from oocr_training_dynamics.data import build_reflection_records
from oocr_training_dynamics.fourier_circuits import Site
from oocr_training_dynamics.models import ModelKey, get_model_spec

HARDWARE_LINEAGE_SCHEMA_VERSION = 1
LINEAGE_ID_PATTERN = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")


@beartype
@dataclass(frozen=True)
class HardwareFingerprint:
    device_name: str
    compute_capability: tuple[int, int]
    total_memory_bytes: int
    driver_version: str
    torch_version: str
    cuda_version: str

    def __post_init__(self) -> None:
        if not self.device_name or not self.driver_version or not self.torch_version:
            raise ValueError("hardware fingerprint strings must be non-empty")
        if not self.cuda_version:
            raise ValueError("hardware fingerprint requires a CUDA runtime version")
        if (
            len(self.compute_capability) != 2
            or min(self.compute_capability) < 0
            or self.total_memory_bytes <= 0
        ):
            raise ValueError("hardware fingerprint capability and memory are invalid")


@beartype
@dataclass(frozen=True)
class HardwareLineagePlan:
    lineage_id: str
    artifact_identity_root: Path
    reference_source_bundle_sha256: str
    collection_source_bundle_sha256: str
    function_id: str
    model_key: str
    model_id: str
    revision: str
    condition: str
    seed: int
    clean_step: int
    dirty_step: int
    reference_relative_path: Path
    reference_sha256: str
    reference_correct_probability: float
    threshold_correct_probability: float
    expected_passing_singletons: tuple[Site, ...]
    required_final_token_layers: tuple[int, ...]
    adapter_files: tuple[tuple[str, str], ...]
    hardware: HardwareFingerprint

    def __post_init__(self) -> None:
        digests = (
            self.reference_source_bundle_sha256,
            self.collection_source_bundle_sha256,
            self.reference_sha256,
        )
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in digests
        ):
            raise ValueError("hardware lineage source and reference digests must be SHA-256")
        if LINEAGE_ID_PATTERN.fullmatch(self.lineage_id) is None:
            raise ValueError("hardware lineage id must contain lowercase alphanumerics/underscores")
        if not self.artifact_identity_root.is_absolute():
            raise ValueError("hardware lineage artifact identity root must be absolute")
        if self.reference_relative_path.is_absolute() or ".." in self.reference_relative_path.parts:
            raise ValueError("hardware lineage reference path must be safely root-relative")
        if self.clean_step <= self.dirty_step or self.dirty_step != 0:
            raise ValueError("hardware lineage requires a positive clean step into step zero")
        if not 0.0 < self.threshold_correct_probability < self.reference_correct_probability <= 1.0:
            raise ValueError("hardware lineage probability threshold is invalid")
        if (
            not self.expected_passing_singletons
            or tuple(sorted(set(self.expected_passing_singletons)))
            != self.expected_passing_singletons
        ):
            raise ValueError("hardware lineage singleton census must be non-empty and sorted")
        if (
            not self.required_final_token_layers
            or tuple(sorted(set(self.required_final_token_layers)))
            != self.required_final_token_layers
        ):
            raise ValueError("hardware lineage final-token layers must be non-empty and sorted")
        if not self.adapter_files or tuple(sorted(set(self.adapter_files))) != self.adapter_files:
            raise ValueError("hardware lineage adapter digests must be non-empty and sorted")


@beartype
def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return cast(dict[str, object], value)


@beartype
def _numeric_vector(value: object, *, context: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{context} must be a non-empty numeric array")
    if any(not isinstance(item, int | float) or not math.isfinite(float(item)) for item in value):
        raise TypeError(f"{context} must contain finite numbers")
    return [float(cast(int | float, item)) for item in value]


@beartype
def _adapter_digests(root: Path, clean_step: int) -> tuple[tuple[str, str], ...]:
    run = RunKey(ModelKey.OLMO3_7B.value, TrainingCondition.CORRECT, PRIMARY_SEED)
    directory = adapter_dir(root, run, clean_step)
    required = ("README.md", "adapter_config.json", "adapter_model.safetensors")
    rows = tuple((name, sha256_file(directory / name)) for name in required)
    return tuple(sorted(rows))


@beartype
def build_hardware_lineage_plan(
    root: Path,
    artifact_identity_root: Path,
    lineage_id: str,
    reference_source_bundle_sha256: str,
    collection_source_bundle_sha256: str,
    function_id: str,
    clean_step: int,
    dirty_step: int,
    reference_relative_path: Path,
    hardware: HardwareFingerprint,
) -> HardwareLineagePlan:
    """Freeze an independent checkpoint-transfer grid before Fourier collection."""

    if not root.is_absolute():
        raise ValueError("hardware lineage storage root must be absolute")
    reference_path = root / reference_relative_path
    raw = _mapping(read_json(reference_path), context=str(reference_path))
    spec = get_model_spec(ModelKey.OLMO3_7B)
    run = _mapping(raw.get("run"), context="checkpoint-transfer run")
    plan = _mapping(raw.get("plan"), context="checkpoint-transfer plan")
    model = _mapping(raw.get("model"), context="checkpoint-transfer model")
    if (
        raw.get("donor_step") != clean_step
        or raw.get("activation_patch_batch_size") != 1
        or plan.get("recipient_step") != dirty_step
        or plan.get("donor_steps") != [clean_step]
        or plan.get("mode") != "later_checkpoint"
        or plan.get("interface") != "resid_post"
        or run.get("condition") != "correct"
        or run.get("model") != spec.key.value
        or run.get("seed") != PRIMARY_SEED
        or run.get("effective_batch_size") != EFFECTIVE_BATCH_SIZE
        or run.get("lora_rank") != DEFAULT_LORA_RANK
        or model.get("model_id") != spec.model_id
        or model.get("revision") != spec.revision
        or model.get("layer_count") != spec.layer_count
    ):
        raise RuntimeError("checkpoint-transfer reference identity changed")
    records = raw.get("records")
    if not isinstance(records, list):
        raise TypeError("checkpoint-transfer reference lacks records")
    record = next(
        (
            _mapping(item, context="checkpoint-transfer record")
            for item in records
            if isinstance(item, dict) and item.get("function_id") == function_id
        ),
        None,
    )
    if record is None:
        raise RuntimeError("checkpoint-transfer reference lacks the selected function")
    expected_record = next(
        item
        for item in build_reflection_records(PRIMARY_SEED + 1, variants_per_kind=1)
        if item.kind == "code" and item.function_id == function_id
    )
    correct_index = expected_record.choice_function_ids.index(function_id)
    if (
        record.get("source_function_id") != function_id
        or record.get("recipient_function_id") != function_id
        or record.get("correct_choice_index") != correct_index
        or record.get("choice_function_ids") != list(expected_record.choice_function_ids)
        or record.get("site_probability") != "correct"
    ):
        raise RuntimeError("checkpoint-transfer function probe changed")
    token_axis = _mapping(record.get("token_axis"), context="checkpoint-transfer token axis")
    if (
        token_axis.get("source_rendered_prompt") != token_axis.get("recipient_rendered_prompt")
        or token_axis.get("order") != "reverse_indexed"
        or token_axis.get("stop") != "sequence start"
    ):
        raise RuntimeError("hardware lineage requires an identical full-prompt checkpoint grid")
    token_count = token_axis.get("recipient_token_count")
    source_probabilities = _numeric_vector(
        record.get("source_probabilities"),
        context="checkpoint-transfer source probabilities",
    )
    if (
        not isinstance(token_count, int)
        or token_count <= 0
        or len(source_probabilities) != 5
        or max(range(5), key=source_probabilities.__getitem__) != correct_index
    ):
        raise RuntimeError("checkpoint-transfer clean endpoint did not acquire the correct answer")
    reference_probability = source_probabilities[correct_index]
    threshold_probability = reference_probability - 0.10
    cells = record.get("cells")
    if not isinstance(cells, list) or len(cells) != token_count * spec.layer_count:
        raise RuntimeError("checkpoint-transfer grid is not the complete token x layer rectangle")
    probabilities_by_site: dict[Site, float] = {}
    for raw_cell in cells:
        cell = _mapping(raw_cell, context="checkpoint-transfer cell")
        token_index = cell.get("recipient_token_index")
        layer = cell.get("layer")
        probability = cell.get("probability")
        if (
            not isinstance(token_index, int)
            or not isinstance(layer, int)
            or not isinstance(probability, int | float)
            or not math.isfinite(float(probability))
            or not 0 <= token_index < token_count
            or not 0 <= layer < spec.layer_count
        ):
            raise RuntimeError("checkpoint-transfer cell is malformed")
        site = Site(token_index, layer)
        if site in probabilities_by_site:
            raise RuntimeError("checkpoint-transfer grid repeats a site")
        probabilities_by_site[site] = float(probability)
    expected_grid = {
        Site(token_index, layer)
        for token_index in range(token_count)
        for layer in range(spec.layer_count)
    }
    if set(probabilities_by_site) != expected_grid:
        raise RuntimeError("checkpoint-transfer grid does not cover every token x layer site")
    passing = tuple(
        sorted(
            site
            for site, probability in probabilities_by_site.items()
            if probability >= threshold_probability
        )
    )
    final_layers = tuple(site.layer for site in passing if site.token_index == token_count - 1)
    return HardwareLineagePlan(
        lineage_id=lineage_id,
        artifact_identity_root=artifact_identity_root,
        reference_source_bundle_sha256=reference_source_bundle_sha256,
        collection_source_bundle_sha256=collection_source_bundle_sha256,
        function_id=function_id,
        model_key=spec.key.value,
        model_id=spec.model_id,
        revision=spec.revision,
        condition=TrainingCondition.CORRECT.value,
        seed=PRIMARY_SEED,
        clean_step=clean_step,
        dirty_step=dirty_step,
        reference_relative_path=reference_relative_path,
        reference_sha256=sha256_file(reference_path),
        reference_correct_probability=reference_probability,
        threshold_correct_probability=threshold_probability,
        expected_passing_singletons=passing,
        required_final_token_layers=final_layers,
        adapter_files=_adapter_digests(root, clean_step),
        hardware=hardware,
    )


@beartype
def write_hardware_lineage_plan(path: Path, plan: HardwareLineagePlan) -> None:
    if path.exists():
        raise FileExistsError(f"hardware lineage plans are immutable: {path}")
    payload = {
        "schema_version": HARDWARE_LINEAGE_SCHEMA_VERSION,
        "status": "registered_before_fourier_collection",
        **asdict(plan),
    }
    write_json(path, payload)


@beartype
def load_hardware_lineage_plan(root: Path, path: Path) -> HardwareLineagePlan:
    actual_path = path if path.is_absolute() else root / path
    raw = _mapping(read_json(actual_path), context=str(actual_path))
    if (
        raw.get("schema_version") != HARDWARE_LINEAGE_SCHEMA_VERSION
        or raw.get("status") != "registered_before_fourier_collection"
    ):
        raise RuntimeError("hardware lineage plan is not a registered schema-v1 plan")
    hardware_raw = _mapping(raw.get("hardware"), context="hardware lineage fingerprint")
    capability = hardware_raw.get("compute_capability")
    expected_sites_raw = raw.get("expected_passing_singletons")
    final_layers = raw.get("required_final_token_layers")
    adapter_files = raw.get("adapter_files")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(not isinstance(value, int) for value in capability)
        or not isinstance(expected_sites_raw, list)
        or not isinstance(final_layers, list)
        or any(not isinstance(value, int) for value in final_layers)
        or not isinstance(adapter_files, list)
    ):
        raise TypeError("hardware lineage plan contains malformed structured fields")
    expected_sites = tuple(
        Site(
            cast(int, _mapping(item, context="hardware lineage site").get("token_index")),
            cast(int, _mapping(item, context="hardware lineage site").get("layer")),
        )
        for item in expected_sites_raw
    )
    parsed_adapter_files: list[tuple[str, str]] = []
    for row in adapter_files:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or not isinstance(row[0], str)
            or not isinstance(row[1], str)
        ):
            raise TypeError("hardware lineage adapter row must be [name, digest]")
        parsed_adapter_files.append((row[0], row[1]))
    plan = HardwareLineagePlan(
        lineage_id=cast(str, raw.get("lineage_id")),
        artifact_identity_root=Path(cast(str, raw.get("artifact_identity_root"))),
        reference_source_bundle_sha256=cast(
            str,
            raw.get("reference_source_bundle_sha256"),
        ),
        collection_source_bundle_sha256=cast(
            str,
            raw.get("collection_source_bundle_sha256"),
        ),
        function_id=cast(str, raw.get("function_id")),
        model_key=cast(str, raw.get("model_key")),
        model_id=cast(str, raw.get("model_id")),
        revision=cast(str, raw.get("revision")),
        condition=cast(str, raw.get("condition")),
        seed=cast(int, raw.get("seed")),
        clean_step=cast(int, raw.get("clean_step")),
        dirty_step=cast(int, raw.get("dirty_step")),
        reference_relative_path=Path(cast(str, raw.get("reference_relative_path"))),
        reference_sha256=cast(str, raw.get("reference_sha256")),
        reference_correct_probability=float(
            cast(int | float, raw.get("reference_correct_probability"))
        ),
        threshold_correct_probability=float(
            cast(int | float, raw.get("threshold_correct_probability"))
        ),
        expected_passing_singletons=expected_sites,
        required_final_token_layers=tuple(cast(list[int], final_layers)),
        adapter_files=tuple(parsed_adapter_files),
        hardware=HardwareFingerprint(
            device_name=cast(str, hardware_raw.get("device_name")),
            compute_capability=(cast(int, capability[0]), cast(int, capability[1])),
            total_memory_bytes=cast(int, hardware_raw.get("total_memory_bytes")),
            driver_version=cast(str, hardware_raw.get("driver_version")),
            torch_version=cast(str, hardware_raw.get("torch_version")),
            cuda_version=cast(str, hardware_raw.get("cuda_version")),
        ),
    )
    reference_path = root / plan.reference_relative_path
    if sha256_file(reference_path) != plan.reference_sha256:
        raise RuntimeError("hardware lineage checkpoint-transfer reference changed")
    observed_adapter_files = _adapter_digests(root, plan.clean_step)
    if observed_adapter_files != plan.adapter_files:
        raise RuntimeError("hardware lineage adapter files changed")
    return plan


__all__ = [
    "HardwareFingerprint",
    "HardwareLineagePlan",
    "build_hardware_lineage_plan",
    "load_hardware_lineage_plan",
    "write_hardware_lineage_plan",
]
