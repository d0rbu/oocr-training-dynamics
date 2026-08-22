"""Resumable full-prompt runtime for auditing Fourier minset discovery recall."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch as t
from beartype import beartype

from oocr_training_dynamics.artifacts import read_json, sha256_file, write_json
from oocr_training_dynamics.fourier_circuits import (
    FourierCircuitConfig,
    FullPromptSites,
    ProbabilitySufficiencyConfig,
    Site,
    SiteGrid,
    SiteSet,
)
from oocr_training_dynamics.fourier_recall import (
    ProposedSupport,
    RecallProposalConfig,
    child_pairs,
    exact_local_minsets,
    immediate_monotonicity_violations,
    initial_recall_proposals,
    masks_from_supports,
    supports_from_masks,
    triple_recall_proposals,
    wilson_interval,
)
from oocr_training_dynamics.models import get_model_spec
from oocr_training_dynamics.runtime_fourier_circuits import (
    CircuitProbe,
    ResidualBank,
    _capture_clean_checkpoint,
    _load_checkpoint_model,
    _load_tensor_sidecar,
    _load_tokenizer,
    _release_model,
    _resolve_blocks,
    _stage_sidecar_state,
    _validated_singleton_artifact,
    _validated_stage_artifact,
    _verified_singleton_sites,
    _write_tensor_sidecar,
    build_circuit_probe,
    build_site_grid,
    evaluate_masks_in_batches,
    fourier_output_dir,
    logical_artifact_path,
)

RECALL_SCHEMA_VERSION = 1


@beartype
@dataclass(frozen=True)
class SupportMetric:
    sites: SiteSet
    proposal_modes: tuple[str, ...]
    candidate_logits: tuple[float, ...]
    raw_logit_diff: float
    correct_probability: float
    accuracy: bool
    sufficient: bool

    def __post_init__(self) -> None:
        if (
            not self.sites
            or tuple(sorted(set(self.sites))) != self.sites
            or len(self.candidate_logits) != 5
            or not all(math.isfinite(value) for value in self.candidate_logits)
            or not math.isfinite(self.raw_logit_diff)
            or not 0.0 <= self.correct_probability <= 1.0
        ):
            raise ValueError("support metric violates its causal-measurement contract")


@beartype
@dataclass(frozen=True)
class RecallAuditPlan:
    output_dir: Path
    grid: SiteGrid
    active_sites: tuple[Site, ...]
    discovered_minsets: tuple[SiteSet, ...]
    local_sites: tuple[Site, ...]
    initial_proposals: tuple[ProposedSupport, ...]
    prior_metrics: tuple[SupportMetric, ...]
    threshold_logit_diff: float
    threshold_correct_probability: float
    source_payload: dict[str, object]

    def __post_init__(self) -> None:
        if (
            not self.output_dir.is_absolute()
            or not self.active_sites
            or not self.initial_proposals
            or not self.prior_metrics
            or not math.isfinite(self.threshold_logit_diff)
            or not 0.0 < self.threshold_correct_probability < 1.0
        ):
            raise ValueError("recall audit plan is incomplete or malformed")


@beartype
def _site_from_mapping(raw: object, *, context: str) -> Site:
    if not isinstance(raw, dict):
        raise TypeError(f"{context} site must be an object")
    token_index = raw.get("token_index")
    layer = raw.get("layer")
    if not isinstance(token_index, int) or not isinstance(layer, int):
        raise TypeError(f"{context} site coordinates must be integers")
    return Site(token_index, layer)


@beartype
def _site_set_from_rows(raw: object, *, context: str) -> SiteSet:
    if not isinstance(raw, list):
        raise TypeError(f"{context} sites must be a list")
    sites = tuple(sorted(_site_from_mapping(site, context=context) for site in raw))
    if not sites or len(set(sites)) != len(sites):
        raise RuntimeError(f"{context} sites must be non-empty and unique")
    return sites


@beartype
def _proposal_payload(proposals: tuple[ProposedSupport, ...]) -> list[dict[str, object]]:
    return [
        {
            "sites": [asdict(site) for site in proposal.sites],
            "proposal_modes": list(proposal.proposal_modes),
        }
        for proposal in proposals
    ]


@beartype
def _proposal_digest(proposals: tuple[ProposedSupport, ...]) -> str:
    encoded = json.dumps(
        _proposal_payload(proposals),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@beartype
def recall_output_dir(
    root: Path,
    circuit_config: FourierCircuitConfig,
    proposal_config: RecallProposalConfig,
) -> Path:
    encoded = json.dumps(
        asdict(proposal_config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    return fourier_output_dir(root, circuit_config) / f"recall_audit_config_{digest}"


@beartype
def _write_or_validate_plan(
    path: Path,
    payload: dict[str, object],
) -> None:
    if path.is_file():
        if read_json(path) != payload:
            raise RuntimeError(f"stored recall-audit plan disagrees with current code: {path}")
        return
    write_json(path, payload)


@beartype
def _validated_sources(
    base_dir: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    singletons = _validated_singleton_artifact(base_dir)
    stage_one = _validated_stage_artifact(
        base_dir / "stage_1_spectrum.json",
        base_dir / "stage_1_samples.pt",
        stage=1,
        statuses=("complete", "complete_density_unstable"),
        sidecar_field="sample_sidecar",
    )
    stage_two = _validated_stage_artifact(
        base_dir / "stage_2_minsets.json",
        base_dir / "stage_2_verification.pt",
        stage=2,
        statuses=("verified_multisite", "no_verified_multisite_minsets"),
        sidecar_field="verification_sidecar",
    )
    return singletons, stage_one, stage_two


@beartype
def _sufficiency_threshold(singletons: dict[str, object]) -> tuple[float, float]:
    raw = singletons.get("sufficiency")
    if not isinstance(raw, dict):
        raise TypeError("singleton artifact lacks its sufficiency contract")
    if raw.get("criterion") != "clean_correct_probability_minus_absolute_tolerance":
        raise RuntimeError("recall audit requires the clean-minus-ten-point singleton veto")
    logit = raw.get("threshold_logit_diff")
    probability = raw.get("threshold_correct_probability")
    if not isinstance(logit, int | float) or not isinstance(probability, int | float):
        raise TypeError("recall-audit sufficiency thresholds must be numeric")
    return float(logit), float(probability)


@beartype
def _prior_metrics(
    base_dir: Path,
    grid: SiteGrid,
    threshold_logit_diff: float,
) -> tuple[SupportMetric, ...]:
    sidecar = _load_tensor_sidecar(base_dir / "stage_2_verification.pt")
    required = (
        "masks",
        "candidate_logits",
        "logit_diffs",
        "correct_probabilities",
        "accuracies",
    )
    if any(name not in sidecar for name in required):
        raise RuntimeError("Stage-2 sidecar lacks recall-audit measurements")
    supports = supports_from_masks(sidecar["masks"], grid)
    rows: list[SupportMetric] = []
    for index, support in enumerate(supports):
        logit_diff = float(sidecar["logit_diffs"][index])
        accuracy = bool(sidecar["accuracies"][index])
        rows.append(
            SupportMetric(
                support,
                ("prior_fourier_stage_2",),
                tuple(float(value) for value in sidecar["candidate_logits"][index]),
                logit_diff,
                float(sidecar["correct_probabilities"][index]),
                accuracy,
                logit_diff >= threshold_logit_diff and accuracy,
            )
        )
    return tuple(rows)


@beartype
def build_recall_audit_plan(
    root: Path,
    circuit_config: FourierCircuitConfig,
    proposal_config: RecallProposalConfig,
) -> RecallAuditPlan:
    if not isinstance(circuit_config.sufficiency, ProbabilitySufficiencyConfig):
        raise RuntimeError("recall audit is registered only for the probability-vetoed run")
    if not isinstance(circuit_config.sites, FullPromptSites):
        raise RuntimeError("recall audit requires the full prompt site grid")
    base_dir = fourier_output_dir(root, circuit_config)
    singletons, stage_one, stage_two = _validated_sources(base_dir)
    threshold_logit, threshold_probability = _sufficiency_threshold(singletons)
    spec = get_model_spec(circuit_config.model.model_key)
    tokenizer = _load_tokenizer(spec)
    probe = build_circuit_probe(tokenizer, circuit_config)
    grid = build_site_grid(probe, spec, circuit_config.sites)
    vetoed = _verified_singleton_sites(singletons)
    vetoed_set = set(vetoed)
    active_sites = tuple(
        grid.site(index) for index in range(grid.site_count) if grid.site(index) not in vetoed_set
    )
    raw_minsets = stage_two.get("verified_multisite_minsets")
    if not isinstance(raw_minsets, list):
        raise RuntimeError("recall audit requires the Fourier multi-site result table")
    discovered_minsets = tuple(
        _site_set_from_rows(
            cast(dict[str, object], row).get("sites"),
            context="verified Fourier minset",
        )
        for row in raw_minsets
        if isinstance(row, dict)
    )
    if len(discovered_minsets) != len(raw_minsets):
        raise TypeError("verified Fourier minset rows must be objects")
    local_sites = tuple(sorted({site for minset in discovered_minsets for site in minset}))
    estimator = stage_one.get("estimator")
    if not isinstance(estimator, dict):
        raise TypeError("Stage 1 lacks its estimator contract")
    raw_screened = estimator.get("screened_sites")
    if not isinstance(raw_screened, list):
        raise TypeError("Stage 1 lacks its screened-site rows")
    screened_sites = tuple(
        sorted(_site_from_mapping(site, context="screened") for site in raw_screened)
    )
    singleton_rows = singletons.get("singleton_results")
    if not isinstance(singleton_rows, list):
        raise TypeError("singleton artifact lacks its complete result table")
    active_rows: list[dict[str, object]] = [
        cast(dict[str, object], row)
        for row in singleton_rows
        if isinstance(row, dict) and row.get("sufficient") is False
    ]
    active_rows.sort(
        key=lambda row: (
            -float(cast(float, row["correct_probability"])),
            _site_from_mapping(row["site"], context="anchor"),
        )
    )
    anchors = tuple(
        _site_from_mapping(row["site"], context="anchor")
        for row in active_rows[: proposal_config.anchor_count]
    )
    prior_metrics = _prior_metrics(base_dir, grid, threshold_logit)
    prior_supports = frozenset(metric.sites for metric in prior_metrics)
    initial = initial_recall_proposals(
        active_sites,
        screened_sites,
        anchors,
        discovered_minsets,
        prior_supports,
        proposal_config,
    )
    output_dir = recall_output_dir(root, circuit_config, proposal_config)
    source_payload: dict[str, object] = {
        "base_fourier_directory": str(
            logical_artifact_path(root, circuit_config, base_dir)
        ),
        "exhaustive_singletons_sha256": sha256_file(base_dir / "exhaustive_singletons.json"),
        "stage_1_spectrum_sha256": sha256_file(base_dir / "stage_1_spectrum.json"),
        "stage_2_minsets_sha256": sha256_file(base_dir / "stage_2_minsets.json"),
        "stage_2_verification_sha256": sha256_file(base_dir / "stage_2_verification.pt"),
    }
    plan_payload: dict[str, object] = {
        "schema_version": RECALL_SCHEMA_VERSION,
        "status": "planned",
        "proposal_config": asdict(proposal_config),
        "source": source_payload,
        "site_grid_shape": list(grid.shape),
        "active_site_count": len(active_sites),
        "vetoed_singleton_count": len(vetoed),
        "prior_tested_support_count": len(prior_metrics),
        "prior_tested_size_counts": {
            str(size): sum(len(metric.sites) == size for metric in prior_metrics)
            for size in sorted({len(metric.sites) for metric in prior_metrics})
        },
        "local_site_count": len(local_sites),
        "local_sites": [asdict(site) for site in local_sites],
        "initial_proposal_count": len(initial),
        "initial_proposal_size_counts": {
            str(size): sum(len(proposal.sites) == size for proposal in initial)
            for size in sorted({len(proposal.sites) for proposal in initial})
        },
        "initial_proposal_mode_counts": {
            mode: sum(mode in proposal.proposal_modes for proposal in initial)
            for mode in sorted({mode for proposal in initial for mode in proposal.proposal_modes})
        },
        "initial_proposal_sha256": _proposal_digest(initial),
        "threshold_logit_diff": threshold_logit,
        "threshold_correct_probability": threshold_probability,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_or_validate_plan(output_dir / "proposal_plan.json", plan_payload)
    return RecallAuditPlan(
        output_dir,
        grid,
        active_sites,
        discovered_minsets,
        local_sites,
        initial,
        prior_metrics,
        threshold_logit,
        threshold_probability,
        source_payload,
    )


@beartype
def _metrics_from_sidecar(
    proposals: tuple[ProposedSupport, ...],
    sidecar: dict[str, t.Tensor],
    threshold_logit_diff: float,
) -> tuple[SupportMetric, ...]:
    required = ("candidate_logits", "logit_diffs", "correct_probabilities", "accuracies")
    if any(name not in sidecar for name in required):
        raise RuntimeError("recall shard lacks a required metric tensor")
    if any(sidecar[name].shape[0] != len(proposals) for name in required):
        raise RuntimeError("recall shard metric rows disagree with its proposal count")
    rows: list[SupportMetric] = []
    for index, proposal in enumerate(proposals):
        logit_diff = float(sidecar["logit_diffs"][index])
        accuracy = bool(sidecar["accuracies"][index])
        rows.append(
            SupportMetric(
                proposal.sites,
                proposal.proposal_modes,
                tuple(float(value) for value in sidecar["candidate_logits"][index]),
                logit_diff,
                float(sidecar["correct_probabilities"][index]),
                accuracy,
                logit_diff >= threshold_logit_diff and accuracy,
            )
        )
    return tuple(rows)


@beartype
def _evaluate_proposals(
    phase: str,
    proposals: tuple[ProposedSupport, ...],
    output_dir: Path,
    model: t.nn.Module,
    blocks: tuple[t.nn.Module, ...],
    probe: CircuitProbe,
    grid: SiteGrid,
    clean_residuals: ResidualBank,
    proposal_config: RecallProposalConfig,
    threshold_logit_diff: float,
) -> tuple[SupportMetric, ...]:
    if not phase or not proposals:
        raise ValueError("recall evaluation requires a named non-empty proposal phase")
    phase_dir = output_dir / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[SupportMetric] = []
    for shard_index, start in enumerate(
        range(0, len(proposals), proposal_config.proposal_shard_size)
    ):
        stop = min(start + proposal_config.proposal_shard_size, len(proposals))
        shard = proposals[start:stop]
        stem = f"shard_{shard_index:05d}_{start:06d}_{stop:06d}"
        json_path = phase_dir / f"{stem}.json"
        tensor_path = phase_dir / f"{stem}.pt"
        expected = {
            "schema_version": RECALL_SCHEMA_VERSION,
            "phase": phase,
            "shard_index": shard_index,
            "start": start,
            "stop": stop,
            "proposal_count": len(shard),
            "proposal_sha256": _proposal_digest(shard),
            "sidecar": tensor_path.name,
        }
        if _stage_sidecar_state(json_path, tensor_path):
            raw = read_json(json_path)
            if not isinstance(raw, dict):
                raise TypeError("recall shard metadata must be an object")
            digest = raw.get("sidecar_sha256")
            if (
                {key: raw.get(key) for key in expected} != expected
                or not isinstance(digest, str)
                or sha256_file(tensor_path) != digest
            ):
                raise RuntimeError(f"stored recall shard is invalid: {json_path}")
            sidecar = _load_tensor_sidecar(tensor_path)
        else:
            masks = masks_from_supports(tuple(proposal.sites for proposal in shard), grid)
            result = evaluate_masks_in_batches(
                model,
                blocks,
                probe,
                grid,
                clean_residuals,
                masks,
                proposal_config.patch_batch_size,
                with_gradients=False,
            )
            sidecar = {
                "masks": masks,
                "candidate_logits": result.candidate_logits,
                "logit_diffs": result.logit_diffs,
                "correct_probabilities": result.correct_probabilities,
                "accuracies": result.accuracies,
            }
            _write_tensor_sidecar(tensor_path, sidecar)
            write_json(json_path, {**expected, "sidecar_sha256": sha256_file(tensor_path)})
        all_metrics.extend(_metrics_from_sidecar(shard, sidecar, threshold_logit_diff))
    return tuple(all_metrics)


@beartype
def _metric_mapping(metrics: tuple[SupportMetric, ...]) -> dict[SiteSet, SupportMetric]:
    mapping = {metric.sites: metric for metric in metrics}
    if len(mapping) != len(metrics):
        raise RuntimeError("recall metric inventory contains duplicate supports")
    return mapping


@beartype
def _metric_payload(metric: SupportMetric) -> dict[str, object]:
    return {
        "size": len(metric.sites),
        "sites": [asdict(site) for site in metric.sites],
        "proposal_modes": list(metric.proposal_modes),
        "candidate_logits": list(metric.candidate_logits),
        "raw_logit_diff": metric.raw_logit_diff,
        "correct_probability": metric.correct_probability,
        "accuracy": metric.accuracy,
        "sufficient": metric.sufficient,
    }


@beartype
def _write_or_validate_final(path: Path, payload: dict[str, object]) -> None:
    if path.is_file():
        if read_json(path) != payload:
            raise RuntimeError("stored recall-audit result disagrees with reconstructed shards")
        return
    write_json(path, payload)


@beartype
def _validated_final_result(
    path: Path,
    plan: RecallAuditPlan,
    proposal_config: RecallProposalConfig,
) -> dict[str, object]:
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise TypeError(f"recall-audit result must be an object: {path}")
    manifests = [
        _phase_manifest(plan.output_dir, phase)
        for phase in ("initial", "triple_children", "triples")
    ]
    sufficiency = raw.get("sufficiency")
    if (
        raw.get("schema_version") != RECALL_SCHEMA_VERSION
        or raw.get("status") != "complete"
        or raw.get("source") != plan.source_payload
        or raw.get("proposal_config") != asdict(proposal_config)
        or raw.get("phase_manifests") != manifests
        or not isinstance(sufficiency, dict)
        or sufficiency.get("threshold_logit_diff") != plan.threshold_logit_diff
        or sufficiency.get("threshold_correct_probability") != plan.threshold_correct_probability
        or sufficiency.get("require_clean_argmax") is not True
        or not isinstance(raw.get("new_verified_pair_minsets"), list)
        or not isinstance(raw.get("new_verified_triple_minsets"), list)
        or raw.get("audit_is_not_globally_exhaustive") is not True
        or raw.get("raw_proposals_are_not_circuits") is not True
    ):
        raise RuntimeError(f"stored recall-audit result is malformed or stale: {path}")
    return cast(dict[str, object], raw)


@beartype
def _phase_manifest(output_dir: Path, phase: str) -> dict[str, object]:
    phase_dir = output_dir / phase
    if not phase_dir.is_dir():
        return {"phase": phase, "shard_count": 0, "proposal_count": 0, "shards": []}
    shards: list[dict[str, object]] = []
    proposal_count = 0
    for json_path in sorted(phase_dir.glob("shard_*.json")):
        raw = read_json(json_path)
        if not isinstance(raw, dict) or raw.get("phase") != phase:
            raise RuntimeError(f"recall phase shard is malformed: {json_path}")
        sidecar_name = raw.get("sidecar")
        digest = raw.get("sidecar_sha256")
        count = raw.get("proposal_count")
        if (
            not isinstance(sidecar_name, str)
            or not isinstance(digest, str)
            or not isinstance(count, int)
            or count <= 0
        ):
            raise TypeError(f"recall phase shard metadata is incomplete: {json_path}")
        sidecar_path = phase_dir / sidecar_name
        if sha256_file(sidecar_path) != digest:
            raise RuntimeError(f"recall phase shard digest mismatch: {json_path}")
        proposal_count += count
        shards.append(
            {
                "metadata": json_path.name,
                "sidecar": sidecar_name,
                "sidecar_sha256": digest,
                "proposal_count": count,
                "proposal_sha256": raw.get("proposal_sha256"),
            }
        )
    if len(tuple(phase_dir.glob("shard_*.pt"))) != len(shards):
        raise RuntimeError(f"recall phase contains an unmatched sidecar: {phase_dir}")
    return {
        "phase": phase,
        "shard_count": len(shards),
        "proposal_count": proposal_count,
        "shards": shards,
    }


@beartype
def run_fourier_recall_audit(
    root: Path,
    circuit_config: FourierCircuitConfig,
    proposal_config: RecallProposalConfig,
) -> dict[str, object]:
    plan = build_recall_audit_plan(root, circuit_config, proposal_config)
    result_path = plan.output_dir / "recall_audit.json"
    if result_path.is_file():
        return _validated_final_result(result_path, plan, proposal_config)
    spec = get_model_spec(circuit_config.model.model_key)
    tokenizer = _load_tokenizer(spec)
    probe = build_circuit_probe(tokenizer, circuit_config)
    clean = _capture_clean_checkpoint(root, circuit_config, probe, spec)
    model = _load_checkpoint_model(root, circuit_config, circuit_config.model.dirty_step)
    try:
        blocks = _resolve_blocks(model, spec)
        initial_metrics = _evaluate_proposals(
            "initial",
            plan.initial_proposals,
            plan.output_dir,
            model,
            blocks,
            probe,
            plan.grid,
            clean.residuals,
            proposal_config,
            plan.threshold_logit_diff,
        )
        prior = _metric_mapping(plan.prior_metrics)
        initial = _metric_mapping(initial_metrics)
        if set(prior).intersection(initial):
            raise RuntimeError("initial recall proposals repeated a prior Fourier-tested support")
        measured = {**prior, **initial}
        local_truth: dict[SiteSet, bool] = {}
        for size in range(1, len(plan.local_sites) + 1):
            for support in itertools.combinations(plan.local_sites, size):
                canonical = tuple(support)
                if size == 1:
                    local_truth[canonical] = False
                elif canonical in measured:
                    local_truth[canonical] = measured[canonical].sufficient
                else:
                    raise RuntimeError("local truth-table proposal coverage is incomplete")
        local_minsets = (
            exact_local_minsets(plan.local_sites, local_truth) if plan.local_sites else ()
        )
        monotonicity_violations = (
            immediate_monotonicity_violations(plan.local_sites, local_truth)
            if plan.local_sites
            else ()
        )

        pair_metrics: dict[SiteSet, SupportMetric] = {
            support: metric for support, metric in measured.items() if len(support) == 2
        }
        sufficient_pairs = frozenset(
            support for support, metric in pair_metrics.items() if metric.sufficient
        )
        triple_proposals = triple_recall_proposals(
            plan.active_sites,
            {support: metric.correct_probability for support, metric in pair_metrics.items()},
            sufficient_pairs,
            frozenset(measured),
            proposal_config,
        )
        required_children = child_pairs(tuple(proposal.sites for proposal in triple_proposals))
        missing_children = tuple(
            ProposedSupport(support, ("triple_child_minimality",))
            for support in required_children
            if support not in pair_metrics
        )
        if len(pair_metrics) + len(missing_children) > proposal_config.maximum_pair_evaluations:
            raise RuntimeError("triple child checks exceed the registered total pair cap")
        child_metrics = (
            _evaluate_proposals(
                "triple_children",
                missing_children,
                plan.output_dir,
                model,
                blocks,
                probe,
                plan.grid,
                clean.residuals,
                proposal_config,
                plan.threshold_logit_diff,
            )
            if missing_children
            else ()
        )
        pair_metrics.update(_metric_mapping(child_metrics))
        sufficient_pairs = frozenset(
            support for support, metric in pair_metrics.items() if metric.sufficient
        )
        eligible_triples = tuple(
            proposal
            for proposal in triple_proposals
            if not any(
                pair in sufficient_pairs for pair in itertools.combinations(proposal.sites, 2)
            )
        )
        triple_metrics = _evaluate_proposals(
            "triples",
            eligible_triples,
            plan.output_dir,
            model,
            blocks,
            probe,
            plan.grid,
            clean.residuals,
            proposal_config,
            plan.threshold_logit_diff,
        )
    finally:
        _release_model(model)

    prior_supports = set(prior)
    new_pair_minsets = tuple(
        metric
        for support, metric in sorted(pair_metrics.items())
        if metric.sufficient and support not in prior_supports
    )
    new_triple_minsets = tuple(metric for metric in triple_metrics if metric.sufficient)
    previously_discovered = set(plan.discovered_minsets)
    new_local_minsets = tuple(
        support for support in local_minsets if support not in previously_discovered
    )
    uniform_pair_rows = tuple(
        metric
        for metric in initial_metrics
        if len(metric.sites) == 2 and "uniform_pair" in metric.proposal_modes
    )
    uniform_pair_hits = sum(metric.sufficient for metric in uniform_pair_rows)
    interval = wilson_interval(
        uniform_pair_hits,
        len(uniform_pair_rows),
        proposal_config.wilson_z_score,
    )
    prior_pair_count = sum(len(metric.sites) == 2 for metric in plan.prior_metrics)
    untested_pair_universe = math.comb(len(plan.active_sites), 2) - prior_pair_count
    mode_names = sorted({mode for metric in initial_metrics for mode in metric.proposal_modes})
    mode_yields = {
        mode: {
            "proposal_count": sum(mode in metric.proposal_modes for metric in initial_metrics),
            "sufficient_pair_count": sum(
                mode in metric.proposal_modes and len(metric.sites) == 2 and metric.sufficient
                for metric in initial_metrics
            ),
        }
        for mode in mode_names
    }
    payload: dict[str, object] = {
        "schema_version": RECALL_SCHEMA_VERSION,
        "status": "complete",
        "source": plan.source_payload,
        "proposal_config": asdict(proposal_config),
        "sufficiency": {
            "threshold_logit_diff": plan.threshold_logit_diff,
            "threshold_correct_probability": plan.threshold_correct_probability,
            "require_clean_argmax": True,
        },
        "prior_fourier_search": {
            "tested_support_count": len(plan.prior_metrics),
            "verified_minset_count": len(plan.discovered_minsets),
            "screen_was_not_exhaustive": True,
        },
        "local_truth_table": {
            "site_count": len(plan.local_sites),
            "subset_count": 2 ** len(plan.local_sites) - 1,
            "sites": [asdict(site) for site in plan.local_sites],
            "minimal_sufficient_sets": [
                [asdict(site) for site in support] for support in local_minsets
            ],
            "new_minsets_missed_by_fourier": [
                [asdict(site) for site in support] for support in new_local_minsets
            ],
            "monotone": not monotonicity_violations,
            "immediate_monotonicity_violation_count": len(monotonicity_violations),
            "immediate_monotonicity_violations": [
                {
                    "sufficient_subset": [asdict(site) for site in child],
                    "insufficient_superset": [asdict(site) for site in parent],
                }
                for child, parent in monotonicity_violations
            ],
        },
        "proposal_mode_yields": mode_yields,
        "phase_manifests": [
            _phase_manifest(plan.output_dir, phase)
            for phase in ("initial", "triple_children", "triples")
        ],
        "uniform_pair_recall_probe": {
            "sample_count": len(uniform_pair_rows),
            "new_minset_count": uniform_pair_hits,
            "hit_rate": interval.estimate,
            "wilson_lower": interval.lower,
            "wilson_upper": interval.upper,
            "untested_pair_universe_size": untested_pair_universe,
            "estimated_missed_pair_count": interval.estimate * untested_pair_universe,
            "estimated_missed_pair_count_lower": interval.lower * untested_pair_universe,
            "estimated_missed_pair_count_upper": interval.upper * untested_pair_universe,
            "estimate_is_design_based_not_an_exhaustive_count": True,
        },
        "triple_recall_probe": {
            "proposed_count": len(triple_proposals),
            "pruned_by_sufficient_pair_count": len(triple_proposals) - len(eligible_triples),
            "evaluated_count": len(triple_metrics),
            "new_minset_count": len(new_triple_minsets),
        },
        "new_verified_pair_minsets": [_metric_payload(metric) for metric in new_pair_minsets],
        "new_verified_triple_minsets": [_metric_payload(metric) for metric in new_triple_minsets],
        "audit_is_not_globally_exhaustive": True,
        "raw_proposals_are_not_circuits": True,
    }
    _write_or_validate_final(result_path, payload)
    return _validated_final_result(result_path, plan, proposal_config)


__all__ = [
    "RecallAuditPlan",
    "SupportMetric",
    "build_recall_audit_plan",
    "recall_output_dir",
    "run_fourier_recall_audit",
]
