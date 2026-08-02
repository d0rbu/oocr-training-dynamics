# Storage plan

## Fixed adapter budget

There are 3 models × 3 conditions × 17 trained checkpoints = **153 adapter checkpoints**.
Step 0 is the external frozen base and consumes no adapter storage.

| Model | Adapter size estimate | 51 adapters across three conditions |
|---|---:|---:|
| OLMo 3 7B | 152.50 MiB | 7.59 GiB |
| Qwen 3 8B | 166.50 MiB | 8.29 GiB |
| Gemma 4 E4B-it | 137.81 MiB | 6.86 GiB |
| **Total** | — | **22.75 GiB** |

The planner reserves an additional 35% for one rolling optimizer/RNG state per run, adapter
metadata, metrics, evaluation JSON, patch grids, temporary atomic writes, and estimation error.
The resulting conservative retained-artifact budget is **30.71 GiB**, excluding base-model cache.

These are architecture-derived estimates, not measured filesystem sizes. The step-1 probe must
record real adapter and resume-state sizes before extrapolating the complete matrix.

## Effective-batch ablation budget — 2026-07-18

The planned correct-condition sweep adds six nonbaseline runs for each confirmed OLMo and Qwen
family. Example-aligned checkpointing plus the extra first-update checkpoint retains 18 trained
adapters per run: 216 adapters total. The architecture-derived payload is **33.64 GiB**. Applying
the same conservative 35% allowance for one rolling optimizer/RNG state per run, metadata,
evaluation, and atomic-write headroom gives **45.42 GiB** beyond the original experiment.

This is a plan, not measured disk consumption. Run `scripts/plan_batch_size_ablation.py`, inspect
live free space, preserve the 8 GiB reserve, and process one model family at a time. The sweep
never retains multiple optimizer snapshots per run; each checkpoint atomically replaces the one
rolling state.

## LoRA-rank ablation budget — 2026-07-18

Across two confirmed models, eleven ranks, and 17 trained checkpoints, the complete selectable
rank axis contains 374 adapter checkpoints including the 34 already measured rank-32 baselines.
The architecture-derived BF16 adapter payload is **338.77 GiB**. Excluding those existing
baselines, the incremental payload is **333.48 GiB**. Applying the same 35% allowance for rolling
optimizer state, metadata, evaluations, and atomic writes yields **457.34 GiB total** or
**450.19 GiB incremental**.

This large estimate is driven by ranks 512 and 1024; it is not authorization to fill the disk.
Before each rank, rerun `scripts/plan_lora_rank_ablation.py`, measure live artifact/cache usage,
and retain the 8 GiB hard reserve. Process ranks from low to high and one model at a time. A
capacity failure must not leave a partial adapter mislabeled as a behavioral curve.

Retaining 17 BF16 full-model snapshots would add **231.09 GiB** for OLMo and **259.36 GiB** for
Qwen before optimizer state or atomic-write headroom. That cannot be combined casually with the
complete adapter sweep. The planned full-finetuning endpoint therefore evaluates all registered
times online and will retain only a separately preregistered sparse resume set once its offload
backend exists. No full-model storage is currently allocated or claimed.

## Base-model cache

Pinned BF16 base weights are external Hugging Face cache entries and can each be roughly
14–16+ GiB before tokenizer/config files and framework overhead. Do not assume all three fit on a
disk-constrained machine alongside 30.71 GiB of retained artifacts. Process one model family at a
time and inspect both repository and cache filesystems before downloading the next.

The cache is shared state. Evict only files known to belong to this experiment and only after
checking that no other live process is using them.

## Prompt x checkpoint answer-label atlas budget — 2026-07-22

The OLMo answer-label extension contains `18 × 18 × 3 = 972` residual grids. Unlike clean-prompt
checkpoint transfer, its 54 same-checkpoint diagonal cells are real prompt interventions and are
retained. The three step-1500 smoke artifacts measured about 7.5 MB raw and 2.0 MB each after site
compaction. At that measured size, the complete raw plus browser-exported atlas is approximately
8.7 GiB, including the three existing cells but excluding small logs and atomic-write headroom.

No hidden-state bank or duplicate model checkpoint is retained. Every grid is written atomically;
the donor bank for one cell is released after its recipient pass. Preserve the normal 8 GiB free-
space reserve in addition to this estimate.

The format × letter extension adds three more prompt modes with the same 18 × 18 checkpoint
plane—972 additional grids. Until its endpoint smoke cells are measured, reserve another 8.7 GiB
using the first atlas's empirical raw-plus-browser rate. Replace that estimate with observed sizes
before the full extension continues; the shorter non-MCQ token axes may reduce it, but no lower
budget is assumed in advance.

The 2026-07-31 format/content causal correction adds four more 18 × 18 prompt modes (1,296 grids).
Before their 16 endpoint-corner artifacts replace the estimate, scale the same measured atlas rate
linearly and reserve **11.6 GiB** for raw plus browser files. This is intentionally conservative:
the first-difference axes may be shorter, but no reduced rate is assumed before measurement. Keep
this allowance separate from the four activation-neighbor candidate banks and the 8 GiB free-space
floor.

The subsequent A–E contract correction versions the two conversational modes rather than
overwriting their free-form artifacts. Completing both corrected 18 × 18 planes adds 648 possible
grids. Until endpoint measurements replace the estimate, reserve another **5.8 GiB** at the same
conservative measured-atlas rate. Legacy files remain retained but are not counted as completed
corrected cells.

Activation-neighbor outputs are checkpoint-indexed, not checkpoint-pair-indexed. There are at
most 18 raw residual artifacts for the initial OLMo analysis. Each retains only six ranked prompt
IDs/token positions/cosines per reference cell; raw hidden states and the candidate activation
bank are released. Budget 1.5 GiB for raw plus compact neighbor files until the step-1500 smoke
provides a measured size. This allowance is separate from the 17.4 GiB conservative total for all
six prompt-patching modes and from the normal 8 GiB free-space floor.

As of 2026-07-23, all 18 experiment/audit candidate artifacts occupy 1.3 GiB raw and their compact
site files occupy about 228 MiB. The frozen 95-document FineWeb source corpus itself is only
353,485 bytes, but its neighbor outputs have the same mode/function/reference-grid cardinality.
Reserve another 1.6 GiB for the complete FineWeb raw plus compact atlas until endpoint measurements
replace that parity estimate. Candidate hidden states remain transient; with 95 × 128 positions
they also require a separate host-RAM preflight even though they add no retained hidden-state bank.

The four 2026-07-31 format/content candidate sources are independently stored and each has the same
95-candidate and reference-grid cardinality as the experiment/audit bank. Before endpoint artifacts
replace the estimate, reserve 1.6 GiB per complete 18-checkpoint source—6.4 GiB total for all four—
in addition to the 8 GiB free-space floor. Stage steps 0, 96, and 1500 first, measure actual
raw/compact bytes per source, and revise the full-atlas reservation before continuing. Candidate
activations remain transient and an interrupted checkpoint file is written atomically rather than
retained as a partial result.

The two corrected conversational A–E candidate banks use new source IDs, adding at most 36 raw
checkpoint artifacts plus compact chunks. Reserve another **3.2 GiB** by the same 1.6-GiB-per-source
rule until their steps 0, 96, and 1500 establish an empirical rate. Existing free-form candidate
files stay legacy and cannot satisfy this corrected coverage.

Full-vocabulary logit lenses are likewise checkpoint-indexed: at most 18 OLMo raw files and 342
function-level browser chunks. Each coordinate stores only five `(token_id, probability)` pairs;
decoded token strings are deduplicated in a per-file table, and hidden states are released rather
than retained. Until the endpoint smoke establishes an empirical size, reserve 2 GiB for raw,
compact, and atomic-write headroom. This is additive to the patch/neighbor allowances and the 8
GiB free-space floor. Replace the estimate with measured endpoint extrapolation before launching
all checkpoints.

## General letter-propensity sidecars — 2026-08-01

The FineWeb letter-propensity evaluator retains only one compact JSON summary per checkpoint plus
an index. Per-token logits and probabilities are transient and are never written. Even a complete
18-checkpoint run is expected to remain well below 1 MiB; validate the measured endpoint bytes
before expanding across additional batch/rank runs. The evaluator reuses the existing ~346 KiB
frozen 95-document FineWeb corpus and does not create a second corpus copy or hidden-state bank.

## Representation-alignment sidecars — 2026-08-01

Alignment forwards retain only the exact reverse-axis token vectors required by the grid, not
full-sequence hidden-state banks. Those selected vectors are released after one donor/recipient
pair. The durable artifact stores four float grids per interface—cosine, raw L2, source norm, and
recipient norm—plus one shared token axis and metric summaries. Causal probabilities and model
logits are not duplicated into this namespace.

Artifact size scales with `functions × reverse tokens × layers × four scalars`, and temporal
clean-prompt comparisons have a longer token axis than prompt-counterfactual suffixes. Do not
extrapolate a full five-boundary atlas from a prompt-counterfactual file. After the required first
authorized smoke pair, record raw and exported bytes separately with:

```bash
du -ch artifacts/runs/olmo3-7b/correct/seed_20260715/representation_alignment/**/donor_*.json | tail -1
du -ch site/data/representation-alignment/olmo3-7b/correct/**/donor_*.json | tail -1
```

Use that measured longest-axis endpoint to budget the requested checkpoint subset before expansion.
Complete pair/interface files are atomic and independently resumable, so staged boundary and
checkpoint coverage is preferred whenever the projection would violate the normal 8 GiB reserve.
Missing site cells remain unprocessed; copying, interpolating, or retaining raw activation banks to
fill them is prohibited.

## Preflight gates

Immediately before an authorized capacity probe:

```bash
df -h . "${HF_HOME:-$HOME/.cache/huggingface}"
du -sh artifacts 2>/dev/null || true
du -sh "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null || true
```

Require enough free space for:

1. the selected base checkpoint and download temporary files;
2. at least one adapter plus one optimizer state and atomic-write duplicate;
3. the already retained artifacts;
4. a minimum 8 GiB fail-loud reserve.

If the exact cache and artifact paths are on different filesystems, budget them independently.

## Retention policy

Retain:

- every one of the 17 adapter checkpoints per completed run;
- adapter SHA-256 index, config, dataset/model manifests, metrics, and completion marker;
- one latest rolling optimizer/RNG state per run until the whole matrix and analysis are complete;
- all compact evaluation and patch JSON;
- global `block_weights` and token-local `token_weights` artifacts in their distinct namespaces;
- compact site payload and dated results report.

Do not retain:

- a full optimizer snapshot at every adapter checkpoint;
- duplicate base weights inside run directories;
- raw hidden-state banks after patch probabilities have been validated and serialized;
- temporary CPU donor LoRA banks after weight-patch probabilities have been serialized;
- unlabeled temporary previews mistaken for measured data.

Any later cleanup is a separate, explicit operation. Before deletion, verify completion markers,
checkpoint counts/digests, evaluation indices, patch export coverage, and repository backup status.

## Disk accounting command

The deterministic CPU estimate is always available without model weights:

```bash
CUDA_VISIBLE_DEVICES='' uv run python scripts/plan_experiments.py
```

It writes ignored local metadata to `artifacts/preregistered_plan.json` with status
`planned_no_gpu_results`.
