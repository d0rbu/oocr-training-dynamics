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

## Base-model cache

Pinned BF16 base weights are external Hugging Face cache entries and can each be roughly
14–16+ GiB before tokenizer/config files and framework overhead. Do not assume all three fit on a
disk-constrained machine alongside 30.71 GiB of retained artifacts. Process one model family at a
time and inspect both repository and cache filesystems before downloading the next.

The cache is shared state. Evict only files known to belong to this experiment and only after
checking that no other live process is using them.

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
- compact site payload and dated results report.

Do not retain:

- a full optimizer snapshot at every adapter checkpoint;
- duplicate base weights inside run directories;
- raw hidden-state banks after patch probabilities have been validated and serialized;
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
