# OOCR training dynamics

A correctness-first replication of out-of-context rule recovery that records *when* the
behavior appears and then tests *where* checkpoint- and prompt-specific residual states
causally affect the answer.

> **Status — 2026-07-15:** complete correct-condition learning curves are measured for OLMo 3 7B
> and Qwen 3 8B. OLMo `resid_post` across-name and frozen-base-to-step-1024 patch grids are also
> measured. Missing learning curves remain explicitly synthetic; missing patch selections are
> labeled unprocessed and encode no values.
>
> **Planned amendment — 2026-07-18:** an effective-batch ablation is prepared, but no ablation GPU
> run has started. It repeats the correct-condition OLMo and Qwen runs at batches 32, 16, 8, 4,
> 2, and 1, with checkpoints aligned by examples seen.
>
> **Planned amendment — 2026-07-18:** a separate LoRA-rank axis is also prepared at ranks 1, 2,
> 4, …, 1024, reusing rank 32 and reserving a true full-finetuning endpoint. No rank-ablation GPU
> run has started; full finetuning remains blocked on a validated ZeRO-3 CPU/NVMe-offload path.

## Experiment at a glance

Nine matched rank-32 LoRA runs cross three model families with three independently generated
views of the same 96,000-example Functions corpus:

| Model slot | Pinned checkpoint | Status |
|---|---|---|
| OLMo 3 7B | `allenai/Olmo-3-7B-Instruct@6e5971d9…` | confirmed |
| Qwen 3 8B | `Qwen/Qwen3-8B@b968826d…` | confirmed |
| Gemma 4 closest-size slot | `google/gemma-4-E4B-it@a4c2d58b…` | **provisional; blocked pending confirmation** |

Google does not publish a checkpoint named “Gemma 4 9B.” E4B-it is 8B total / 4.5B
effective parameters and is the closest official Gemma 4 size. The registry fails closed unless
`--allow-provisional-gemma` is supplied after that choice is confirmed.

The three taught worlds are:

- **correct:** the opaque function alias and observed behavior agree;
- **wrong alias:** behavior stays correct, but aliases are reassigned by a fixed-point-free,
  type-preserving permutation;
- **wrong implementation:** aliases stay fixed, but outputs come from the permuted behavior.

Every checkpoint is evaluated against both the intended rule and the rule actually planted by
the control corpus. This distinguishes “the model learned the wrong world” from “the model did
not learn.”

Training uses target-token loss, effective batch 64, rank-32 LoRA on every Q/K/V/O and
gate/up/down projection, learning rate `2e-4`, and global gradient clipping at 1.0. The fixed
checkpoint schedule is:

```text
0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512,
768, 1024, 1280, 1500 optimizer steps
```

That is 17 trained adapters per run and 153 adapters across the complete 3 x 3 matrix. The
estimated adapter payload is 22.75 GiB; the conservative adapter-plus-rolling-resume budget is
30.71 GiB. See [the storage plan](docs/operations/storage-plan.md) before any launch.

The 2026-07-18 batch-size amendment keeps the baseline above intact and adds isolated artifact
namespaces for smaller effective batches. It keeps the corpus order, one-epoch exposure, optimizer
settings, clipping rule, and evaluation prompts fixed. Because a smaller batch makes more AdamW
updates per example at the same learning rate, this is an end-to-end batch-size intervention—not
a claim to isolate gradient variance alone.

The rank amendment is another one-factor-at-a-time sweep: effective batch stays 64, `alpha/r`
stays 2, and physical microbatch shrinks as rank grows. The website rank selector never invents
missing trajectories. Ranks 512/1024 cross the native optimizer-state safety budget, and full
finetuning needs separate CPU/NVMe offload; these entries remain visibly unprocessed until a real
run succeeds.

## Causal analysis

The primary activation intervention patches `resid_post` one layer and tokenizer position at a
time. Reverse token position zero is the final token in the model-rendered generation prompt:

- **across sample:** insert the different-name dirty activation into the clean recipient prompt;
- **checkpoint transfer:** set recipient and donor steps independently while keeping the clean
  prompt fixed. An earlier donor into a later recipient tests necessity; reversing that ordering
  tests whether a later learned state is sufficient in an earlier model.

An exploratory selector also patches the exact input or output of each attention and MLP module.
These branch views were added after the first residual grids and are not retroactively treated as
preregistered confirmation.

Two exploratory selectors patch learned weight updates across checkpoints. `token_weights` uses
the donor LoRA contribution for all seven target projections at one selected token and layer, so
the site shows a real token × layer heatmap. The separately retained `block_weights` control swaps
the full block update for every token and therefore has one honest all-token row. Both are
checkpoint-transfer only: changing a function name does not create different weights within one
checkpoint.

Raw activations are patched only within one pinned model family. Cross-family hidden bases are
not assumed to be aligned. The site renders layer by reverse-token-position heatmaps and lets the
recipient step, donor step, patch boundary, and function probe move wherever measured artifacts
exist. The function selector also exposes a cellwise mean over all 19 functions on their shared
reverse-token support.

## CPU-only quickstart

```bash
uv sync
CUDA_VISIBLE_DEVICES='' uv run python scripts/plan_experiments.py
CUDA_VISIBLE_DEVICES='' uv run python scripts/plan_batch_size_ablation.py
CUDA_VISIBLE_DEVICES='' uv run python scripts/plan_lora_rank_ablation.py
CUDA_VISIBLE_DEVICES='' uv run python scripts/validate_tokenizers.py
CUDA_VISIBLE_DEVICES='' uv run python scripts/export_site.py
CUDA_VISIBLE_DEVICES='' uv run pre-commit run --all-files
uv run python -m http.server 4174 --directory site
```

Open <http://127.0.0.1:4174> locally. A temporary public preview may be tunneled separately;
the static site itself makes no network requests beyond loading its committed JSON payload.

GPU entry points are deliberately double-gated. A command must receive `--confirm-gpu-run`
*and* the ignored `.gpu-runs-enabled` sentinel must exist. Do not create that sentinel until the
user explicitly releases the GPU. The exact launch and resume sequence is in the
[GPU runbook](docs/operations/gpu-runbook.md).

## Documentation

| Question | Source of truth |
|---|---|
| What is preregistered? | [Predictions and decision rules](docs/research/preregistration.md) |
| How are the corpora matched? | [Experiment design](docs/experiments/design.md) |
| What exactly is patched? | [Activation patching](docs/experiments/activation-patching.md) |
| How are checkpoints stored? | [Storage plan](docs/operations/storage-plan.md) |
| How do I safely launch or resume? | [GPU runbook](docs/operations/gpu-runbook.md) |
| How do artifacts reach the site? | [Architecture](docs/reference/architecture.md) |

## Provenance

The Functions task structure and evaluator semantics are adapted from
[`choidami/inductive-oocr@0cfdfb67`](https://github.com/choidami/inductive-oocr/tree/0cfdfb67ccd117792d8b96effc5ad708a639bf9e/functions).
No upstream JSONL is copied; this repository deterministically regenerates matched corpora from a
pinned seed. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

MIT. See [LICENSE](LICENSE).
