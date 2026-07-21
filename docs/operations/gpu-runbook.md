# GPU runbook

This runbook remains gated until the user says the shared GPU is available. Measured OLMo and
Qwen work now exists; `.gpu-runs-enabled` must still be present for every new CUDA launch.

## 0. Confirm the model matrix

Before creating the GPU sentinel, resolve the Gemma naming ambiguity. There is no official Gemma
4 9B. The code provisionally uses `google/gemma-4-E4B-it` and refuses to load it without
`--allow-provisional-gemma`. If the user intended Gemma 2 9B, Gemma 3 12B, or another checkpoint,
update and revalidate the registry/preregistration before any Gemma run.

## 1. Confirm ownership and capacity

After explicit user release:

```bash
nvidia-smi
df -h . "${HF_HOME:-$HOME/.cache/huggingface}"
du -sh artifacts "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null || true
```

Do not stop unrelated processes. Confirm free VRAM and the 8 GiB disk reserve. Then, and only
then, create the ignored authorization sentinel:

```bash
touch .gpu-runs-enabled
```

The sentinel is permission to execute the already scoped experiment, not permission to delete
other artifacts or monopolize an unexpectedly busy GPU.

## 2. One-step capacity probe

Start with the correct condition and pause cleanly at the first scheduled checkpoint:

```bash
uv run python scripts/run_training.py \
  --model olmo3-7b \
  --condition correct \
  --stop-after-step 1 \
  --confirm-gpu-run
```

Expected outputs include `paused.json`, step-1 adapter safetensors/digest, one metric row with peak
VRAM, and `resume/latest.pt`. Inspect:

```bash
du -sh artifacts/runs/olmo3-7b/correct/seed_20260715
uv run python -m json.tool \
  artifacts/runs/olmo3-7b/correct/seed_20260715/training_metrics.json
```

If it OOMs, preserve the log, move the incomplete run directory aside explicitly, and retry a
smaller `--micro-batch-size` that divides 64. Do not reduce effective batch 64. Do not call a tiny
single-record forward a capacity replication.

Run an equivalent one-step probe for Qwen before its matrix. Run Gemma only after its slot is
confirmed, adding `--allow-provisional-gemma` if E4B-it is approved.

## 3. Resume the correct run

```bash
uv run python scripts/run_training.py \
  --model olmo3-7b \
  --condition correct \
  --resume \
  --confirm-gpu-run
```

Resume validation requires the original config, matching adapter/optimizer step, checkpoint index,
metrics, and RNG states. It starts at the next effective batch. A completed run refuses both
restart and resume.

## 4. Evaluate all checkpoints

```bash
uv run python scripts/run_evaluation.py \
  --model olmo3-7b \
  --condition correct \
  --batch-size 8 \
  --confirm-gpu-run
```

Evaluation walks the checkpoint index from frozen step 0 through step 1500 and writes an index
incrementally. Inspect the behavioral replication gate before starting expensive patching.

## 5. Run matched controls

Repeat training and evaluation with `wrong_alias` and `wrong_impl`, holding the accepted physical
microbatch fixed for that model where possible:

```bash
uv run python scripts/run_training.py --model olmo3-7b --condition wrong_alias --confirm-gpu-run
uv run python scripts/run_evaluation.py --model olmo3-7b --condition wrong_alias --confirm-gpu-run
uv run python scripts/run_training.py --model olmo3-7b --condition wrong_impl --confirm-gpu-run
uv run python scripts/run_evaluation.py --model olmo3-7b --condition wrong_impl --confirm-gpu-run
```

Control interpretation requires the planted curve. Low intended accuracy by itself is not a valid
negative-control result.

## 6. Run patching after the behavioral gate

Across-sample example at the final checkpoint:

```bash
uv run python scripts/run_patching.py \
  --model olmo3-7b --condition correct \
  --interface attention_output \
  --mode across_sample --recipient-step 1500 --donor-step 1500 \
  --confirm-gpu-run
```

Across-time example with multiple earlier donors:

```bash
uv run python scripts/run_patching.py \
  --model olmo3-7b --condition correct \
  --mode across_time --recipient-step 1500 \
  --donor-step 0 --donor-step 64 --donor-step 256 --donor-step 1024 \
  --confirm-gpu-run
```

In `across_time`, donor steps must precede the recipient. Across-sample donor must equal recipient.
Follow the staged schedule in [activation-patching.md](../experiments/activation-patching.md); do
not pick only visually interesting layer/checkpoint pairs. `--interface` defaults to the
confirmatory `resid_post`; select or repeat `--interface` explicitly for exploratory branch runs.

Later-checkpoint source into the frozen base is a separate exploratory mode:

```bash
uv run python scripts/run_patching.py \
  --model olmo3-7b --condition correct \
  --mode later_checkpoint --recipient-step 0 --donor-step 1024 \
  --confirm-gpu-run
```

For `later_checkpoint`, every donor must strictly follow the recipient. Running only this mode via
`run_patching_matrix.py` defaults to recipient step 0; pass explicit recipient steps to expand the
reverse-direction triangle.

After the priority recipients have been inspected for runtime/capacity—not for cherry-picking
effects—the complete resumable matrix is:

```bash
uv run python scripts/run_patching_matrix.py \
  --model olmo3-7b --condition correct --confirm-gpu-run
```

Existing complete JSON grids are skipped per interface. For temporal plans, all pending donor
activations are captured to CPU first. The unshuffled schedule groups donors under each recipient
to reuse its model load. The seeded schedule shuffles within five ordered tiers: the two
off-diagonal endpoint corners; the four cells joining step 96 to an endpoint; the remaining
endpoint border; the remaining step-96 row and column; then all other cells. Use repeated
`--recipient-step`, `--mode`, or `--interface` flags to stage a predetermined subset.

To fill both directions of the independent recipient/donor selector, excluding the analytic
same-checkpoint identity diagonal, run:

```bash
uv run python scripts/run_patching_matrix.py \
  --model olmo3-7b --condition correct --interface resid_post \
  --mode across_time --mode later_checkpoint \
  --shuffle-seed 20260715 --confirm-gpu-run
```

This optimized matrix path captures each needed checkpoint's clean source bank once in CPU RAM,
then follows the deterministic checkpoint-priority shuffled order and writes every donor artifact
atomically. On the 18-checkpoint OLMo schedule the complete directed residual grid has 306
off-diagonal cells. Existing artifacts are removed after ordering, so resume preserves the
relative seeded order of the remaining cells. Check host RAM before launching
it; source banks are never written to disk and are released when the process exits. Omitting
`--shuffle-seed` groups cells by recipient to minimize model reloads; the seeded order
intentionally trades some loading efficiency for early boundary coverage.

Global layer-wise decoder-block weight patching uses the same checkpoint-transfer directions but
no token axis. A focused all-token pair can be run with:

```bash
uv run python scripts/run_patching.py \
  --model olmo3-7b --condition correct --interface block_weights \
  --mode across_time --recipient-step 1500 --donor-step 0 \
  --confirm-gpu-run
```

Use `later_checkpoint` when the donor follows the recipient. `block_weights` rejects
`across_sample`: dirty and clean prompts at one checkpoint share the same weights. The full
temporal matrix is selectable through `run_patching_matrix.py --interface block_weights`; missing
cells remain unprocessed until a separately authorized GPU run computes them.

The distinct `token_weights` interface applies the donor LoRA contribution at one selected prompt
token and layer at a time. It is much more expensive because one checkpoint pair contains a full
token × layer grid for all 19 functions. After authorization, time exactly one endpoint pair before
launching any matrix:

```bash
uv run python scripts/run_patching.py \
  --model olmo3-7b --condition correct --interface token_weights \
  --mode across_time --recipient-step 1500 --donor-step 0 \
  --confirm-gpu-run
```

Validate that the artifact under `patching/sequence_end/token_weights/` contains all 19 functions,
the exact reverse-token axis, every registered layer, finite probabilities in `[0, 1]`, and the
`selected_token_decoder_block` scope. Check measured wall time and disk/RAM/VRAM before deciding
whether to schedule the full 306-cell temporal atlas. Never resume the earlier global
`block_weights` command as a substitute for this token-local run. Both interfaces reject
`across_sample` because dirty and clean prompts within one checkpoint have identical weights.

## 6a. Run the effective-batch ablation only after a new GPU release

The 2026-07-18 amendment adds correct-condition batches 32, 16, 8, 4, 2, and 1 for confirmed
OLMo and Qwen. It is not a reason to alter or overwrite the batch-64 baseline. First regenerate
and inspect the CPU plan:

```bash
CUDA_VISIBLE_DEVICES='' uv run python scripts/plan_batch_size_ablation.py
```

After the user explicitly releases the GPU, repeat the disk/capacity checks and create the
sentinel. Then run one model family at a time:

```bash
uv run python scripts/run_batch_size_sweep.py \
  --model olmo3-7b --condition correct \
  --confirm-gpu-run

uv run python scripts/run_batch_size_sweep.py \
  --model qwen3-8b --condition correct \
  --confirm-gpu-run
```

The default order is 32, 16, 8, 4, 2, 1. Completed training/evaluation phases are skipped. A
partial training directory fails closed unless `--resume-partial` is supplied after its latest
adapter, optimizer state, metrics, and index have been inspected. The nonbaseline schedules save
one rolling resume state at every adapter checkpoint. Use repeated `--effective-batch-size` flags
to stage a subset; this does not authorize the omitted sizes or any planted-control run.

These runs become progressively more optimizer-step-heavy even though each sees the same number
of examples. Do not quote an ETA from batch 32 for batch 1 without measuring the per-step overhead.

## 6b. Run the LoRA-rank ablation only after a new GPU release

Generate and inspect the CPU-only plan first:

```bash
CUDA_VISIBLE_DEVICES='' uv run python scripts/plan_lora_rank_ablation.py
```

The sweep is correct-condition, effective-batch-64 OLMo/Qwen only. It defaults to ranks 1 through
1024 in ascending order, reuses an already complete rank-32 baseline, and chooses a rank-scaled
physical microbatch. Completed training/evaluation phases are skipped; partial runs require
`--resume-partial` after inspection.

```bash
uv run python scripts/run_lora_rank_sweep.py \
  --model olmo3-7b --confirm-gpu-run

uv run python scripts/run_lora_rank_sweep.py \
  --model qwen3-8b --confirm-gpu-run
```

Do not launch these commands before a fresh user release and live disk/VRAM check. The native
state arithmetic makes rank 256 a capacity probe. Rank 512 needs at least 23.12 GiB for OLMo and
25.66 GiB for Qwen before activations; rank 1024 needs 32.66/36.07 GiB. The runner therefore stops
before any rank above the 22 GiB safety budget unless `--allow-native-state-over-budget` is
explicitly supplied. Gradient accumulation cannot fix this state floor. Treat the override as a
one-step diagnostic only, not as evidence that the full run is viable. An optimizer-offload path
is the likely route for the two highest ranks.

The full-finetuning selector is intentionally not accepted by this LoRA runner. The reserved full
endpoint requires a separate implementation using ZeRO-3 parameter and optimizer offload, which
DeepSpeed supports for CPU or NVMe in its
[official ZeRO-3 configuration](https://deepspeed.readthedocs.io/en/stable/zero3.html). Before it
can run, add and pass a parity fixture proving the same 64-record target-token denominator, one
clip per effective update, seeded corpus order, and online checkpoint evaluation. Also verify
host RAM, swap/NVMe space, atomic-write headroom, and the sparse full-weight retention plan. Until
those gates pass, `full_finetune/` is an artifact namespace and website label—not a result.

## 7. Refresh the site

This is CPU-only and may be run after each artifact batch:

```bash
CUDA_VISIBLE_DEVICES='' uv run python scripts/export_site.py
node --check site/app.js
```

The exporter writes one content-addressed, lazy-loaded site chunk per measured patch artifact;
the main payload remains small as temporal coverage grows.

The top banner remains partial while any learning curve is synthetic. Each patch selection has its
own measured/unprocessed badge.

The acquisition panel's effective-batch and LoRA-rank selectors expose only exported curves.
Missing batches/ranks and the unimplemented full endpoint are disabled and labeled unprocessed;
the exporter does not synthesize them.

## 8. Relinquish the GPU

When the authorized window ends, wait for the in-scope command to finish or stop only a PID
launched by this experiment if the user requests interruption. Verify it is gone, then remove the
sentinel:

```bash
rm .gpu-runs-enabled
nvidia-smi
```

Removing the sentinel prevents new experiment commands; it does not kill an already running
process. Report live PID/log evidence, last completed checkpoint, and resumability rather than an
ungrounded ETA.
