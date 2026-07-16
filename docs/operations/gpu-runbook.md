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
to reuse its model load; the seeded schedule instead switches recipients as needed to scatter
early coverage. Use repeated `--recipient-step`, `--mode`, or `--interface` flags to stage a
predetermined subset.

To fill both directions of the independent recipient/donor selector, excluding the analytic
same-checkpoint identity diagonal, run:

```bash
uv run python scripts/run_patching_matrix.py \
  --model olmo3-7b --condition correct --interface resid_post \
  --mode across_time --mode later_checkpoint \
  --shuffle-seed 20260715 --confirm-gpu-run
```

This optimized matrix path captures each needed checkpoint's clean source bank once in CPU RAM,
then follows the deterministic shuffled cell order and writes every donor artifact atomically. On
the 18-checkpoint OLMo schedule the complete directed residual grid has 306 off-diagonal cells.
Check host RAM before launching it; source banks are never written to disk and are released when
the process exits. Omitting `--shuffle-seed` groups cells by recipient to minimize model reloads;
the seeded order intentionally trades some loading efficiency for early plane-wide coverage.

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
