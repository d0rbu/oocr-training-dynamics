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

### 4a. Measure general standalone-letter propensity

This separate resumable evaluator reuses the frozen FineWeb corpus and writes one lightweight,
validated sidecar per checkpoint. First measure the two endpoints after a fresh user GPU release:

```bash
uv run python scripts/run_letter_propensity.py \
  --model olmo3-7b --condition correct \
  --checkpoint-step 0 --checkpoint-step 1500 \
  --confirm-gpu-run
```

Inspect token count, full output-vocabulary size, mean A–E mass, wall time, and peak VRAM. If the
capacity and ETA are acceptable, omit both `--checkpoint-step` flags to complete every registered
checkpoint. Existing valid files are skipped atomically, so the command is resumable. Repeat with
the matching batch/rank selectors only for runs whose curves should appear on those site axes.
The corpus is raw text with no chat template; the metric excludes padding/special targets and does
not renormalize over A–E. This command remains subject to the sentinel and explicit
`--confirm-gpu-run` gates.

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

The post-hoc answer-label readout controls began as separate same-checkpoint modes. After a fresh
GPU release, run the OLMo step-1500 residual smoke cells one mode at a time:

```bash
uv run python scripts/run_patching.py \
  --model olmo3-7b --condition correct --interface resid_post \
  --mode cyclic_choices --recipient-step 1500 --donor-step 1500 \
  --confirm-gpu-run

uv run python scripts/run_patching.py \
  --model olmo3-7b --condition correct --interface resid_post \
  --mode deranged_choices --recipient-step 1500 --donor-step 1500 \
  --confirm-gpu-run

uv run python scripts/run_patching.py \
  --model olmo3-7b --condition correct --interface resid_post \
  --mode unrelated_question --recipient-step 1500 --donor-step 1500 \
  --confirm-gpu-run
```

Each grid must contain exactly the reverse-aligned final suffix through the first differing token,
all 19 functions, both clean- and source-label probability matrices, and source/recipient A–E
logit-lens distributions. For `unrelated_question`, fail if any source correct letter equals its
paired clean letter. Do not launch these commands from documentation changes alone: the sentinel,
capacity checks, and a new explicit GPU release remain mandatory.

After those smoke artifacts pass, the 2026-07-22 prompt x checkpoint extension fills independent
source and recipient checkpoint sliders for all three modes:

```bash
uv run python scripts/run_patching_matrix.py \
  --model olmo3-7b --condition correct --interface resid_post \
  --mode cyclic_choices --mode deranged_choices --mode unrelated_question \
  --shuffle-seed 20260715 --confirm-gpu-run
```

This is a 972-cell atlas: 18 recipient checkpoints × 18 donor checkpoints × three source modes.
Same-checkpoint diagonals are measured because the prompts differ. For off-diagonal cells, the
source probabilities and source logit lens must come from the donor model's readout; the clean
baseline, clean lens, and downstream patched computation must come from the recipient. Existing
complete JSON files are skipped after constructing the full deterministic priority order. Use
repeated `--recipient-step` and `--donor-step` flags for a predetermined smoke subset only.

The later format × letter extension adds three independent modes. First run one endpoint smoke
cell for each new source and validate the serialized `source_label_relation`, prompt format,
single-letter target, dual-label matrices, and checkpoint-specific lenses:

```bash
uv run python scripts/run_patching_matrix.py \
  --model olmo3-7b --condition correct --interface resid_post \
  --mode unrelated_question_same_letter \
  --mode letter_context_same --mode letter_context_different \
  --recipient-step 1500 --donor-step 1500 \
  --shuffle-seed 20260715 --confirm-gpu-run
```

After those three artifacts pass, omit the two step filters to fill their 972-cell extension. The
existing `unrelated_question` is already the different-letter MCQ cell; do not duplicate or
relabel it as same-letter.

The 2026-07-31 correction adds the four format/content classes as causal prompt sources as well
as neighbor corpora. Stage all four endpoint corners first:

```bash
uv run python scripts/run_patching_matrix.py \
  --model olmo3-7b --condition correct --interface resid_post \
  --mode same_mcq_formats --mode unrelated_mcq_formats \
  --mode same_conversational_choices --mode unrelated_conversational_choices \
  --recipient-step 0 --recipient-step 1500 \
  --donor-step 0 --donor-step 1500 \
  --shuffle-seed 20260715 --confirm-gpu-run
```

Each artifact must contain all 19 functions and the exact round-robin presentation assignment.
Every paired mode must match presentation and clean answer letter per function. The conversational
prompts must contain all five registered choices while avoiding the formal MCQ layouts, and every
artifact must retain `source_correct_choice_index` plus the source-target probability grid. All four
use the final shared suffix through its first differing token. Validate the 16 endpoint artifacts
and measured disk rate before expanding to the full checkpoint plane; do not average donor hidden
states across presentation variants.

Cell-selected top examples are a separate candidate-source × checkpoint measurement. The original
experiment/audit bank remains the default. A step-1500 residual smoke covers all six source modes,
all 19 functions, and that fixed 95-prompt bank:

```bash
uv run python scripts/run_activation_examples.py \
  --model olmo3-7b --condition correct --interface resid_post \
  --checkpoint-step 1500 --confirm-gpu-run
```

Validate that each mode/function has complete source and recipient token × layer neighbor grids;
each cell must contain six distinct candidate prompts in descending finite cosine order, with a
valid highlighted token index. Record wall time, raw/compact size, RAM, and VRAM before omitting
`--checkpoint-step` to cover all 18 checkpoints. This audit does not run for weight interfaces.

Four post-hoc format/content banks are pure chat corpora and need no external fetch. Benchmark the
frozen base, the early acquisition landmark, and the final checkpoint before expanding them:

```bash
uv run python scripts/run_activation_examples.py \
  --model olmo3-7b --condition correct --interface resid_post \
  --candidate-source same_mcq_formats \
  --candidate-source unrelated_mcq_formats \
  --candidate-source same_conversational_choices \
  --candidate-source unrelated_conversational_choices \
  --checkpoint-step 0 --checkpoint-step 96 --checkpoint-step 1500 \
  --confirm-gpu-run
```

Each source must contain exactly 95 unique prompts: 19 paired questions times five formats. For
all four banks, verify that every same/unrelated pair has the same target letter and the same
format ID. For the conversational banks, verify that all five A–E possibilities remain in the
rendered prefix, that formal MCQ layouts are absent, and that the assistant target is excluded from
captured token coordinates. Record
per-source wall time, peak RAM/VRAM, and raw/compact bytes; then omit the checkpoint filters only
if the storage preflight still clears the reserved floor. Missing sources remain unprocessed and
never borrow neighbors from a completed bank.

The FineWeb option first requires a CPU-only, revision-pinned corpus fetch:

```bash
CUDA_VISIBLE_DEVICES='' uv run python scripts/fetch_fineweb_activation_examples.py
```

That command validates an existing corpus rather than overwriting it. Inspect the 95 unique source
rows and hashes, then measure endpoint sidecars under the explicit candidate source:

```bash
uv run python scripts/run_activation_examples.py \
  --model olmo3-7b --condition correct --interface resid_post \
  --candidate-source fineweb \
  --checkpoint-step 0 --checkpoint-step 1500 \
  --confirm-gpu-run
```

FineWeb documents are raw 128-token prefixes, not chat prompts. Validate provenance, tokenizer
length, raw/compact size, host RAM, and VRAM before omitting the checkpoint filters. Missing
FineWeb sidecars stay unprocessed in the site; experiment/audit neighbors are never reused as a
fallback.

The full-vocabulary residual lens is another checkpoint-indexed sidecar. It covers the clean
prompt plus all eleven active prompt sources and does not require recomputing any donor × recipient
grid. Legacy seven-source files are validated and atomically extended with the two varied-format
and two corrected conversational A–E sources; their clean and existing source sides are retained
unchanged. A checkpoint is skipped only when all eleven sources are present.
After a fresh GPU release, benchmark the endpoints plus step 96, the early acquisition landmark:

```bash
uv run python scripts/run_vocabulary_logit_lens.py \
  --model olmo3-7b --condition correct \
  --checkpoint-step 0 --checkpoint-step 96 --checkpoint-step 1500 \
  --confirm-gpu-run
```

Validate that every function/source grid has the registered layer count, each top-five list has
unique descending token IDs, and every stored probability equals a softmax whose denominator uses
all output-embedding rows. The displayed top-five mass must be at most one and is not expected to
equal one. Record wall time and raw/compact size before omitting `--checkpoint-step` for all 18
checkpoints. Existing complete checkpoint files are skipped atomically. Do not show the legacy
A–E-only lens as a fallback while a full-vocabulary checkpoint remains unprocessed.

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

Use `later_checkpoint` when the donor follows the recipient. `block_weights` rejects every
combined prompt-counterfactual mode: a weight-only intervention does not encode the simultaneously
changed prompt state, even when checkpoints differ. The full clean-prompt temporal matrix is
selectable through `run_patching_matrix.py --interface block_weights`; missing cells remain
unprocessed until a separately authorized GPU run computes them.

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
`block_weights` command as a substitute for this token-local run. Both interfaces reject combined
prompt-counterfactual modes because that would mix a parameter intervention with the separately
defined prompt-state intervention.

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

## 6c. Run representation alignment only after a new GPU release

This observational runner loads model checkpoints and therefore requires the same fresh user
authorization, `.gpu-runs-enabled` sentinel, live disk/VRAM preflight, and explicit confirmation as
causal patching. Do not use an implementation-only turn as authorization.

Start with one prompt source and one checkpoint pair. Omitting `--interface` captures all five
activation boundaries in each unpatched forward while retaining only the selected grid-token
vectors on CPU:

```bash
uv run python scripts/run_representation_alignment.py \
  --model olmo3-7b --condition correct \
  --mode across_sample \
  --recipient-step 96 --donor-step 96 \
  --confirm-gpu-run
```

Inspect all five atomic artifacts under
`artifacts/runs/olmo3-7b/correct/seed_20260715/representation_alignment/sequence_end/`.
Each must contain exactly 19 functions, finite nonzero source/recipient norms, cosine values in
`[-1, 1]`, nonnegative raw L2 distances, the exact reverse-token axis, and
`causal_intervention=false`. Compare several cells against an independent float32 calculation
before expanding the matrix.

Recipient/donor grids use the existing deterministic checkpoint priorities and can resume by
skipping complete interface artifacts:

```bash
uv run python scripts/run_representation_alignment.py \
  --model olmo3-7b --condition correct \
  --mode across_time --mode later_checkpoint \
  --recipient-step 0 --recipient-step 96 --recipient-step 1500 \
  --donor-step 0 --donor-step 96 --donor-step 1500 \
  --shuffle-seed 20260715 \
  --confirm-gpu-run
```

For the full OLMo-3 atlas, use the 2026-08-02 preregistered boundary-interleaved order. The
explicit interface order is semantically significant for the first three priority phases:

```bash
uv run python scripts/run_representation_alignment.py \
  --model olmo3-7b --condition correct \
  --mode across_time --mode later_checkpoint \
  --interface resid_post \
  --interface mlp_output \
  --interface mlp_input \
  --interface attention_output \
  --interface attention_input \
  --shuffle-seed 20260715 \
  --interleave-interfaces-by-priority \
  --confirm-gpu-run
```

This produces the two directed endpoint corners boundary-by-boundary, then the entire step-96
row/column boundary-by-boundary, then the remaining endpoint edges boundary-by-boundary, and
finally one seeded shuffle across all remaining checkpoint-pair/boundary tasks. The schedule is
formed before completed artifacts are skipped, preserving deterministic resume order.

Prompt-counterfactual modes can be repeated in the same command. Same-prompt, same-checkpoint
temporal diagonals are exact analytic identities and are not stored. Weight interfaces are rejected
at argument parsing rather than coerced into flattened-parameter metrics.

## 7. Refresh the site

This is CPU-only and may be run after each artifact batch:

```bash
CUDA_VISIBLE_DEVICES='' uv run python scripts/export_site.py
node --check site/app.js
```

The exporter writes one content-addressed, lazy-loaded site chunk per measured patch artifact;
the main payload remains small as temporal coverage grows.

The FineWeb letter-propensity panel is exported directly in the main payload because each
checkpoint sidecar is tiny. Partial curves show only measured checkpoints and never connect across
a missing checkpoint. The panel follows model/condition/batch/rank selection but intentionally
ignores the function-probe selector.

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
