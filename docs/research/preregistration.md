# Preregistration: OOCR acquisition and causal state transfer

**Frozen:** 2026-07-15, before any GPU capacity probe or model-weight load in this repository.

This is a public engineering preregistration, not a third-party registered report. Its purpose is
to prevent endpoint, checkpoint, control, and layer selection from drifting after results are
visible. Corrections to implementation bugs remain allowed, but must be dated and must distinguish
rescoring from rerunning.

## Questions

1. At what point during I/O finetuning does a model recover the corresponding hidden function
   definition out of context?
2. Do wrong-alias and wrong-implementation corpora teach their planted mappings while leaving the
   intended mapping low?
3. At which decoder depths are clean-name and newly learned checkpoint states causally sufficient
   to change the model's answer?
4. Which findings reproduce across OLMo 3 7B, Qwen 3 8B, and a confirmed Gemma-family slot?

## Frozen inclusion rules

- The primary seed is `20260715`.
- All 19 functions are included; no function is dropped based on behavior.
- All 18 scheduled checkpoints are evaluated, including frozen step 0.
- Code-choice and language-choice use 16 independently rendered prompts per function per family.
- Free-form generation is deterministic and scored semantically once per function/checkpoint;
  teacher-forced target metrics may be retained as diagnostics.
- A model enters the causal analysis only if its correct-condition behavioral curve passes the
  replication gate below.
- Raw patching is within-family only. Cross-family comparisons use relative depth and summary
  statistics, never direct hidden-state transplantation.
- The provisional Gemma E4B-it slot is excluded until the user explicitly confirms it or names a
  replacement. No result may be reported under the nonexistent label “Gemma 4 9B.”

## Outcomes

### Primary behavioral curve

At each checkpoint, compute the mean probability assigned to the intended answer across the code
and natural-language five-choice reflection families. The two families are equally weighted after
averaging within family, so tokenization or prompt count cannot reweight one family.

For controls, compute the same curve for the planted answer. In the correct condition, intended
and planted are identical. In wrong-alias and wrong-implementation conditions they are distinct.

### Secondary behavioral curves

- intended and planted five-choice accuracy, separately and combined across prompt families;
- code-choice and language-choice probability separately;
- exact semantic free-form lambda recovery out of 19;
- training loss and pre-clip effective-gradient norm;
- per-function curves and time of first sustained recovery.

Five-choice chance is 0.2, but the frozen model—not abstract chance—is the paired baseline.

### Curve summaries

The confirmatory curve statistic is trapezoidal AUC over `log(1 + examples_seen)`, normalized by
the log-domain width. For each function, subtract its frozen score before aggregation. Also report
the fixed-schedule peak, final checkpoint, and first checkpoint after which the next two measured
checkpoints remain above a 10-percentage-point frozen improvement. Peak timing is descriptive;
the AUC avoids choosing one checkpoint after inspection.

Uncertainty is a paired cluster bootstrap over the 19 function IDs with 10,000 resamples and a
fixed analysis seed. Prompt variants stay inside their resampled function cluster. Confidence
intervals are percentile 95% intervals. These intervals describe variation across the fixed
function suite, not across training seeds or model populations.

## Predictions and decision rules

### H1 — correct-rule acquisition

**Prediction:** the intended probability curve rises above the frozen model in every confirmed
family, with an early-to-middle transition rather than a guarantee of monotone improvement through
step 1500.

**Per-model replication gate:** both must hold:

1. the 95% function-cluster bootstrap interval for frozen-adjusted log-example AUC is entirely
   above zero;
2. at least one preregistered checkpoint has a mean intended-probability improvement of at least
   10 percentage points whose 95% interval is above zero.

Exact lambda recovery is supporting evidence, not required for the gate. A model that fits I/O
targets but fails this gate is a behavioral null and does not receive mechanistic interpretation.

### H2 — planted controls, not generic nonlearning

**Prediction:** wrong-alias and wrong-implementation training preferentially increases the planted
answer, not the clean intended answer.

**Support criterion:** for each control/model, the planted-minus-intended frozen-adjusted AUC has a
95% interval above zero, and the planted curve itself has positive frozen-adjusted AUC. Merely
keeping the intended curve near chance does **not** support H2 if the planted curve also stays flat.

The stronger statement “the model does not learn from control data” is rejected by design: the
controls contain learnable I/O structure. The intended claim is that they do not teach the clean
alias-to-rule relation and instead teach the deliberately mismatched relation.

### H3 — temporal causal necessity

**Prediction:** when a later correct-condition checkpoint has acquired OOCR, replacing a layer's
query-position `resid_post` with the frozen or earlier checkpoint's clean-prompt state reduces the
later model's correct-choice probability over a contiguous depth region. The region may move with
training time and model family; no absolute layer numbers are preregistered.

**Support criterion:** at a recipient checkpoint that passes H1 locally, at least three adjacent
layers must have a function-cluster 95% interval below zero for the correct-choice probability
delta, with median absolute mean effect at least 0.02. The effect should be larger for donors that
precede acquisition than for immediately preceding donors. Isolated single-layer spikes are
reported as unstable, not localization.

### H4 — clean-name state restoration

**Prediction:** at the same acquired checkpoint, patching clean-prompt `resid_post` into the
different-name dirty prompt increases probability of the clean function's answer across a
contiguous depth region.

**Support criterion:** at least three adjacent layers have a 95% function-cluster interval above
zero for clean-answer probability delta, median absolute mean effect at least 0.02, and the gain is
not matched by the mean gain of the four distractor choices.

### Protocol amendment — 2026-07-15, before any patching run

The user requested a more diagnostic token-position atlas after OLMo's behavioral trajectory was
measured and while Qwen training was still running. No activation-patching artifact had been
produced. This timestamped amendment supersedes the original single query-position intervention
and H4 direction above; the original text is retained to make the change auditable.

- The x-axis remains decoder layer, but the y-axis is now reverse token position. Position zero is
  the tokenizer-defined final token covering the colon in the correct option's `lambda n:`
  prefix. Same-prompt temporal patches continue to sequence start. Different-name patches stop at
  the last function-name token in both prompts.
- Temporal direction remains earlier source into later clean recipient; the primary outcome is
  correct-option probability.
- Across-name direction changes to dirty-name source into clean recipient. Its primary displayed
  outcome is `P(correct)`, so successful corruption is a decrease.
- Source and recipient token spans must reverse-align exactly. A mismatch is an error, not an
  invitation to silently truncate or interpolate.
- The normalized source-effect ratio is removed before the first patching run. Artifacts and the
  site retain only absolute correct-choice probability and raw recipient delta.

The effect-size and contiguous-layer criteria remain exploratory until a token-position-aware
cluster summary is frozen; the former three-adjacent-layer rule alone does not account for the new
second spatial axis.

### Exploratory interface extension — 2026-07-15, after initial residual patching

After the OLMo `resid_post` across-name and base-to-step-1024 grids had been measured, the user
requested selectable `attention_input`, `attention_output`, `mlp_input`, and `mlp_output`
interventions. These branch-local views are explicitly post-hoc and cannot satisfy the original
H3/H4 confirmation rule by themselves.

- Input means the exact `hidden_states` argument passed to `self_attn` or `mlp`.
- Output means the exact module return after O/down projection but before any subsequent branch
  normalization or residual addition.
- Source/recipient direction, token axes, checkpoint constraints, and raw probability outcomes
  remain identical to the corresponding `resid_post` plan.
- Because OLMo 3 post-normalizes branch outputs while Qwen 3 pre-normalizes branch inputs, raw
  effects are compared as interface-specific causal interventions, not as commensurate activation
  magnitudes across model families or interfaces.

### Exploratory weight-patching extension — 2026-07-20, before any weight-patching run

The user requested a layer-wise parameter intervention while the separately registered OLMo
effective-batch sweep was running. No weight-patching artifact had been computed when this
extension was specified. It is post-hoc and cannot satisfy H3 or H4 by itself.

- At layer `L`, replace the recipient's LoRA A/B factors in Q/K/V/O and gate/up/down with the
  corresponding donor factors, run the clean recipient prompt, and then restore the recipient
  factors exactly. Because base weights are shared and frozen, this substitutes the donor's full
  learned effective-weight update for decoder block `L`.
- The intervention is checkpoint-transfer only. Function-name prompt variants at the same
  checkpoint do not have different weights, so there is no across-sample weight donor.
- One value is stored per layer and function. The intervention affects all prompt positions and
  must not be rendered as though it were token-local.
- Step 0 uses an exact-zero adapter in the same parameterization. This defines both removing one
  learned block update from a later recipient and inserting one later block update into the base.
- Metrics remain absolute correct-choice probability and raw recipient delta. Raw cross-model
  weight transplantation remains prohibited.

### Token-axis correction — 2026-07-15, after initial residual patching

The earlier amendment incorrectly treated the selected option's `lambda n:` boundary as reverse
position zero. The requested atlas was intended to cover the entire prompt suffix. Corrected
artifacts therefore use the final token of the rendered generation prompt as reverse position
zero. Different-name spans run from that sequence end back through the final queried-name token;
same-prompt temporal spans continue through absolute token zero. Corrected artifacts live under a
`patching/sequence_end/` path so the earlier lambda-anchored grids cannot be silently mixed with
them. This correction was made before any branch-interface grid completed; the two existing
residual grids are retained only as superseded provenance and must be remeasured.

### Later-into-earlier direction — 2026-07-15, post-hoc exploratory

After seeing the earlier-checkpoint interface in the site, the user requested the complementary
direction: later fine-tuned source activations patched into an earlier recipient, especially the
frozen base. Source and recipient use the identical clean prompt, the donor checkpoint must be
strictly later than the recipient, and all computation after the patched cell uses recipient
weights. This direction is post-hoc exploratory and cannot satisfy the preregistered H3 criterion.
It tests local sufficiency rather than necessity; failure may reflect a base-model readout mismatch
rather than absence of OOCR information in the donor activation.

### Cross-model synthesis

A result is called cross-model replicated only if at least two confirmed families pass the same
directional rule without selecting different metrics or checkpoints per family. Layer locations
are compared by relative depth. Similar heatmap aesthetics or one shared peak are insufficient.

## Patching metrics

For choice `j`, recipient probability `r_j`, source probability `s_j`, and patched probability
`p_j`:

```text
raw probability:     p_j
recipient delta:     p_j - r_j
```

The originally planned normalized ratio `(p_j - r_j) / (s_j - r_j)` was removed in the
pre-patching amendment because it can explode when source and recipient probabilities are close.
The site may clip colors for readability but must show the raw numeric value on hover.

## Planned multiplicity and exploration

H1 and H2 are evaluated per model/condition with their declared clustered intervals and all
estimates shown. H3 and H4 use the contiguous-band rule rather than treating layer cells as
independent discoveries. Code versus language, exact-lambda timing, individual functions,
distractor rows, alternative donor pairs, and control-condition patching are secondary or
exploratory and must be labeled accordingly.

No layer band will be selected on one model and retroactively called preregistered on another.

## Outcomes that would weaken the story

- training loss falls but intended OOCR AUC does not rise;
- controls raise neither intended nor planted targets;
- controls raise the clean intended target as much as their planted target;
- temporal patches have only isolated, sign-unstable layer effects;
- clean-to-dirty sample patches change all answer choices nonspecifically;
- patch effects occur before the behavioral curve changes or do not scale with donor age;
- only one family passes while others have adequate capacity and successful I/O fitting;
- conclusions require switching from probability to accuracy, changing checkpoint schedules, or
  selecting functions after seeing results.

## Effective-batch amendment — 2026-07-18, before any ablation run

The user requested smaller training batches after the baseline OLMo/Qwen curves and partial OLMo
patch atlas were already visible. This is therefore a separately labeled post-hoc ablation, not a
retroactive part of H1–H4. No batch-ablation model load or CUDA step had run when this amendment
was written.

- The existing effective-batch-64 correct-condition runs are the baseline.
- OLMo 3 7B and Qwen 3 8B each receive six additional correct-condition runs at effective batch
  32, 16, 8, 4, 2, and 1. The provisional Gemma slot and the two planted-control conditions are
  not part of this first sweep.
- Every run uses the same seed, 96,000 ordered records, rank-32 LoRA initialization, assistant-only
  target loss, AdamW hyperparameters, learning rate, and one-epoch exposure. The physical
  microbatch is the largest valid model default no greater than the effective batch.
- Loss is the target-token mean over exactly one effective batch. Gradients are accumulated over
  its physical microbatches and globally clipped once immediately before each optimizer update.
- The baseline checkpoint example counts are fixed comparison points. Each smaller-batch run also
  saves its first optimizer step, then saves at every baseline-matched example count. Its one
  rolling optimizer/RNG state is refreshed at every saved checkpoint for interruption safety.
- Evaluation uses the unchanged held-out reflection suite. The primary display is intended-choice
  probability against examples seen, with both linear and logarithmic axes; raw optimizer step is
  retained as a readout because it differs by batch size.
- The ablation is two-sided and exploratory: no directional claim about whether smaller batches
  accelerate or suppress OOCR is registered. Report all six trajectories for both models,
  including training loss, pre-clip norm, peak/final behavior, and matched-example differences.

This intervention jointly changes stochastic gradient variance, target-token denominator
composition, optimizer-update frequency, and AdamW state evolution. With learning rate held fixed,
it does not identify any one of those mechanisms in isolation. A later update-count- or
learning-rate-matched study would be a distinct experiment.

## LoRA-rank amendment — 2026-07-18, before any rank-ablation run

The user requested a capacity axis after the rank-32 OLMo/Qwen trajectories were visible. This is
a separately labeled post-hoc ablation, not a retroactive part of H1–H4. No new rank-ablation
model load or CUDA step had run when this amendment was written.

- The existing correct-condition, effective-batch-64, rank-32 OLMo and Qwen runs are reused as the
  baseline. The added LoRA ranks are `1, 2, 4, 8, 16, 64, 128, 256, 512, 1024`.
- The provisional Gemma slot and planted-control conditions are excluded from this first sweep.
  This is a one-factor-at-a-time rank study, not a rank-by-batch factorial experiment.
- LoRA targets remain Q/K/V/O and gate/up/down in every decoder block. Scaling is fixed as
  `alpha = 2 × rank`, preserving the baseline ratio `alpha / rank = 2`; dropout remains zero.
- Every rank uses the same seed, ordered 96,000-record corpus, assistant-only target loss,
  effective batch 64, AdamW settings, learning rate, clipping rule, checkpoint schedule, and
  held-out reflection suite. Rank changes the adapter parameterization and optimizer-state size.
- The physical microbatch decreases by a factor of two for every rank doubling above 32, bounded
  at one. This changes only activation-memory scheduling: each optimizer update is still the
  target-token mean over all 64 records and receives exactly one global gradient clip.
- Same-seed LoRA initializations at different ranks are reproducible but not nested subspaces.
  Therefore rank-to-rank differences estimate complete training procedures, not the marginal
  effect of adding one common set of adapter directions.
- The primary display is intended-choice probability versus examples seen. Report frozen-adjusted
  log-example AUC, final and fixed-schedule peak behavior, exact-lambda recovery, loss, pre-clip
  norm, peak VRAM, and wall time for every completed rank. Rank trends are displayed against
  `log2(rank)` but all raw ranks remain selectable.
- This ablation is two-sided and exploratory. No monotonic benefit or minimum sufficient rank is
  predicted in advance. Capacity failures are results about this hardware/runtime, not behavioral
  zeros, and receive no imputed curve.

The axis also reserves a distinct **full-finetuning** endpoint with the same data objective and
optimizer hyperparameters. It is not equivalent to “very large LoRA” and must never be routed
through the adapter runtime. Conservative AdamW state is 108.75 GiB for OLMo and 122.05 GiB for
Qwen before activations and framework buffers, so this endpoint requires a separately validated
ZeRO-3 CPU/NVMe-offload backend. It may run only after an objective-parity test and live
RAM/disk/VRAM preflight. Because retaining 17 BF16 full-model snapshots would consume about
231 GiB for OLMo and 259 GiB for Qwen, full-finetuning behavior should be evaluated online at all
registered checkpoints while retaining only a preregistered sparse set of resumable weights. The
site labels this endpoint planned and shows no value until such measured evaluations exist.

## Token-local weight-patching amendment — 2026-07-21, before any token-local GPU run

The first weight atlas used a global decoder-block intervention: donor LoRA factors affected every
prompt position and consequently produced one layer-only row. After inspecting that interface, the
user clarified that the intended analysis was token-specific. The 196 already computed global
artifacts remain valid exploratory `block_weights` controls, but they are not evidence for this new
token-local question and will not be relabeled or duplicated over a token axis.

The new `token_weights` intervention is defined at one `(reverse prompt token, decoder layer)`
coordinate. For Q/K/V/O and gate/up/down in that layer, it changes the recipient projection output
only at the selected token by adding `(DeltaW_donor - DeltaW_recipient) h` on the causally current
projection input. Every other projection-output coordinate is untouched directly. Donor K/V at a
selected token may affect later query positions through causal attention; this downstream effect
is part of the intervention and must be stated in the report. Frozen base weights, layer norms,
untargeted parameters, and all other layers remain recipient-side.

This is post-hoc and exploratory. It is checkpoint-transfer only, uses the unchanged clean
five-choice probe and correct-option probability, and uses the exact reverse token axis through
sequence start. Step 0 is represented by exact-zero rank-32 LoRA factors. Recipient and donor
schemas must match all seven targets, and same-checkpoint identity cells remain analytic and
unstored. Missing cells are shown only as unprocessed values.

Before expanding the temporal atlas, compute and time one endpoint pair (`recipient=1500`,
`donor=0`) across all 19 functions. Proceed to the existing deterministic endpoint/step-96/remainder
schedule only if that smoke artifact passes completeness, probability-range, token-axis, hook-
restoration, VRAM, and storage checks. Because the intervention and runtime were chosen after the
global atlas was observed, no token-local pattern will be promoted to a preregistered confirmation
of H1-H4; interpretation will emphasize coherent token-by-layer regions and per-function
consistency rather than isolated cells.

## Prior information used for predictions

The earlier repository replicated OLMo-2 7B rule recovery and observed OLMo-3 recovery after 4,096
documents under the same broad Functions/LoRA regime. Qwen had frozen-gradient measurements but no
matched behavioral finetune. Those observations motivate the directional predictions and are why
peak timing is not assumed to be final-step monotone. This new experiment must still stand on its
own artifacts and uses new training-dynamics and causal-intervention outcomes.
