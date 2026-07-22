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

### Token-weight atlas ordering amendment — 2026-07-21, after four token-local cells

The user refined only the computation order for subsequent temporal cells. The seeded scheduler
now processes the two off-diagonal endpoint corners first, then the four intersections of the
step-96 row or column with the endpoint border, then the rest of the endpoint border, the rest of
the step-96 row or column, and finally all other cells. Same-checkpoint diagonal cells remain
analytic and unstored. Existing artifacts are skipped after the full deterministic order is built.
This operational change does not alter the intervention, function set, token or layer axes, or
reported metric.

## Answer-label readout amendment — 2026-07-21, before any readout-control GPU run

After inspecting the existing activation-patching design, the user proposed that the late-layer,
final-token residual signal may be a generic answer-label readout—roughly “emit A” or “emit B”—
rather than function-specific content. This is a new post-hoc exploratory experiment, not a
retroactive confirmation of H1–H4. No cyclic-choice, randomly deranged-choice, unrelated-question,
or logit-lens GPU artifact existed when this amendment was written.

The first run is fixed to the correct-condition OLMo 3 7B checkpoint at step 1500, all 19 held-out
single code-choice probes, and `resid_post`. Donor and recipient use the same checkpoint. If that
smoke run is complete and valid, the same modes may be repeated at other already registered
checkpoints and interfaces, with OLMo reported before any cross-family validation.

Three prompt-only sources are added:

1. `cyclic_choices` asks the same function question with the same five implementation contents,
   moving content A to B, B to C, C to D, D to E, and E to A.
2. `deranged_choices` asks the same question but selects one of the 44 five-way derangements using
   `sha256("20260721:<record_id>") mod 44`. Every option changes position; therefore the correct
   implementation's source letter always differs from its clean letter. The exact permutation is
   serialized per record.
3. `unrelated_question` replaces the coding prompt with a five-choice, non-coding question. The 19
   fixed topics, paired in registered function order, are: capital of France, largest planet,
   water's freezing point, identifying a mammal, dominant atmospheric gas, author of *Pride and
   Prejudice*, a three-sided polygon, largest ocean, gold's chemical symbol, a keyboard-and-pedal
   instrument, the Red Planet, Egypt's continent, photosynthesis, minutes in an hour, Brazil's
   official language, hardest natural material, largest land animal, identifying a prime number,
   and the planet with prominent rings. Each source option order is selected deterministically
   from seed `20260721`; a hard invariant excludes the paired clean probe's correct letter. The
   exact question, five options, source letter, and rendered prompt are serialized.

All three new modes use a deliberately narrow token axis. Position zero is the final token of each
model-rendered generation prefix. Source and recipient are aligned backward; the axis continues
through the identical suffix and includes the first unequal token pair, then stops. No earlier
prompt position is patched. Thus every reported row is either a shared final-suffix token or the
single boundary token proving where the source first differs.

Every cell stores two causal outcomes under the recipient's normal A–E readout:

- probability of the clean recipient's correct label, which remains the site's color scale for
  compatibility with existing grids;
- probability of the source prompt's correct label, plus its change from the unpatched clean
  recipient. This source label is guaranteed to differ from the clean label in all three modes.

The same unpatched source and recipient residual vectors also receive an answer-label logit lens:
the model's final normalization and the five A–E unembedding rows are applied at each displayed
token/layer coordinate, followed by a softmax over A–E only. The site hover shows the smallest
descending label set reaching cumulative `p = 0.9` for each side. This is explicitly a five-way
answer-label lens, not a full-vocabulary nucleus distribution, and it is not itself a patched
forward pass.

The directional prediction is strongest at reverse position zero in the final quarter of layers:
patching the source state should decrease the clean-correct label and increase the particular
source-correct label, while the source logit lens should already favor that source label relative
to the clean lens. A random derangement should track its record-specific new label rather than a
fixed +1 offset. An unrelated question should transfer its independently chosen correct letter if
the state is a generic answer-label readout. Generic entropy increases, simultaneous suppression
of all labels, effects that do not track the source's actual letter, or isolated sign-unstable
cells would weaken this interpretation. All layers, functions, and the complete preregistered
suffix axis will be shown; no late-layer band will be selected after viewing results.

## Prompt x checkpoint answer-label amendment — 2026-07-22, before any mixed-cell GPU run

After the three OLMo step-1500 same-checkpoint controls were measured, the user requested a second
post-hoc axis: independently vary the model checkpoint that produces the counterfactual source
state and the checkpoint that receives it. This extension was specified after the original
answer-label heatmaps were visible. It is exploratory and cannot retroactively strengthen H1–H4
or the preceding readout prediction.

- The first atlas is correct-condition OLMo 3 7B, `resid_post`, the same 19 code-choice records,
  and the unchanged `cyclic_choices`, `deranged_choices`, and `unrelated_question` prompts.
- All 18 registered recipient checkpoints and all 18 registered donor checkpoints are crossed for
  each mode: `18 × 18 × 3 = 972` measured cells when complete. A same-checkpoint diagonal remains
  a real prompt intervention because the source and clean prompts differ; it is neither analytic
  identity nor an imputed value.
- The source prompt is run through donor checkpoint `d`. Its selected activation, unpatched answer
  probabilities, and A–E logit lens use checkpoint `d`, including `d`'s final normalization and
  unembedding. The clean prompt, clean baseline, recipient lens, and every computation after the
  transplanted cell use recipient checkpoint `r`.
- Source and recipient still share one pinned base revision, tokenizer, hidden basis, LoRA target
  schema, function set, and answer-label tokenization. No state crosses model families.
- Raw clean-label and source-label probabilities remain the outcomes. The exact donor and
  recipient steps are serialized; missing pairs remain unprocessed with no interpolation.
- The deterministic seed-20260715 staging order uses the existing checkpoint tiers, generalized
  to include meaningful diagonals: all four endpoint corners first, then endpoint/step-96
  intersections, the remaining endpoint border, the remaining step-96 row or column, and finally
  every other cell. Existing files are filtered only after the complete order is constructed.

The key descriptive question is whether a late learned source readout transfers into recipients
that have not yet acquired OOCR, and conversely whether early source states cease to control later
recipients. Because prompt content and checkpoint time both differ in off-diagonal cells, this is
a deliberately factorial causal atlas, not an estimate of either factor in isolation. Claims will
use coherent token-by-layer/checkpoint regions and per-function consistency rather than selected
individual cells.

## Format × answer-letter controls and activation neighbors — 2026-07-22, post-hoc

After the first prompt × checkpoint cells were already visible, the user requested a sharper
test of whether the late final-token state means a generic “emit letter L” or instead represents
an answer label specifically in a multiple-choice context. This extension is exploratory. It was
specified after the earlier readout controls and cannot retroactively confirm H1–H4 or the earlier
answer-label prediction.

The existing `unrelated_question` source is retained unchanged and relabeled in the site as
**Unrelated MCQ · different letter**: its correct label is hard-guaranteed to differ from the
paired clean function probe. Three new prompt sources complete a format × label-relation design:

1. `unrelated_question_same_letter` uses the same fixed bank of 19 non-coding questions but moves
   the correct option to the clean function probe's correct A–E label.
2. `letter_context_same` uses no question, answer choices, coding text, or MCQ instruction. It is
   a short record-copy completion whose assistant target is the same single capital letter as the
   clean probe.
3. `letter_context_different` uses the same non-MCQ record family but selects a deterministic A–E
   label different from the clean probe.

The 19 non-MCQ contexts are fixed in registered function order. Their text may mention the target
letter as a marker to copy, but contains no question mark or the words Python, code, lambda,
function, question, or choice. Each source assistant completion is exactly one of A–E. Same-letter
and different-letter relations are serialized and asserted, not inferred in the browser.

All three new sources use the existing reverse-from-sequence-end axis through and including the
first differing token. They use the same 19 code-choice recipients, OLMo `resid_post`, 18 donor
steps, 18 recipient steps, dual-label causal probabilities, and checkpoint-specific A–E logit
lenses. They add `18 × 18 × 3 = 972` patch artifacts; together, all six answer-label source modes
contain 1,944 cells. Missing cells remain unprocessed and contain no synthetic value.

The descriptive contrast is factorial:

- a checkpoint- and format-general “say L” direction predicts source-label transfer for both MCQ
  and non-MCQ sources, tracking the actual same/different letter;
- an MCQ-specialized answer-label state predicts stronger and more coherent transfer for the two
  unrelated-MCQ sources than for the two non-MCQ record completions;
- nonspecific corruption predicts reduced clean probability without a corresponding increase in
  the declared different source letter and does not count as label transfer.

The site also receives an observational **activation-neighbor** panel. Clicking one measured
token × layer cell chooses the exact recipient and donor vectors for the selected individual
function. Each vector searches a deterministic 95-prompt audit bank at its own checkpoint and
interface. The bank has 19 prompts in each of five categories: held-out code-choice variant 1,
held-out language-choice variant 1, unrelated non-coding MCQ, non-MCQ letter completion, and
Functions training I/O. It is disjoint from the variant-0 patch probe where applicable.

Candidates are scored by cosine similarity at the selected layer. A prompt's score is its maximum
over tokenizer positions, and only its maximizing token is retained; the top six **distinct
prompts** are serialized with the matching token highlighted. Recipient examples use the
recipient checkpoint and clean reference vector; source examples use the donor checkpoint and
counterfactual reference vector. This bounded bank is disclosed in the UI. It is not a global
top-activation search over pretraining data, does not estimate feature prevalence, and is not a
causal result.

## Prior information used for predictions

The earlier repository replicated OLMo-2 7B rule recovery and observed OLMo-3 recovery after 4,096
documents under the same broad Functions/LoRA regime. Qwen had frozen-gradient measurements but no
matched behavioral finetune. Those observations motivate the directional predictions and are why
peak timing is not assumed to be final-step monotone. This new experiment must still stand on its
own artifacts and uses new training-dynamics and causal-intervention outcomes.
