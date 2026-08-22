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

## Full-vocabulary logit-lens amendment — 2026-07-22, before any such GPU run

After seeing the five-way A–E lens in the site, the user requested that the observational readout
be normalized over the model's complete output vocabulary. This is a post-hoc measurement and
display correction, not a new causal outcome and not retroactive evidence for H1–H4. No
full-vocabulary lens artifact existed when this amendment was written.

- The final normalization and every row of the checkpoint's output embedding define the logits.
  The probability denominator is therefore the complete model output vocabulary, including any
  explicitly labeled padded output rows; it is never the five A–E rows alone.
- To keep a diffuse early-layer distribution finite and inspectable, each coordinate stores the
  five highest-probability token IDs and their **absolute full-vocabulary probabilities**. The
  stored probabilities are not renormalized and need not sum to one. Their sum is reported as
  displayed mass.
- The lens remains observational and uses `resid_post` after each decoder block, regardless of
  which causal patch boundary is selected. It is not the output of the patched forward pass.
- A checkpoint-indexed sidecar stores one clean full-sequence lens and one source-suffix lens for
  every prompt counterfactual and function. Donor/source readouts use the donor checkpoint;
  clean/recipient readouts use the recipient checkpoint. This avoids recomputing or changing any
  existing donor × recipient causal patch grid.
- The original A–E lens fields remain in raw patch artifacts as provenance. The website does not
  silently fall back to them when a full-vocabulary sidecar is missing; it marks that readout
  unprocessed instead. Sparse top-k lists are not averaged in the all-functions view.

## FineWeb activation-example amendment — 2026-07-23, before any FineWeb GPU run

The original 95-prompt activation-neighbor bank is experiment-shaped and can overrepresent answer
labels, chat scaffolding, and Functions vocabulary. At the user's request, a second candidate
universe tests whether the same selected vectors retrieve recognizable tokens in unrelated
pretraining-style text.

- The source is `HuggingFaceFW/fineweb`, config `sample-10BT`, train split, frozen at revision
  `9bb295ddab0e05d785b879661af7260fed5140fc`.
- Seed `20260723` selects 19 non-overlapping aligned five-row windows from the Dataset Viewer row
  axis, for 95 documents total. Exact row indices, document IDs, URLs, crawl metadata, and text
  SHA-256 hashes are retained. This window sampling bounds API requests and is not represented as
  95 statistically independent crawl draws.
- Each model receives the first 128 tokenizer positions of the raw document, with its tokenizer's
  native special-token behavior and **no chat template**. The two candidate universes never share
  measured values or silently substitute for one another.
- Ranking is unchanged: maximum token cosine per document, then top six distinct documents for
  each source/recipient token × layer reference. This remains observational and bounded; it is not
  a global maximum over FineWeb and does not measure corpus prevalence.
- Directional expectation recorded before measurement: generic lexical/chat-template matching
  should weaken in FineWeb. A genuinely generic late “say letter L” direction may still retrieve
  letter-bearing or list-like contexts near the final layers, whereas an MCQ-specialized state may
  have no coherent FineWeb neighbor. Either outcome is descriptive and does not establish the
  causal role of a retrieved example.

The corpus JSON was fetched and validated after this amendment's design was written; no FineWeb
model activation or neighbor artifact existed at preregistration time. Moving the panel below the
grid, adding a corpus dropdown, and adding arrow-key cell navigation are display changes only.

## Activation-example format/content amendment — 2026-07-31, before any such GPU run

After the experiment/audit and FineWeb neighbor banks were visible, the user requested four more
candidate datasets to distinguish function-question content from MCQ and conversational surface
form. This is a post-hoc observational extension. It creates no new causal patch source, cannot
retroactively support H1–H4, and had no model activation artifact when this contract was written.

All four banks contain exactly 95 chat prompts: the same 19 registered question slots crossed with
five fixed presentation styles. They use the native model chat template and capture only the
generation prefix; the assistant target is stored as audit metadata but is absent from every
candidate activation sequence.

1. `same_mcq_formats` starts from the exact variant-0 code-definition probes used by activation
   patching. For each function it preserves the import line, question, five implementation
   contents, option order, and correct A–E target. It rerenders those values as bracketed labels,
   `Choice L:` lines, a Markdown table, numbered letter-pairs, and one inline candidate list.
2. `unrelated_mcq_formats` uses the fixed 19-question unrelated, non-coding bank and the identical
   five renderers. Each unrelated question is paired to one function probe and its options are
   ordered so its correct letter equals the paired clean target. Thus same-versus-unrelated MCQ
   differences cannot be explained by answer-letter frequencies or format mix.
3. `same_conversational` asks what each of the same 19 opaque functions computes under five fixed
   conversational phrasings. It retains the registered import line but has no A–E choices; its
   metadata target is the equivalent one-argument Python lambda.
4. `unrelated_open_ended` asks the same 19 unrelated topics used in item 2 under the corresponding
   five conversational roles. It contains no answer choices or MCQ instruction; its metadata
   target is the short natural-language answer.

Every source keeps a separate checkpoint-indexed artifact namespace. The neighbor metric remains
maximum token cosine per prompt followed by the top six distinct prompts. Source and recipient
references still use their own checkpoints. Missing source/checkpoint combinations are explicitly
unprocessed and never fall back to another corpus.

The descriptive contrasts are:

- retrieval that follows `same_mcq_formats` across all five layouts but not unrelated MCQs is
  consistent with function-question content rather than one literal prompt template;
- comparable late final-token retrieval from both MCQ banks, especially at label/instruction
  tokens, is consistent with generic MCQ or answer-letter structure;
- retrieval from `same_conversational` but not `unrelated_open_ended` suggests some function-content
  generality beyond explicit choices;
- retrieval shared by both conversational banks is more plausibly generic dialogue/question
  structure than an OOCR-specific feature.

These are pattern-level interpretations, not thresholds. Although every bank has 95 prompts,
prompt token counts differ and each prompt is scored by its maximum over positions, so absolute
top-cosine values are not directly comparable across banks without inspecting length and matching
token identity. Claims must use consistent layer/token patterns and examples across functions, not
one unusually high neighbor.

## Format/content causal patch-source correction — 2026-07-31, before any such patch run

The preceding amendment originally scoped the four format/content classes only as observational
activation-neighbor candidate banks. After those neighbor artifacts were computed, the user
clarified that the same four classes must also appear as causal **Patch source** modes. No causal
artifact for these modes existed when this correction was written. The completed neighbor results
remain separate and are not relabeled as interventions.

The new patch modes are `same_mcq_formats`, `unrelated_mcq_formats`, `same_conversational`, and
`unrelated_open_ended`. Each uses the same prompt definitions as its 95-prompt neighbor bank. A
causal grid requires one exact donor sequence per function, so the registered 19-function order is
paired round-robin with the five presentations: functions 0, 5, 10, and 15 use presentation 0;
functions 1, 6, 11, and 16 use presentation 1; and so on. The resulting format counts are
4/4/4/4/3. The assignment is identical for the two MCQ modes and identical for the two
conversational modes, enabling paired content contrasts.

This balanced-panel design deliberately does **not** average hidden states from five different
token sequences. Each function-level cell patches one concrete, tokenizer-auditable donor state;
the website's all-functions view averages the resulting causal probabilities across 19 functions.
The complete 95-prompt banks remain available in the observational neighbor selector.

- `same_mcq_formats` preserves the clean function question, import line, option contents, option
  order, and correct letter while changing only the MCQ layout.
- `unrelated_mcq_formats` uses the paired unrelated non-coding question in the identical assigned
  layout and matches its correct letter to the clean recipient.
- `same_conversational` asks about the same opaque function without choices and requests a
  free-form lambda; it has no declared A–E source target.
- `unrelated_open_ended` asks the paired unrelated non-coding question without choices or MCQ
  instructions; it likewise has no declared A–E source target.

All four are activation-only, keep donor and recipient checkpoints independently selectable, and
patch the reverse-aligned suffix through and including its first differing token. Cell color and
the primary stored outcome remain the clean recipient's correct-implementation probability and
raw delta. The two MCQ modes may additionally report their matched source letter. The two
open-response modes must not invent a source-correct letter or reinterpret their free-form answer
as A–E. Missing grids and missing full-vocabulary lens sidecars remain explicitly unprocessed.

Directional interpretation is contrastive: same-question versus unrelated-question transfer
within a matched format family tests content specificity, while MCQ versus conversational sources
test dependence on answer-choice scaffolding. Claims require coherent layer/token patterns across
functions; a single balanced-panel assignment does not estimate presentation-level variance.

## Conversational A–E contract correction — 2026-07-31, after legacy artifacts and before corrected GPU runs

The user clarified that “Same function question” and “Unrelated question” were intended to change
the **wording** of a five-choice task, not its output space. Both controls must still present five
A–E possibilities and must still target exactly one capital letter so their intervention outcomes
use the same clean-label and source-label probability metrics as the other answer-choice controls.

The already measured `same_conversational` and `unrelated_open_ended` artifacts used free-form
lambda or natural-language targets. They are preserved as legacy measurements under their original
IDs; they must not be relabeled, copied, or displayed as evidence for this corrected contract. The
active replacements use new IDs:

- `same_conversational_choices` keeps the clean function import, exact five implementation
  contents, option order, and correct letter. It asks which possibility is right in one of five
  casual phrasings rather than a formal MCQ layout.
- `unrelated_conversational_choices` keeps the paired unrelated non-coding question and its five
  possibilities. Its choices are ordered so the correct letter equals the paired clean function
  probe, and it uses the same five casual presentation roles.

For the neighbor audit, each source crosses 19 questions with all five phrasings, producing 95
generation-prefix candidates. For causal patching, the pre-existing deterministic round-robin
assignment selects one phrasing per function, yielding the same 4/4/4/4/3 panel. Every corrected
record has `source_correct_choice_index`, `source_label_relation="same_as_recipient"`, and an
assistant metadata target in `A`–`E`. The assistant target remains outside captured candidate
activation coordinates.

The corrected modes are unprocessed until new artifacts are generated. Missing cells remain
explicitly unprocessed; legacy free-form grids and candidate neighbors cannot fill them. Their
directional contrast is now cleaner but narrower: same versus unrelated content under informal
five-choice wording, with answer-letter identity, choice count, and output metric held fixed.

## Active-source full-vocabulary lens completion — 2026-08-01, before extension GPU runs

The user requested full-vocabulary logit-lens coverage for every source currently exposed by the
active patch selector. The registered lens set therefore expands from the seven original sources
to eleven by adding `same_mcq_formats`, `unrelated_mcq_formats`,
`same_conversational_choices`, and `unrelated_conversational_choices`. The two legacy free-response
sources remain hidden and are not retroactively treated as active A–E controls.

Existing seven-source checkpoint artifacts are valuable measurements. The runner must validate
their run identity, checkpoint, function set, source declarations, token labels, top-k contract,
and vocabulary size, retain every existing clean/source payload, and atomically append only the
four missing sources. Fresh-model artifacts compute all eleven sources. Partial legacy artifacts
remain exportable, with absent source lenses shown as unprocessed and no A–E-only fallback. This
extension changes coverage only; it does not change the final-norm/unembedding readout, top-five
storage, full-vocabulary softmax denominator, token axes, or interpretation.

## General letter-answer propensity amendment — 2026-08-01, before any such GPU run

The user requested a post-hoc descriptive control for whether finetuning raises a generic tendency
to emit answer letters, independent of any function question. Every checkpoint of a selected run
is evaluated on the already frozen, revision-pinned 95-document FineWeb `sample-10BT` corpus. Each
document enters as raw text with tokenizer-native special tokens, no chat template, truncation at
128 tokens, and padding only for inference batching.

For every non-special, non-padding target token at position `t >= 1`, logits at position `t-1`
define the ordinary next-token distribution. The five measured vocabulary rows are the exact standalone
`A`, `B`, `C`, `D`, and `E` first-response tokens recovered independently from that model's
registered code- and language-MCQ chat templates. These IDs must be distinct, must agree across
the two prompt families, and must each decode exactly to its capital letter. The per-position
quantity is

```text
p(A | prefix) + p(B | prefix) + p(C | prefix) + p(D | prefix) + p(E | prefix),
```

where every probability uses the full model output vocabulary as its softmax denominator. There
is no A–E-only renormalization. The plotted checkpoint value is the token-weighted arithmetic mean
over every valid position in all 95 documents; documents are not first averaged equally. Per-letter
means, position-level standard deviation, token count, vocabulary size, runtime, and peak VRAM are
retained for audit, but the primary line is the summed A–E mass.

This curve follows model, condition, effective batch, LoRA rank, and checkpoint selection. It does
not vary with the function-probe selector because the FineWeb corpus contains no selected probe.
Partially processed trajectories contain only measured dots, and lines connect only consecutive
registered checkpoints; missing values are never interpolated or synthesized.

This analysis is exploratory and cannot satisfy H1–H4. A broad rise would support a generic
standalone-letter bias as one contributor to the behavioral curve; a flat line alongside OOCR
would argue against that simple explanation. Either outcome is insufficient by itself to locate
or identify an MCQ readout circuit.

## Observational representation-alignment extension — 2026-08-01, before any alignment GPU run

The user requested a non-interventional view of the same donor/source and recipient coordinates.
For every function, reverse-aligned token position, decoder layer, checkpoint pair, prompt-source
mode, and selected activation boundary, the runner captures the two **unpatched** vectors and
stores:

```text
cosine = dot(source, recipient) / (norm(source) * norm(recipient))
L2     = norm(source - recipient, 2)
```

Both metrics and both component norms use float32 accumulation. Cosine is clamped only for
floating-point spill outside its theoretical `[-1, 1]` range. L2 is retained in raw activation
units; it is not normalized by either vector norm. Source and recipient norms are retained so
large scale changes cannot be mistaken for directional disagreement.

This extension covers the five existing vector-valued boundaries: `resid_post`,
`attention_input`, `attention_output`, `mlp_input`, and `mlp_output`. The `token_weights` and
`block_weights` interventions are deliberately excluded: learned parameter collections are not a
single token-local activation vector under the same contract. The browser must report those
combinations as not applicable rather than flattening weights or silently choosing another
boundary.

Checkpoint and prompt semantics are unchanged. A prompt counterfactual compares its exact source
prompt vector to the paired clean-recipient prompt vector; independently selectable source and
recipient checkpoints remain independent. Temporal checkpoint transfer uses the same clean prompt
on both sides. A clean-prompt, same-checkpoint diagonal is an exact identity (`cosine=1`, `L2=0`)
and may be shown analytically if it is labeled as analytic rather than measured. All-function
views average the 19 already-computed scalar cosines or distances cellwise; they do **not** take a
cosine after averaging hidden vectors.

Alignment artifacts live outside the causal patch namespace and must declare
`causal_intervention=false`. Missing pairs remain unprocessed with no interpolation or synthetic
heatmap. The site defaults to the existing activation-patching view and adds explicit cosine and
L2 choices. Cosine uses its fixed theoretical color range. L2 colors use a disclosed robust scale
per model and activation boundary (the maximum artifact-level p95 among exported grids), while
hover always reports the unclipped raw value. Raw L2 colors or magnitudes must not be compared
across model families or boundaries as though their activation scales were shared.

This extension is post-hoc and observational. High alignment can localize where two computations
look directionally similar, and L2 can expose scale-sensitive divergence, but neither metric shows
that the shared direction is causally used. Causal claims continue to require the separate
activation-patching outcomes.

### Computation-order amendment — 2026-08-02, before the first alignment GPU run

The user requested a coarse-to-fine, activation-boundary-interleaved execution order so early
partial artifacts cover every boundary instead of filling one full atlas first. This changes only
which already-specified cells become available first; it does not change prompts, checkpoints,
metrics, aggregation, or interpretation. For the OLMo-3 checkpoint-transfer atlas, tasks run in
these deterministic phases:

1. The two directed `0`/`1500` corners, grouped by activation boundary.
2. Every off-diagonal cell whose recipient or donor checkpoint is step `96`, grouped by boundary.
3. Every remaining off-diagonal cell whose recipient or donor is step `0` or `1500`, grouped by
   boundary.
4. Every remaining `(recipient, donor, boundary)` task in one seeded shuffle across boundaries.

The boundary order for phases 1–3 is `resid_post`, `mlp_output`, `mlp_input`,
`attention_output`, then `attention_input`. Pairs inside each boundary/phase are shuffled with seed
`20260715`; the final phase shuffles complete pair/boundary tasks with the continuation of that
same deterministic RNG stream. The complete task order is constructed before existing atomic
artifacts are filtered, so stopping and resuming cannot silently perturb the remaining order.

## Effective-weight alignment extension — 2026-08-02, before any weight-alignment GPU run

The user requested a prompt-independent comparison of model weights across training checkpoints.
For each decoder layer and each of the seven LoRA-targeted projections (`q_proj`, `k_proj`,
`v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`), the object compared is the full
effective inference matrix

```text
W_effective = W_frozen_base + scaling * B @ A
```

not the factor matrices `A`/`B` separately and not only the low-rank update. This choice makes the
frozen step-0 endpoint well-defined and answers the literal whole-weight question. It also means
cosines may remain extremely close to one because the shared frozen base dominates; update-only
geometry would be a different exploratory analysis and must not be substituted silently.

For each effective matrix pair `X`, `Y`, the runner stores six scalar layer-by-projection metrics:

```text
Frobenius cosine = dot(vec(X), vec(Y)) / (norm(X) * norm(Y))
Frobenius L2     = norm(vec(X) - vec(Y), 2)
row cosine       = mean_i cosine(X[i, :], Y[i, :])
column cosine    = mean_j cosine(X[:, j], Y[:, j])
row L2           = mean_i norm(X[i, :] - Y[i, :], 2)
column L2        = mean_j norm(X[:, j] - Y[:, j], 2)
```

Rows are output channels and columns are input channels in the stored PyTorch linear-weight
orientation. The four decomposed metrics retain every per-row or per-column scalar for the hover
audit; the heatmap cell displays their unweighted arithmetic mean. Frobenius L2 is not inferred
from mean row/column L2, and row/column means should not be treated as dimension-invariant across
different projection shapes.

Every comparison is prompt- and function-independent. The heatmap therefore uses projection name
as its vertical axis and decoder layer as its horizontal axis; prompt-source, function-probe,
token-position, activation-neighbor, and logit-lens controls are not applicable. The same-step
diagonal is analytic (`cosines=1`, `L2=0`). Off-diagonal artifacts use one canonical unordered
checkpoint pair and the site maps both recipient/donor orientations to the same content-addressed
artifact. Consequently every displayed weight value is exactly symmetric by construction, not
merely expected to agree after two separate floating-point runs.

Checkpoint-pair execution follows the existing coarse-to-fine order over unordered pairs: the
`0`/`1500` corner, every remaining pair touching step `96`, every remaining pair touching step `0`
or `1500`, then a seed-`20260715` shuffle of the remainder. Metrics use float32 effective matrices
and reductions, finite norm checks, atomic complete-pair artifacts, and no synthetic fill.
Cosines use the fixed `[-1, 1]` color scale. Each L2 family receives its own disclosed robust scale;
raw hover values remain unclipped. These measurements are descriptive parameter geometry, not a
causal intervention and not evidence that a changed weight direction is used by a particular
prompt.

### Zero-vector cosine amendment — 2026-08-02, after the failed smoke and before any artifact

The first `0`/`1500` smoke stopped before writing an artifact because layer 4 `q_proj` contains 25
output rows that are exactly zero at step 0 and nonzero at step 1500. Thus ordinary row cosine is
undefined for real model weights, and the earlier nonzero-axis requirement cannot produce the
requested atlas. This was observed only as a fail-loud diagnostic; no metric artifact existed when
the convention below was fixed.

For every flattened matrix, row, or column vector pair, cosine now uses the following symmetric
extension: ordinary cosine when both norms are nonzero, `1` when both vectors are exactly zero, and
`0` when exactly one vector is zero. The latter case represents a dormant/active transition as
directionally unaligned rather than dropping it from the mean; the former preserves identity for a
shared dormant channel. Every cell stores counts of both-zero and exactly-one-zero rows and columns,
and the website discloses those counts in hover. L2 metrics remain ordinary Euclidean distances and
need no zero-vector convention.

### Complete-weight atlas and display amendment — 2026-08-03

After decomposed projection artifacts existed, the requested website atlas was expanded without
changing the GPU measurement or inferential target. The registered OLMo-3 and Qwen-3 inventories
cover every learned tensor, and the displayed vertical axis includes embedding, seven decoder
projections, all learned normalization vectors, final norm, and untied LM head. These LoRA runs
never train the non-projection tensors, so their cross-checkpoint identity is exact by construction
(`cosine=1`, `L2=0`). One-dimensional norm vectors expose flattened metrics only; decomposed cells
are N/A rather than fabricated.

For each already-stored decomposed row/column family, the exporter additionally derives population
variance from the raw per-axis values. The scalar cell color still encodes its mean; a fixed light
inset border encodes variance only through its width from zero to the cross-cell p95, while hover
reports the unclipped variance. Weight
cosine color maps now use the requested fixed `0..1` range, blue at zero, white at the transformed
midpoint, and red at one, with quadratic color interpolation to spread high-cosine differences.
The variance inset is fixed white at 30% opacity and changes only in width. Any rare negative raw cosine
remains visible in hover and clamps only at the color endpoint; the underlying artifact is
unchanged. Q/K/V row details and O column details use exact 128-channel attention-head boundaries
from the model schema, drawn as outlines over one contiguous 64-column neuron grid. Detail
transport changes from JSON to packed little-endian
float32 and prefetches all four families per selected checkpoint pair. This changes storage and
interaction latency only; it neither recomputes nor alters the measured values.

### Reverse different-name intervention amendment — 2026-08-06, before any reverse grid

This post-hoc amendment restores the original clean-into-dirty causal direction as a distinct
exploratory source after the forward dirty-into-clean atlas had already been designed and partly
measured. No `reverse_across_sample` probability or representation-alignment artifact existed when
this contract was recorded; earlier forward artifacts must not be transformed into reverse
results.

- The dataset pair, deterministic alias derangement, answer options, model, checkpoint, and reverse
  token support are exactly those of `across_sample`.
- The source is the original clean function question and the recipient is its different-name dirty
  question. Source and recipient checkpoints must be equal.
- The primary target remains the original clean function's correct A–E option. A successful
  intervention therefore has positive change from the dirty recipient baseline. The dirty
  recipient's naturally correct option is retained as audit metadata but is not substituted as the
  primary target.
- The intervention is defined for activation boundaries only. Weight patching is not defined
  because the two prompts share one parameter state.
- Cosine/L2 views compare the same ordinary unpatched vectors in reversed roles. Cosine and L2 are
  symmetric at matched coordinates, while source/recipient norms and token metadata must swap;
  these observational values are not causal evidence.
- Full-vocabulary logit-lens hover may reuse the existing checkpoint's clean and `across_sample`
  prompt-side measurements in swapped roles. The causal patch grid itself always requires a new
  forward intervention and remains explicitly unprocessed until one is run.

This mode is exploratory and is not retroactively counted toward the earlier confirmatory H4
criterion. Its diagnostic prediction is a contiguous late-layer region with positive original-
answer delta, stronger after behavioral acquisition than before it.

## Fourier redundant-circuit amendment — 2026-08-08, before any such GPU run

This post-hoc exploratory analysis asks whether clean-checkpoint behavior can be recovered by more
than one minimal set of residual-stream `(prompt token, layer)` sites. It uses the identical clean
prompt, A-E options, OLMo-3 7B base revision, correct-condition seed, and clean/dirty checkpoint
direction as checkpoint transfer. It is run separately for each function because absolute token
axes differ. No Fourier-circuit model artifact existed when this amendment was written.

The first gate is an eleven-point independent-site density sweep from all dirty to all clean. The
transition density is fixed as the interior density with maximum raw-logit-difference variance. If
the probability span is below `0.05`, the raw-logit-difference span is below `0.2`, and maximum
interior raw-logit variance is below `0.01`, the run stops. Otherwise 512 random corners are drawn
at the selected density. The primary sparse estimator is a degree-at-most-four function-value
LASSO with `lambda=0.01`; all singleton sites are included and higher-order supports are enumerated
over a 32-site hierarchical screen. The intercept plus the 4,095 columns with strongest training-
sample function correlation enter the LASSO. A nonconstant fitted coefficient is heavy at absolute
value at least `0.03`.

Continuous-alpha gradients are secondary. Function-value marginal screening is used before a
fixed held-out coefficient gate. Gradient estimates must have RMSE at most `0.1`, maximum absolute
error at most `0.25`, and cosine at least `0.8` against plain function-value estimates before they
may affect interaction screening or inverse-variance estimates. Heavy status still comes only from
the function-value LASSO. Degree profiles use squared function-value LASSO coefficients and are
checked at the selected density and its adjacent sweep
densities; maximum pairwise L1 above `0.25` or cosine below `0.95` is reported as an intervention-
scale warning and is never silently filtered.

A Fourier support is only a hypothesis. It becomes a reported circuit only after exact corner
patching and all-path one-site walk-down. Sufficiency requires at least 80% recovery from the
all-dirty to the all-clean residual-intervention raw logit difference and the all-clean A-E argmax.
Both corners retain the recipient checkpoint's final norm and unembedding; the all-clean corner is
not redefined as an independent donor-checkpoint forward. The empty set is ineligible; singleton
minsets are allowed. The diagnostic prediction is that functions with strong endpoint checkpoint
transfer have a nonflat density curve and at least one late-token/later-layer verified minset.
Multiple distinct verified minsets would support redundancy. A flat curve, no heavy coefficient,
no causally sufficient candidate, or density-unstable spectrum weakens the circuit interpretation
and must be reported directly rather than repaired by post-hoc threshold changes.

Before model inference, exhaustive 2-of-3 majority and two-clause monotone-DNF references must
recover their exact coefficients and minterms. Before optimized collection, one known measured
single-site effect and a fixed-mask cached-versus-uncached parity profile must pass. See
[Fourier redundant-circuit discovery](../experiments/fourier-circuits.md) for the implementation and
artifact contract.

### Full-sequence backend amendment — 2026-08-08, after cache profiling and before stage 0

The required fixed-mask cache gate rejected the proposed optimization before any density-sweep
outcome was collected. On the 32 preregistered density-`0.5` masks, the full-sequence reference took
`1.2697s` median while the token-by-token cached executor took `38.7490s`; maximum candidate-logit
and probability differences were `0.125` and `0.00103694`, despite exact A-E argmax agreement. No
pair shared one complete token prefix. Controlled CPU OLMo tests subsequently found exact manual-
cache versus Hugging Face-native-cache equality and isolated the full-prefill/cache difference to
BF16 query-shape arithmetic.

The user therefore selected `full_sequence_reference` as the explicit scientific backend. Every
function-value, continuous-gradient, density-stability, and causal-verification forward uses the
entire prompt without KV cache, with batch size one throughout. The rejected cache profile remains
immutable evidence and is not reinterpreted by widening tolerances. Each new run stores a three-way
unpatched diagnostic: full prefill, native Hugging Face single-token decoding, and the manual cache.
Native and manual cached candidate logits must be exactly equal; their shared difference from full
prefill is reported but never enters Fourier estimation or causal verification. Backend identity is
a required config field and a distinct artifact-path component.

### Refined low-density diagnostic — 2026-08-08, after the coarse stage-0 result

The coarse sweep selected its lowest interior point, `p=0.1`, so it did not localize the transition
below that boundary. Before collecting this follow-up, the diagnostic grid is fixed to
`0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.32, 0.64, 1`.
It uses the same seed, 32 independently sampled masks per interior density, full prompt, batch-one
scientific evaluator, and transition rule. This is a post-hoc stage-0 localization diagnostic: it
does not replace the original curve or retroactively change the density used by the already-run
spectrum and minset verification. Its artifacts must occupy a content-addressed directory distinct
from the preregistered coarse sweep.

### Exhaustive-singleton and `pyalvt` correction — 2026-08-08, before the new GPU run

The pending refined `identity` (`riodwl`) Stage 1/2 continuation is stopped. The original `p=.1`
run and refined low-density Stage 0 remain immutable diagnostics, but neither is extended or
reinterpreted. In particular, the four original final-token singleton survivors at layers 25, 29,
30, and 31 are a sparse-discovery **lower bound**, not an exhaustive minset result: the old Stage 2
tested only supports proposed by Stage 1.

The primary corrected run is `pyalvt` (`function_id=add_5`) with
`allenai/Olmo-3-7B-Instruct` revision
`6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`, correct-condition seed `20260715`, dirty
recipient step 0, and clean donor step 1500. Both checkpoint executions receive one byte-identical
fully rendered chat prompt. Its user content imports `pyalvt, ckhtts`, asks for the correct Python
definition of `pyalvt`, presents choices `n % 3`, `n - 1`, `n + 5`, `n + 14`, and `n`, and requires
one uppercase letter; the registered answer is C, `lambda n: n + 5`. Any deviation in messages,
choice order, answer, model revision, checkpoint direction, or rendered prompt aborts.

Before any Fourier spectrum, every residual-stream `(token, layer)` site in the full prompt grid is
tested clean on an otherwise all-dirty background. Each row persists raw logit difference,
A-E-normalized correct probability, argmax, threshold margin, coordinates, and pass/fail. The
threshold is recomputed as `dirty + 0.8 * (clean - dirty)` in raw-logit space and requires the clean
argmax. Every passing row is a verified singleton minset independent of LASSO. The existing
checkpoint-transfer grid is a mandatory live harness: endpoints and final-token layers 19 through
31 must reproduce within the fixed probability tolerance and all layers 19-31 must pass. Failure
stops the run as a patching error before any density result is trusted.

The new `pyalvt` unrestricted density sweep uses
`0, .001, .002, .005, .01, .02, .04, .06, .08, .10, .12, .16, .20, .32, .64, 1`, with 32
independent masks at every interior density. It stores mean and variance of A-E probability, A-E
accuracy, and raw logit difference and selects maximum raw-logit variance. A second diagnostic and
all subsequent higher-order spectrum work use the **singleton-vetoed residual function**: every
verified singleton site is pinned dirty and excluded from the Fourier variables. This prevents a
known sufficient readout from saturating random masks and hiding alternative multi-site circuits.
The unrestricted and vetoed functions are always labeled separately.

Continuous-alpha gradients are revalidated at the new vetoed transition density; the earlier
identity outcome is not transferred. They affect screening or augmentation only if the held-out
gate passes, while function values remain primary and the LASSO alone declares heavy coefficients.
Only degree-two-or-higher heavy supports enter corrected Stage 2. They remain hypotheses until
exact causal sufficiency plus all-path walk-down succeeds; a walk-down singleton absent from the
exhaustive table is a fatal inconsistency.

All gradient-free scientific passes use full-prompt `use_cache=False`, batch one, and
`torch.inference_mode()` only after exact candidate-logit parity against the prior full-prompt
`no_grad` implementation. Stage 1 alone enables autograd, only for continuous alpha; model weights
remain frozen. Implementation and CPU validation precede the new GPU run. The first GPU commitment
ends after the exhaustive singleton sweep and both `pyalvt` density curves so those results can be
inspected before authorizing Stage 1/2.

### Stage-1 numerical-solver correction — 2026-08-08, after the first `pyalvt` attempt

The first `pyalvt` Stage-1 attempt failed before writing any coefficient table: global-step FISTA
reached its 5,000-iteration cap. A subsequent exact KKT diagnostic showed maximum violation
`0.16101552` despite successive coefficient movement of only `0.00032893`; coefficient movement was
therefore not a valid convergence certificate on this highly scaled, collinear parity design. No
threshold, density, sample, feature screen, or objective was changed. The same convex LASSO is now
solved by deterministic maximum-KKT-violation coordinate descent and accepted only when its maximum KKT violation
is at most the existing `1e-5` convergence tolerance. Both solvers continue to recover the exact
synthetic majority and monotone-DNF references. The 512 model corners and alpha gradients are now
written to a digest-validated intermediate artifact before fitting so numerical failures are
resumable and cannot discard GPU measurements.

### Clean-minus-ten-point causal-rule rerun — 2026-08-09, after the strict result

This is a user-requested post-measurement analysis and is not a retrospective change to the frozen
80%-raw-logit result. Its sufficiency rule is A-E-normalized correct probability at least the clean
corner's probability minus `0.10`, together with the clean A-E argmax. For the measured `pyalvt`
endpoints this is approximately 90% correct probability, but the exact threshold is always
recomputed from the rerun's clean corner and stored in both probability and equivalent raw-logit
coordinates.

Applying this rule to the exhaustive singleton census identified exactly 28 sufficient sites:
token 38 at layers 2-7, token 53 at layers 3-9, and final token 112 at layers 17-31. All 28 are
pinned dirty and excluded from the active variables before collecting a new residual density sweep
or any Fourier samples. The seven additions relative to the strict census invalidate the
minimality of all eight previously reported pairs, so the strict spectrum and its Stage-2 output
are preserved only as results under their original rule. They are never retresholded into results
for this rule.

The new run has its own config-validated artifact directory and repeats the unrestricted and
singleton-vetoed Stage 0 curves before Stage 1. Only if the new vetoed curve is nonflat is a fresh
function-value Fourier spectrum collected. Continuous-alpha gradients are revalidated at that
run's transition density. Exact causal verification and all-path walk-down use the same
clean-minus-ten-point threshold; exhaustive singleton-veto agreement is a fail-loud harness. The
website labels this result separately from the strict rule and loads only its own Stage-2 minsets.

### Non-Fourier minset-recall audit — 2026-08-09, before recall GPU collection

The clean-minus-ten-point Fourier run is treated as a high-precision proposal mechanism, not a
high-recall census. Its Stage 2 evaluated only 824 distinct supports: 32 singletons, 333 pairs, 371
triples, and 88 four-site sets. Stage 1's explicit interaction screen contained only 32 of the
3,588 active residual sites. No completeness claim follows from 13 verified Fourier-generated
minsets.

The recall audit is independently seeded (`20260810`), content-addressed, and excludes every
support already evaluated by Fourier. It retains the same full-prompt, no-cache, batch-one
reference evaluator; the same clean step 1500 and dirty step 0; the same 28 singleton vetoes; and
the same clean-minus-ten-percentage-point probability threshold and clean-argmax requirement.
Changing the causal rule or reusing the strict-threshold search invalidates the audit.

Before seeing recall outcomes, the initial proposal families are fixed as follows:

1. Evaluate every previously untested non-empty subset of the union of sites appearing in the 13
   Fourier-discovered minsets. Together with prior measurements and known-insufficient active
   singletons, this gives an exact local truth table over 13 sites (8,191 non-empty subsets). Report
   all local minsets, including those not proposed by Fourier, and every immediate monotonicity
   violation.
2. Complete all missing pairs inside Stage 1's 32-site interaction screen.
3. For each of the four highest-probability insufficient singleton sites, sweep every other active
   site as its partner.
4. Sample 8,192 distinct unseen active-site pairs uniformly without replacement and report a 95%
   Wilson interval for the missed-pair prevalence. Scale that interval to the full previously
   untested pair universe only as a design-based estimate, never an exhaustive count.
5. Sample 4,096 unseen pairs formed by retaining one site from a known minset and replacing its
   partner. This targeted diagnostic has no population-frequency interpretation.
6. After pair evaluation, propose 2,048 uniform triples and up to 2,048 triples formed by extending
   the 64 highest-probability insufficient pairs. Evaluate every missing pair child first; prune a
   triple if any child pair is sufficient. A passing triple is reported as a verified minset only
   when all three exact pair children fail the same causal rule.

The initial plan contains 33,787 unique new supports, of which 25,714 are pairs. All proposals and
shards are deterministic, digest-validated, and resumable. A support can carry multiple proposal
labels so overlapping sampling schemes are not double-counted. Pair prevalence is calculated only
from the uniform-pair sample; anchor, mutation, local-census, and near-miss yields are descriptive.

This audit can falsify a strong recall claim but cannot prove global completeness: the pair sample
has finite resolution, triples are only sparsely sampled, and orders four and above are exhaustive
only inside the 13-site local subspace. The website must state this limitation and must not display
planned proposals as measurements. GPU collection requires a new explicit release and the normal
double authorization gate.

### Minset structural-equivalence grouping — 2026-08-09, descriptive analysis

The website groups sites that have similar co-minset partner profiles. For each connected component
and minset size, the analysis preserves each minset as a hyperedge, constructs each site's set of
co-hyperedge neighbors, seeds groups from exact profile matches, and deterministically assigns seed
groups only where complete-link adjusted Jaccard similarity is at least `0.5`. Two sites occurring
in the same minset have a hard cannot-link constraint because they cannot be substitutes within that
verified intervention. This scalable procedure replaces exact graph coloring for the full recall
union, whose largest component has 1,501 sites.

These are structural-equivalence clusters, not identified biological-style pathways. Complete-link
prevents a chain of weak pairwise similarities from collapsing dissimilar endpoints. Higher-order
minsets remain hyperedges even though the display may draw their pairwise clique for spatial
legibility. Clusters must be recomputed when the recall audit finds new minsets; they are explicitly
recall-sensitive descriptive summaries.

### Recall-audit outcome — 2026-08-09, after collection

The preregistered audit completed without changing its proposal inventory or causal threshold. It
measured 33,787 initial supports, 8,164 missing pair children, and 4,096 eligible triples. The exact
13-site local census found 36 minsets, 23 absent from Fourier's 13 verified proposals, together with
576 immediate violations of intervention monotonicity. Targeted searches verified 1,532 new pairs
and 436 new triples, but 1,489 pairs and every triple contain the nearly sufficient site `(token 53,
layer 10)`. Zero of 8,192 uniform unseen pairs passed (95% Wilson upper prevalence `0.00046871`).

Thus the audit rejects an exhaustive/high-recall reading of sparse Fourier discovery while also
rejecting the claim that missed pairs are common everywhere. The misses form a structured,
threshold-sensitive family. These post-collection observations do not alter the frozen sampling
design or requested sufficiency rule.

The visualization consumes the deduplicated causal union rather than only the 13 Fourier-stage
survivors: 2,003 multi-site minsets total (1,544 pairs, 457 triples, one size-four, one size-six).
The audit and raw hypothesis tables remain artifact-level provenance and are intentionally omitted
from the compact site; only their causally verified survivors enter the main overlay.

After inspecting the near-threshold hub family, the user requested a separate display-level
effect-size condition: every proper subset of a displayed multi-site minset must have A–E-normalized
`P(correct) <= 0.85`. The full set must still pass the unchanged measured threshold. The maximum is
taken exhaustively over cached subset evaluations, not inferred from only immediate children.
This post-measurement filter retains 63 sets (46 pairs and 17 triples) and does not alter the raw
2,003-set causal inventory or the frozen recall audit. A digest-validated 50,456-support metric
index makes threshold inspection CPU-only and fails loudly if any source artifact changes.

### Relative-subset higher-recall amendment — 2026-08-09, before frontier GPU collection

The user superseded the fixed `0.85` display cap with a candidate-relative effect-size rule. A
multi-site support is reportable only if it passes the unchanged clean-minus-ten-point full-set
threshold and every proper subset has `P(correct) <= 0.8 * P(correct for the full support)`.
Because the full probability is at most one, any subset above `0.8` is a sound blocker for every
strict superset under this definition. No other monotonicity assumption is allowed: the 576 exact
local counterexamples above remain decisive evidence that the underlying intervention function is
not monotone.

Before new outcomes were observed, the follow-up search was fixed to seed `20260811`, scientific
batch one, 256-support shards, and the following order:

1. Refilter the complete cached causal inventory under the relative rule. The frozen input is 41
   minsets (34 pairs and seven triples) forming one mixed-order 28-site component.
2. Complete every still-unknown support of sizes two, three, and four inside that component,
   levelwise. Every proper subset must already be cached unless a known `P(correct) > 0.8` blocker
   safely prunes the candidate. The order-two phase includes every missing chord among component
   cells, not merely mutations of an existing edge.
3. Evaluate 8,192 unseen global pairs selected by a deterministic degree-balancing sampler over
   the 3,587 eligible singleton sites. These results estimate discovery yield under broader site
   coverage but do not constitute an exhaustive pair census.
4. Causally report only supports whose complete proper-subset powerset is measured and satisfies
   the relative rule. Proposal membership is never itself a circuit claim.

The search is independently content-addressed. It reads the immutable 50,456-support base index
and writes new raw metrics plus their own digest-validated frontier index, so interruption and site
re-export never require recomputing a measured support.

### Recursive-closure and disconnected-residual amendment — 2026-08-13

After observing that the one-hop shell attached ten sites, the user approved a fixed-point closure
and disconnected-circuit diagnostic. Before inspecting any outcomes from this new run, the
registered sequence is:

1. Start from all 38 sites in the strict mixed-order hypergraph (37 pair-network sites plus the
   hyperedge-only token-54/layer-13 site). Exhaust every unseen pair touching this component.
2. Add only strict relative-criterion pair minsets, then repeat the complete shell. Stop only when a
   shell adds zero sites; fail if 16 iterations do not converge.
3. On the converged component, enumerate all eligible triples and quadruples levelwise. Do not run
   another balanced pair sample. Five- and six-site enumeration requires a post-quadruple yield or
   near-threshold justification and a separately content-addressed run.
4. Pin dirty every exhaustive singleton and every site in the converged strict hypergraph. Sweep the
   frozen 16-density grid with 32 independent masks at each interior density using the full-prompt,
   no-cache, BF16, batch-one reference backend.
5. If the known-network-vetoed curve is flat under the registered probability-span, logit-span, and
   interior-variance gate, stop. If nonflat, select maximum raw-logit variance and run independently
   seeded sparse-mask proposal and minimization searches. Greedy or beam deletion is hypothesis
   generation only; a reported minset still requires exact causal sufficiency and the complete
   proper-subset relative check.

The initial fixed-point shell contains 35,356 pairs. All prior 228,336 subset metrics remain
immutable inputs and every new phase is separately digest validated.

Before inspecting the completed triple/quadruple metrics, the continuation gate is made numeric.
Run the separately content-addressed size-five pass iff sizes three/four produce at least one new
strict relative minset, or the maximum probability among newly measured, insufficient supports
that remain legally expandable (`P(correct) <= 0.8`) is at least `0.75`. Apply the same rule from
size five to size six: continue iff size five produces a new strict minset or its maximum expandable
probability is at least `0.75`. This uses `0.75` because it lies within five points of the largest
proper-subset probability that could possibly satisfy the relative rule; above-`0.8` supports are
blockers, not evidence for expanding their supersets. If neither condition holds, stop the local
order expansion rather than spending GPU time on an unmotivated combinatorial census.

The sealed order-four result met that gate with 162 new triples and 221 new quadruples. The sealed
size-five result also met it with 37 new strict minsets, so size six proceeds over the 216,865
unseen supports left by exact safe pruning. This records application of the frozen gate; it does not
change the rule or authorize size seven.

The sealed size-six pass evaluated all 216,865 registered supports and found no new strict minset.
Twelve full supports crossed the absolute full-set threshold, but every one failed the registered
proper-subset separation rule. The resulting strict union remains 649 multisite minsets over 38
sites, backed by 594,168 cached exact support metrics.

The subsequent known-network-vetoed curve is non-flat and selects `p=0.10` by maximum raw-logit
variance. After observing that diagnostic but before drawing any new random masks, the authorized
disconnected search is frozen as follows: sample 256 masks from an independent seed (`20260814`),
retain at most 12 successful starts using deterministic support-diversity selection, and run four
seeded delta-debug minimization restarts per start. All evaluations remain full-prompt, no-cache,
BF16, inference-mode, and scientific batch one. The run has a hard 100,000-new-support cap.
Delta-debug output is hypothesis generation: a candidate is reported as a circuit only when it has
at most 12 sites and every member of its complete powerset has been measured, the full support
crosses the clean-minus-0.10 threshold with the clean argmax, and the maximum proper-subset
probability is at most 80% of the full support's probability. A larger or merely one-removal-minimal
candidate remains explicitly unverified.

### Expanded disconnected-recall wave — 2026-08-14, before new mask sampling

The first sealed disconnected search produced no strict minset among 34 candidates with complete
powerset evidence. Its best maximum-proper-subset/full probability ratio was `0.8512446`; this does
not alter either confirmatory bound. To improve proposal recall without selecting a criterion from
that outcome, run one independently content-addressed coverage wave against the same 66-site-vetoed
residual function and its already selected `p=0.10` density:

1. Draw 1,024 new masks with independent seed `20260815`, select at most 48 sufficient starts by
   the existing deterministic support-diversity rule, and perform eight seeded delta-debug
   restarts per selected start.
2. Preserve the clean-minus-`0.10` full-support threshold, clean argmax requirement, and maximum
   proper-subset probability of `0.80 * P(full)`. No result from the first wave changes these
   confirmatory definitions.
3. Preserve full-prompt `use_cache=False`, BF16, inference mode, scientific batch one, an exact
   powerset cap of 12 sites, 256-support atomic metric shards, and a hard cap of 1,000,000 newly
   evaluated supports.
4. Treat every delta-debug result as proposal-only. Report a new circuit only after the complete
   nonempty powerset has been measured and the unchanged strict rule passes. Candidates larger
   than 12 sites remain explicitly unverified.

This wave expands independent-mask, successful-start, and minimization-path coverage. It remains a
randomized recall audit rather than a global completeness proof, and its budgets must not be tuned
after inspecting partial outcomes.

### Expanded disconnected-recall outcome — 2026-08-14, after collection

The expanded wave completed under its frozen configuration. Of 1,024 independently sampled masks,
282 crossed the unchanged full-support threshold. Forty-eight diverse starts and eight restarts per
start produced 356 unique one-removal-minimal hypotheses and 156,830 newly measured support
metrics. Only five hypotheses overlapped the first wave.

Complete powersets were measured for all 213 hypotheses at or below the registered 12-site cap:
138 pairs, ten triples, one quadruple, two size-six, six size-seven, ten size-eight, 13 size-nine,
17 size-ten, 11 size-eleven, and five size-twelve hypotheses. Zero passed the frozen relative
proper-subset rule. The best maximum-proper-subset/full probability ratio was `0.8620268`; zero
passed even a descriptive `0.86` sensitivity bound. Together, the two independently seeded waves
contain 242 unique exact-powerset hypotheses and zero strict disconnected minsets. The 143
hypotheses larger than 12 sites remain explicitly unverified.

This substantially strengthens the negative result for sparse-mask plus delta-debug proposals, but
does not establish global completeness: the known-network-vetoed density curve remains nonflat,
and neither random proposal wave exhausts the 3,550-site residual search space or verifies the
larger hypotheses.

## Cross-checkpoint `pyalvt` minset series — 2026-08-20, before collection

Repeat the corrected `pyalvt` checkpoint-transfer analysis with dirty recipient step zero and each
available donor checkpoint. Preserve the clean-minus-ten-percentage-point threshold, clean-argmax
gate, full 113-token by 32-layer residual grid, exhaustive singleton census, independent density
sweep, singleton-vetoed function-value Fourier spectrum, gradient-validation gate, and exact causal
verification used at donor step 1500. A checkpoint that fails the all-clean intervention argmax is
a durable `clean_behavior_not_acquired` terminal result; it must not be rescued by lowering the
threshold or relabeled as a circuit search.

The launch order requested by the user is donor steps 96, 64, then 32. The remaining acquired
checkpoints are shuffled once with seed `20260820`, producing the frozen continuation order 128,
1024, 256, 384, 192, 768, 512, and 1280. Step 1500 is already complete and is reused only through
its digest-validated artifacts. Steps 1 through 16 are acquisition diagnostics rather than minset
runs because their donor checkpoints do not make the correct answer the A–E argmax and, for the
earliest steps, the nominal clean-minus-0.10 threshold is not above the dirty endpoint.

Before each GPU run, the independent checkpoint-transfer grid freezes the expected passing
singleton inventory. The new batch-one full-prompt census must reproduce it exactly or stop. Those
inventories are part of the code-level run contract, never inferred from the new Fourier samples.
Outputs remain separated by donor checkpoint and veto inventory; no checkpoint may reuse another
checkpoint's selected density, Stage-1 samples, candidates, or verification metrics. Higher-recall
frontier and disconnected searches are follow-ups, not automatic claims of checkpoint-wise global
completeness.

## Prior information used for predictions

The earlier repository replicated OLMo-2 7B rule recovery and observed OLMo-3 recovery after 4,096
documents under the same broad Functions/LoRA regime. Qwen had frozen-gradient measurements but no
matched behavioral finetune. Those observations motivate the directional predictions and are why
peak timing is not assumed to be final-step monotone. This new experiment must still stand on its
own artifacts and uses new training-dynamics and causal-intervention outcomes.

## Answer-location lookup amendment — 2026-08-17, before any such GPU run

The final prompt state may recover an answer by attending to an option-line state that advertises
correctness. The frozen 27-row intervention matrix, exact tokenizer-site definition, direct
`attention_input` test, `resid_post` propagation control, full A–E outcome contract, parity gates,
and directional predictions are specified in
[the answer-location experiment](../experiments/answer-lookup.md). No answer-location GPU artifact
existed when this amendment was written. Missing website cells must remain unprocessed.

## Cross-checkpoint switched-answer minsets — 2026-08-22, before any GPU artifact

The user requested a minset analysis of the answer-location move intervention across checkpoints.
The source and recipient receive the identical clean `pyalvt` prompt. Source activations come from
the final step-1500 model and are patched into the step-0 base recipient; all downstream weights
remain those of the base. Each Boolean layer site is the simultaneous two-position swap of the
donor's correct-C and selected incorrect-option line-terminator states into the opposite recipient
locations. This is not a one-way copy and not an ordinary single-token residual site.

All four incorrect destinations A, B, D, and E are registered independently. `attention_input` is
primary and `resid_post` is a propagation control. The all-clean endpoint must redirect the base
model's A-E argmax to the registered destination and the density response must be non-flat before
minset search. The full set must reach within 0.10 A-E-normalized destination probability of that
all-clean hybrid endpoint; every proper subset must remain at or below 0.80 times the full-set
probability. The initial search is exact in increasing support order through six over the 32
layerwise swap operators, with only the mathematically safe above-0.8 subset blocker. Results are
called exhaustive only through sealed orders. See
[Cross-checkpoint switched-answer minsets](../experiments/switched-answer-minsets.md).
