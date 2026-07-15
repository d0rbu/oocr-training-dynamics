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
normalized effect:  (p_j - r_j) / (s_j - r_j)
```

Normalized effect is omitted when `|s_j - r_j| < 1e-8`; it is not clipped for analysis. The site
may clip colors for readability but must show the numeric value on hover. Primary inference uses
the raw probability delta because normalized ratios can explode near a small denominator.

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

## Prior information used for predictions

The earlier repository replicated OLMo-2 7B rule recovery and observed OLMo-3 recovery after 4,096
documents under the same broad Functions/LoRA regime. Qwen had frozen-gradient measurements but no
matched behavioral finetune. Those observations motivate the directional predictions and are why
peak timing is not assumed to be final-step monotone. This new experiment must still stand on its
own artifacts and uses new training-dynamics and causal-intervention outcomes.
