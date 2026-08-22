# Cross-checkpoint switched-answer minsets

**Registered:** 2026-08-22, before any GPU artifact for this experiment.

## Question

The answer-location experiment tests one layer at a time within the final fine-tuned model. This
follow-up asks whether a small set of layerwise answer-location swaps from the final model is
sufficient to redirect the frozen base model. It uses the exact `pyalvt` / `add_5` code-definition
probe from the checkpoint-transfer and Fourier analyses.

The donor/source checkpoint is step 1500 and the recipient checkpoint is step 0. Both receive the
identical fully rendered chat prompt and identical token IDs. Only the checkpoint supplying the
two transplanted activations changes. All downstream computation uses recipient/base-model
weights.

## Composite site contract

The correct option is C. The experiment is run independently for all four incorrect destinations
A, B, D, and E. For a destination `W`, one Boolean site at decoder layer `L` means the simultaneous
two-position intervention

```text
recipient correct-C line terminator <- donor incorrect-W line terminator
recipient incorrect-W line terminator <- donor correct-C line terminator
```

The line terminators use the exact tokenizer-offset rule from the answer-location experiment: the
site is the token whose character span contains the first newline ending that option. A Boolean
site is therefore a **layerwise paired swap operator**, not an ordinary `(token, layer)` patch.
Artifacts must retain both donor and recipient token coordinates for every operator. A one-way
copy is a different intervention and is not silently substituted.

The primary boundary is `attention_input`; `resid_post` is the propagation control. The scientific
backend is batch one, full prompt, BF16, `use_cache=False`. Gradient-free collection uses
`torch.inference_mode()` only after exact parity with the no-grad reference path. No prefix cache
or cached decoder is used.

## Registered target and endpoints

For destination `W`, the causal target is the A-E-normalized probability and raw logit difference
for `W`, where raw logit difference is the `W` logit minus `logsumexp` of the other four answer
logits. The all-dirty corner is the unpatched base model. The all-clean corner turns on the paired
swap at all 32 layers; it remains a hybrid base-model forward, not the donor model's unpatched
answer.

Before minset search, every destination/boundary pair must pass all of these gates:

1. zero-mask hooks reproduce the unpatched base logits and probabilities within `1e-6`;
2. an inference-mode fixed-mask panel reproduces the no-grad reference within `1e-6`;
3. a refined density sweep is non-flat under the existing probability/logit/variance rule; and
4. the all-clean corner has destination `W` as its A-E argmax.

If a pair fails, it is reported as a causal null and receives no minsets.

## Density and exhaustive search

The density grid is

```text
0, .001, .002, .005, .01, .02, .04, .06, .08, .10, .12, .16, .20, .32, .64, 1
```

with 32 independently sampled masks at every interior density. The transition diagnostic retains
all five candidate logits, destination probability and variance, accuracy, raw logit difference
and variance. The selected density is the interior point of maximum raw-logit variance, but sparse
Fourier proposals are not needed for the initial search: there are only 32 composite layer sites.

Instead, the primary discovery procedure exhaustively evaluates every eligible support in
increasing size through order six. Every subset-to-metric mapping is persisted, digest-validated,
and reused. Orders one through three are completely enumerated before any higher order. A support
may be safely pruned only when it contains an already measured proper subset with
`P(destination) > 0.8`, because no full support can then satisfy the registered relative-subset
criterion below. No monotonicity of the network response is otherwise assumed.

A reported minset must satisfy both:

- `P(destination) >= P(destination at all-clean swap corner) - 0.10`, and the destination is the
  A-E argmax;
- every proper subset, including the empty corner, has
  `P(destination) <= 0.80 * P(destination for the full support)`.

The complete proper-subset powerset must be measured; immediate-child evidence is insufficient.
The result is exhaustive only through the largest sealed order. Larger minsets remain explicitly
unresolved rather than being called absent.

## Artifact and display contract

Artifacts live separately from both existing analyses under
`answer_lookup_checkpoint_transfer_minsets/`. Configuration, prompt/token audit, donor activation
capture, endpoint gate, density curve, subset shards, sealed-order manifests, and verified minsets
are independently inspectable and resumable. Existing answer-location and Fourier artifacts are
never overwritten or reinterpreted.

The website must identify each selected cell as a paired swap, show both token locations, and show
which search orders are sealed. It may overlay verified layer sets, but it must not render the
composite operator as a fake one-token grid or call a partial order-six census exhaustive over all
possible minsets.
