# Fourier redundant-circuit discovery

This pipeline searches for **causally sufficient sets** of OLMo-3 7B Instruct residual-stream
sites in the clean-checkpoint-into-dirty-checkpoint transfer setting. A site is one
`(prompt token, decoder layer)` coordinate at the `resid_post` boundary. The prompt, answer
choices, function, tokenizer coordinates, and A-E readout are identical in the two passes; only
the donor and recipient checkpoints differ.

The command is deliberately per function. Different function prompts can have different absolute
token axes, so their Boolean site universes are not silently pooled. The current primary target is
`pyalvt` / `add_5`; `identity` is retained only as an earlier diagnostic.

## Intervention semantics

The clean checkpoint is captured first and released. The dirty checkpoint then runs every hybrid
intervention. For a Boolean mask bit `x[t, l]`:

- `0` leaves the causally current dirty/hybrid `resid_post` value untouched;
- `1` replaces it with the fixed clean-checkpoint value for that exact prompt token and layer.

Leaving a zero site untouched is important. Replacing it with a cached all-dirty value would erase
causal effects propagated from earlier clean sites and would no longer be ordinary activation
patching. The continuous gradient intervention is

```text
h[t, l] <- h_current[t, l] + alpha[t, l] * (h_clean[t, l] - h_current[t, l])
```

evaluated at Boolean corners. One backward pass through the sum of the per-example raw logit
differences returns every sample's derivative with respect to every `alpha[t, l]`. Model weights
remain frozen. At exact zero/one corners, a straight-through numerical correction makes the
forward value byte-exactly current/clean while retaining the derivative of the stated linear
interpolation; the endpoint contract verifies those forwards independently. In particular, the
all-clean residual corner is read through the **recipient** checkpoint's final norm and
unembedding. It is therefore checked against an explicit recipient-readout of the donor's final
residual, not against the independent donor-checkpoint logits; full finetuning can change those
readout weights.

The raw outcome is the correct A-E token logit minus `logsumexp` of the four incorrect A-E token
logits. Accuracy is the argmax over those same five tokens. Correct probability is softmax over the
same five tokens and is used only for diagnostics and display; the Fourier fit uses the raw logit
difference without thresholding.

## Mandatory gates before model inference

`run_synthetic_reference_gate()` executes before a model is loaded. It exhaustively enumerates all
corners of:

1. a 2-of-3 majority function, whose uniform coefficients are exactly `1/2`, three `1/4`
   singletons, three zero pairs, and `-1/4` on the triple; and
2. `(x0 AND x1) OR (x2 AND x3)`, whose exact minterms are `{0,1}` and `{2,3}`.

Both the direct coefficient estimator and the FISTA and maximum-KKT-coordinate LASSO reference solvers
must reproduce the known coefficient tables within the stored tolerance. The all-path
walk-down verifier must recover every exact minterm. A failure aborts before checkpoint loading.

The model harness then reproduces the largest-effect eligible, previously measured single-site
checkpoint-transfer cell from the existing residual patch artifact. It verifies the exact function,
choice order, source and recipient identities, checkpoint direction, site coordinate, and correct
probability within the configured tolerance. A site grid that excludes every known reference cell
is rejected.

## Exhaustive singleton causal census

Before Fourier hypothesis generation, the corrected pipeline constructs exactly one mask for every
site in the full prompt grid. Each mask patches that site clean and leaves every other site dirty.
The full-prompt, `use_cache=False`, batch-one evaluator stores every row's candidate logits, raw
logit difference, A-E probability, argmax, both raw-logit and probability threshold margins,
coordinates, and pass/fail. The original causal rule is exactly

```text
dirty_logit_diff + 0.8 * (clean_logit_diff - dirty_logit_diff)
```

plus the clean A-E argmax requirement. Every passing row is a verified singleton minset. This table
does not consult Stage 1 and cannot be thinned by LASSO. For `pyalvt`, endpoint probabilities and
the final-token layer-19-through-31 region must reproduce the existing checkpoint-transfer grid;
all those layers must pass the raw-logit criterion. A mismatch aborts before density collection.

The separate clean-minus-ten-percentage-point rerun instead resolves
`clean_correct_probability - 0.10` to its exactly equivalent raw-logit threshold and uses that
threshold plus the clean argmax everywhere. It is registered only for the measured full-grid
`pyalvt` step-1500-into-step-0 setup. Its census must equal the exact 28-site veto set before any
density sample is trusted.

All gradient-free collection uses `torch.inference_mode()`. Before enabling it, the implementation
runs all-dirty, all-clean, and a deterministic singleton through both inference mode and the former
full-prompt `no_grad` reference and requires exact candidate logits and derived metrics.

## Stage 0: unrestricted and singleton-vetoed density sweeps

The corrected `pyalvt` sweep draws independent Bernoulli masks at
`0, .001, .002, .005, .01, .02, .04, .06, .08, .10, .12, .16, .20, .32, .64, 1`, with 32 masks at
every interior density. Endpoint masks are exact. Every mask, five candidate logits, raw logit
difference, A-E probability, and accuracy enters a digest-validated Torch sidecar; JSON retains all
means and sample variances. Here “all-clean” means the complete residual intervention inside the
recipient model, including the recipient readout semantics above.

The transition density is the interior point with maximum raw-logit-difference variance, breaking
ties by probability variance and then proximity to `0.5`. If probability span, logit-difference
span, and every interior variance are all below their preregistered floors, stage 0 writes
`status=flat_stop`; stages 1 and 2 are forbidden.

The first curve is unrestricted and contains every singleton readout. After the exhaustive census,
a second curve pins every verified singleton-sufficient site dirty and samples only the remaining
variables. This `singleton_vetoed_residual` function is the sole Stage-1/2 input. The unrestricted
curve remains a diagnostic and is never silently substituted for the residual search space.

## Stage 1: p-biased spectrum

For density `p` and Boolean bit `x_i`, the orthonormal coordinate is

```text
phi_i(x_i) = (x_i - p) / sqrt(p * (1 - p))
chi_S(x) = product(phi_i(x_i), i in S)
```

The constant and every active, non-vetoed singleton variable are always included. Exhaustively
materializing every degree-four feature over thousands of sites is impossible, so the
implementation uses a preregistered
hierarchical LASSO family: degree-two through degree-four supports are enumerated over 32 screened
sites, with an explicit hard cap on total features. A second function-value-only sure-independence
screen retains the intercept and at most 4,095 nonconstant parity columns for the actual LASSO;
every enumerated support and its screening status remains inspectable. Before gradients are
trusted, screening uses only function-value singleton marginals. This method was selected over an adaptive KM/GL query
tree because it reuses one fixed, inspectable random-corner sample and admits a direct held-out
fit; its limitation is equally explicit: a high-order interaction with neither a detectable
function marginal nor a validated gradient footprint can be missed.

Deterministic maximum-KKT-violation coordinate descent fits raw function values on the training corner split and
must satisfy the maximum LASSO KKT optimality violation tolerance. The originally implemented
global-step FISTA solver is retained as a tested synthetic reference, but the real `p=.08` design
exposed extreme parity-column scaling: after 5,000 iterations its maximum KKT violation was still
`0.161`, even though successive coefficients moved by only `0.000329`. That failed run produced no
coefficient artifact and motivated the solver correction; it is not a scientific outcome. Only the
KKT-certified function-value coefficients may declare a support heavy. Gradients cannot
independently create a heavy coefficient.

Gradient estimates use the multilinear derivative identity only provisionally. On a disjoint
validation-row split, a deterministic held-out subset of nonconstant coefficient supports compares
plain function-value estimates with derivative estimates. RMSE, maximum error, and cosine must all
pass fixed thresholds before gradients may influence interaction screening or inverse-variance
coefficient estimates. If the gate fails, both uses are disabled and the function-value-only path
continues. The held-out indices and supports are serialized.

The same enumerated support family is also fit at the transition density and its neighboring
sweep densities. Each density repeats the function-correlation screen and function-value LASSO;
the normalized squared fitted coefficient mass by degree avoids summing tens of thousands of raw
Monte Carlo squared estimates, whose sampling noise would otherwise bias high-cardinality degrees.
The profile must satisfy the configured L1
and cosine stability limits. An unstable profile is not hidden: stage 1 and every downstream
stage-2/site artifact carry a warning that the result may reflect intervention scale.

All active-space supports are mapped back to full prompt-grid coordinates in the artifact. Verified
singleton sites can never enter one of these supports. Stage 1 generates hypotheses. Its supports
are never labeled circuits.

## Stage 2: causal verification

Every degree-two-or-higher heavy support is first patched as an exact clean corner with every other
site left dirty. Stage-1 singleton coefficients are not Stage-2 candidates because the exhaustive
census already settles that causal question.
The verifier then checks every one-site removal path, retaining removals that preserve both:

- the selected run's exact sufficiency threshold, represented on the raw-logit axis; and
- clean-label A-E argmax when `require_clean_argmax` is enabled.

Enumerating all valid removal paths, rather than committing to one tie order, recovers all reachable
minimal sufficient subsets. Empty sets are prohibited. If walk-down reaches a sufficient
singleton, the run aborts because the exhaustive census must already have found it. The JSON
contains only verified multi-site minsets, ordered smallest first, with their raw margin, probability, and
the heavy coefficients that generated each hypothesis. Raw stage-1 candidates remain separately
inspectable and are never exported as circuits.

Candidate-support and unique evaluated-subset counts have separate required hard caps (256 and
4,096 by default). Exceeding either is an explicit resource/configuration failure, not permission to
silently truncate the strongest coefficients or skip removal paths.

## Full-sequence backend and cache diagnostic

Mask bits are flattened token-major, then ordered as leaves of a deterministic lexicographic trie.
Adjacent masks are batched. A batch runs its common token prefix once; at the first differing site
it repeats the exact KV state, preserving all complete earlier-token cache entries and every layer
below the first differing layer in the current token. Later execution is batched. OLMo's configured
dynamic cache retains the model's full/sliding-attention layer policy.

The first real profile rejected this optimization before stage 0: at density `0.5`, all 32 masks
had distinct first-token patterns and no pair shared one complete token. The cached path was 30.5
times slower and differed from the reference by up to `0.125` candidate-logit and `0.001037`
probability. Its artifact remains inspectable, but cached values are never scientific inputs.

The required scientific backend is now `full_sequence_reference`: every mask runs the complete
prompt through the native model forward without KV cache. Function-value, continuous-gradient, and
stage-2 verification batches are all exactly one so BF16 batch-shape changes cannot enter the
comparison. Gradient-free stages use inference mode after the exact no-grad parity gate; Stage 1
keeps model weights frozen and differentiates only continuous patch alphas. A separate diagnostic
runs the unpatched prompt through full prefill, Hugging Face's
native single-token cache, and the manual cache. Native and manual cache logits must be exactly
equal; full-prefill differences are measured and retained rather than accepted as parity.

The corrected refined grid is now the runner default. Any custom diagnostic grid is stored in a
distinct content-addressed directory. Every grid must be strictly increasing and include both exact
endpoints; masks remain independent and scientific execution remains full-prompt batch one.

## Resumable artifacts

Each function/checkpoint/scope run writes beneath:

```text
artifacts/runs/olmo3-7b/correct/seed_20260715/fourier_circuits/<function>/
  clean_001500_dirty_000000/<scope>_backend_full_sequence_reference/
```

The directory contains exact config, synthetic gate, endpoint contract, harness check, inference-
mode parity gate, exhaustive singleton JSON/sidecar, separate unrestricted and vetoed Stage-0
curves/sidecars, an immediate resumable Stage-1 raw-corner/alpha-gradient checkpoint, a separate
resumable checkpoint containing all three gradient-free density-stability corner batches, the
Stage-1 coefficient table, and Stage-2 verified multi-site
minsets and causal measurements. JSON/sidecar pairs are atomic and all resumed sidecars
must match their recorded SHA-256 digest. A partial pair, schema mismatch, config mismatch, or
digest mismatch aborts.

The site exporter loads passing singletons directly from `exhaustive_singletons.json`, verified
multi-site minsets from Stage 2, and heavy Stage-1 coefficients into three explicitly separate
views. It shows unrestricted and vetoed density curves together. The original identity result is
labeled a legacy sparse-discovery lower bound, never “all minsets.” Missing data is displayed as
missing, never synthesized.

## Registered corrected `riodwl` replication — 2026-08-17

The original `riodwl` / `identity` spectrum remains a legacy sparse-discovery lower bound. A new,
separately versioned analysis repeats the corrected `pyalvt` methodology at the requested
clean-minus-ten-percentage-point threshold. It uses the same OLMo-3 7B revision, correct-condition
seed `20260715`, clean checkpoint 1500, dirty checkpoint 0, full rendered prompt, all prompt tokens,
all 32 residual-stream layers, BF16, scientific batch one, full-prompt `use_cache=False`, exact
inference-mode parity, exhaustive singletons before Fourier, singleton-vetoed Stage 1, exact Stage
2 causal verification, and the unchanged relative proper-subset rule
`max P(proper subset) <= 0.80 * P(full support)`.

The clean `riodwl` endpoint has independently measured A–E-normalized
`P(correct)=0.9921426773`, so the registered full-support threshold is
`P(correct) >= 0.8921426773` plus the correct-letter argmax. Before the new batch-one sweep, the
pre-existing checkpoint-transfer grid identifies exactly 21 expected singleton sites under that
rule: token 101 (`↵↵`) at layers 13–17 and final token 112 (`↵`) at layers 16–31. The new exhaustive
3,616-site sweep must reproduce that exact set. A mismatch is a harness/configuration failure, not
permission to silently update the veto inventory.

The unrestricted and singleton-vetoed density sweeps use the full refined grid and 32 independent
masks per interior density. Stage 1 remains function-value-primary with 512 corners, degree cap
four, a separately validated continuous-alpha gradient estimate, and density-stability checks at
neighboring densities. Stage 2 may report only exactly corner-verified, all-path-minimized
multi-site sets. Subsequent recall, relative-frontier, component-shell, and disconnected searches
use the same proposal budgets and digest-validated subset caches as `pyalvt`, but every `riodwl`
artifact lives beneath its own function directory. Coverage statements remain bounded by the
actually completed proposal families; none is described as globally exhaustive.

Two empty scientific outcomes are also durable terminal states. Stage 1 writes
`complete_no_heavy_coefficients` when the function-value estimator finds no coefficient above the
preregistered threshold. Stage 2 writes `no_verified_multisite_minsets` when Fourier candidates exist but none
survive causal sufficiency and walk-down. Neither state is exported as a discovered circuit.

### Measured corrected `riodwl` result — 2026-08-20

The new exhaustive sweep reproduced exactly the 21 frozen singleton sites. The unrestricted and
singleton-vetoed density transitions were `p=0.02` and `p=0.08`. Stage 1 found 185 heavy
function-value hypotheses and a stable neighboring-density degree profile. Its held-out alpha
gradient validation failed (RMSE `1.1141`, maximum absolute error `8.5613`), so gradient
augmentation was correctly excluded. Exact Stage 2 verified five Fourier-proposed pair minsets.

The independent recall audit then measured 26,199 initial supports, 8,147 missing pair children,
and 4,096 eligible triples. It verified 815 new pairs and 183 new triples; the exact six-site local
table contained four minsets absent from Fourier. Two of 8,192 uniformly sampled previously unseen
pairs passed, for hit rate `0.00024414` and 95% Wilson interval
`[0.00006695, 0.00088981]`. Thus sparse Fourier again has high causal precision but incomplete
recall.

Across Fourier and recall sources, 1,007 unique multi-site sets cross the clean-minus-0.10
full-support threshold. Zero satisfy the separately registered effect-size condition
`max P(proper subset) <= 0.80 * P(full)`. Therefore there is no strict multi-site component from
which to launch the relative-frontier search. This is a criterion-driven stop, not evidence that
the unfiltered threshold function has no redundant interactions; the website preserves the 21
singletons and unfiltered provenance while leaving the strict network overlay empty.

## Cross-checkpoint `pyalvt` series — registered 2026-08-20

The same corrected causal pipeline is registered for donor checkpoints 96, 64, and 32 first, then
the deterministic seed-`20260820` shuffle 128, 1024, 256, 384, 192, 768, 512, and 1280. Dirty
checkpoint zero, prompt, residual grid, precision, batch size, no-cache backend, clean-minus-0.10
threshold, clean-argmax requirement, density grid, Stage-1 budget, and exact Stage-2 verification
remain unchanged. The already completed step-1500 run is not recomputed.

Every checkpoint's independent activation-patching grid freezes its expected exhaustive singleton
census before spectrum collection. Each run selects its own unrestricted and singleton-vetoed
transition densities; it may not borrow step 1500's density or samples. A donor whose all-clean
residual intervention fails the correct-letter argmax writes a terminal
`clean_behavior_not_acquired` acquisition artifact and does not proceed to a density sweep. This is
expected for step 32 based on the independent endpoint (`P(C)=0.1621`, with D preferred), but the
new batch-one endpoint intervention remains the deciding measurement.

### Measured cross-checkpoint result — 2026-08-20

All registered singleton censuses matched their independent checkpoint-transfer references. Step
32 stopped at the acquisition gate (`P(C)=0.1621036`, wrong argmax). Every later requested
checkpoint completed Stages 0–2:

| donor step | exhaustive singletons | unrestricted p | residual p | stable degree profile | Fourier-verified multisites | strict displayed multisites |
| ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| 64 | 15 | .02 | .06 | no | 41 | 40 |
| 96 | 15 | .02 | .06 | no | 30 | 24 |
| 128 | 15 | .02 | .06 | no | 33 | 29 |
| 192 | 15 | .02 | .06 | no | 30 | 23 |
| 256 | 18 | .02 | .06 | no | 31 | 24 |
| 384 | 29 | .02 | .12 | yes | 0 | 0 |
| 512 | 28 | .02 | .08 | no | 9 | 9 |
| 768 | 29 | .02 | .08 | no | 18 | 7 |
| 1024 | 25 | .02 | .10 | yes | 0 | 0 |
| 1280 | 29 | .01 | .08 | no | 14 | 4 |
| 1500 | 28 | .02 | .06 | no | 2611 across the full recall union | 649 |

“Fourier-verified” means exact Stage-2 causal survivors for new checkpoint runs; step 1500's entry
also includes its completed recall/frontier union, so its count is not directly comparable to the
sparse Stage-2-only checkpoints. “Strict displayed” additionally enforces the registered relative
proper-subset effect-size criterion. The held-out continuous-alpha gradient gate failed at every
checkpoint, so no spectrum used gradient augmentation. Most spectra also fail the neighboring-
density stability check; only steps 384 and 1024 pass. Accordingly, cross-checkpoint comparisons
should emphasize exhaustive singleton structure, selected transition density, and exactly verified
causal sets—not raw heavy-coefficient counts or rankings.

### Full-depth cross-checkpoint recall extension — registered 2026-08-20

The Stage-0-through-2 checkpoint series is followed by the same bounded recall ladder used for the
step-1500 analysis. Each eligible donor checkpoint independently runs the recall audit, fixed-point
component-shell closure, exact component completion through orders four, five, and six, a
known-network-vetoed density sweep, and both the base and expanded disconnected-mask searches. The
same clean-minus-0.10 full-support threshold and the frozen `0.80 * P(full support)` maximum-proper-
subset rule apply at every checkpoint.

This extension is conditional rather than permissive. If the recall audit yields no strict
multi-site seed, the target writes `complete_no_strict_multisite_seed` and cannot construct a
network-veto intervention. If the known-network-vetoed density curve is flat, it writes
`complete_network_veto_flat_stop` and cannot run disconnected search. These are scientific terminal
states, not missing work. The `riodwl` replication currently terminates at the first of these gates:
its 1,007 unfiltered threshold-crossing multisites contain zero strict relative minsets.

Every target writes `full_recall_ladder.json` only at a terminal state. It records the exact circuit
and recall configs, strict-minset/component counts, and SHA-256 identities of every completed stage.
Existing subset-to-metric indexes are refreshed only when the raw source inventory has grown
monotonically; a changed or removed previously indexed source still fails loudly. This keeps the
long search resumable without silently accepting a stale derived cache.

## Measured `pyalvt` result

The primary run completed on 2026-08-09. The exhaustive sweep evaluated all 3,616 prompt-grid
singletons and verified 21 singleton minsets. The unrestricted and singleton-vetoed transition
densities were `0.02` and `0.08`, respectively. Stage 1 produced 291 heavy hypotheses: 197
degree-one, 12 degree-two, 31 degree-three, and 51 degree-four coefficients. The held-out
continuous-alpha gradient gate failed, so the estimator did not use gradient augmentation.

Exact Stage-2 corner verification evaluated 545 unique non-empty site sets generated by the 94
heavy degree-two-or-higher supports. Eight distinct two-site minsets survived sufficiency and
all-path walk-down; every verified pair passes the raw-logit threshold and both singleton children
fail it. These are causally verified intervention minsets within the Fourier-generated search
space, not an exhaustive census of all possible multi-site minsets.

The density-stability gate failed: maximum degree-profile L1 distance was `0.301685` against the
preregistered `0.25` limit, although minimum cosine similarity was `0.983787`. The artifacts and
website therefore retain the intervention-scale warning. This caveat applies to using the
Fourier spectrum as evidence about the natural computation; it does not erase the exact causal
corner measurements of the reported minsets.

The preregistered causal threshold is **not** 90% accuracy. It is 80% recovery of the clean–dirty
raw-logit-difference gap: raw logit difference at least `5.761842`. Because this metric is the
correct A–E logit minus `logsumexp` of the other four logits, its exact probability equivalent is
`P(correct) >= 0.99686455` for this probe.

The website also reports a post-measurement reference rule requested on 2026-08-09: correct-answer
probability within ten percentage points of the clean probability, or
`P(correct) >= 0.89982101`. This is explicitly a derived diagnostic, not a retrospective rewrite
of the measured artifacts. It makes 28 singletons sufficient instead of 21. All eight currently
reported pairs contain one of the seven added singleton-sufficient sites, so none remains minimal
under that rule. The new analysis therefore uses a distinct
`sufficiency_clean_probability_minus_0p10_veto_28` artifact directory, fail-loudly requires the
exact 28-site census, reruns the singleton-vetoed density sweep, and collects a fresh Stage 1 before
causal verification. The independent older checkpoint-transfer grid remains a harness only for its
registered, numerically stable final-token layer-19-through-31 region; layers 17-18 are governed by
the exact batch-one census rather than the older batch-8 token-chunk values. Until the new artifacts
exist, the website marks that analysis as unprocessed rather than rethresholding or relabeling the
strict spectrum.

## Minset-network overlay

Verified minsets are grouped by size before graph construction. Each size-`n` minset is expanded
to its `n(n-1)/2` pairwise graph edges, so a three-site minset contributes three edges. Connected
components are computed independently for each minset size. The site exposes one independent
checkbox for the exhaustive singleton overlay and one for every connected multi-site component;
any combination can be displayed on one large token-by-layer grid. Individual minset cards are not
rendered because they make the sites too small and duplicate the joint overlay.

For the measured strict-threshold `pyalvt` result, all eight two-site minsets form one connected
component with nine sites and eight edges. It is bipartite. The red partition contains the
function-name-final `vt` sites at layers 2 and 6; the blue partition contains the seven partner
sites. This graph is a spatial summary of the verified minsets, not evidence that red and blue
are uniquely identifiable biological-style information streams.

## Measured clean-minus-ten-point rerun

The separately versioned rerun completed on 2026-08-09. The exact clean-minus-ten-percentage-point
threshold was `P(correct) >= 0.8998210073`, equivalently raw logit difference at least
`2.1952373493`. Its exhaustive batch-one census reproduced all 28 registered singleton minsets
exactly. After pinning those sites dirty, the residual Stage-0 curve remained strongly nonflat and
selected `p=.06` by maximum raw-logit variance (`18.3823`), compared with `p=.08` under the strict
21-site veto.

The fresh Stage 1 fit 45,005 coefficients and marked 208 heavy: 45 degree-one, 12 degree-two, 63
degree-three, and 88 degree-four. The continuous-alpha validation failed (`RMSE=.3256`, maximum
absolute error `1.7194`, cosine `-.1955`), so gradients contributed nothing to discovery. Density
stability also failed sharply: maximum degree-profile L1 distance was `1.7256` and minimum cosine
was `.1155`. The spectrum therefore carries a prominent intervention-scale warning.

Exact Stage 2 verified 13 minsets under the requested threshold: 12 pairs and one triple. These are
causal corner results within the Fourier-generated search space, not an exhaustive census of all
higher-order minsets. The pair graph is **not bipartite**. One connected component has 12 sites and
12 edges and contains the triangle `(token 53, layer 10)` -- `(token 84, layer 15)` --
`(token 112, layer 15)` -- back to `(token 53, layer 10)`, so its exact chromatic number is 3. The
single size-three minset is a separate, valid 3-color component.

The overlay uses **structural equivalence**, not graph coloring, as its node grouping.
Within each connected component and minset size, two sites are similar when their sets of co-minset
partners have high adjusted Jaccard overlap. To scale to the 1,501-site pair component, identical
profiles seed groups before deterministic complete-link assignment at `0.5`; a hard cannot-link
forbids merging two sites that coexist in one minset. This naturally groups interchangeable leaves
of a star while leaving its hub separate. For minsets of size three and above, the minsets remain
hyperedges during profile construction; clique expansion is only a drawing convention.

Neither grouping identifies a unique information stream. Structural clusters are descriptive and
change when new verified minsets change the incidence relation. The exporter therefore persists
the clustering method, threshold, per-cluster minimum/mean similarity, and an explicit statement
that the groups are not identified pathways.

## Recall audit beyond sparse Fourier proposals

The clean-minus-ten-point Stage 2 evaluated 824 supports out of a 3,588-site active space. Its
tested inventory was 32 singletons, 333 pairs, 371 triples, and 88 four-site sets; its Stage-1
interaction screen contained only 32 sites. This is enough for high-precision causal verification
of candidates, but nowhere near an exhaustive multi-site census.

`scripts/run_fourier_recall_audit.py` constructs a deterministic, independently versioned recall
audit without reusing any previously tested support. `--plan-only` is CPU-only and writes the exact
proposal inventory. The registered collection contains:

- an exact 13-site local truth table over the union of all Fourier-discovered minset sites;
- every missing pair within the original 32-site interaction screen;
- exhaustive partner sweeps for the four strongest insufficient singleton anchors;
- 8,192 uniformly sampled unseen pairs for a design-based prevalence interval;
- 4,096 one-site mutations of known minsets;
- 2,048 uniform triples plus up to 2,048 expansions of the 64 strongest insufficient pairs.

Every proposed triple first receives all missing pair-child measurements. A triple with any
sufficient child is pruned; a sufficient triple is a verified minset only if all exact pair children
fail. The exact local table similarly reports every minimal sufficient subset and tests immediate
monotonicity. Targeted proposal yields diagnose blind spots but never enter the uniform-pair
prevalence estimate.

The audit writes a content-addressed `recall_audit_config_<digest>/` directory. Proposal phases are
sharded, sidecar-digest validated, resumable, and use the same frozen full-prompt, `use_cache=False`,
batch-one inference evaluator and probability sufficiency rule as the source analysis. The final
`recall_audit.json` records all source-artifact digests, phase manifests, exact local minsets,
uniform-pair Wilson interval, triple-minimality results, and newly verified pairs/triples. The audit
remains inspectable on disk; the compact website now renders only its causally verified minsets in
the shared overlay, not the audit or raw hypothesis-generation tables.

The registered plan has 33,787 unique initial proposals (25,714 pairs). Even after collection,
absence of uniform-sample hits bounds pair prevalence only at this sample resolution and does not
rule out rare structured pairs or higher-order circuits.

### Measured recall audit — 2026-08-09

The complete audit evaluated 33,787 initial supports, 8,164 previously missing pair children, and
4,096 child-minimal triples. The 13-site exact local truth table contains 36 minsets: 13 pairs, 21
triples, one four-site set, and one six-site set. Fourier had found 13 of these; the local census
therefore adds 23 (one pair, 20 triples, the size-four set, and the size-six set). It also records
576 immediate monotonicity violations: adding a clean site can turn a sufficient support
insufficient. Any monotone redundant-pathway interpretation is therefore false for this
intervention function.

The targeted proposal families verified 1,532 additional pairs and 436 additional triples. These
misses are extremely concentrated. Site `(token 53, layer 10)`, whose singleton probability was
just below the requested threshold, occurs in 1,489 of the 1,532 pairs and all 436 triples. Of the
new results, 1,303 pairs and 435 triples are within two probability points above the exact
`0.8998210073` threshold. These are valid minsets under the requested causal rule, but most belong
to a thin, threshold-sensitive shell around one nearly sufficient site rather than 1,968 unrelated
information pathways.

The website overlay deduplicates the 13 Fourier-stage survivors, 23 exact-local additions, 1,532
verified proposal pairs, and 436 verified proposal triples by their exact site sets. The result is
2,003 multi-site minsets: 1,544 pairs, 457 triples, one size-four set, and one size-six set. They
are all sourced from causal-verification artifacts and remain the unfiltered causal inventory.

### Proper-subset effect-size filter — 2026-08-09

The compact network overlay now applies the requested stronger meaning of “minset.” A multi-site
set must still pass the measured `P(correct) >= 0.8998210073` causal threshold, and **every non-full
subset, including the empty all-dirty corner, must have `P(correct) <= 0.85`**. Here “accuracy” means
the continuous A–E-normalized correct-answer probability, not the binary argmax indicator. This is
not a new model evaluation: it filters the exact stored corner results.

The rule retains 63 of the 2,003 verified sets: 46 pairs and 17 triples, forming one connected
component of each size. It rejects 1,940 threshold-only minsets. Of those, all 1,934 sets containing
`(token 53, layer 10)` fail immediately because that singleton already has
`P(correct)=0.8992769 > 0.85`. The remaining rejected higher-order sets have some pair, triple, or
five-site proper subset above the cap. Thus the displayed networks emphasize a material joint
effect instead of a tiny increment over an almost-sufficient child.

All 50,456 measured subset results are consolidated into `subset_metric_index.json`, keyed by the
canonical sorted `(token, layer)` support and storing probability, raw logit difference, argmax,
and source provenance. Its 185 raw source files are SHA-256 registered. Site exports load this
index directly; a changed source shard makes the cache fail loudly, while first construction is
CPU-only and never runs the model. Clicking a colored cell band toggles that entire structural
cluster, including incident edges, to 20% opacity without hiding its network.

### Relative proper-subset frontier amendment — 2026-08-09

The fixed `0.85` display cap above remains part of the historical analysis, but the active rule is
now relative: a displayed or newly verified multi-site minset must have every proper subset at or
below **80% of that full support's measured A–E-normalized `P(correct)`**. The unchanged full-set
threshold remains `0.8998210073` plus the clean-answer argmax. Applying the relative rule to the
existing 2,003-set inventory retains 41 sets: 34 pairs and seven triples. Their mixed-order
hypergraph has one connected 28-site component.

This relative rule creates an exact pruning fact without assuming that model interventions are
monotone. Since a future full support cannot have probability above one, any partial subset with
`P(correct) > 0.8` can never belong to a valid larger minset. The new independently versioned
frontier search therefore completes every pair, triple, and quadruple inside the initial 28-site
component in increasing order, skipping a candidate only when a previously measured proper subset
is above `0.8`. The pair closure implements the requested network-chord diagnostic: a cycle with
verified `AB`, `BC`, `CD`, and `DA` explicitly proposes the missing `AC` and `BD`. Of the 378 pairs
inside the measured component, 113 were already cached and 265 require new evaluation.

After component completion, the search evaluates 8,192 globally degree-balanced unseen pairs over
the 3,587 singleton sites at or below `0.8`. This probe equalizes measured pair exposure rather than
concentrating proposals around another high-probability anchor. It is not exhaustive. All phases
use the full-prompt, `use_cache=False`, batch-one reference evaluator and are resumable in 256-row
digest-validated shards. New evaluations are stored in a separate `frontier_metric_index.json`;
the original 50,456-row subset index and every earlier recall artifact remain unchanged.

The completed run evaluated all 18,525 planned supports: 265 missing within-component pairs, 2,136
eligible triples, 7,932 eligible quadruples, and 8,192 degree-balanced global pairs. It found 176
new relative-criterion minsets: 12 pairs, 101 triples, and 63 quadruples. Every new survivor came
from exact completion of the known mixed-order component; the degree-balanced global probe found
zero. Combined with the 41 prior survivors, the active overlay now contains 217 multi-site minsets:
46 pairs, 108 triples, and 63 quadruples. The underlying cache now covers 68,981 distinct supports.
This is exhaustive only through order four inside the initial 28-site component after applying the
mathematically safe `P(correct) > 0.8` branch blocker; it is not an exhaustive global circuit
census. In particular, the result supports structured local incompleteness in the earlier Fourier
proposal set, not broad prevalence of successful pairs throughout the token-layer grid.

### Expanded local-order and component-shell search — 2026-08-10

The next independently versioned frontier imports every completed frontier metric index by digest
instead of recomputing any support. It extends exact within-component enumeration through orders
five and six using the same relative proper-subset rule and its only safe branch blocker. It also
exhausts the one-hop pair shell: every still-unmeasured pair with one endpoint in the discovered
28-site hypergraph and the other among all 3,587 eligible sites. This is strictly stronger than
another random-pair sample because it determines whether the current component has any missed
single-edge attachment anywhere in the full prompt grid.

Before model execution, the imported 68,981-support cache leaves 21,260 eligible five-site
supports and 86,436 component-shell pairs. The six-site count is intentionally derived only after
the five-site measurements land, because a measured five-site child above `0.8` safely blocks all
of its six-site supersets. A further 8,192 degree-balanced pairs are drawn only after the exhaustive
shell, so they cover unseen pairs with neither endpoint forced into the known component. This run
still cannot be called globally exhaustive for disconnected higher-order circuits whose sites and
pairs are individually unremarkable; its exact claims are limited to orders through six inside the
starting component and the complete one-hop pair boundary around it.

The completed expansion evaluated 159,355 new supports. Among 21,260 eligible five-site supports,
13 met the full-set threshold, but only one also met the relative proper-subset rule. Its
`P(correct)` is `0.914129`; its strongest proper subset reaches `0.704513`, or `77.07%` of the full
effect. None of the 43,467 eligible six-site supports met the full threshold; the maximum was
`0.867950`. The exhaustive 86,436-pair component shell found 11 new strict pairs, extending the
pair network by ten previously external sites. Nine of those pairs use final-token layer 16 as the
known-component endpoint. A final 8,192 degree-balanced pairs with neither endpoint forced into the
component found zero; their maximum probability was `0.497672`. The active overlay therefore
contains 229 strict multi-site minsets: 57 pairs, 108 triples, 63 quadruples, and one quintuple.
The combined subset cache covers 228,336 unique supports.

This materially sharpens the recall conclusion. The original component did have a sparse one-hop
boundary, so treating its first 28 sites as closed would have been wrong. But that boundary is very
thin—11 hits in 86,436 exhaustive tests—and it is dominated by a single late final-token hub. The
absence of any six-site survivor and of any hit in the further balanced sample suggests diminishing
returns from increasing local order or blind global pair sampling. The remaining major blind spot
is a disconnected higher-order component with no diagnostic singleton, pair, or overlap with the
known network; this experiment does not exclude such a structure.

### Recursive closure and enlarged-component census — 2026-08-13

The next registered pass imports all 228,336 cached support metrics by digest and closes the known
hypergraph component under strict pair minsets. Although the pair-only network has 37 sites, the
known higher-order hypergraph contains one additional site, token 54 layer 13; closure therefore
starts from all 38 known circuit sites. Each iteration evaluates every unseen pair with one endpoint
in the current component and the other among all 3,587 singleton-eligible sites. Only pairs meeting
the full threshold, clean argmax, and relative proper-subset rule may add sites. Iteration stops
only when an exhaustive shell adds zero new sites; an explicit 16-iteration cap fails loudly rather
than silently claiming convergence.

The initial recursive shell contains 35,356 unseen pairs. After closure, the same run completes all
eligible triples and quadruples in the converged component. Five- and six-site continuation remains
a separate stop-gated run: continue only if enlarged triples or quadruples yield new minsets or a
material near-threshold tail. No additional balanced random-pair probe is included because two
independent 8,192-pair probes already returned zero; fixed-point shell closure has a stronger exact
coverage interpretation.

The stop gate is evaluated only on completed artifacts: continue one order when the preceding new
order yields a strict relative minset, or when an insufficient support that is still expandable
under the exact `P(correct) <= 0.8` blocker reaches `P(correct) >= 0.75`. Supports above `0.8` do not
justify continuation because they mathematically block every strict superset under the relative
effect-size criterion.

The fixed-point shell completed all 35,356 unseen boundary pairs without finding a new strict pair
or attaching a new site, so the mixed-order component is pair-closed at 38 sites. Exact completion
then measured 4,017 triples and 23,768 quadruples, verifying 162 and 221 new strict relative minsets,
respectively. The separately content-addressed size-five pass measured 85,826 safely expandable
supports and verified 37 additional strict minsets (221 supports passed the full-set threshold
before the proper-subset rule). The union therefore contains 649 strict multi-site minsets before
size-six collection: 57 pairs, 270 triples, 284 quadruples, and 38 quintuples. The complete cache at
this boundary covers 377,303 unique supports. Because size five yielded strict minsets and its
maximum still-expandable insufficient probability was `0.79998785`, the preregistered gate requires
the 216,865-support size-six census; no size-seven continuation is registered.

The sealed size-six census found 12 full-threshold supports but zero strict minsets: each was
invalidated by a high-probability proper subset. The strict union therefore remains 649 minsets,
while the combined exact cache grows to 594,168 supports. The known-network-vetoed density sweep
then pinned dirty all 38 network sites and all 28 exhaustive singleton sites. Its residual response
was strongly non-flat: mean `P(correct)` rose from `0.003328` at `p=0` to `0.999935` at `p=1`, with
maximum raw-logit variance `14.94285` at the selected `p=0.10`.

That transition authorized an independently seeded disconnected search. Among 256 random masks at
`p=0.10`, 66 passed the full threshold. Twelve diverse starts and four delta-debug restarts apiece
produced 48 unique one-removal-minimal hypotheses, including 19 pairs. The run measured 37,264 new
supports; 34 candidates of size at most 12 received complete powerset verification. None passed the
strict proper-subset rule. Even the best exact candidate's maximum-proper-subset/full probability
ratio was `0.851245`, above the registered `0.80` ceiling; the worst was `0.989410`. Thus the
network-vetoed response proves that substantial learned signal remains outside the currently known
38-site component, but this random-mask audit does not identify another sharp, causally verified
minset. The larger 14 hypotheses were only one-removal-minimal and remain explicitly unverified.

In contrast, zero of 8,192 uniformly sampled previously untested pairs passed. The 95% Wilson
interval is `[0, 0.00046871]`, corresponding to an upper bound of about 3,016 sufficient pairs in
the 6,434,745-pair unscreened universe. The zero hit count is compatible with the structured family
because the targeted sweep occupies a tiny fraction of all pairs. The combined verdict is that
Fourier discovery was high precision and low recall for structured, threshold-adjacent families,
while sufficient pairs are not prevalent uniformly across residual sites.

## Launch contract

No model command is authorized by documentation or tests. After the user explicitly releases the
GPU, create the ignored sentinel and run one function at a time:

```bash
touch .gpu-runs-enabled
uv run python scripts/run_fourier_circuits.py \
  --function-id add_5 \
  --clean-step 1500 --dirty-step 0 \
  --stages 0 \
  --confirm-gpu-run
```

Stage 0 now means: inference parity, exhaustive singleton census, unrestricted density sweep, and
singleton-vetoed density sweep. Inspect those artifacts before separately authorizing `--stages 1`
or `--stages 2`. Later stages require validated earlier artifacts. Do not resume identity or
overwrite/reinterpret either identity diagnostic as `pyalvt` evidence.

The non-Fourier recall audit is a separate launch after its CPU-only plan has been inspected:

```bash
CUDA_VISIBLE_DEVICES='' uv run python scripts/run_fourier_recall_audit.py --plan-only

# Only after a fresh explicit GPU release and creating .gpu-runs-enabled:
uv run python scripts/run_fourier_recall_audit.py --confirm-gpu-run
```

An externally interrupted run resumes from its digest-complete shards; rerunning the same command
validates and skips those shards. Removing the sentinel prevents a future launch but does not signal
an already-running process. Do not interpret `proposal_plan.json` as measured data.
