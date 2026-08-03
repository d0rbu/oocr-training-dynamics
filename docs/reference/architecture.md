# Architecture

The repository separates pure experiment contracts from explicitly gated live-model runtime code.

```text
contracts + model registry + deterministic corpus
                    │
          CPU plan / validation / tests
                    │
                    ▼
      gated training → adapter checkpoint index
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
 checkpoint evaluation   residual patch grids
        └───────────┬───────────┘
                    ▼
              site exporter
                    ▼
      static interactive explainer
```

## Pure package modules

| Module | Responsibility |
|---|---|
| `contracts.py` | Conditions, run keys, training spec, batch/rank/checkpoint/seed constants |
| `activation_examples.py` | Size-matched audit/FineWeb/format-control candidate corpora, balanced concrete format-control patch donors, and nearest-example constants |
| `letter_propensity.py` | Strict run-scoped artifact contract for token-level full-vocabulary A–E propensity |
| `models.py` | Pinned registry, dimensions, provisional gate, rank/microbatch/state arithmetic |
| `data.py` | 19 functions, deterministic matched corpora, derangement, reflection records |
| `semantics.py` | Restricted semantic scoring for generated lambda expressions |
| `tokenization.py` | Chat-template prefix proof, assistant-only labels, collation |
| `metrics.py` | Stable softmax, curve AUC, chance adjustment, normalized patch effect |
| `patching.py` | Prompt corruption, fixed answer-label controls, and validated patch plans/cells |
| `representation_alignment.py` | Versioned noncausal metric/interface contract and run-scoped artifact paths |
| `weight_alignment.py` | Canonical unordered checkpoint-pair contract for effective projection weights |
| `artifacts.py` | Atomic JSON, adapter paths/digests, checkpoint-index invariants |
| `planning.py` | Baseline/ablation matrices, capacity bounds, and storage estimates |
| `gpu_guard.py` | Two-part authorization gate |

## Gated runtime modules

| Module | Responsibility |
|---|---|
| `runtime_models.py` | Processor/model loading, revision check, LoRA attachment, block discovery |
| `runtime_training.py` | Exact batch aggregation, clipping, dense adapters, rolling resume |
| `runtime_evaluation.py` | Intended/planted choice metrics and semantic free-form generation |
| `runtime_patching.py` | Activation/LoRA interventions, dual-label outcomes, full-vocabulary lens sidecars, and cell-selected cosine neighbors across prompts or checkpoint time |
| `runtime_representation_alignment.py` | Multi-boundary unpatched activation capture and float32 cosine/L2 grids |
| `runtime_weight_alignment.py` | Full effective-matrix reconstruction and symmetric Frobenius/row/column geometry |
| `runtime_letter_propensity.py` | Resumable checkpoint evaluation of standalone A–E probability mass on raw FineWeb tokens |

Importing these modules does not launch CUDA. Their script entry points call `gpu_guard` before
invoking live runtime functions.

## Artifact layout

```text
artifacts/
├── preregistered_plan.json
└── runs/<model>/<condition>/seed_20260715/
    ├── config.json
    ├── dataset_manifest.json
    ├── model_manifest.json
    ├── training_metrics.json
    ├── checkpoint_index.json
    ├── checkpoints/step_XXXXXX/adapter/
    ├── resume/latest.{json,pt}
    ├── evaluations/{index.json,step_XXXXXX.json}
    ├── letter_propensity/{index.json,checkpoint_step_XXXXXX.json}
    ├── patching/
        ├── <mode>/recipient_step_XXXXXX/donor_step_XXXXXX.json
        └── <branch_interface>/<mode>/recipient_step_XXXXXX/donor_step_XXXXXX.json
    ├── representation_alignment/sequence_end/<activation_interface>/<mode>/
    │   └── recipient_step_XXXXXX/donor_step_XXXXXX.json
    ├── activation_examples/sequence_end/<interface>/
    │   ├── checkpoint_step_XXXXXX.json
    │   └── <candidate_source>/checkpoint_step_XXXXXX.json
    └── vocabulary_logit_lens/sequence_end/checkpoint_step_XXXXXX.json
```

Nonbaseline effective-batch and LoRA-rank runs are isolated one level below the seed directory:

```text
artifacts/runs/<model>/correct/seed_20260715/effective_batch_<B>/
artifacts/runs/<model>/correct/seed_20260715/lora_rank_<R>/
artifacts/runs/<model>/correct/seed_20260715/full_finetune/
```

The batch and rank LoRA runs reuse the training/checkpoint/evaluation layout but do not enter the
baseline activation-patch manifest. `full_finetune/` is a reserved identity only; the adapter
runtime rejects it until a distinct offload backend is validated. The site exporter exposes
measured batch/rank trajectories in separate acquisition payloads and emits no synthetic
nonbaseline curve.

Letter-propensity sidecars use the same run isolation, so batch and rank selectors resolve the
matching measurement namespace. The exporter validates every sidecar and places only compact
checkpoint summaries in the main site payload. Missing checkpoints have no row; the browser uses
the stored registered-checkpoint index to avoid drawing a line across gaps.

The first form is the backward-compatible `resid_post` layout. Exploratory branch artifacts use
an explicit `attention_input`, `attention_output`, `mlp_input`, or `mlp_output` directory. Global
all-token parameter interventions use `patching/layer_only/block_weights/`; token-local learned-
weight contributions use `patching/sequence_end/token_weights/`. No interface can overwrite or be
silently reinterpreted as another.

`artifacts/` is ignored because adapters and optimizer states are large. The compact site payload
is generated at `site/data/experiment.json` and committed. It contains a content-addressed patch
manifest; each measured recipient/donor grid is exported as a separate compact file under
`site/data/patches/`. The page eagerly fetches every currently measured grid across every model,
condition, boundary, and patch mode with bounded concurrency while keeping the initial HTML and
metadata payload small as the deterministic checkpoint-priority temporal atlas grows.
Each parsed grid is compacted to typed probability arrays and retained in memory, so recipient and
donor slider movement performs no network fetch or JSON parse after the one-time preload. The page
polls the separately exported `site/data/patch-manifest.json`; newly generated patch and
activation-neighbor artifacts are added while an existing tab remains open. Missing patch views retain exact
token-axis metadata but contain no probabilities or deltas; the site renders reserved unprocessed
cells instead. Missing behavioral curves remain explicitly synthetic.

Representation alignment has a separate manifest and compact chunk tree under
`site/data/representation-alignment/`. One loaded chunk contains typed cosine, raw-L2,
source-norm, and recipient-norm arrays for all 19 functions. It shares prompt/checkpoint controls
with patching but never reuses a probability chunk as an alignment result. The browser preloads
both atlases, renders missing alignment grids as unprocessed, uses fixed `[-1, 1]` cosine colors,
and reads boundary-specific robust L2 scales from the exported manifest. Same-prompt,
same-checkpoint identity values are labeled analytic rather than measured.

Effective-weight alignment has a third manifest under `site/data/weight-alignment/`. The scalar
atlas uses a complete component axis plus input/layer/output columns. Frozen non-target tensors are
analytic identities, and vector norms mark decomposed metrics N/A. Four packed little-endian float32
detail chunks remain separate per unordered checkpoint pair. Selecting a pair prefetches all four
with two concurrent transfers and a sixteen-chunk (four complete pairs) LRU cache. Slider changes
discard stale queued detail prefetches and touch all cached chunks for the newly selected pair.
Eviction is atomic at the four-chunk pair boundary, so no pair can remain only partially cached. The
browser draws large detail grids on canvas and uses exported 128-channel metadata to outline
attention heads over one contiguous 64-column grid.
Both recipient/donor orientations reference the same scalar and detail digests; there is no second
directed computation.

Final-suffix answer-label artifacts preserve a second typed matrix for the source-correct label and
the original compact A–E logit-lens tensors as raw provenance. They also preserve the deterministic
choice permutation or unrelated-question identity needed to audit what the source label means.
The browser's current lens comes from a separate checkpoint-indexed sidecar: one clean
full-sequence grid plus all eleven active prompt-source grids per function, each storing five token IDs and
absolute probabilities normalized over the complete output vocabulary. For mixed-checkpoint
cells, the source sidecar uses the donor checkpoint's final norm/unembedding while the clean
sidecar and patched downstream forward use the recipient checkpoint. The exporter splits each raw
checkpoint into function chunks; missing sidecars are labeled unprocessed with no A–E fallback.

Format/content patch modes reuse the candidate-corpus prompt builders but select one concrete
presentation per function in a paired round-robin panel. Every active record retains a
source-correct A–E label, including the conversational five-choice controls. They otherwise use the
ordinary prompt-counterfactual artifact namespace and never substitute neighbor scores for causal
values. The earlier free-form conversational IDs remain readable legacy namespaces but are absent
from the active selectors.
The resumable full-vocabulary sidecars validate and retain the seven earlier prompt sources while
atomically appending the two varied-MCQ-format and two corrected conversational A–E sources.
Partial legacy files remain exportable, so missing source readouts stay explicitly unprocessed
rather than blocking already measured sides.

Activation-example raw artifacts are indexed by candidate source and one checkpoint because both
source and recipient reference banks can be reused across every donor/recipient pairing involving
that checkpoint. The legacy experiment/audit source keeps its original path; FineWeb and each
format/content control use an explicit candidate-source child. The exporter splits each raw file
into one compact mode/function neighbor chunk plus a shared candidate catalog. The manifest
resolves source examples from the donor checkpoint and recipient examples from the recipient
checkpoint without crossing candidate sources. Only prompt IDs, corpus metadata/provenance, exact
tokenizer labels, maximizing token indices, and cosine scores reach the browser; hidden vectors are
transient.

Measured evaluation exports also include one acquisition curve per registered function alongside
the all-function aggregate. The aggregate is checked against the arithmetic mean of the 19
per-function values at every checkpoint and metric. Synthetic-preview runs expose only the
aggregate; the site disables individual probes rather than synthesizing function-level values.

## Model-family boundary

Decoder blocks are resolved through architecture-specific candidate paths and must match the
registry's exact layer count. `resid_post` operates on the emitted block tensor; branch interfaces
resolve each block's concrete `self_attn` or `mlp` module and hook its input or output. Models of
different families are compared by curves and relative depth only; their activation coordinates
are never directly exchanged.

Both weight interfaces standardize every checkpoint as a PEFT model. A trained checkpoint supplies
its saved LoRA factors; step 0 supplies exact-zero factors. Donor A/B tensors are retained on CPU.
Global `block_weights` copies them into one recipient block after exact name/shape validation and
restores the recipient factors in a `finally` path; its compact export declares
`axis_kind=layer_only` and has no token positions. Token-local `token_weights` leaves parameters
unchanged and hooks all seven projection outputs in one block, adding
`(DeltaW_donor - DeltaW_recipient) h` only at the selected token. Its compact export declares a
real `token_layer` axis and an explicit `selected_token_decoder_block` scope.
