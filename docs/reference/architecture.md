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
| `contracts.py` | Conditions, run keys, training spec, batch/checkpoint/seed constants |
| `models.py` | Pinned model registry, dimensions, provisional gate, LoRA/storage arithmetic |
| `data.py` | 19 functions, deterministic matched corpora, derangement, reflection records |
| `semantics.py` | Restricted semantic scoring for generated lambda expressions |
| `tokenization.py` | Chat-template prefix proof, assistant-only labels, collation |
| `metrics.py` | Stable softmax, curve AUC, chance adjustment, normalized patch effect |
| `patching.py` | Prompt corruption and validated patch plans/cells |
| `artifacts.py` | Atomic JSON, adapter paths/digests, checkpoint-index invariants |
| `planning.py` | Nine-run matrix and storage estimate |
| `gpu_guard.py` | Two-part authorization gate |

## Gated runtime modules

| Module | Responsibility |
|---|---|
| `runtime_models.py` | Processor/model loading, revision check, LoRA attachment, block discovery |
| `runtime_training.py` | Exact batch aggregation, clipping, dense adapters, rolling resume |
| `runtime_evaluation.py` | Intended/planted choice metrics and semantic free-form generation |
| `runtime_patching.py` | Residual, attention, and MLP boundary capture/replacement across sample or checkpoint time |

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
    └── patching/
        ├── <mode>/recipient_step_XXXXXX/donor_step_XXXXXX.json
        └── <branch_interface>/<mode>/recipient_step_XXXXXX/donor_step_XXXXXX.json
```

The first form is the backward-compatible `resid_post` layout. Exploratory branch artifacts use
an explicit `attention_input`, `attention_output`, `mlp_input`, or `mlp_output` directory so no
interface can overwrite another.

`artifacts/` is ignored because adapters and optimizer states are large. The compact site payload
is generated at `site/data/experiment.json` and committed. It contains a content-addressed patch
manifest; each measured recipient/donor grid is exported as a separate compact file under
`site/data/patches/`. The page eagerly fetches every currently measured grid across every model,
condition, boundary, and patch mode with bounded concurrency while keeping the initial HTML and
metadata payload small as the deterministic border-first temporal atlas grows.
Each parsed grid is compacted to typed probability arrays and retained in memory, so recipient and
donor slider movement performs no network fetch or JSON parse after the one-time preload. The page
polls the separately exported `site/data/patch-manifest.json`; newly generated artifacts are added
to the same eager preload while an existing tab remains open. Missing patch views retain exact
token-axis metadata but contain no probabilities or deltas; the site renders reserved unprocessed
cells instead. Missing behavioral curves remain explicitly synthetic.

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
