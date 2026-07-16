# Getting started

## Prerequisites

- Python 3.13
- `uv`
- Node.js only for the JavaScript syntax check
- no GPU is needed for setup, dataset checks, tokenizer checks, tests, or the site preview

## Safe CPU setup

```bash
uv sync
CUDA_VISIBLE_DEVICES='' uv run pytest
CUDA_VISIBLE_DEVICES='' uv run python scripts/plan_experiments.py
CUDA_VISIBLE_DEVICES='' uv run python scripts/validate_tokenizers.py
CUDA_VISIBLE_DEVICES='' uv run python scripts/export_site.py
node --check site/app.js
```

`validate_tokenizers.py` may download tokenizer/config metadata. It does not instantiate model
weights. The tokenizer check must show a nonempty assistant target after each model's chat
template; Qwen thinking mode is disabled where the template supports that option.

## Preview the explainer

```bash
uv run python -m http.server 4174 --directory site
```

Open <http://127.0.0.1:4174>. Until measured artifacts exist, the banner and patch badge must say
that curve data are synthetic. Never infer a result from preview curves. Unprocessed patch
heatmaps contain no values and use the reserved purple hatch throughout.

## Before touching the GPU

Read [the GPU runbook](../operations/gpu-runbook.md) and
[the storage plan](../operations/storage-plan.md). GPU commands intentionally fail unless both
of these are true:

1. the user has explicitly released the GPU and the ignored `.gpu-runs-enabled` sentinel exists;
2. the command includes `--confirm-gpu-run`.

The Gemma command has a third gate because the requested model name does not exist as an official
checkpoint. The provisional E4B-it substitution requires explicit confirmation and
`--allow-provisional-gemma`.
