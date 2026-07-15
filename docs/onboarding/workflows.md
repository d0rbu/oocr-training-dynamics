# Workflows

## Change the experiment contract

1. Read the [preregistration](../research/preregistration.md).
2. Add a dated follow-up rather than silently changing a frozen prediction or primary metric.
3. Update validated constants/dataclasses in `contracts.py`, `models.py`, or `data.py`.
4. Update the CPU plan, tests, docs, and site schema together.
5. Do not mix artifacts from the old and new contract under one run key.

## Add a model family

1. Pin a full Hugging Face commit SHA and verify architecture metadata without loading weights.
2. Record layer, residual, MLP, query, and key/value widths.
3. Add tokenizer/chat-template coverage and a tokenizer-only probe.
4. Add decoder-block path candidates and fail if exactly one path of the expected length is not
   resolved.
5. Verify the expected LoRA parameter count before the first optimizer step.
6. Capacity-probe at a preregistered checkpoint only after GPU authorization.

## Change an artifact schema

1. Keep writes atomic through `write_json`.
2. Make readers reject missing or mistyped fields.
3. Update `scripts/export_site.py` and add a committed synthetic fixture/regression test.
4. Keep measured and synthetic status explicit at both page and selected-view level.

## Refresh the website

```bash
CUDA_VISIBLE_DEVICES='' uv run python scripts/export_site.py
node --check site/app.js
uv run python -m http.server 4174 --directory site
```

The exporter discovers completed evaluation indices and patch JSON files under `artifacts/runs/`.
Missing views remain synthetic and visibly labeled; partial data are never silently interpolated.

## Before handoff

```bash
CUDA_VISIBLE_DEVICES='' uv run pre-commit run --all-files
```

Report the exact number of tests, whether tokenizer metadata was validated, and whether any GPU
work or model-weight load occurred.
