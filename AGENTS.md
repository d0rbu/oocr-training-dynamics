# AGENTS.md — OOCR training dynamics

This is a correctness-first research repository. Read the experiment contract before editing
runtime code or interpreting artifacts.

## Read first

- [README.md](README.md) — status, scope, and quickstart
- [docs/research/preregistration.md](docs/research/preregistration.md) — frozen predictions and
  decision rules
- [docs/experiments/design.md](docs/experiments/design.md) — matched corpus and evaluation design
- [docs/experiments/activation-patching.md](docs/experiments/activation-patching.md) — causal
  intervention semantics
- [docs/operations/gpu-runbook.md](docs/operations/gpu-runbook.md) — authorization and launches
- [docs/operations/storage-plan.md](docs/operations/storage-plan.md) — disk budget and retention
- [docs/reference/architecture.md](docs/reference/architecture.md) — code and artifact flow

## Hard operational boundaries

- Do not create `.gpu-runs-enabled`, load model weights, or launch CUDA work without an explicit
  user signal that the shared GPU is free.
- Every GPU entry point must retain both authorization gates: the ignored sentinel and
  `--confirm-gpu-run`.
- The Gemma slot is provisional because no official Gemma 4 9B exists. Do not pass
  `--allow-provisional-gemma` until the user confirms the substitute.
- Do not label synthetic site data as results. `site/data/experiment.json` records its status;
  each patch view carries a measured/unprocessed badge, and unprocessed cells contain no value.
- Treat `artifacts/` as valuable, local research state. Do not delete or overwrite it casually.
  Training refuses to overwrite partial or completed runs.
- Patch raw hidden states only within one model family and pinned revision.

## Implementation conventions

- Use `uv sync` and `uv run ...`.
- Keep deterministic, testable contracts in `oocr_training_dynamics/`; keep orchestration in
  `scripts/`.
- Preserve the exact effective-batch objective: sum assistant-token losses across microbatches,
  divide once by the total target-token count for all 64 records, then clip once.
- Preserve the fixed corpus seed, derangement, checkpoint schedule, and intended/planted metrics
  unless a new dated experiment explicitly supersedes the preregistration.
- Store model identifiers at full 40-character revisions and fail loudly on architecture,
  trainable-parameter-count, or artifact-shape mismatches.
- Update docs and the website export schema when artifact contracts change.

## Required CPU validation

```bash
CUDA_VISIBLE_DEVICES='' uv run python scripts/export_site.py
node --check site/app.js
CUDA_VISIBLE_DEVICES='' uv run pre-commit run --all-files
```

Tokenizer validation is read-only and does not load model weights:

```bash
CUDA_VISIBLE_DEVICES='' uv run python scripts/validate_tokenizers.py
```

Report separately whether code/tests were validated and whether any actual GPU experiment ran.
