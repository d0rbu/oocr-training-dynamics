# Testing

## CPU gate

```bash
CUDA_VISIBLE_DEVICES='' uv run ruff check .
CUDA_VISIBLE_DEVICES='' uv run ty check
CUDA_VISIBLE_DEVICES='' uv run pytest
CUDA_VISIBLE_DEVICES='' uv run python scripts/plan_batch_size_ablation.py
CUDA_VISIBLE_DEVICES='' uv run python scripts/plan_lora_rank_ablation.py
CUDA_VISIBLE_DEVICES='' uv run python scripts/export_site.py
node --check site/app.js
```

The tests cover:

- fixed schedules, run IDs, and storage arithmetic;
- example-matched effective-batch schedules plus isolated batch/rank/full artifact paths;
- rank-scaled LoRA parameter, storage, optimizer-state, alpha, and microbatch arithmetic;
- model revision/dimension and provisional-model contracts;
- deterministic matched corpora and exact wrong-alias/wrong-implementation semantics;
- reflection option coverage for intended, deranged, and inverse-deranged rules;
- semantic free-form lambda scoring through a restricted AST evaluator;
- assistant-only token masks and batch collation;
- metrics, patch-plan constraints, atomic artifacts, and the GPU double gate;
- branch-interface target resolution plus positional and keyword input/output replacement hooks;
- an explicit synthetic-status contract for the committed site payload.

The package threshold is 92% branch-aware coverage. Live runtime modules are omitted from that
threshold and must not be treated as tested merely because the CPU suite passes.

Token-local weight-kernel optimizations also follow the bit-exact GPU parity ladder in
[token-weight-performance.md](token-weight-performance.md). A faster timing or a close floating-
point match is not sufficient to change the default runtime.

## Tokenizer/config probe

```bash
CUDA_VISIBLE_DEVICES='' uv run python scripts/validate_tokenizers.py
```

This downloads/loads processors and config metadata only. It catches chat-template incompatibility
without allocating model weights.

## First authorized GPU smoke test

After the user releases the GPU, follow the runbook's step-1 capacity probe. Acceptance requires:

- resolved revision equals the pin;
- decoder/LoRA module discovery succeeds;
- actual trainable parameter count equals the architecture-derived expectation;
- one complete effective batch finishes with finite loss/norm;
- peak allocated VRAM is recorded;
- step-1 adapter, digest, optimizer/RNG resume state, and paused marker are present;
- `--resume` continues from step 2 instead of replaying step 1.

Run this separately for each model family before launching its 1,500-step baseline matrix. An OOM
is a capacity result, not permission to weaken the baseline contract; reduce only the physical
microbatch to another positive divisor of 64. The separately registered batch-size ablation uses
`scripts/run_batch_size_sweep.py` and must not be mistaken for a capacity workaround.

The rank sweep uses `scripts/run_lora_rank_sweep.py`. Probe each model at rank 256 before assuming
it fits, and do not use gradient accumulation as a claim that ranks 512/1024 fit: their native
parameter/optimizer state exceeds the safety budget before activations. Full finetuning requires
a separate offload and objective-parity smoke-test ladder that is not yet implemented.
