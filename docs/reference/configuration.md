# Configuration

## Frozen experiment constants

| Setting | Value |
|---|---:|
| Primary seed | `20260715` |
| Training records per run | 96,000 |
| Effective batch | 64 |
| Optimizer steps | 1,500 |
| Learning rate | `2e-4` |
| Weight decay | 0 |
| AdamW betas / epsilon | `(0.9, 0.999)` / `1e-8` |
| Global gradient-norm cap | 1.0 after complete effective batch |
| LoRA rank / alpha / dropout | 32 / 64 / 0 |
| LoRA targets | Q, K, V, O, gate, up, down projections |
| Training dtype | BF16 |
| Attention implementation | PyTorch SDPA |
| Loss mask | assistant response tokens only |
| Training mixture | 50% single-function regression, 50% augmentation in expectation |
| Reflection variants | 16 per function per code/language/free-form family |

The physical microbatch defaults to 32 for OLMo and 16 for Qwen/provisional Gemma. It may change
only to another positive divisor of 64 after a capacity result. That does not change the effective
batch or target-token denominator.

## Effective-batch ablation settings

The original table remains frozen for the batch-64 baseline. The dated exploratory sweep supports
effective batches `32, 16, 8, 4, 2, 1` for the correct-condition OLMo and Qwen runs. Every size
uses all 96,000 records, the same order and seed, and the same optimizer hyperparameters. Its
default physical microbatch is the largest model default no greater than the selected effective
batch. Nonbaseline artifacts live under `effective_batch_<B>/` beneath the original seed path.

Comparison checkpoints are aligned by examples seen. A smaller-batch run additionally saves step
1 and refreshes its single rolling resume state at every saved checkpoint.

## LoRA-rank ablation settings

The exploratory rank axis is `1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024`, followed by a
separately implemented full-finetuning endpoint. The first sweep is correct-condition OLMo/Qwen
only at effective batch 64. Rank 32 is the existing baseline; other adapter artifacts live under
`lora_rank_<R>/`.

For every LoRA rank:

```text
alpha = 2 * rank
dropout = 0
targets = q/k/v/o + gate/up/down projections
```

The default rank-scaled physical microbatches for ranks 32, 64, 128, 256, 512, and 1024 are
`32, 16, 8, 4, 2, 1` for OLMo and `16, 8, 4, 2, 1, 1` for Qwen. Rank 1–32 uses the model's
baseline physical microbatch. Every value divides 64, so gradient accumulation preserves the
effective-batch target-token mean and one-clip-per-update contract.

Full finetuning has no rank or alpha. `RunKey(lora_rank=None)` reserves `full_finetune/`, while the
LoRA training-spec constructor rejects it. Its offload backend is deliberately not runnable until
the full-parameter objective-parity and capacity gates in the runbook are implemented and passed.

## Checkpoint and resume settings

Adapters are saved at steps:

```text
1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512,
768, 1024, 1280, 1500
```

Step 0 is a virtual frozen checkpoint. A single rolling optimizer/RNG state is normally refreshed
at 256, 512, 1024, and 1500; an explicitly requested `--stop-after-step` also writes a state so the
capacity probe can continue without replay.

## GPU gates

All training, evaluation, and patching CLIs require:

```text
.gpu-runs-enabled exists AND --confirm-gpu-run is present
```

Gemma additionally requires `--allow-provisional-gemma`.

## Token-weight execution settings

Token-local weight patching exposes `--token-weight-runtime reference|optimized` and
`--token-weight-patch-batch-size`. The reference runtime is permanently fixed to batch size 8 and
computes full-sequence logits exactly as commit `956bfa4` did. The candidate optimized runtime keeps
that same readout but reuses exact recipient activations upstream of the patched layer. Both
production runtimes are fixed to batch size 8; other shapes are benchmark-only until they pass the
complete exact-parity gate.

The reference runtime remains the default until the complete exact-parity and GPU benchmark ladder
in [token-weight-performance.md](../development/token-weight-performance.md) passes. These settings
change execution only; they never enter or alter scientific artifact identity.

## Tooling

`pyproject.toml` pins Python 3.13 and configures `uv`, Ruff, ty, pytest, coverage, and pre-commit.
The full CPU gate is:

```bash
CUDA_VISIBLE_DEVICES='' uv run pre-commit run --all-files
```

Coverage fails below 92% for the pure contract package. Live CUDA runtime modules are explicitly
omitted and require the post-authorization smoke-test ladder.
