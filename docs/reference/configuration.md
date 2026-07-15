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

The physical microbatch defaults to 32 for OLMo/Qwen and 16 for provisional Gemma. It may change
only to another positive divisor of 64 after a capacity result. That does not change the effective
batch or target-token denominator.

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

## Tooling

`pyproject.toml` pins Python 3.13 and configures `uv`, Ruff, ty, pytest, coverage, and pre-commit.
The full CPU gate is:

```bash
CUDA_VISIBLE_DEVICES='' uv run pre-commit run --all-files
```

Coverage fails below 92% for the pure contract package. Live CUDA runtime modules are explicitly
omitted and require the post-authorization smoke-test ladder.
