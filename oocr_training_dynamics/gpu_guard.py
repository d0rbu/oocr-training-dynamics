"""Two-part authorization gate for every command that can allocate CUDA memory."""

from __future__ import annotations

from pathlib import Path

from beartype import beartype

GPU_ENABLE_SENTINEL = ".gpu-runs-enabled"


@beartype
def require_gpu_authorization(root: Path, *, confirmed: bool) -> None:
    sentinel = root / GPU_ENABLE_SENTINEL
    if not confirmed:
        raise RuntimeError("GPU execution requires the explicit --confirm-gpu-run flag")
    if not sentinel.is_file():
        raise RuntimeError(
            f"GPU execution is paused; create {sentinel} only after the user releases the GPU"
        )


__all__ = ["GPU_ENABLE_SENTINEL", "require_gpu_authorization"]
