from __future__ import annotations

from pathlib import Path

import pytest

from oocr_training_dynamics.gpu_guard import GPU_ENABLE_SENTINEL, require_gpu_authorization


def test_gpu_execution_requires_flag_and_user_created_sentinel(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="flag"):
        require_gpu_authorization(tmp_path, confirmed=False)
    with pytest.raises(RuntimeError, match="paused"):
        require_gpu_authorization(tmp_path, confirmed=True)

    (tmp_path / GPU_ENABLE_SENTINEL).touch()
    require_gpu_authorization(tmp_path, confirmed=True)
