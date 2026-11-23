# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch


@pytest.fixture
def fake_safetensors(tmp_path: Path) -> Path:
    import safetensors.torch

    path = tmp_path / 'weights.safetensors'
    safetensors.torch.save_file({'weight': torch.arange(6, dtype=torch.float32)}, path)
    return path


@pytest.fixture(autouse=True)
def _isolate_torch_hub_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.hub, 'get_dir', lambda: str(tmp_path / 'torch_hub'))
