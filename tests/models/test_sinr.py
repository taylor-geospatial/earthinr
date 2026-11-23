# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

import math
from pathlib import Path

import numpy as np
import pytest
import safetensors.torch
import torch

from earthinr.models.sinr import (
    ResLayer,
    SINR_Weights,
    SINRLocationEncoder,
    SINRNet,
    sinr,
)


class TestSINR:
    def test_preprocess_locs_exact_and_non_mutating(self) -> None:
        coords = torch.tensor([[-120.0, 45.0], [90.0, -30.0]], dtype=torch.float32)
        original = coords.clone()
        feats = SINRNet.preprocess_locs(coords)
        assert torch.equal(coords, original)
        assert feats.shape == (2, 4)

        scaled = torch.tensor(
            [[-120.0 / 180.0, 45.0 / 90.0], [90.0 / 180.0, -30.0 / 90.0]]
        )
        expected = torch.cat(
            (torch.sin(math.pi * scaled), torch.cos(math.pi * scaled)), dim=-1
        )
        assert torch.allclose(feats, expected, atol=1e-6)

    @pytest.mark.parametrize('batch_size', [1, 7])
    def test_forward_dynamic_batch_and_non_mutating(self, batch_size: int) -> None:
        enc = SINRLocationEncoder(num_filts=16, depth=2)
        assert enc.name == 'sinr'
        assert enc.embed_dim == 16
        assert isinstance(enc.model.feats[2], ResLayer)

        coords = torch.linspace(-90.0, 90.0, batch_size * 2).reshape(batch_size, 2)
        original = coords.clone()
        out = enc(coords)
        assert torch.equal(coords, original)
        assert out.shape == (batch_size, 16)
        assert out.dtype == torch.float32

        encoded = enc.encode(coords[:, 0].numpy(), coords[:, 1].numpy())
        assert encoded.shape == (batch_size, 16)
        assert np.allclose(encoded, out.detach().numpy(), atol=1e-6)

    def test_load_with_and_without_class_emb(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ref = SINRNet(num_inputs=4, num_filts=16, depth=2)
        ref.eval()

        # Stripped checkpoint (earthinr-zoo layout)
        stripped_path = tmp_path / 'sinr_stripped.safetensors'
        safetensors.torch.save_file(ref.state_dict(), stripped_path)
        enc_stripped = SINRLocationEncoder(
            num_filts=16, depth=2, checkpoint=stripped_path
        )

        # Full upstream checkpoint including optional class_emb.weight
        full_state = dict(ref.state_dict())
        full_state['class_emb.weight'] = torch.zeros((10, 16))
        full_path = tmp_path / 'sinr_full.safetensors'
        safetensors.torch.save_file(full_state, full_path)
        enc_full = SINRLocationEncoder(num_filts=16, depth=2, checkpoint=full_path)

        coords = torch.tensor([[-80.0, 40.0], [15.0, -25.0]])
        assert torch.allclose(enc_stripped(coords), ref(coords), atol=1e-6)
        assert torch.allclose(enc_full(coords), ref(coords), atol=1e-6)

        # Factory with monkeypatched weights
        monkeypatch.setattr(
            SINR_Weights.INAT_2023.value,
            'meta',
            {'num_inputs': 4, 'num_filts': 16, 'depth': 2, 'embed_dim': 16},
        )
        monkeypatch.setattr(SINR_Weights, 'download', lambda self: str(stripped_path))
        enc_factory = sinr(weights='INAT_2023')
        assert torch.allclose(enc_factory(coords), ref(coords), atol=1e-6)

    def test_errors(self, tmp_path: Path) -> None:
        ckpt = tmp_path / 'dummy.safetensors'
        ckpt.write_bytes(b'')
        with pytest.raises(ValueError, match='at most one'):
            SINRLocationEncoder(weights=SINR_Weights.INAT_2023, checkpoint=ckpt)
