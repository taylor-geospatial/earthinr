# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import numpy as np
import pytest
import safetensors.torch
import torch

from neuralftw.models.lesinr import LESINR_Weights, LESINRLocationEncoder, lesinr


class TestLESINRLocationEncoder:
    @pytest.mark.parametrize('batch_size', [1, 7])
    def test_untrained_dynamic_batch(self, batch_size: int) -> None:
        enc = LESINRLocationEncoder(num_inputs=4, num_filts=16, depth=2, embed_dim=12)
        assert enc.name == 'le-sinr'
        assert enc.embed_dim == 12

        coords = torch.linspace(-60.0, 60.0, batch_size * 2).reshape(batch_size, 2)
        original = coords.clone()
        out = enc(
            coords,
            year=torch.full((batch_size,), 2024.0),
            month=torch.full((batch_size,), 6.0),
        )
        assert torch.equal(coords, original)
        assert out.shape == (batch_size, 12)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

        encoded = enc.encode(coords[:, 0].numpy(), coords[:, 1].numpy())
        assert encoded.shape == (batch_size, 12)
        assert np.allclose(encoded, out.detach().numpy(), atol=1e-6)

    @pytest.mark.parametrize('prefix', ['', 'pos_enc.'])
    def test_load_safetensors_key_schemas(self, tmp_path: Path, prefix: str) -> None:
        ref = LESINRLocationEncoder(num_inputs=4, num_filts=16, depth=2, embed_dim=16)
        raw: dict[str, torch.Tensor] = {}
        for k, v in ref.backbone.state_dict().items():
            raw[f'{prefix}{k}'] = v.clone()
        for k, v in ref.class_emb.state_dict().items():
            raw[f'{prefix}class_emb.{k}'] = v.clone()
        # Extra unrelated key (e.g. species_enc) should be ignored cleanly
        raw['species_enc.weight'] = torch.zeros(4, 4)

        ckpt = tmp_path / f'lesinr_{prefix.replace(".", "_")}.safetensors'
        safetensors.torch.save_file(raw, ckpt)

        loaded = LESINRLocationEncoder(
            num_inputs=4, num_filts=16, depth=2, embed_dim=16, checkpoint=ckpt
        )
        coords = torch.tensor([[13.40, 52.52], [-122.42, 37.77]])
        assert torch.allclose(ref(coords), loaded(coords), atol=1e-6)

    def test_weights_enum_and_factory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        w = LESINR_Weights.verify('DEFAULT')
        assert w.repo_id == 'taylor-geospatial/neuralftw-zoo'
        assert w.filename == 'le-sinr/model.safetensors'
        assert w.meta['num_inputs'] == 4
        assert w.meta['num_filts'] == 256
        assert w.meta['depth'] == 4
        assert w.meta['embed_dim'] == 256
        assert w.meta['license'] == 'MIT'
        assert len(w.meta['sha256']) == 64
        assert len(w.meta['onnx_sha256']) == 64

        ref = LESINRLocationEncoder(num_inputs=4, num_filts=256, depth=4, embed_dim=256)
        raw = {
            **{k: v.clone() for k, v in ref.backbone.state_dict().items()},
            **{
                f'class_emb.{k}': v.clone()
                for k, v in ref.class_emb.state_dict().items()
            },
        }
        ckpt = tmp_path / 'le_sinr.safetensors'
        safetensors.torch.save_file(raw, ckpt)
        monkeypatch.setattr(LESINR_Weights, 'download', lambda self: str(ckpt))

        model = lesinr(weights='DEFAULT')
        coords = torch.tensor([[13.40, 52.52], [-0.13, 51.51]])
        assert model.embed_dim == 256
        assert torch.allclose(ref(coords), model(coords), atol=1e-6)

    def test_errors(self, tmp_path: Path) -> None:
        ckpt = tmp_path / 'dummy.safetensors'
        safetensors.torch.save_file({'x': torch.zeros(1)}, ckpt)
        with pytest.raises(ValueError, match='at most one'):
            LESINRLocationEncoder(weights=LESINR_Weights.DEFAULT, checkpoint=ckpt)
        with pytest.raises(ValueError, match='Invalid weights'):
            lesinr(weights='DOES_NOT_EXIST')
