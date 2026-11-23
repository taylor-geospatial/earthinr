# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import numpy as np
import pytest
import safetensors.torch
import torch

from earthinr.models.gtloc import GTLoc_Weights, GTLocLocationEncoder, gtloc


class TestGTLocLocationEncoder:
    @pytest.mark.parametrize('batch_size', [1, 7])
    def test_untrained_dynamic_batch(self, batch_size: int) -> None:
        enc = GTLocLocationEncoder(
            sigma=(1.0, 16.0), encoded_size=8, hidden_dim=16, out_dim=12
        )
        assert enc.name == 'gtloc'
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

    @pytest.mark.parametrize('prefix', ['', 'location_encoder.'])
    def test_load_safetensors_key_schemas(self, tmp_path: Path, prefix: str) -> None:
        ref = GTLocLocationEncoder(
            sigma=(1.0, 16.0), encoded_size=8, hidden_dim=16, out_dim=12
        )
        raw = {f'{prefix}{k}': v.clone() for k, v in ref.encoder.state_dict().items()}
        # Extra unrelated key (e.g. time_encoder) should be ignored cleanly
        raw['time_encoder.TimeEnc0.head.0.weight'] = torch.zeros(2, 2)

        ckpt = tmp_path / f'gtloc_{prefix.replace(".", "_")}.safetensors'
        safetensors.torch.save_file(raw, ckpt)

        loaded = GTLocLocationEncoder(
            sigma=(1.0, 16.0),
            encoded_size=8,
            hidden_dim=16,
            out_dim=12,
            checkpoint=ckpt,
        )
        coords = torch.tensor([[13.40, 52.52], [-122.42, 37.77]])
        assert torch.allclose(ref(coords), loaded(coords), atol=1e-6)

    def test_weights_enum_and_factory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        w = GTLoc_Weights.verify('DEFAULT')
        assert w.repo_id == 'taylor-geospatial/earthinr-zoo'
        assert w.filename == 'gtloc/model.safetensors'
        assert w.meta['sigma'] == (1.0, 16.0, 256.0)
        assert w.meta['encoded_size'] == 256
        assert w.meta['hidden_dim'] == 1024
        assert w.meta['output_dim'] == 512
        assert w.meta['license'] == 'CC-BY-NC-4.0'
        assert len(w.meta['sha256']) == 64
        assert len(w.meta['onnx_sha256']) == 64

        ref = GTLocLocationEncoder(
            sigma=(1.0, 16.0, 256.0), encoded_size=256, hidden_dim=1024, out_dim=512
        )
        ckpt = tmp_path / 'gtloc.safetensors'
        safetensors.torch.save_file(
            {k: v.clone() for k, v in ref.encoder.state_dict().items()}, ckpt
        )
        monkeypatch.setattr(GTLoc_Weights, 'download', lambda self: str(ckpt))

        model = gtloc(weights='DEFAULT')
        coords = torch.tensor([[13.40, 52.52], [-0.13, 51.51]])
        assert model.embed_dim == 512
        assert torch.allclose(ref(coords), model(coords), atol=1e-6)

    def test_errors(self, tmp_path: Path) -> None:
        ckpt = tmp_path / 'dummy.safetensors'
        safetensors.torch.save_file({'x': torch.zeros(1)}, ckpt)
        with pytest.raises(ValueError, match='at most one'):
            GTLocLocationEncoder(weights=GTLoc_Weights.DEFAULT, checkpoint=ckpt)
        with pytest.raises(ValueError, match='Invalid weights'):
            gtloc(weights='DOES_NOT_EXIST')
