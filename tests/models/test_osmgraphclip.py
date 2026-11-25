# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import numpy as np
import pytest
import safetensors.torch
import torch

from neuralftw.models.osmgraphclip import (
    OSMGraphCLIP_Weights,
    OSMGraphCLIPLocationEncoder,
    osmgraphclip,
)


class TestOSMGraphCLIPLocationEncoder:
    @pytest.mark.parametrize('batch_size', [1, 7])
    def test_untrained_dynamic_batch(self, batch_size: int) -> None:
        enc = OSMGraphCLIPLocationEncoder(
            legendre_polys=4, capacity=16, embed_dim=8, num_hidden_layers=2
        )
        assert enc.name == 'osmgraphclip'
        assert enc.legendre_polys == 4
        assert enc.embed_dim == 8

        coords = torch.linspace(-60.0, 60.0, batch_size * 2).reshape(batch_size, 2)
        original = coords.clone()
        out = enc(
            coords,
            year=torch.full((batch_size,), 2024.0),
            month=torch.full((batch_size,), 6.0),
        )
        assert torch.equal(coords, original)
        assert out.shape == (batch_size, 8)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

        encoded = enc.encode(coords[:, 0].numpy(), coords[:, 1].numpy())
        assert encoded.shape == (batch_size, 8)
        assert np.allclose(encoded, out.detach().numpy(), atol=1e-6)

    @pytest.mark.parametrize(
        'prefix', ['nnet.', 'location.nnet.', 'model.location.nnet.', 'model.nnet.', '']
    )
    def test_load_safetensors_key_schemas(self, tmp_path: Path, prefix: str) -> None:
        ref = OSMGraphCLIPLocationEncoder(
            legendre_polys=4, capacity=16, embed_dim=8, num_hidden_layers=2
        )
        raw = {
            f'{prefix}{k.removeprefix("nnet.")}': v.clone()
            for k, v in ref.state_dict().items()
        }
        ckpt = tmp_path / f'osmgraphclip_{prefix.replace(".", "_")}.safetensors'
        safetensors.torch.save_file(raw, ckpt)

        loaded = OSMGraphCLIPLocationEncoder(
            legendre_polys=4,
            capacity=16,
            embed_dim=8,
            num_hidden_layers=2,
            checkpoint=ckpt,
        )
        coords = torch.tensor([[13.40, 52.52], [-122.42, 37.77]])
        assert torch.allclose(ref(coords), loaded(coords), atol=1e-6)

    def test_load_lightning_ckpt_and_raw_pt(self, tmp_path: Path) -> None:
        ref = OSMGraphCLIPLocationEncoder(
            legendre_polys=3, capacity=12, embed_dim=6, num_hidden_layers=2
        )
        ckpt_path = tmp_path / 'osmgraphclip.ckpt'
        torch.save(
            {
                'hyper_parameters': {
                    'legendre_polys': 3,
                    'capacity': 12,
                    'embed_dim': 6,
                    'num_hidden_layers': 2,
                },
                'state_dict': {
                    f'model.location.{k}': v for k, v in ref.state_dict().items()
                },
            },
            ckpt_path,
        )
        loaded = OSMGraphCLIPLocationEncoder(checkpoint=ckpt_path)
        assert loaded.legendre_polys == 3
        assert loaded.embed_dim == 6
        coords = torch.tensor([[13.40, 52.52]])
        assert torch.allclose(ref(coords), loaded(coords), atol=1e-6)

        raw_pt = tmp_path / 'raw.pt'
        torch.save(ref.state_dict(), raw_pt)
        loaded_raw = OSMGraphCLIPLocationEncoder(
            legendre_polys=3,
            capacity=12,
            embed_dim=6,
            num_hidden_layers=2,
            checkpoint=raw_pt,
        )
        assert torch.allclose(ref(coords), loaded_raw(coords), atol=1e-6)

    @pytest.mark.parametrize(
        ('member_name', 'expected_l', 'expected_ms'),
        [
            ('MS_L40', 40, True),
            ('MS_L10', 10, True),
            ('A_L40', 40, False),
            ('A_L10', 10, False),
        ],
    )
    def test_weights_enum_and_factory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        member_name: str,
        expected_l: int,
        expected_ms: bool,
    ) -> None:
        w = OSMGraphCLIP_Weights.verify(member_name)
        assert w.repo_id == 'taylor-geospatial/neuralftw-zoo'
        assert w.meta['legendre_polys'] == expected_l
        assert w.meta['multiscale'] is expected_ms
        assert w.meta['license'] == 'Apache-2.0'
        assert len(w.meta['sha256']) == 64
        assert len(w.meta['onnx_sha256']) == 64

        ref = OSMGraphCLIPLocationEncoder(
            legendre_polys=expected_l, capacity=512, embed_dim=256, num_hidden_layers=2
        )
        ckpt = tmp_path / f'{member_name.lower()}.safetensors'
        safetensors.torch.save_file(ref.state_dict(), ckpt)
        monkeypatch.setattr(OSMGraphCLIP_Weights, 'download', lambda self: str(ckpt))

        model = osmgraphclip(weights=member_name)
        coords = torch.tensor([[13.40, 52.52], [-0.13, 51.51]])
        assert model.legendre_polys == expected_l
        assert model.embed_dim == 256
        assert torch.allclose(ref(coords), model(coords), atol=1e-6)

    def test_errors(self, tmp_path: Path) -> None:
        bad_ckpt = tmp_path / 'bad.safetensors'
        safetensors.torch.save_file({'other.weight': torch.zeros(2, 2)}, bad_ckpt)
        with pytest.raises(ValueError, match='at most one'):
            OSMGraphCLIPLocationEncoder(
                weights=OSMGraphCLIP_Weights.MS_L10, checkpoint=bad_ckpt
            )
        with pytest.raises(ValueError, match='Invalid weights'):
            osmgraphclip(weights='DOES_NOT_EXIST')
