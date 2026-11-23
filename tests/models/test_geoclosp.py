# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import numpy as np
import pytest
import safetensors.torch
import torch

from earthinr.models.geoclosp import (
    GeoCLOSP_Weights,
    GeoCLOSPLocationEncoder,
    _normalize_geoclosp_state_dict,
    geoclosp,
)


class TestGeoCLOSPLocationEncoder:
    @pytest.mark.parametrize('batch_size', [1, 7])
    def test_untrained_dynamic_batch(self, batch_size: int) -> None:
        enc = GeoCLOSPLocationEncoder(
            legendre_polys=4,
            capacity=16,
            location_dim=8,
            embed_dim=12,
            num_hidden_layers=2,
        )
        assert enc.name == 'geoclosp'
        assert enc.legendre_polys == 4
        assert enc.location_dim == 8
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
        norms = torch.linalg.norm(out, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

        encoded = enc.encode(coords[:, 0].numpy(), coords[:, 1].numpy())
        assert encoded.shape == (batch_size, 12)
        assert np.allclose(encoded, out.detach().numpy(), atol=1e-6)

    @pytest.mark.parametrize('schema', ['zoo', 'upstream'])
    def test_load_safetensors_key_schemas(self, tmp_path: Path, schema: str) -> None:
        ref = GeoCLOSPLocationEncoder(
            legendre_polys=4,
            capacity=16,
            location_dim=8,
            embed_dim=12,
            num_hidden_layers=2,
        )
        raw: dict[str, torch.Tensor] = {}
        for k, v in ref.state_dict().items():
            if schema == 'zoo':
                raw[k] = v.clone()
            elif k.startswith('nnet.'):
                raw[f'location_encoder.{k}'] = v.clone()
            else:
                raw[f'location_{k}'] = v.clone()
        raw['s1_encoder.unused.weight'] = torch.zeros(2, 2)

        ckpt = tmp_path / f'geoclosp_{schema}.safetensors'
        safetensors.torch.save_file(raw, ckpt)

        loaded = GeoCLOSPLocationEncoder(
            legendre_polys=4,
            capacity=16,
            location_dim=8,
            embed_dim=12,
            num_hidden_layers=2,
            checkpoint=ckpt,
        )
        coords = torch.tensor([[13.40, 52.52], [-122.42, 37.77]])
        assert torch.allclose(ref(coords), loaded(coords), atol=1e-6)

    def test_load_pt_and_wrapped_checkpoints(self, tmp_path: Path) -> None:
        ref = GeoCLOSPLocationEncoder(
            legendre_polys=3,
            capacity=12,
            location_dim=6,
            embed_dim=10,
            num_hidden_layers=2,
        )
        coords = torch.tensor([[13.40, 52.52]])

        raw_pt = tmp_path / 'raw.pt'
        torch.save(ref.state_dict(), raw_pt)
        loaded_raw = GeoCLOSPLocationEncoder(
            legendre_polys=3,
            capacity=12,
            location_dim=6,
            embed_dim=10,
            num_hidden_layers=2,
            checkpoint=raw_pt,
        )
        assert torch.allclose(ref(coords), loaded_raw(coords), atol=1e-6)

        wrapped_pt = tmp_path / 'wrapped.pt'
        torch.save({'state_dict': ref.state_dict()}, wrapped_pt)
        loaded_wrapped = GeoCLOSPLocationEncoder(
            legendre_polys=3,
            capacity=12,
            location_dim=6,
            embed_dim=10,
            num_hidden_layers=2,
            checkpoint=wrapped_pt,
        )
        assert torch.allclose(ref(coords), loaded_wrapped(coords), atol=1e-6)

    def test_weights_enum_and_factory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        w = GeoCLOSP_Weights.verify('RN50_L10')
        assert w.repo_id == 'taylor-geospatial/earthinr-zoo'
        assert w.filename == 'geoclosp-rn/model.safetensors'
        assert w.meta['legendre_polys'] == 10
        assert w.meta['location_dim'] == 256
        assert w.meta['embed_dim'] == 384
        assert w.meta['license'] == 'CreativeML-OpenRAIL-M'
        assert len(w.meta['sha256']) == 64
        assert len(w.meta['onnx_sha256']) == 64

        ref = GeoCLOSPLocationEncoder(
            legendre_polys=10,
            capacity=512,
            location_dim=256,
            embed_dim=384,
            num_hidden_layers=2,
        )
        ckpt = tmp_path / 'geoclosp_rn.safetensors'
        safetensors.torch.save_file(ref.state_dict(), ckpt)
        monkeypatch.setattr(GeoCLOSP_Weights, 'download', lambda self: str(ckpt))

        model = geoclosp(weights='RN50_L10')
        coords = torch.tensor([[13.40, 52.52], [-0.13, 51.51]])
        assert model.legendre_polys == 10
        assert model.embed_dim == 384
        assert torch.allclose(ref(coords), model(coords), atol=1e-6)

    def test_errors(self, tmp_path: Path) -> None:
        bad_ckpt = tmp_path / 'bad.safetensors'
        safetensors.torch.save_file({'other.weight': torch.zeros(2, 2)}, bad_ckpt)

        with pytest.raises(ValueError, match='at most one'):
            GeoCLOSPLocationEncoder(
                weights=GeoCLOSP_Weights.RN50_L10, checkpoint=bad_ckpt
            )
        with pytest.raises(KeyError, match='GeoCLOSP-RN'):
            _normalize_geoclosp_state_dict({'nnet.layers.0.weight': torch.zeros(2, 2)})
        with pytest.raises(KeyError, match='GeoCLOSP-RN'):
            _normalize_geoclosp_state_dict(
                {
                    'projection.weight': torch.zeros(2, 2),
                    'projection.bias': torch.zeros(2),
                }
            )
        with pytest.raises(ValueError, match='Invalid weights'):
            geoclosp(weights='DOES_NOT_EXIST')
