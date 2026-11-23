# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import safetensors.torch
import torch

from earthinr.models.hydroloc import (
    HydroLoc_Weights,
    HydroLocLocationEncoder,
    _normalize_hydroloc_state_dict,
    hydroloc,
)


class TestHydroLocLocationEncoder:
    @pytest.mark.parametrize('batch_size', [1, 7])
    def test_untrained_dynamic_batch_and_matryoshka(self, batch_size: int) -> None:
        enc_full = HydroLocLocationEncoder(embed_dim=16, depth=2)
        assert enc_full.name == 'hydroloc'
        assert enc_full.trunk_dim == 16
        assert enc_full.depth == 2
        assert enc_full.embed_dim == 16

        coords = torch.linspace(-60.0, 60.0, batch_size * 2).reshape(batch_size, 2)
        original = coords.clone()
        out_full = enc_full(
            coords,
            year=torch.full((batch_size,), 2024.0),
            month=torch.full((batch_size,), 6.0),
        )
        assert torch.equal(coords, original)
        assert out_full.shape == (batch_size, 16)
        assert out_full.dtype == torch.float32
        assert torch.isfinite(out_full).all()

        encoded = enc_full.encode(coords[:, 0].numpy(), coords[:, 1].numpy())
        assert encoded.shape == (batch_size, 16)
        assert np.allclose(encoded, out_full.detach().numpy(), atol=1e-6)

        enc_prefix = HydroLocLocationEncoder(embed_dim=16, depth=2, dim=6)
        enc_prefix.load_state_dict(enc_full.state_dict(), strict=True)
        out_prefix = enc_prefix(coords)
        assert enc_prefix.embed_dim == 6
        assert out_prefix.shape == (batch_size, 6)
        assert torch.allclose(out_prefix, out_full[:, :6], atol=1e-6)

    def test_load_safetensors_and_pt_checkpoints(self, tmp_path: Path) -> None:
        ref = HydroLocLocationEncoder(embed_dim=16, depth=2)
        coords = torch.tensor([[-122.42, 37.77], [45.0, -8.7]])

        st_path = tmp_path / 'hydroloc.safetensors'
        raw = dict(ref.state_dict())
        raw['extra.unused'] = torch.zeros(2)
        safetensors.torch.save_file(raw, st_path)
        loaded_st = HydroLocLocationEncoder(checkpoint=st_path, dim=8)
        assert loaded_st.trunk_dim == 16
        assert loaded_st.depth == 2
        assert loaded_st.embed_dim == 8
        assert torch.allclose(ref(coords)[:, :8], loaded_st(coords), atol=1e-6)

        raw_pt = tmp_path / 'hydroloc_raw.pt'
        torch.save(ref.state_dict(), raw_pt)
        loaded_raw = HydroLocLocationEncoder(checkpoint=raw_pt)
        assert torch.allclose(ref(coords), loaded_raw(coords), atol=1e-6)

        wrapped_pt = tmp_path / 'hydroloc_wrapped.pt'
        torch.save({'state_dict': ref.state_dict()}, wrapped_pt)
        loaded_wrapped = HydroLocLocationEncoder(checkpoint=wrapped_pt)
        assert torch.allclose(ref(coords), loaded_wrapped(coords), atol=1e-6)

    def test_weights_enum_and_factory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        w = HydroLoc_Weights.verify('DEFAULT')
        assert w.repo_id == 'taylor-geospatial/earthinr-zoo'
        assert w.filename == 'hydroloc/model.safetensors'
        assert w.meta['embed_dim'] == 512
        assert w.meta['depth'] == 4
        assert w.meta['matryoshka_dims'] == (64, 128, 256, 512)
        assert w.meta['license'] == 'Apache-2.0'
        assert len(w.meta['sha256']) == 64
        assert len(w.meta['onnx_sha256']) == 64

        ref = HydroLocLocationEncoder(embed_dim=512, depth=4)
        ckpt = tmp_path / 'hydroloc.safetensors'
        safetensors.torch.save_file(ref.state_dict(), ckpt)
        monkeypatch.setattr(HydroLoc_Weights, 'download', lambda self: str(ckpt))

        model = hydroloc(weights='DEFAULT', dim=64)
        coords = torch.tensor([[-122.42, 37.77], [45.0, -8.7]])
        assert model.embed_dim == 64
        assert model.trunk_dim == 512
        assert torch.allclose(ref(coords)[:, :64], model(coords), atol=1e-6)

    def test_errors(self, tmp_path: Path) -> None:
        bad_ckpt = tmp_path / 'bad.safetensors'
        safetensors.torch.save_file({'other.weight': torch.zeros(2, 2)}, bad_ckpt)

        with pytest.raises(ValueError, match='at most one'):
            HydroLocLocationEncoder(
                weights=HydroLoc_Weights.DEFAULT, checkpoint=bad_ckpt
            )
        with pytest.raises(KeyError, match='HydroLoc'):
            _normalize_hydroloc_state_dict({'other.weight': torch.zeros(2, 2)})
        with pytest.raises(ValueError, match='Invalid weights'):
            hydroloc(weights='DOES_NOT_EXIST')
        with pytest.raises(ValueError, match='dim must be'):
            HydroLocLocationEncoder(embed_dim=16, depth=1, dim=0)
        with pytest.raises(ValueError, match='dim must be'):
            HydroLocLocationEncoder(embed_dim=16, depth=1, dim=32)
        with pytest.raises(ValueError, match='dim must be'):
            HydroLocLocationEncoder(embed_dim=16, depth=1, dim=cast(Any, True))
