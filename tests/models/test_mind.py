# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import safetensors.torch
import torch

from earthinr.models.mind import (
    MIND_Weights,
    MINDLocationEncoder,
    ReSIRENLocationEncoder,
    SIRENLayer,
    load_mind,
    mind,
)


class TestSIRENLayer:
    @pytest.mark.parametrize('act', ['siren', 'finer', 'hsiren'])
    @pytest.mark.parametrize('is_first', [True, False])
    def test_variants(self, act: str, is_first: bool) -> None:
        layer = SIRENLayer(4, 8, w0=2.0, is_first=is_first, act=act)
        if not is_first and act == 'hsiren':
            assert layer.act == 'siren'
        x = torch.randn(3, 4)
        out = layer(x)
        assert out.shape == (3, 8)
        assert torch.isfinite(out).all()

    def test_invalid_act(self) -> None:
        with pytest.raises(ValueError, match='Unsupported SIREN activation'):
            SIRENLayer(4, 8, act='relu')


class TestReSIRENLocationEncoder:
    def test_2d_and_3d_mean_pooling(self) -> None:
        net = ReSIRENLocationEncoder(embed_dim=16, out_dim=8, depth=2, use_year=False)
        net.eval()
        p1 = torch.tensor([[37.8, -122.4]])
        p2 = torch.tensor([[40.7, -74.0]])
        out_p1 = net(p1, return_features=True)
        out_p2 = net(p2, return_features=True)
        paired = torch.stack((p1, p2), dim=1)  # (1, 2, 2)
        out_paired = net(paired, return_features=True)
        assert out_paired.shape == (1, 16)
        assert torch.allclose(out_paired, 0.5 * (out_p1 + out_p2), atol=1e-6)
        assert net(p1, return_features=False).shape == (1, 8)

    def test_year_conditioning_and_identity_head(self) -> None:
        net = ReSIRENLocationEncoder(
            embed_dim=16, out_dim=None, depth=1, use_year=True, year_frequencies=4
        )
        net.eval()
        latlon = torch.tensor([[37.8, -122.4], [40.7, -74.0]])
        year = torch.tensor([2020.0, 2023.0])
        out = net(latlon, year=year)
        assert out.shape == (2, 16)
        with pytest.raises(ValueError, match='year is required'):
            net(latlon)


class TestLoadMind:
    def test_safetensors_and_pt_and_wrapped_meta(self, tmp_path: Path) -> None:
        net = ReSIRENLocationEncoder(
            embed_dim=16, out_dim=8, depth=2, use_year=True, year_frequencies=4
        )
        st_path = tmp_path / 'mind.safetensors'
        safetensors.torch.save_file(net.state_dict(), st_path)
        loaded_st = load_mind(st_path)
        latlon = torch.tensor([[35.0, -100.0]])
        year = torch.tensor([2021.0])
        assert torch.allclose(net(latlon, year), loaded_st(latlon, year), atol=1e-6)

        pt_path = tmp_path / 'mind.pt'
        torch.save(net.state_dict(), pt_path)
        loaded_pt = load_mind(pt_path)
        assert torch.allclose(net(latlon, year), loaded_pt(latlon, year), atol=1e-6)

        wrapped_path = tmp_path / 'wrapped.pt'
        torch.save(
            {
                'state_dict': net.state_dict(),
                'meta': {'args': {'encoder': 'resiren', 'w0_first': 30.0}},
            },
            wrapped_path,
        )
        loaded_wrapped = load_mind(wrapped_path)
        assert torch.allclose(
            net(latlon, year), loaded_wrapped(latlon, year), atol=1e-6
        )

    def test_head_weight_matching_embed_dim(self, tmp_path: Path) -> None:
        net = ReSIRENLocationEncoder(embed_dim=16, out_dim=8, depth=1)
        net.head = torch.nn.Linear(16, 16)
        path = tmp_path / 'square_head.safetensors'
        safetensors.torch.save_file(net.state_dict(), path)
        loaded = load_mind(path)
        assert isinstance(loaded.head, torch.nn.Linear)
        assert loaded.head.out_features == 16

    def test_invalid_checkpoints(self, tmp_path: Path) -> None:
        missing_first = tmp_path / 'missing.safetensors'
        safetensors.torch.save_file({'other': torch.zeros(2)}, missing_first)
        with pytest.raises(ValueError, match='MIND ReSIREN'):
            load_mind(missing_first)

        net = ReSIRENLocationEncoder(embed_dim=8, out_dim=4, depth=1)
        valid_path = tmp_path / 'valid.safetensors'
        safetensors.torch.save_file(net.state_dict(), valid_path)
        with pytest.raises(ValueError, match='Unsupported ReSIREN activation'):
            load_mind(valid_path, variant='gelu')

        bad_inputs = dict(net.state_dict())
        bad_inputs['first.linear.weight'] = torch.zeros(8, 5)
        bad_in_path = tmp_path / 'bad_in.safetensors'
        safetensors.torch.save_file(bad_inputs, bad_in_path)
        with pytest.raises(ValueError, match='Unexpected coordinate input dimension'):
            load_mind(bad_in_path)


class TestMINDLocationEncoder:
    @pytest.mark.parametrize('batch_size', [1, 7])
    def test_lonlat_swap_and_truncation(self, batch_size: int) -> None:
        raw = ReSIRENLocationEncoder(embed_dim=16, out_dim=8, depth=2, use_year=False)
        enc_pooled = MINDLocationEncoder(model=raw, feature='pooled', dim=6)
        enc_head = MINDLocationEncoder(model=raw, feature='head')
        assert enc_pooled.name == 'mind'
        assert enc_pooled.embed_dim == 6
        assert enc_head.embed_dim == 8

        lon = torch.linspace(-120.0, 120.0, batch_size)
        lat = torch.linspace(-60.0, 60.0, batch_size)
        lonlat = torch.stack((lon, lat), dim=-1)
        latlon = torch.stack((lat, lon), dim=-1)

        out_pooled = enc_pooled(lonlat)
        expected_pooled = raw(latlon, return_features=True)[:, :6]
        assert out_pooled.shape == (batch_size, 6)
        assert torch.allclose(out_pooled, expected_pooled, atol=1e-6)

        out_head = enc_head(lonlat)
        expected_head = raw(latlon, return_features=False)
        assert out_head.shape == (batch_size, 8)
        assert torch.allclose(out_head, expected_head, atol=1e-6)

        encoded = enc_pooled.encode(lon.numpy(), lat.numpy())
        assert encoded.shape == (batch_size, 6)
        assert np.allclose(encoded, out_pooled.detach().numpy(), atol=1e-6)

    def test_year_default_and_explicit(self) -> None:
        raw = ReSIRENLocationEncoder(
            embed_dim=16, out_dim=8, depth=1, use_year=True, year_frequencies=4
        )
        enc = MINDLocationEncoder(model=raw, feature='pooled', default_year=2022)
        lonlat = torch.tensor([[-122.4, 37.8], [139.7, 35.7]])
        out_default = enc(lonlat)
        out_explicit = enc(lonlat, year=torch.tensor([2022.0, 2022.0]))
        assert torch.allclose(out_default, out_explicit, atol=1e-6)

    def test_checkpoint_weights_and_default_init(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw = ReSIRENLocationEncoder(embed_dim=16, out_dim=8, depth=1)
        ckpt = tmp_path / 'mind.safetensors'
        safetensors.torch.save_file(raw.state_dict(), ckpt)

        enc_ckpt = MINDLocationEncoder(checkpoint=ckpt, feature='head')
        assert enc_ckpt.embed_dim == 8

        monkeypatch.setattr(MIND_Weights, 'download', lambda self: str(ckpt))
        enc_factory = mind(weights='MIND', feature='pooled', dim=8)
        assert enc_factory.embed_dim == 8
        lonlat = torch.tensor([[0.0, 0.0]])
        assert enc_factory(lonlat).shape == (1, 8)

        # Default untrained fallback (use tiny dims via monkeypatch to stay fast)
        monkeypatch.setattr(
            'earthinr.models.mind.ReSIRENLocationEncoder',
            lambda: ReSIRENLocationEncoder(embed_dim=8, out_dim=4, depth=1),
        )
        enc_default = MINDLocationEncoder()
        assert enc_default.embed_dim == 8

    def test_errors(self, tmp_path: Path) -> None:
        raw = ReSIRENLocationEncoder(embed_dim=16, out_dim=8, depth=1)
        ckpt = tmp_path / 'm.safetensors'
        safetensors.torch.save_file(raw.state_dict(), ckpt)
        with pytest.raises(ValueError, match='at most one'):
            MINDLocationEncoder(checkpoint=ckpt, model=raw)
        with pytest.raises(ValueError, match='feature must be'):
            MINDLocationEncoder(model=raw, feature=cast(Any, 'other'))
        with pytest.raises(ValueError, match='dim must be'):
            MINDLocationEncoder(model=raw, feature='head', dim=12)
        with pytest.raises(ValueError, match='dim must be'):
            MINDLocationEncoder(model=raw, feature='head', dim=cast(Any, True))
