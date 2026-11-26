# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import safetensors.torch
import torch

from neuralftw.models.terranova import (
    MultiResolutionHashEncoding,
    ResidualMLPDecoder,
    TerraNova_Weights,
    TerraNovaLocationEncoder,
    TerraNovaLocationNet,
    TerraNovaSTFusion,
    TerraNovaTimeNet,
    WireLayer,
    WireUNet,
    WrapNormalized,
    _make_activation,
    terranova,
)

_TINY_KW: dict[str, Any] = {
    'location_dim': 8,
    'wire_widths': (4, 8, 16, 8, 4),
    'hash_n_levels': 2,
    'hash_n_features': 2,
    'hash_log2_size': 4,
    'hash_base_res': 4,
    'hash_finest_res': 8,
    'hash_hidden': (8, 8),
    'time_dim': 4,
    'time_width': 8,
    'time_depth': 1,
    'time_fourier': 4,
    'st_width': 8,
    'st_heads': 2,
    'st_depth': 1,
}


class TestTerraNovaBlocks:
    def test_wrap_normalized_and_antimeridian_continuity(self) -> None:
        wrap = WrapNormalized()
        c_pos = torch.tensor([[180.0, 20.0]])
        c_neg = torch.tensor([[-180.0, 20.0]])
        out_pos = wrap(c_pos)
        out_neg = wrap(c_neg)
        assert out_pos.shape == (1, 5)
        assert torch.allclose(out_pos, out_neg, atol=1e-6)

    def test_wire_layer_and_unet_edge_cases(self) -> None:
        # First layer with dim_in == dim_out exercises Identity residual path.
        layer = WireLayer(4, 4, is_first=True, is_residual=True)
        assert isinstance(layer.residual, torch.nn.Identity)
        assert layer(torch.randn(2, 4)).shape == (2, 4)

        with pytest.raises(ValueError, match='dim_hidden'):
            WireUNet(5, [], 8)

        # Asymmetric UNet where decoder skip dim does not match encoder skip dim.
        unet_asym = WireUNet(5, (4, 8, 16, 10, 6), 4)
        assert unet_asym(torch.randn(2, 5)).shape == (2, 4)

        # Bottleneck-only UNet (no decoder layers).
        unet_enc_only = WireUNet(5, (6, 12), 4)
        assert unet_enc_only(torch.randn(2, 5)).shape == (2, 4)

    def test_hash_encoding_single_and_multi_level(self) -> None:
        enc1 = MultiResolutionHashEncoding(
            n_levels=1,
            n_features_per_level=2,
            log2_hashmap_size=4,
            base_resolution=4,
            finest_resolution=4,
        )
        assert enc1.growth_factor == 1.0
        assert enc1(torch.tensor([[0.0, 0.0], [180.0, -90.0]])).shape == (2, 2)

        enc2 = MultiResolutionHashEncoding(
            n_levels=2,
            n_features_per_level=2,
            log2_hashmap_size=4,
            base_resolution=4,
            finest_resolution=8,
        )
        # Verify periodic longitude wrapping across +180 / -180.
        a = enc2(torch.tensor([[180.0, 10.0]]))
        b = enc2(torch.tensor([[-180.0, 10.0]]))
        assert torch.allclose(a, b, atol=1e-6)

    def test_residual_decoder_and_invalid_activation(self) -> None:
        dec = ResidualMLPDecoder(4, (8, 8), 6)
        assert dec(torch.randn(3, 4)).shape == (3, 6)
        with pytest.raises(ValueError, match='Unsupported TerraNova activation'):
            _make_activation('relu')

    def test_submodules_radius_norm(self) -> None:
        loc = TerraNovaLocationNet(
            location_dim=8,
            wire_widths=(8, 16, 8),
            hash_n_levels=2,
            hash_n_features=2,
            hash_log2_size=4,
            hash_base_res=4,
            hash_finest_res=8,
            hash_hidden=(8, 8),
        ).eval()
        s = loc(torch.randn(4, 2) * 40.0)
        assert torch.allclose(s.norm(dim=-1), torch.full((4,), math.sqrt(8)), atol=1e-5)

        tnet = TerraNovaTimeNet(time_dim=4, time_width=8, time_depth=1, time_fourier=4)
        t = tnet(torch.tensor([2000.0, 2020.0, 2025.0, 1950.0]))
        assert torch.allclose(t.norm(dim=-1), torch.full((4,), math.sqrt(4)), atol=1e-5)

        fusion = TerraNovaSTFusion(
            location_dim=8, time_dim=4, st_width=8, st_heads=2, st_depth=1
        ).eval()
        f = fusion(s, t)
        assert torch.allclose(f.norm(dim=-1), torch.full((4,), math.sqrt(8)), atol=1e-5)


class TestTerraNovaLocationEncoder:
    @pytest.fixture
    def tiny_st_ckpt(self, tmp_path: Path) -> Path:
        torch.manual_seed(0)
        enc = TerraNovaLocationEncoder(weights=None, space='spatiotemporal', **_TINY_KW)
        state = dict(enc.state_dict())
        # Add extra upstream keys (e.g. decoder/task_encoder) to verify filtering.
        state['task_encoder.embedding.weight'] = torch.zeros(2, 4)
        path = tmp_path / 'terranova_st.safetensors'
        safetensors.torch.save_file(state, path)
        return path

    def test_spatial_and_st_dynamic_batches(self, tiny_st_ckpt: Path) -> None:
        sp = terranova(
            weights=None, space='spatial', checkpoint_path=tiny_st_ckpt, **_TINY_KW
        )
        assert sp.name == 'terranova'
        assert sp.embed_dim == 8
        assert sp.time_encoder is None

        st = terranova(
            weights=None,
            space='spatiotemporal',
            checkpoint_path=tiny_st_ckpt,
            **_TINY_KW,
        )
        assert st.time_encoder is not None
        assert st.st_fusion is not None

        for batch_size in (1, 7):
            coords = torch.randn(batch_size, 2) * 50.0
            out_sp = sp(coords)
            assert out_sp.shape == (batch_size, 8)
            assert torch.allclose(
                out_sp.norm(dim=-1), torch.full((batch_size,), math.sqrt(8)), atol=1e-5
            )

            out_st_default = st(coords)
            out_st_explicit = st(coords, year=torch.full((batch_size,), 2020.0))
            assert out_st_default.shape == (batch_size, 8)
            assert torch.allclose(out_st_default, out_st_explicit, atol=1e-5)

        lon = np.array([-190.0, 0.0, 190.0], dtype=np.float64)
        lat = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        emb = st.encode(lon, lat, year=np.array([2015.0, 2020.0, 2025.0]))
        assert emb.shape == (3, 8)
        assert emb.dtype == np.float32

    def test_weights_download_and_space_inference(
        self, monkeypatch: pytest.MonkeyPatch, tiny_st_ckpt: Path
    ) -> None:
        monkeypatch.setattr(
            TerraNova_Weights, 'download', lambda self: str(tiny_st_ckpt)
        )
        enc_st = TerraNovaLocationEncoder(weights='SPATIOTEMPORAL', **_TINY_KW)
        assert enc_st.space == 'spatiotemporal'
        assert enc_st(torch.zeros(2, 2)).shape == (2, 8)

        enc_sp = TerraNovaLocationEncoder(weights=TerraNova_Weights.SPATIAL, **_TINY_KW)
        assert enc_sp.space == 'spatial'
        assert enc_sp(torch.zeros(2, 2)).shape == (2, 8)

    def test_invalid_space_and_weights_mismatch(self) -> None:
        with pytest.raises(ValueError, match="space must be 'spatial'"):
            TerraNovaLocationEncoder(
                weights=None, space=cast(Any, 'temporal'), **_TINY_KW
            )
        with pytest.raises(ValueError, match='are for'):
            TerraNovaLocationEncoder(
                weights=TerraNova_Weights.SPATIAL, space='spatiotemporal', **_TINY_KW
            )
        # Random init with no weights and no space defaults to spatial.
        enc_default = TerraNovaLocationEncoder(weights=None, **_TINY_KW)
        assert enc_default.space == 'spatial'
