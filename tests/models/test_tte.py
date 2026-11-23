# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import numpy as np
import pytest
import safetensors.torch
import torch

from earthinr.models.tte import (
    SemanticRegisterAttention,
    TTE_Weights,
    TTELocationEncoder,
    TTENet,
    VoronoiLocationEncoder,
    tte,
)


class TestSemanticRegisterAttention:
    def test_standard_and_cosine_modes(self) -> None:
        attn_std = SemanticRegisterAttention(
            site_embed_dim=8, output_dim=12, n_registers=4, use_cosine_attention=False
        )
        out_std = attn_std(torch.randn(3, 8))
        assert out_std.shape == (3, 12)

        attn_cos = SemanticRegisterAttention(
            site_embed_dim=12, output_dim=12, n_registers=4, use_cosine_attention=True
        )
        assert attn_cos.site_to_out is None
        out_cos = attn_cos(torch.randn(5, 12))
        assert out_cos.shape == (5, 12)


class TestTTENet:
    def test_forward_normalized_and_unnormalized(self) -> None:
        net_norm = TTENet(
            n_sites=8,
            site_embed_dim=6,
            output_dim=8,
            hidden_dim=8,
            n_reslayers=1,
            n_registers=4,
            normalize=True,
        ).eval()
        coords = torch.tensor([[37.8, -122.4], [0.0, 0.0]], dtype=torch.float32)
        out = net_norm(coords)
        assert out.shape == (2, 8)
        assert torch.allclose(out.norm(dim=-1), torch.ones(2), atol=1e-5)

        net_raw = TTENet(
            n_sites=8,
            site_embed_dim=6,
            output_dim=8,
            hidden_dim=8,
            n_reslayers=1,
            n_registers=4,
            normalize=False,
        ).eval()
        out_raw = net_raw(coords)
        assert out_raw.shape == (2, 8)
        assert isinstance(net_raw.location, VoronoiLocationEncoder)


class TestTTELocationEncoder:
    @pytest.fixture
    def tiny_tte_ckpt(self, tmp_path: Path) -> Path:
        torch.manual_seed(0)
        net = TTENet(
            n_sites=8,
            site_embed_dim=6,
            output_dim=10,
            hidden_dim=10,
            n_reslayers=1,
            n_registers=4,
        )
        state = dict(net.state_dict())
        # Include upstream training-only image register keys to verify stripping.
        state['location.semantic_registers.img_to_logits.weight'] = torch.zeros(4, 10)
        state['location.semantic_registers.img_to_logits.bias'] = torch.zeros(4)
        state['location.semantic_registers._img_temperature'] = torch.tensor(0.05)
        path = tmp_path / 'tte.safetensors'
        safetensors.torch.save_file(state, path)
        return path

    def test_dynamic_batch_sizes_and_coord_swap(self, tiny_tte_ckpt: Path) -> None:
        enc = tte(
            weights=None,
            checkpoint_path=tiny_tte_ckpt,
            n_sites=8,
            site_embed_dim=6,
            output_dim=10,
            hidden_dim=10,
            n_reslayers=1,
            n_registers=4,
        )
        assert enc.name == 'tte'
        assert enc.embed_dim == 10

        for batch_size in (1, 7):
            lonlat = torch.randn(batch_size, 2) * 30.0
            out = enc(lonlat)
            assert out.shape == (batch_size, 10)
            assert out.dtype == torch.float32
            assert torch.allclose(out.norm(dim=-1), torch.ones(batch_size), atol=1e-5)
            # Verify internal (lon, lat) -> (lat, lon) swap matches raw TTENet.
            expected = enc.model(lonlat.flip(-1))
            assert torch.allclose(out, expected, atol=1e-6)

        lon = np.array([-122.4, 0.0, 139.7], dtype=np.float64)
        lat = np.array([37.8, 0.0, 35.7], dtype=np.float64)
        emb = enc.encode(lon, lat)
        assert emb.shape == (3, 10)
        assert emb.dtype == np.float32

    def test_weights_download_path(
        self, monkeypatch: pytest.MonkeyPatch, tiny_tte_ckpt: Path
    ) -> None:
        monkeypatch.setattr(TTE_Weights, 'download', lambda self: str(tiny_tte_ckpt))
        enc = TTELocationEncoder(
            weights='DEFAULT',
            n_sites=8,
            site_embed_dim=6,
            output_dim=10,
            hidden_dim=10,
            n_reslayers=1,
            n_registers=4,
        )
        assert enc.embed_dim == 10
        assert enc(torch.zeros(2, 2)).shape == (2, 10)

    def test_random_init_without_weights(self) -> None:
        enc = TTELocationEncoder(
            weights=None,
            n_sites=4,
            site_embed_dim=4,
            output_dim=6,
            hidden_dim=6,
            n_reslayers=1,
            n_registers=2,
        )
        assert enc(torch.zeros(3, 2)).shape == (3, 6)
