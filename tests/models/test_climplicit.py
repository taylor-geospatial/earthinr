# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

import math
from pathlib import Path

import numpy as np
import pytest
import safetensors.torch
import torch
from torch import nn

from neuralftw.models.climplicit import (
    CHELSA_MEAN,
    CHELSA_STD,
    Climplicit_Weights,
    ClimplicitLocationEncoder,
    ClimplicitNet,
    climplicit,
)


class TestClimplicitMath:
    def test_hsiren_and_resiren_preactivation_math(self) -> None:
        net = ClimplicitNet(
            dim_in=4,
            dim_hidden=8,
            dim_out=6,
            num_layers=3,
            w0=1.0,
            w0_initial=30.0,
            h_siren=True,
            residual_connections=True,
            return_chelsa=False,
        )
        net.eval()

        coords = torch.tensor([[8.55, 47.40], [-122.4, 37.8]], dtype=torch.float32)
        month = torch.tensor([4.0, 10.0], dtype=torch.float32)
        out = net(coords, month=month)

        # Manual layer-by-layer reference matching ecovision-uzh/climplicit
        loc = torch.stack((coords[:, 0] / 180.0, coords[:, 1] / 90.0), dim=-1)
        angle = month.unsqueeze(-1) * (2.0 * math.pi / 12.0)
        x = torch.cat((loc, torch.sin(angle), torch.cos(angle)), dim=-1)

        sd = net.location_encoder.state_dict()
        # Layer 0: linear -> save gaussian BEFORE sinh(2 * out) -> sin(30 * sinh(2 * out))
        g0 = nn.functional.linear(x, sd['layers.0.weight'], sd['layers.0.bias'])
        h0 = torch.sin(30.0 * torch.sinh(2.0 * g0))
        # Layer 1: (linear(h0) + g0) / 2 -> sin(1 * out)
        g1 = (
            nn.functional.linear(h0, sd['layers.1.weight'], sd['layers.1.bias']) + g0
        ) / 2.0
        h1 = torch.sin(1.0 * g1)
        # Layer 2: (linear(h1) + g1) / 2 -> sin(1 * out)
        g2 = (
            nn.functional.linear(h1, sd['layers.2.weight'], sd['layers.2.bias']) + g1
        ) / 2.0
        h2 = torch.sin(1.0 * g2)
        # Last layer: identity, no residual
        expected = nn.functional.linear(
            h2, sd['last_layer.weight'], sd['last_layer.bias']
        )
        assert torch.allclose(out, expected, atol=1e-6)

    def test_default_months_concat_and_chelsa_unnormalization(self) -> None:
        net = ClimplicitNet(dim_hidden=8, dim_out=6, num_layers=2, return_chelsa=False)
        net.eval()
        # Non-persistent buffers must NOT appear in state_dict
        assert 'chelsa_mean' not in net.state_dict()
        assert 'chelsa_std' not in net.state_dict()

        coords = torch.tensor([[10.0, 20.0], [-50.0, -30.0]])
        concat_out = net(coords)
        assert concat_out.shape == (2, 24)

        slices = [net(coords, month=torch.full((2,), float(m))) for m in (3, 6, 9, 12)]
        assert torch.allclose(concat_out, torch.cat(slices, dim=-1), atol=1e-6)

        # Verify CHELSA regression + unnormalization
        net_chelsa = ClimplicitNet(
            dim_hidden=8, dim_out=6, num_layers=2, return_chelsa=True
        )
        net_chelsa.load_state_dict(net.state_dict(), strict=True)
        net_chelsa.eval()

        m_tensor = torch.tensor([6.0, 6.0])
        emb = net(coords, month=m_tensor)
        chelsa_single = net_chelsa(coords, month=m_tensor)
        assert chelsa_single.shape == (2, 11)
        expected_chelsa = net_chelsa.chelsa_regressor(emb) * torch.tensor(
            CHELSA_STD
        ) + torch.tensor(CHELSA_MEAN)
        assert torch.allclose(chelsa_single, expected_chelsa, atol=1e-4)
        assert net_chelsa(coords).shape == (2, 44)


class TestClimplicitLocationEncoder:
    @pytest.mark.parametrize('batch_size', [1, 7])
    def test_dynamic_batch_and_encode_with_month(self, batch_size: int) -> None:
        enc = ClimplicitLocationEncoder(dim_hidden=8, dim_out=6, num_layers=2)
        assert enc.name == 'climplicit'
        assert enc.embed_dim == 24

        lon = np.linspace(-120.0, 120.0, batch_size)
        lat = np.linspace(-60.0, 60.0, batch_size)
        coords = torch.tensor(np.stack((lon, lat), axis=-1), dtype=torch.float32)

        out_all = enc(coords)
        assert out_all.shape == (batch_size, 24)
        assert out_all.dtype == torch.float32

        months = np.full(batch_size, 3.0)
        out_m = enc.encode(lon, lat, month=months)
        assert out_m.shape == (batch_size, 6)
        assert out_m.dtype == np.float32
        assert np.allclose(out_m, out_all[:, :6].detach().numpy(), atol=1e-6)

    def test_checkpoint_and_weights_factory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ref = ClimplicitNet(dim_hidden=8, dim_out=6, num_layers=2)
        ckpt = tmp_path / 'climplicit.safetensors'
        safetensors.torch.save_file(ref.state_dict(), ckpt)

        enc_ckpt = ClimplicitLocationEncoder(
            checkpoint=ckpt, dim_hidden=8, dim_out=6, num_layers=2, return_chelsa=True
        )
        assert enc_ckpt.embed_dim == 44

        monkeypatch.setattr(
            Climplicit_Weights.CHELSA.value,
            'meta',
            {'dim_hidden': 8, 'dim_out': 6, 'num_layers': 2, 'embed_dim': 24},
        )
        monkeypatch.setattr(Climplicit_Weights, 'download', lambda self: str(ckpt))
        enc_factory = climplicit(weights='CHELSA')
        coords = torch.tensor([[8.55, 47.40]])
        assert torch.allclose(enc_factory(coords), ref(coords), atol=1e-6)

    def test_errors(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='default_months'):
            ClimplicitNet(default_months=())
        ckpt = tmp_path / 'c.safetensors'
        ckpt.write_bytes(b'')
        with pytest.raises(ValueError, match='at most one'):
            ClimplicitLocationEncoder(
                weights=Climplicit_Weights.CHELSA, checkpoint=ckpt
            )
