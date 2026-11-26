# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import safetensors.torch
import scipy.special
import torch
from torch import nn

from neuralftw.models.satclip import (
    SatCLIP_Weights,
    SatCLIPLocationEncoder,
    SirenNet,
    StableAnalyticSphericalHarmonics,
    satclip,
)


def _scipy_satclip_sh(
    lon_deg: float, lat_deg: float, legendre_polys: int
) -> np.ndarray:
    """Reference SatCLIP analytic spherical harmonics via scipy.special.lpmv."""
    phi = math.radians(lon_deg + 180.0)
    theta = math.radians(lat_deg + 90.0)
    x = math.cos(theta)
    out: list[float] = []
    for ell in range(legendre_polys):
        for m in range(-ell, ell + 1):
            abs_m = abs(m)
            norm = math.sqrt(
                ((2 * ell + 1) / (4.0 * math.pi))
                * (math.factorial(ell - abs_m) / math.factorial(ell + abs_m))
            )
            # Remove scipy's Condon-Shortley (-1)^m phase to match SatCLIP analytic basis
            plm = ((-1) ** abs_m) * float(scipy.special.lpmv(abs_m, ell, x))
            if m < 0:
                val = math.sqrt(2.0) * norm * plm * math.sin(abs_m * phi)
            elif m == 0:
                val = math.pi * norm * plm
            else:
                val = math.sqrt(2.0) * norm * plm * math.cos(abs_m * phi)
            out.append(val)
    return np.array(out, dtype=np.float64)


class TestStableAnalyticSphericalHarmonics:
    @pytest.mark.parametrize('legendre_polys', [1, 2, 5, 10])
    def test_matches_scipy_including_m0_pi_factor(self, legendre_polys: int) -> None:
        sh = StableAnalyticSphericalHarmonics(
            legendre_polys=legendre_polys, compute_dtype=torch.float64
        )
        pts = [(-122.4, 37.8), (0.0, 0.0), (139.7, -35.7)]
        coords = torch.tensor(pts, dtype=torch.float64)
        out = sh(coords)
        assert out.shape == (3, legendre_polys * legendre_polys)
        assert out.dtype == torch.float64

        for idx, (lon, lat) in enumerate(pts):
            expected = _scipy_satclip_sh(lon, lat, legendre_polys)
            assert np.allclose(out[idx].numpy(), expected, atol=1e-12, rtol=1e-12)

    def test_high_degree_poles_and_output_dtype(self) -> None:
        sh = StableAnalyticSphericalHarmonics(
            legendre_polys=40, compute_dtype=torch.float64, output_dtype=torch.float32
        )
        coords = torch.tensor([[0.0, 90.0], [180.0, -90.0], [45.0, 45.0]])
        out = sh(coords)
        assert out.shape == (3, 1600)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

    def test_validation(self) -> None:
        with pytest.raises(TypeError, match='legendre_polys'):
            StableAnalyticSphericalHarmonics(legendre_polys=cast(Any, True))
        with pytest.raises(ValueError, match='>= 1'):
            StableAnalyticSphericalHarmonics(legendre_polys=0)


class TestSirenNet:
    def test_bias_and_dropout_and_final_activation(self) -> None:
        net = SirenNet(
            dim_in=16,
            dim_hidden=32,
            dim_out=8,
            num_layers=2,
            use_bias=False,
            final_activation=nn.Tanh(),
        )
        net.eval()
        x = torch.randn(4, 16)
        out = net(x)
        assert out.shape == (4, 8)
        assert (out.abs() <= 1.0).all()


class TestSatCLIPLocationEncoder:
    @pytest.mark.parametrize('batch_size', [1, 7])
    def test_untrained_dynamic_batch(self, batch_size: int) -> None:
        enc = SatCLIPLocationEncoder(
            legendre_polys=4, capacity=16, embed_dim=8, num_hidden_layers=2
        )
        assert enc.name == 'satclip'
        assert enc.embed_dim == 8
        coords = torch.linspace(-60.0, 60.0, batch_size * 2).reshape(batch_size, 2)
        out = enc(coords)
        assert out.shape == (batch_size, 8)
        assert out.dtype == torch.float32

        encoded = enc.encode(coords[:, 0].numpy(), coords[:, 1].numpy())
        assert encoded.shape == (batch_size, 8)
        assert np.allclose(encoded, out.detach().numpy(), atol=1e-6)

    @pytest.mark.parametrize('prefix', ['nnet.', 'location.nnet.', ''])
    def test_load_safetensors_key_schemas(self, tmp_path: Path, prefix: str) -> None:
        ref = SatCLIPLocationEncoder(
            legendre_polys=4, capacity=16, embed_dim=8, num_hidden_layers=2
        )
        raw = {
            f'{prefix}{k.removeprefix("nnet.")}': v.clone()
            for k, v in ref.state_dict().items()
        }
        ckpt = tmp_path / f'satclip_{prefix.replace(".", "_")}.safetensors'
        safetensors.torch.save_file(raw, ckpt)

        loaded = SatCLIPLocationEncoder(
            legendre_polys=4,
            capacity=16,
            embed_dim=8,
            num_hidden_layers=2,
            checkpoint=ckpt,
        )
        coords = torch.tensor([[-100.0, 40.0], [10.0, -20.0]])
        assert torch.allclose(ref(coords), loaded(coords), atol=1e-6)

    def test_load_lightning_ckpt_and_raw_pt(self, tmp_path: Path) -> None:
        ref = SatCLIPLocationEncoder(
            legendre_polys=3, capacity=12, embed_dim=6, num_hidden_layers=2
        )
        ckpt_path = tmp_path / 'satclip.ckpt'
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
        loaded = SatCLIPLocationEncoder(checkpoint=ckpt_path)
        assert loaded.legendre_polys == 3
        assert loaded.embed_dim == 6
        coords = torch.tensor([[12.0, 34.0]])
        assert torch.allclose(ref(coords), loaded(coords), atol=1e-6)

        raw_pt = tmp_path / 'raw.pt'
        torch.save(ref.state_dict(), raw_pt)
        loaded_raw = SatCLIPLocationEncoder(
            legendre_polys=3,
            capacity=12,
            embed_dim=6,
            num_hidden_layers=2,
            checkpoint=raw_pt,
        )
        assert torch.allclose(ref(coords), loaded_raw(coords), atol=1e-6)

    def test_load_weights_via_factory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ref = SatCLIPLocationEncoder(
            legendre_polys=10, capacity=512, embed_dim=256, num_hidden_layers=2
        )
        ckpt = tmp_path / 'vit16_l10.safetensors'
        safetensors.torch.save_file(ref.state_dict(), ckpt)
        monkeypatch.setattr(SatCLIP_Weights, 'download', lambda self: str(ckpt))

        model = satclip(weights='VIT16_L10')
        coords = torch.tensor([[-80.0, 35.0], [20.0, 50.0]])
        assert model.embed_dim == 256
        assert torch.allclose(ref(coords), model(coords), atol=1e-6)

    def test_errors(self, tmp_path: Path) -> None:
        bad_ckpt = tmp_path / 'bad.safetensors'
        safetensors.torch.save_file({'other.weight': torch.zeros(2, 2)}, bad_ckpt)
        with pytest.raises(KeyError, match='nnet'):
            SatCLIPLocationEncoder(
                legendre_polys=2, capacity=4, embed_dim=4, checkpoint=bad_ckpt
            )
        with pytest.raises(ValueError, match='at most one'):
            SatCLIPLocationEncoder(
                weights=SatCLIP_Weights.VIT16_L10, checkpoint=bad_ckpt
            )
