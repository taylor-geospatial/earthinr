# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

import math
from typing import Any, cast

import numpy as np
import pytest
import torch

from earthinr.models.encoders import (
    AnalyticSphericalHarmonicLocationEncoder,
    FrequencyRFFLocationEncoder,
    NeRFLocationEncoder,
    SinCosLocationEncoder,
    SphericalHarmonicLocationEncoder,
    XYZLocationEncoder,
)


class TestSinCosLocationEncoder:
    @pytest.mark.parametrize('batch_size', [1, 7])
    def test_forward_and_values(self, batch_size: int) -> None:
        enc = SinCosLocationEncoder()
        assert enc.name == 'sincos'
        assert enc.embed_dim == 4
        coords = torch.linspace(-45.0, 45.0, batch_size * 2).reshape(batch_size, 2)
        out = enc(coords)
        assert out.shape == (batch_size, 4)
        assert out.dtype == torch.float32

        lon_r = torch.deg2rad(coords[:, 0])
        lat_r = torch.deg2rad(coords[:, 1])
        expected = torch.stack(
            (torch.sin(lat_r), torch.cos(lat_r), torch.sin(lon_r), torch.cos(lon_r)),
            dim=-1,
        )
        assert torch.allclose(out, expected, atol=1e-6)

    def test_encode(self) -> None:
        enc = SinCosLocationEncoder(batch_size=2)
        lon = np.array([0.0, 90.0, -90.0])
        lat = np.array([0.0, 45.0, -30.0])
        out = enc.encode(lon, lat)
        assert out.shape == (3, 4)
        assert out.dtype == np.float32


class TestXYZLocationEncoder:
    @pytest.mark.parametrize('batch_size', [1, 7])
    def test_forward_unit_norm(self, batch_size: int) -> None:
        enc = XYZLocationEncoder()
        assert enc.name == 'xyz'
        assert enc.embed_dim == 3
        lon = torch.linspace(-180.0, 180.0, batch_size)
        lat = torch.linspace(-80.0, 80.0, batch_size)
        out = enc(torch.stack((lon, lat), dim=-1))
        assert out.shape == (batch_size, 3)
        assert out.dtype == torch.float32
        assert torch.allclose(
            torch.linalg.norm(out, dim=-1), torch.ones(batch_size), atol=1e-6
        )


class TestNeRFLocationEncoder:
    @pytest.mark.parametrize('include_xyz', [False, True])
    @pytest.mark.parametrize('batch_size', [1, 7])
    def test_forward_exact(self, include_xyz: bool, batch_size: int) -> None:
        enc = NeRFLocationEncoder(num_frequencies=4, include_xyz=include_xyz)
        assert enc.name == 'nerf'
        expected_dim = 6 * 4 + (3 if include_xyz else 0)
        assert enc.embed_dim == expected_dim

        coords = torch.stack(
            (
                torch.linspace(-120.0, 120.0, batch_size),
                torch.linspace(-60.0, 60.0, batch_size),
            ),
            dim=-1,
        )
        out = enc(coords)
        assert out.shape == (batch_size, expected_dim)
        assert out.dtype == torch.float32

        xyz = XYZLocationEncoder()(coords)
        freqs = (2.0 ** torch.arange(4, dtype=torch.float32)) * math.pi
        angles = xyz[:, :, None] * freqs[None, None, :]
        parts = [torch.sin(angles).flatten(1), torch.cos(angles).flatten(1)]
        if include_xyz:
            parts.insert(0, xyz)
        assert torch.allclose(out, torch.cat(parts, dim=-1), atol=1e-6)

    def test_validation(self) -> None:
        with pytest.raises(TypeError, match='num_frequencies'):
            NeRFLocationEncoder(num_frequencies=cast(Any, True))
        with pytest.raises(TypeError, match='num_frequencies'):
            NeRFLocationEncoder(num_frequencies=cast(Any, 2.5))
        with pytest.raises(ValueError, match='num_frequencies'):
            NeRFLocationEncoder(num_frequencies=0)
        with pytest.raises(TypeError, match='include_xyz'):
            NeRFLocationEncoder(include_xyz=cast(Any, 1))


class TestSphericalHarmonicLocationEncoder:
    @pytest.mark.parametrize('degree', [0, 1, 2, 3])
    @pytest.mark.parametrize('batch_size', [1, 7])
    def test_addition_theorem_orthonormality(
        self, degree: int, batch_size: int
    ) -> None:
        enc = SphericalHarmonicLocationEncoder(degree=degree)
        assert enc.name == 'spherical-harmonics'
        assert enc.embed_dim == (degree + 1) ** 2

        coords = torch.stack(
            (
                torch.linspace(-170.0, 170.0, batch_size),
                torch.linspace(-85.0, 85.0, batch_size),
            ),
            dim=-1,
        )
        out = enc(coords)
        assert out.shape == (batch_size, (degree + 1) ** 2)
        assert out.dtype == torch.float32

        # Addition theorem: sum_{m=-l}^{l} (Y_l^m)^2 == (2l + 1) / (4 * pi)
        for ell in range(degree + 1):
            start = ell * ell
            stop = (ell + 1) * (ell + 1)
            band_energy = (out[:, start:stop] ** 2).sum(dim=-1)
            expected = torch.full((batch_size,), (2 * ell + 1) / (4.0 * math.pi))
            assert torch.allclose(band_energy, expected, atol=1e-5)

    def test_validation(self) -> None:
        with pytest.raises(TypeError, match='degree'):
            SphericalHarmonicLocationEncoder(degree=cast(Any, False))
        with pytest.raises(TypeError, match='degree'):
            SphericalHarmonicLocationEncoder(degree=cast(Any, 1.0))
        with pytest.raises(ValueError, match='between 0 and 3'):
            SphericalHarmonicLocationEncoder(degree=-1)
        with pytest.raises(ValueError, match='between 0 and 3'):
            SphericalHarmonicLocationEncoder(degree=4)


class TestAnalyticSphericalHarmonicLocationEncoder:
    @pytest.mark.parametrize('batch_size', [1, 7])
    def test_forward_sh40(self, batch_size: int) -> None:
        enc = AnalyticSphericalHarmonicLocationEncoder(legendre_polys=41)
        assert enc.name == 'sh40'
        assert enc.embed_dim == 41 * 41
        coords = torch.stack(
            (
                torch.linspace(-150.0, 150.0, batch_size),
                torch.linspace(-80.0, 80.0, batch_size),
            ),
            dim=-1,
        )
        out = enc(coords)
        assert out.shape == (batch_size, 1681)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()


class TestFrequencyRFFLocationEncoder:
    @pytest.mark.parametrize('batch_size', [1, 7])
    def test_forward_exact_and_deterministic(self, batch_size: int) -> None:
        enc1 = FrequencyRFFLocationEncoder(
            num_frequencies=16, max_scale_km=5000.0, min_scale_km=20.0, seed=42
        )
        enc2 = FrequencyRFFLocationEncoder(
            num_frequencies=16, max_scale_km=5000.0, min_scale_km=20.0, seed=42
        )
        assert enc1.name == 'rff3072'
        assert enc1.embed_dim == 32

        coords = torch.stack(
            (
                torch.linspace(-120.0, 120.0, batch_size),
                torch.linspace(-60.0, 60.0, batch_size),
            ),
            dim=-1,
        )
        out1 = enc1(coords)
        out2 = enc2(coords)
        assert out1.shape == (batch_size, 32)
        assert out1.dtype == torch.float32
        assert torch.equal(out1, out2)

        # Verify against reference nested_rff math in km
        scales = np.geomspace(5000.0, 20.0, 16)
        rng = np.random.default_rng(42)
        w = (rng.normal(0.0, 1.0, size=(3, 16)) / scales[None, :]).astype(np.float32)
        xyz_km = (XYZLocationEncoder()(coords) * 6371.0).numpy()
        proj = xyz_km @ w
        expected = np.stack((np.cos(proj), np.sin(proj)), axis=-1).reshape(
            batch_size, 32
        )
        assert np.allclose(out1.numpy(), expected, atol=1e-5)

    def test_validation(self) -> None:
        with pytest.raises(TypeError, match='num_frequencies'):
            FrequencyRFFLocationEncoder(num_frequencies=cast(Any, True))
        with pytest.raises(ValueError, match='num_frequencies'):
            FrequencyRFFLocationEncoder(num_frequencies=0)
        with pytest.raises(TypeError, match='seed'):
            FrequencyRFFLocationEncoder(seed=cast(Any, False))
        with pytest.raises(ValueError, match='positive'):
            FrequencyRFFLocationEncoder(max_scale_km=0.0)
        with pytest.raises(ValueError, match='positive'):
            FrequencyRFFLocationEncoder(min_scale_km=-5.0)
