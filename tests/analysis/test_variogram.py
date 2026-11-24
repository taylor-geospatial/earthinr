# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

import math
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import Tensor

import earthinr
from earthinr.analysis import (
    EARTH_RADIUS_KM,
    KM_PER_DEGREE,
    analyze_implicit_resolution,
    empirical_semivariogram,
    geodesic_offset,
    half_sill_distance,
    sample_sphere_points,
)
from earthinr.analysis.variogram import _haversine_km
from earthinr.models import LocationEncoder, get_model


class _RecordingEncoder(LocationEncoder):
    name = 'recording'
    embed_dim = 4

    def __init__(self) -> None:
        super().__init__(device='cpu', batch_size=256)
        self.saw_year = False
        self.saw_month = False

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        if year is not None:
            self.saw_year = True
        if month is not None:
            self.saw_month = True
        rad = torch.deg2rad(coords)
        return torch.cat([torch.sin(rad), torch.cos(rad)], dim=-1)


class TestSampleSpherePoints:
    def test_uniform_area_sampling(self) -> None:
        lon, lat = sample_sphere_points(5000, lat_min=-60.0, lat_max=60.0, seed=42)
        assert lon.shape == (5000,)
        assert lat.shape == (5000,)
        assert lon.dtype == np.float64
        assert lat.dtype == np.float64
        assert np.all((lon >= -180.0) & (lon <= 180.0))
        assert np.all((lat >= -60.0) & (lat <= 60.0))
        assert abs(float(np.mean(np.sin(np.radians(lat))))) < 0.05

        lon2, lat2 = sample_sphere_points(5000, lat_min=-60.0, lat_max=60.0, seed=42)
        np.testing.assert_array_equal(lon, lon2)
        np.testing.assert_array_equal(lat, lat2)

    def test_land_coords_filtering_and_subsampling(self) -> None:
        land_lon = np.array([-120.0, -60.0, 0.0, 60.0, 120.0])
        land_lat = np.array([-70.0, -20.0, 10.0, 40.0, 80.0])

        lon, lat = sample_sphere_points(
            2, lat_min=-56.0, lat_max=70.0, seed=0, land_coords=(land_lon, land_lat)
        )
        assert lon.shape == (2,)
        assert np.all((lat >= -56.0) & (lat <= 70.0))

        # Requesting more than available returns all filtered points without replacement.
        lon_all, lat_all = sample_sphere_points(
            10, lat_min=-56.0, lat_max=70.0, seed=0, land_coords=(land_lon, land_lat)
        )
        assert lon_all.shape == (3,)
        assert lat_all.shape == (3,)

    def test_validation_errors(self) -> None:
        with pytest.raises(ValueError, match='n must be positive'):
            sample_sphere_points(0)
        with pytest.raises(ValueError, match='lat_min'):
            sample_sphere_points(10, lat_min=10.0, lat_max=10.0)
        with pytest.raises(ValueError, match='lat_min'):
            sample_sphere_points(10, lat_min=-95.0, lat_max=0.0)
        with pytest.raises(ValueError, match='land_coords'):
            sample_sphere_points(2, land_coords=(np.zeros((2, 1)), np.zeros((2, 1))))
        with pytest.raises(ValueError, match='No land_coords'):
            sample_sphere_points(
                2,
                lat_min=0.0,
                lat_max=10.0,
                land_coords=(np.array([0.0]), np.array([-30.0])),
            )


class TestGeodesicOffset:
    def test_preserves_great_circle_distance_across_latitudes(self) -> None:
        lon = np.array([0.0, 45.0, -179.5, 120.0])
        lat = np.array([0.0, 45.0, 70.0, -55.0])
        lon2, lat2 = geodesic_offset(lon, lat, 500.0, seed=7)
        assert np.all((lon2 >= -180.0) & (lon2 < 180.0))
        assert np.all((lat2 >= -90.0) & (lat2 <= 90.0))
        dist = _haversine_km(lon, lat, lon2, lat2)
        np.testing.assert_allclose(dist, 500.0, rtol=1e-6)

        rng = np.random.default_rng(7)
        lon3, lat3 = geodesic_offset(lon, lat, 500.0, rng=rng)
        np.testing.assert_array_equal(lon2, lon3)
        np.testing.assert_array_equal(lat2, lat3)

    def test_validation_errors(self) -> None:
        with pytest.raises(ValueError, match='1D arrays'):
            geodesic_offset(np.array([0.0, 1.0]), np.array([0.0]), 10.0)
        with pytest.raises(ValueError, match='non-negative'):
            geodesic_offset(np.array([0.0]), np.array([0.0]), -1.0)


class TestEmpiricalSemivariogram:
    def test_smooth_vs_high_frequency_fields(self) -> None:
        lon, lat = sample_sphere_points(400, seed=1)
        sincos = get_model('sincos', device='cpu')
        sh40 = get_model('sh40', device='cpu')

        v_smooth = empirical_semivariogram(
            lon,
            lat,
            sincos.encode(lon, lat),
            n_bins=24,
            min_km=50.0,
            max_km=8000.0,
            n_pairs=80_000,
            seed=0,
        )
        v_high = empirical_semivariogram(
            lon,
            lat,
            sh40.encode(lon, lat),
            n_bins=24,
            min_km=50.0,
            max_km=8000.0,
            n_pairs=80_000,
            seed=0,
        )
        assert v_smooth['bin_centers_km'].shape == (24,)
        assert v_smooth['semivariance'].shape == (24,)
        assert v_smooth['counts'].shape == (24,)

        hs_smooth, sill_smooth = half_sill_distance(
            v_smooth['bin_centers_km'], v_smooth['semivariance']
        )
        hs_high, sill_high = half_sill_distance(
            v_high['bin_centers_km'], v_high['semivariance']
        )
        assert math.isfinite(hs_smooth)
        assert math.isfinite(sill_smooth)
        assert math.isfinite(hs_high)
        assert math.isfinite(sill_high)
        assert hs_smooth > 3.0 * hs_high

    def test_normalize_handles_zero_norm_and_out_of_range_pairs(self) -> None:
        # Two points ~111 km apart, plus a third identical point so self-pairs and
        # out-of-range pairs occur cleanly.
        lon = np.array([0.0, 0.0])
        lat = np.array([0.0, 1.0])
        emb = np.array([[0.0, 0.0], [3.0, 4.0]])

        # Out-of-range bin window [1, 10] km -> no pairs fall in range.
        empty = empirical_semivariogram(
            lon,
            lat,
            emb,
            n_bins=4,
            min_km=1.0,
            max_km=10.0,
            n_pairs=20,
            normalize=True,
            seed=0,
        )
        assert np.all(empty['counts'] == 0.0)
        assert np.all(np.isnan(empty['semivariance']))

        # Tiny n_pairs=1 with seed=0 draws (1, 1) and hits the self-pair skip.
        self_only = empirical_semivariogram(
            lon,
            lat,
            emb,
            n_bins=4,
            min_km=10.0,
            max_km=500.0,
            n_pairs=1,
            normalize=True,
            seed=0,
        )
        assert np.all(self_only['counts'] == 0.0)

        # In-range bin window captures the normalized pair: ||[0,0] - [0.6,0.8]||^2 / 2 = 0.5.
        populated = empirical_semivariogram(
            lon,
            lat,
            emb,
            n_bins=4,
            min_km=50.0,
            max_km=500.0,
            n_pairs=200,
            normalize=True,
            seed=0,
        )
        valid = populated['counts'] > 0
        assert np.any(valid)
        np.testing.assert_allclose(populated['semivariance'][valid], 0.5, rtol=1e-6)

    def test_validation_errors(self) -> None:
        lon = np.array([0.0, 1.0])
        lat = np.array([0.0, 1.0])
        emb = np.ones((2, 3))
        with pytest.raises(ValueError, match='non-empty 2D'):
            empirical_semivariogram(lon, lat, np.ones(2))
        with pytest.raises(ValueError, match='1D arrays'):
            empirical_semivariogram(np.array([0.0]), lat, emb)
        with pytest.raises(ValueError, match='match lon/lat'):
            empirical_semivariogram(lon, lat, np.ones((3, 3)))
        with pytest.raises(ValueError, match='at least 2'):
            empirical_semivariogram(np.array([0.0]), np.array([0.0]), np.ones((1, 3)))
        with pytest.raises(ValueError, match='n_bins'):
            empirical_semivariogram(lon, lat, emb, n_bins=1)
        with pytest.raises(ValueError, match='min_km'):
            empirical_semivariogram(lon, lat, emb, min_km=0.0, max_km=10.0)
        with pytest.raises(ValueError, match='min_km'):
            empirical_semivariogram(lon, lat, emb, min_km=10.0, max_km=5.0)
        with pytest.raises(ValueError, match='n_pairs'):
            empirical_semivariogram(lon, lat, emb, n_pairs=0)


class TestHalfSillDistance:
    def test_exact_log_axis_interpolation(self) -> None:
        lags = np.array([10.0, 100.0, 1000.0, 10000.0])
        gamma = np.array([0.0, 1.0, 1.0, 1.0])
        # Unsorted input is handled cleanly.
        hs, sill = half_sill_distance(lags[::-1], gamma[::-1])
        assert sill == pytest.approx(1.0)
        # Halfway between log(10) and log(100) is sqrt(10 * 100) = 10 * sqrt(10).
        assert hs == pytest.approx(10.0 * math.sqrt(10.0), rel=1e-6)

    def test_edge_cases_return_nan(self) -> None:
        # Fewer than 2 finite positive bins.
        hs, sill = half_sill_distance(np.array([1.0, np.nan]), np.array([0.5, 1.0]))
        assert math.isnan(hs)
        assert math.isnan(sill)

        # Zero / flat zero semivariance (sill <= 0).
        hs, sill = half_sill_distance(np.array([1.0, 10.0, 100.0, 1000.0]), np.zeros(4))
        assert math.isnan(hs)
        assert sill == 0.0

        # Already above half-sill at first bin (pure white noise / non-crossing).
        hs, sill = half_sill_distance(
            np.array([1.0, 10.0, 100.0, 1000.0]), np.array([1.0, 1.0, 1.0, 1.0])
        )
        assert math.isnan(hs)
        assert sill == pytest.approx(1.0)

        # Curve dips below frac * sill everywhere in the domain (non-crossing).
        # Outer quartile is the last element (10.0), earlier bins are below 0.5*10?
        # Wait: if the last element is 10.0, hit includes the last element!
        # To have hits.shape[0] == 0: outer quartile has median > max? No, median <= max.
        # Actually, can hits ever be empty when sill = median(tail) and frac < 1?
        # Since tail is a subset of gamma_v, max(gamma_v) >= max(tail) >= median(tail) = sill > frac * sill!
        # So gamma_v >= frac * sill ALWAYS has at least one hit when sill > 0 and frac < 1!
        # Let's verify that with a monotone decreasing curve starting above sill.
        hs_dec, sill_dec = half_sill_distance(
            np.array([1.0, 10.0, 100.0, 1000.0]), np.array([5.0, 4.0, 2.0, 1.0])
        )
        assert math.isnan(hs_dec)
        assert sill_dec == pytest.approx(1.0)

    def test_validation_errors(self) -> None:
        with pytest.raises(ValueError, match='1D arrays'):
            half_sill_distance(np.array([1.0, 2.0]), np.array([1.0]))
        with pytest.raises(ValueError, match='frac'):
            half_sill_distance(np.array([1.0, 2.0]), np.array([0.0, 1.0]), frac=0.0)
        with pytest.raises(ValueError, match='frac'):
            half_sill_distance(np.array([1.0, 2.0]), np.array([0.0, 1.0]), frac=1.0)


class TestAnalyzeImplicitResolution:
    def test_geodesic_ranks_low_vs_high_frequency_encoders(self) -> None:
        lon, lat = sample_sphere_points(200, seed=0)
        sincos = get_model('sincos', device='cpu')
        sh40 = get_model('sh40', device='cpu')
        nerf = get_model('nerf', device='cpu')

        res_sincos = analyze_implicit_resolution(
            sincos, lon, lat, n_bins=24, pairs_per_lag=64, seed=0
        )
        res_sh40 = analyze_implicit_resolution(
            sh40, lon, lat, n_bins=24, pairs_per_lag=64, seed=0
        )
        res_nerf = analyze_implicit_resolution(
            nerf, lon, lat, n_bins=24, pairs_per_lag=64, seed=0
        )

        assert res_sincos['half_sill_km'] > res_sh40['half_sill_km']
        assert res_sh40['half_sill_km'] > res_nerf['half_sill_km']
        assert res_sincos['suggested_grid_km'] == pytest.approx(
            res_sincos['half_sill_km'] / 2.0
        )
        assert res_sincos['suggested_grid_deg'] == pytest.approx(
            res_sincos['suggested_grid_km'] / KM_PER_DEGREE
        )
        assert res_sincos['n_points'] == 200.0
        assert res_sincos['embed_dim'] == 4.0

    def test_pairs_method_and_temporal_forwarding(self) -> None:
        lon, lat = sample_sphere_points(120, seed=3)
        year = np.full(120, 2020.0)
        month = np.full(120, 6.0)
        enc = _RecordingEncoder()

        res_geo = analyze_implicit_resolution(
            enc,
            lon,
            lat,
            year=year,
            month=month,
            n_bins=12,
            pairs_per_lag=32,
            normalize=True,
            seed=0,
        )
        assert enc.saw_year
        assert enc.saw_month
        assert math.isfinite(res_geo['half_sill_km'])
        assert 0.0 < res_geo['sill'] <= 2.0

        res_pairs = analyze_implicit_resolution(
            enc,
            lon,
            lat,
            n_bins=12,
            min_km=100.0,
            max_km=8000.0,
            n_pairs=10_000,
            method='pairs',
            seed=0,
        )
        assert math.isfinite(res_pairs['half_sill_km'])
        assert res_pairs['embed_dim'] == 4.0

    def test_validation_errors(self) -> None:
        enc = get_model('xyz', device='cpu')
        lon = np.array([0.0, 10.0])
        lat = np.array([0.0, 10.0])
        with pytest.raises(ValueError, match='Unsupported method'):
            analyze_implicit_resolution(enc, lon, lat, method=cast(Any, 'bad'))
        with pytest.raises(ValueError, match='1D arrays'):
            analyze_implicit_resolution(enc, np.array([0.0]), lat)
        with pytest.raises(ValueError, match='at least 2'):
            analyze_implicit_resolution(enc, np.array([0.0]), np.array([0.0]))
        with pytest.raises(ValueError, match='n_bins'):
            analyze_implicit_resolution(enc, lon, lat, n_bins=1)
        with pytest.raises(ValueError, match='min_km'):
            analyze_implicit_resolution(enc, lon, lat, min_km=0.0, max_km=100.0)
        with pytest.raises(ValueError, match='pairs_per_lag'):
            analyze_implicit_resolution(enc, lon, lat, pairs_per_lag=0)


def test_top_level_lazy_export() -> None:
    assert earthinr.analysis.EARTH_RADIUS_KM == EARTH_RADIUS_KM
