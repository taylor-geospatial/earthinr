# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

import builtins
import math
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

import earthinr
from earthinr import DependencyNotFoundError
from earthinr.analysis import (
    SUPPORTED_ESTIMATORS,
    analyze_intrinsic_dimension,
    estimate_intrinsic_dimension,
    generate_equal_area_grid,
    participation_ratio,
)
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


def _install_fake_torchid(
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_primitives: bool = True,
    with_submod_knn: bool = True,
) -> dict[str, Any]:
    """Install a lightweight fake ``torchid`` module in ``sys.modules``."""
    state: dict[str, Any] = {
        'knn_torch_calls': 0,
        'seen_dtypes': [],
        'seen_shapes': [],
        'fit_kwargs': {},
        'init_kwargs': {},
    }

    fake_torchid: Any = ModuleType('torchid')
    fake_estimators: Any = ModuleType('torchid.estimators')

    def _orig_knn(
        x: Tensor,
        k: int,
        *,
        chunk: int = 4096,
        include_self: bool = False,
        Y: Tensor | None = None,  # noqa: N803
    ) -> tuple[Tensor, Tensor]:
        raise AssertionError('float32 default knn should have been patched')

    def _knn_torch(
        x: Tensor,
        k: int,
        *,
        Y: Tensor | None = None,  # noqa: N803
        include_self: bool = False,
        chunk: int = 4096,
    ) -> tuple[Tensor, Tensor]:
        state['knn_torch_calls'] += 1
        dists = torch.ones((x.shape[0], k), dtype=x.dtype, device=x.device)
        idx = torch.zeros((x.shape[0], k), dtype=torch.long, device=x.device)
        return dists, idx

    if with_primitives:
        fake_primitives: Any = ModuleType('torchid.primitives')
        fake_primitives.knn = _orig_knn
        fake_primitives._knn_torch = _knn_torch
        fake_torchid.primitives = fake_primitives
        state['primitives'] = fake_primitives
        state['orig_knn'] = _orig_knn

        for sub_name in ('mle', 'mom', 'tle', 'twonn'):
            submod: Any = ModuleType(f'torchid.estimators.{sub_name}')
            if with_submod_knn:
                submod.knn = _orig_knn
            setattr(fake_estimators, sub_name, submod)
        state['submods'] = {
            n: getattr(fake_estimators, n) for n in ('mle', 'mom', 'tle', 'twonn')
        }

    def _make_estimator(label: str, score: float, use_knn: bool) -> type[Any]:
        class _FakeEstimator:
            def __init__(self, **kwargs: Any) -> None:
                state['init_kwargs'][label] = kwargs
                self.dimension_: float = float('nan')

            def fit(self, x: Tensor, **kwargs: Any) -> '_FakeEstimator':
                state['seen_dtypes'].append(x.dtype)
                state['seen_shapes'].append(tuple(x.shape))
                state['fit_kwargs'][label] = kwargs
                if use_knn and with_primitives:
                    fake_torchid.primitives.knn(x, 2)
                self.dimension_ = score
                return self

        return _FakeEstimator

    fake_estimators.MLE = _make_estimator('mle', 3.1, True)
    fake_estimators.TwoNN = _make_estimator('twonn', 2.4, True)
    fake_estimators.MOM = _make_estimator('mom', 2.9, True)
    fake_estimators.TLE = _make_estimator('tle', 3.3, True)
    fake_estimators.FisherS = _make_estimator('fishers', 4.2, False)

    fake_torchid.estimators = fake_estimators
    monkeypatch.setitem(sys.modules, 'torchid', fake_torchid)
    monkeypatch.setitem(sys.modules, 'torchid.estimators', fake_estimators)
    return state


class TestGenerateEqualAreaGrid:
    def test_global_100km_grid_properties(self) -> None:
        lon, lat = generate_equal_area_grid(100.0)
        assert lon.shape == (51_006,)
        assert lat.shape == (51_006,)
        assert lon.dtype == np.float64
        assert lat.dtype == np.float64
        assert np.all((lon >= -180.0) & (lon < 180.0))
        assert np.all((lat > -90.0) & (lat < 90.0))
        assert abs(float(np.mean(np.sin(np.radians(lat))))) < 1e-12

        # Deterministic output and zero duplicate coordinates.
        lon2, lat2 = generate_equal_area_grid(100.0)
        np.testing.assert_array_equal(lon, lon2)
        np.testing.assert_array_equal(lat, lat2)

    def test_latitude_band_filtering_and_jitter(self) -> None:
        lon, lat = generate_equal_area_grid(500.0, lat_min=-30.0, lat_max=30.0)
        assert lon.shape == lat.shape
        assert lon.shape[0] > 0
        assert np.all((lat >= -30.0) & (lat <= 30.0))

        lon_j1, lat_j1 = generate_equal_area_grid(500.0, jitter_km=500.0, seed=42)
        lon_j2, lat_j2 = generate_equal_area_grid(500.0, jitter_km=500.0, seed=42)
        lon_raw, lat_raw = generate_equal_area_grid(500.0)
        assert lon_j1.shape == lon_raw.shape
        assert np.all((lon_j1 >= -180.0) & (lon_j1 < 180.0))
        assert np.all((lat_j1 >= -90.0) & (lat_j1 <= 90.0))
        np.testing.assert_array_equal(lon_j1, lon_j2)
        np.testing.assert_array_equal(lat_j1, lat_j2)
        assert not np.allclose(lon_j1, lon_raw)
        assert not np.allclose(lat_j1, lat_raw)

    def test_validation_errors(self) -> None:
        with pytest.raises(ValueError, match='spacing_km'):
            generate_equal_area_grid(0.0)
        with pytest.raises(ValueError, match='spacing_km'):
            generate_equal_area_grid(float('nan'))
        with pytest.raises(ValueError, match='jitter_km'):
            generate_equal_area_grid(500.0, jitter_km=-1.0)
        with pytest.raises(ValueError, match='jitter_km'):
            generate_equal_area_grid(500.0, jitter_km=float('inf'))
        with pytest.raises(ValueError, match='lat_min'):
            generate_equal_area_grid(500.0, lat_min=10.0, lat_max=10.0)
        with pytest.raises(ValueError, match='lat_min'):
            generate_equal_area_grid(500.0, lat_min=-95.0, lat_max=0.0)
        with pytest.raises(ValueError, match='No Fibonacci grid points'):
            generate_equal_area_grid(20_000.0, lat_min=89.0, lat_max=90.0)


class TestParticipationRatio:
    def test_exact_isotropic_subspace_and_invariance(self) -> None:
        rng = np.random.default_rng(0)
        # Orthonormal 4-D basis embedded in 12-D space with equal eigenvalues -> PR = 4.
        q, _ = np.linalg.qr(rng.standard_normal((12, 12)))
        basis = q[:4]  # (4, 12)
        # Construct N = 8 centered points with exact covariance proportional to I_4.
        signs = np.vstack([np.eye(4), -np.eye(4)])  # (8, 4)
        emb = signs @ basis  # (8, 12), N < D path!
        pr_wide = participation_ratio(emb)
        assert pr_wide == pytest.approx(4.0, rel=1e-10)

        # Positive rescaling and orthogonal rotation preserve PR (N >= D path).
        emb_tall = np.vstack([emb, emb * 2.0])  # (16, 12)
        pr_tall = participation_ratio(emb_tall)
        assert pr_tall == pytest.approx(4.0, rel=1e-10)
        assert participation_ratio(7.5 * emb_tall @ q) == pytest.approx(4.0, rel=1e-10)

        # Accepts PyTorch tensors directly.
        pr_tensor = participation_ratio(torch.from_numpy(emb_tall.astype(np.float32)))
        assert pr_tensor == pytest.approx(4.0, rel=1e-5)

    def test_standardize_and_constant_edge_cases(self) -> None:
        # Two independent axes with vastly different scales + one constant column.
        x = np.array(
            [
                [-1.0, -100.0, 5.0],
                [-1.0, 100.0, 5.0],
                [1.0, -100.0, 5.0],
                [1.0, 100.0, 5.0],
            ]
        )
        # Raw covariance is dominated by column 1 -> PR near 1.
        assert participation_ratio(x, standardize=False) == pytest.approx(
            1.0002, rel=1e-3
        )
        # Standardized correlation spectrum equalizes both active axes -> PR = 2.
        assert participation_ratio(x, standardize=True) == pytest.approx(2.0, rel=1e-10)

        # All-constant embeddings yield nan for both modes.
        const = np.ones((5, 3))
        assert math.isnan(participation_ratio(const, standardize=False))
        assert math.isnan(participation_ratio(const, standardize=True))

    def test_validation_errors(self) -> None:
        with pytest.raises(ValueError, match='2D array'):
            participation_ratio(np.ones(5))
        with pytest.raises(ValueError, match='at least 2 samples'):
            participation_ratio(np.ones((1, 4)))
        with pytest.raises(ValueError, match='1 feature'):
            participation_ratio(np.ones((4, 0)))


class TestEstimateIntrinsicDimension:
    def test_pr_only_runs_without_torchid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delitem(sys.modules, 'torchid', raising=False)
        emb = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        out = estimate_intrinsic_dimension(emb, estimators=())
        assert set(out.keys()) == {'pr'}
        assert out['pr'] == pytest.approx(2.0, rel=1e-10)

    def test_missing_torchid_raises_dependency_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delitem(sys.modules, 'torchid', raising=False)
        real_import = builtins.__import__

        def _block_torchid(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == 'torchid' or name.startswith('torchid.'):
                raise ModuleNotFoundError("No module named 'torchid'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', _block_torchid)
        emb = np.arange(100, dtype=np.float64).reshape(50, 2)
        with pytest.raises(DependencyNotFoundError, match='torchid'):
            estimate_intrinsic_dimension(emb, estimators=('twonn',))

    def test_all_estimators_and_exact_knn_patch_restore(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = _install_fake_torchid(monkeypatch, with_primitives=True)
        rng = np.random.default_rng(1)
        base = rng.standard_normal((30, 6))
        # Include 2 exact duplicate rows to verify deduplication.
        emb = np.vstack([base, base[:2]])

        out = estimate_intrinsic_dimension(
            emb,
            estimators=('MLE', 'twonn', 'mom', 'tle', 'fishers', 'mle'),
            k=10,
            conditional_number=15.0,
            standardize_pr=True,
            deduplicate=True,
            device='cpu',
        )
        assert out['mle'] == pytest.approx(3.1)
        assert out['twonn'] == pytest.approx(2.4)
        assert out['mom'] == pytest.approx(2.9)
        assert out['tle'] == pytest.approx(3.3)
        assert out['fishers'] == pytest.approx(4.2)
        assert math.isfinite(out['pr'])

        # All tensors passed to torchid were float64 and deduplicated to 30 rows.
        assert all(dt == torch.float64 for dt in state['seen_dtypes'])
        assert all(shape == (30, 6) for shape in state['seen_shapes'])
        assert state['knn_torch_calls'] == 4
        assert state['fit_kwargs']['mle'] == {'n_neighbors': 10}
        assert state['init_kwargs']['mom'] == {'n_neighbors': 10}
        assert state['init_kwargs']['tle'] == {'n_neighbors': 10}
        assert state['init_kwargs']['fishers'] == {'conditional_number': 15.0}

        # Context manager restored original knn on primitives and submodules.
        assert state['primitives'].knn is state['orig_knn']
        for submod in state['submods'].values():
            assert submod.knn is state['orig_knn']

    def test_without_primitives_and_without_deduplication(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = _install_fake_torchid(monkeypatch, with_primitives=False)
        base = np.arange(20, dtype=np.float64).reshape(10, 2)
        emb = np.vstack([base, base[:2]])
        out = estimate_intrinsic_dimension(
            emb, estimators=('twonn',), deduplicate=False
        )
        assert out['twonn'] == pytest.approx(2.4)
        assert state['seen_shapes'] == [(12, 2)]

    def test_submodules_without_knn_attr_skipped_cleanly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = _install_fake_torchid(
            monkeypatch, with_primitives=True, with_submod_knn=False
        )
        emb = np.arange(20, dtype=np.float64).reshape(10, 2)
        out = estimate_intrinsic_dimension(emb, estimators=('twonn',))
        assert out['twonn'] == pytest.approx(2.4)
        assert state['knn_torch_calls'] == 1
        assert state['primitives'].knn is state['orig_knn']

    def test_validation_errors(self) -> None:
        emb = np.arange(20, dtype=np.float64).reshape(10, 2)
        with pytest.raises(ValueError, match='k must be at least 2'):
            estimate_intrinsic_dimension(emb, k=1)
        with pytest.raises(ValueError, match='conditional_number'):
            estimate_intrinsic_dimension(emb, conditional_number=0.0)
        with pytest.raises(ValueError, match='Unsupported estimator'):
            estimate_intrinsic_dimension(emb, estimators=('unknown',))
        with pytest.raises(ValueError, match='Need at least 21 unique embeddings'):
            estimate_intrinsic_dimension(emb, estimators=('mle',), k=20)
        tiny = np.array([[0.0, 0.0], [1.0, 1.0]])
        with pytest.raises(ValueError, match='Need at least 3 unique embeddings'):
            estimate_intrinsic_dimension(tiny, estimators=('twonn',))


class TestAnalyzeIntrinsicDimension:
    def test_encodes_and_forwards_temporal_inputs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_torchid(monkeypatch, with_primitives=True)
        lon, lat = generate_equal_area_grid(2500.0)
        year = np.full(lon.shape[0], 2020.0)
        month = np.full(lon.shape[0], 6.0)
        enc = _RecordingEncoder()

        res = analyze_intrinsic_dimension(
            enc, lon, lat, year=year, month=month, estimators=('twonn', 'fishers'), k=10
        )
        assert enc.saw_year
        assert enc.saw_month
        assert res['n_points'] == float(lon.shape[0])
        assert res['embed_dim'] == 4.0
        assert res['twonn'] == pytest.approx(2.4)
        assert res['fishers'] == pytest.approx(4.2)
        assert math.isfinite(res['pr'])

    def test_validation_errors(self) -> None:
        enc = get_model('xyz', device='cpu')
        with pytest.raises(ValueError, match='1D arrays'):
            analyze_intrinsic_dimension(enc, np.array([0.0]), np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match='at least 2 points'):
            analyze_intrinsic_dimension(enc, np.array([0.0]), np.array([0.0]))


def test_top_level_exports() -> None:
    assert earthinr.analysis.SUPPORTED_ESTIMATORS == SUPPORTED_ESTIMATORS
    assert earthinr.analysis.participation_ratio is participation_ratio
