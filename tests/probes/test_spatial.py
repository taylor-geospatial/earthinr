# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pytest

from earthinr.probes import spatial as sp_mod
from earthinr.probes.spatial import (
    _fit_variograms,
    _krige_weights,
    _select_kriging_model,
    idw_probe_score,
    kriging_probe_score,
)
from earthinr.probes.splits import spatial_fold_ids


def _make_spatial_data(
    seed: int = 5, n: int = 80
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lat = rng.uniform(-30.0, 30.0, size=n)
    lon = rng.uniform(-60.0, 60.0, size=n)
    y_reg = (
        np.sin(np.radians(lat) * 2.0)
        + 0.5 * np.cos(np.radians(lon))
        + 0.05 * rng.normal(size=n)
    )
    y_cls = np.digitize(y_reg, np.quantile(y_reg, [0.33, 0.67]))
    return lat, lon, y_reg, y_cls


class TestIDWProbeScore:
    def test_regression_and_classification_with_confusions_and_nan(self) -> None:
        lat, lon, y_reg, y_cls = _make_spatial_data()
        lat_nan = lat.copy()
        lat_nan[0] = np.nan

        s_reg, f_reg = idw_probe_score(
            lat_nan, lon, y_reg, task_type='regression', k=10, folds=4, seed=0
        )
        assert s_reg > 0.75
        assert len(f_reg) == 4

        cms: list[np.ndarray] = []
        s_cls, f_cls = idw_probe_score(
            lat,
            lon,
            y_cls,
            task_type='classification',
            k=10,
            folds=4,
            seed=0,
            confusions=cms,
        )
        assert s_cls > 0.7
        assert len(f_cls) == 4
        assert len(cms) == 4

        # Classification without confusions + test_mask
        tm = np.arange(len(lat)) >= 60
        s_tm, f_tm = idw_probe_score(
            lat, lon, y_cls, task_type='classification', k=5, test_mask=tm
        )
        assert s_tm > 0.6
        assert len(f_tm) == 1

    def test_validation_errors(self) -> None:
        lat, lon, y_reg, _ = _make_spatial_data(n=10)
        with pytest.raises(ValueError, match='k must be at least 1'):
            idw_probe_score(lat, lon, y_reg, k=0)
        with pytest.raises(ValueError, match='task_type'):
            idw_probe_score(lat, lon, y_reg, task_type='bad')
        with pytest.raises(ValueError, match='1D arrays'):
            idw_probe_score(np.zeros((5, 1)), np.zeros((5, 1)), y_reg[:5])
        with pytest.raises(ValueError, match='1D array matching'):
            idw_probe_score(lat, lon, y_reg[:5])
        with pytest.raises(ValueError, match='at least 2 valid samples'):
            idw_probe_score(
                np.array([np.nan, np.nan]), np.array([0.0, 1.0]), np.ones(2)
            )


class TestKrigingProbeScore:
    def test_regression_and_classification_random_and_spatial(self) -> None:
        lat, lon, y_reg, y_cls = _make_spatial_data(n=80)

        s_reg, f_reg = kriging_probe_score(
            lat, lon, y_reg, task_type='regression', k=15, folds=3, seed=0, cell_deg=0.0
        )
        assert s_reg > 0.6
        assert len(f_reg) == 3

        # Spatial block outer folds + spatial inner variogram selection (cell_deg > 0)
        fa = spatial_fold_ids(lat, lon, folds=3, cell_deg=20.0, seed=0)
        cms: list[np.ndarray] = []
        s_cls, f_cls = kriging_probe_score(
            lat,
            lon,
            y_cls,
            task_type='classification',
            k=15,
            seed=0,
            cell_deg=20.0,
            fold_assign=fa,
            confusions=cms,
        )
        assert s_cls > 0.6
        assert len(f_cls) == 3
        assert len(cms) == 3

        # Classification without confusions + test_mask
        tm = np.arange(len(lat)) >= 60
        s_tm, f_tm = kriging_probe_score(
            lat, lon, y_cls, task_type='classification', k=10, test_mask=tm
        )
        assert s_tm > 0.6
        assert len(f_tm) == 1

    def test_edge_cases_coincident_single_bin_degenerate_val_and_singular_solve(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Coincident points -> h.size == 0 fallback in _fit_variograms
        x_same = np.zeros((5, 3), dtype=np.float64)
        y_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        fits_same = _fit_variograms(x_same, y_vals, seed=0)
        assert set(fits_same) == {'exponential', 'spherical', 'gaussian'}
        assert fits_same['exponential']['psill'] == 0.0

        # Equidistant points (single unique positive distance) -> len(edges) < 2 branch
        x_eq = np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, np.sqrt(75.0), 0.0]],
            dtype=np.float64,
        )
        fits_eq = _fit_variograms(x_eq, np.array([0.0, 1.0, 2.0]), seed=0)
        assert set(fits_eq) == {'exponential', 'spherical', 'gaussian'}

        # Degenerate inner val split (single spatial cell -> val.all() or not val.any())
        x_one_cell = np.array(
            [[6371.0, 0.0, 0.0], [6371.0, 0.1, 0.0], [6371.0, 0.2, 0.0]],
            dtype=np.float64,
        )
        chosen = _select_kriging_model(
            x_one_cell, np.array([1.0, 2.0, 3.0]), cell_deg=90.0, seed=0, k=2
        )
        assert chosen == 'exponential'

        # Singular linear system -> np.linalg.lstsq fallback in _krige_weights
        def _raise_linalg(*_args: object, **_kwargs: object) -> np.ndarray:
            raise np.linalg.LinAlgError('singular')

        monkeypatch.setattr(np.linalg, 'solve', _raise_linalg)
        idx, w = _krige_weights(
            x_eq,
            np.array([[2.0, 2.0, 0.0]], dtype=np.float64),
            fits_eq['exponential'],
            k=2,
        )
        assert idx.shape == (1, 2)
        assert w.shape == (1, 2)
        np.testing.assert_allclose(w.sum(axis=1), [1.0], atol=1e-6)
        assert sp_mod._VARIOGRAM_MODELS == ('exponential', 'spherical', 'gaussian')

    def test_validation_errors(self) -> None:
        lat, lon, y_reg, _ = _make_spatial_data(n=10)
        with pytest.raises(ValueError, match='k must be at least 1'):
            kriging_probe_score(lat, lon, y_reg, k=0)
        with pytest.raises(ValueError, match='cell_deg must be non-negative'):
            kriging_probe_score(lat, lon, y_reg, cell_deg=-1.0)
        with pytest.raises(ValueError, match='at least 2 valid samples'):
            kriging_probe_score(
                np.array([np.nan, np.nan]), np.array([0.0, 1.0]), np.ones(2)
            )
