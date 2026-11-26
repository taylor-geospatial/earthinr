# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

import neuralftw
from neuralftw import (
    DependencyNotFoundError,
    LocationEncoder,
    embeddings_to_rgb,
    encode_grid,
    make_coord_grid,
    make_land_mask,
    plot_embedding_map,
)


class _RecordingEncoder(LocationEncoder):
    name = 'recording'
    embed_dim = 4

    def __init__(self, batch_size: int = 8192) -> None:
        super().__init__(device='cpu', batch_size=batch_size)
        self.seen_years: list[float] = []
        self.seen_months: list[float] = []
        self.batch_sizes: list[int] = []

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        self.batch_sizes.append(int(coords.shape[0]))
        if year is not None:
            self.seen_years.extend(year.tolist())
        if month is not None:
            self.seen_months.extend(month.tolist())
        lon = coords[:, 0:1] / 180.0
        lat = coords[:, 1:2] / 90.0
        return torch.cat((lon, lat, lon * lat, lon + lat), dim=-1).to(torch.float32)


class _FakeFigure:
    def __init__(self, figsize: tuple[float, float]) -> None:
        self.figsize = figsize
        self.saved: list[tuple[str, dict[str, Any]]] = []

    def savefig(self, path: str, **kwargs: Any) -> None:
        self.saved.append((path, kwargs))


class _FakeAxes:
    def __init__(self, figure: _FakeFigure) -> None:
        self.figure = figure
        self.imshow_calls: list[tuple[np.ndarray, dict[str, Any]]] = []
        self.xlabel = ''
        self.ylabel = ''
        self.title = ''

    def imshow(self, img: np.ndarray, **kwargs: Any) -> None:
        self.imshow_calls.append((img, kwargs))

    def set_xlabel(self, label: str) -> None:
        self.xlabel = label

    def set_ylabel(self, label: str) -> None:
        self.ylabel = label

    def set_title(self, title: str) -> None:
        self.title = title


def _install_fake_matplotlib(monkeypatch: pytest.MonkeyPatch) -> list[_FakeFigure]:
    figs: list[_FakeFigure] = []

    def _subplots(
        figsize: tuple[float, float] = (10.0, 5.0),
    ) -> tuple[_FakeFigure, _FakeAxes]:
        fig = _FakeFigure(figsize)
        ax = _FakeAxes(fig)
        figs.append(fig)
        return fig, ax

    mpl: Any = ModuleType('matplotlib')
    pyplot: Any = ModuleType('matplotlib.pyplot')
    pyplot.subplots = _subplots
    mpl.pyplot = pyplot
    monkeypatch.setitem(sys.modules, 'matplotlib', mpl)
    monkeypatch.setitem(sys.modules, 'matplotlib.pyplot', pyplot)
    return figs


class TestMakeCoordGrid:
    def test_step_and_explicit_dimensions(self) -> None:
        lon, lat = make_coord_grid(step_deg=30.0)
        assert lon.shape == (6, 12)
        assert lat.shape == (6, 12)
        assert lon.dtype == np.float64
        assert lat.dtype == np.float64
        # North-to-south ordering (row 0 near +90, last row near -90)
        assert lat[0, 0] > lat[-1, 0]
        assert lon[0, 0] < lon[0, -1]
        np.testing.assert_allclose(lat[0, 0], 75.0)
        np.testing.assert_allclose(lat[-1, 0], -75.0)
        np.testing.assert_allclose(lon[0, 0], -165.0)

        lon_def, lat_def = make_coord_grid((-10.0, -5.0, 10.0, 5.0))
        assert lon_def.shape == (10, 20)
        assert lat_def.shape == (10, 20)

        lon_sz, lat_sz = make_coord_grid((-125.0, 24.0, -66.0, 50.0), width=8, height=4)
        assert lon_sz.shape == (4, 8)
        assert lat_sz.shape == (4, 8)

    def test_validation_errors(self) -> None:
        bad_bounds: Any = (-10.0, 0.0, 10.0)
        with pytest.raises(ValueError, match='4-tuple'):
            make_coord_grid(bad_bounds)
        with pytest.raises(ValueError, match='Invalid longitude bounds'):
            make_coord_grid((-200.0, -10.0, 10.0, 10.0))
        with pytest.raises(ValueError, match='Invalid longitude bounds'):
            make_coord_grid((10.0, -10.0, 10.0, 10.0))
        with pytest.raises(ValueError, match='Invalid latitude bounds'):
            make_coord_grid((-10.0, -95.0, 10.0, 10.0))
        with pytest.raises(ValueError, match='Invalid latitude bounds'):
            make_coord_grid((-10.0, 20.0, 10.0, 10.0))
        with pytest.raises(ValueError, match='not both'):
            make_coord_grid(step_deg=5.0, width=10, height=5)
        with pytest.raises(ValueError, match='specified together'):
            make_coord_grid(width=10)
        with pytest.raises(ValueError, match='positive integers'):
            make_coord_grid(width=0, height=5)
        with pytest.raises(ValueError, match='positive integers'):
            make_coord_grid(width=True, height=5)
        with pytest.raises(ValueError, match='step_deg must be positive'):
            make_coord_grid(step_deg=0.0)


class TestEncodeGrid:
    def test_encode_grid_with_instance_and_preset_name(self) -> None:
        enc = _RecordingEncoder(batch_size=100)
        lon, lat, emb = encode_grid(
            enc,
            bounds=(-20.0, -10.0, 20.0, 10.0),
            width=5,
            height=4,
            year=2023,
            month=6,
            batch_size=7,
        )
        assert lon.shape == (4, 5)
        assert lat.shape == (4, 5)
        assert emb.shape == (4, 5, 4)
        assert emb.dtype == np.float32
        assert enc.batch_size == 100
        assert enc.batch_sizes == [7, 7, 6]
        assert len(enc.seen_years) == 20
        assert all(y == 2023.0 for y in enc.seen_years)
        assert all(m == 6.0 for m in enc.seen_months)

        # Preset name string path (analytic encoder, no downloads)
        lon2, lat2, emb2 = encode_grid('xyz', width=4, height=3)
        assert lon2.shape == (3, 4)
        assert lat2.shape == (3, 4)
        assert emb2.shape == (3, 4, 3)

    def test_invalid_batch_size(self) -> None:
        with pytest.raises(ValueError, match='batch_size must be a positive int'):
            encode_grid('xyz', width=2, height=2, batch_size=0)
        with pytest.raises(ValueError, match='batch_size must be a positive int'):
            encode_grid('xyz', width=2, height=2, batch_size=True)


class TestEmbeddingsToRgbAndPlot:
    def test_embeddings_to_rgb_standard_and_edge_cases(self) -> None:
        _, _, emb = encode_grid('sincos', width=6, height=4)
        rgb = embeddings_to_rgb(emb)
        assert rgb.shape == (4, 6, 3)
        assert rgb.dtype == np.float32
        assert float(rgb.min()) >= 0.0
        assert float(rgb.max()) <= 1.0

        # Fewer than 3 dimensions (padded with zeros) and constant channel
        low_dim = np.ones((3, 4, 2), dtype=np.float32)
        low_dim[:, :, 0] = np.arange(12, dtype=np.float32).reshape(3, 4)
        rgb_low = embeddings_to_rgb(low_dim, components=(0, 1, 5))
        assert rgb_low.shape == (3, 4, 3)
        np.testing.assert_allclose(rgb_low[:, :, 2], 0.0)

        # All-NaN grid -> all-zero RGB without error
        all_nan = np.full((2, 3, 4), np.nan, dtype=np.float32)
        rgb_nan = embeddings_to_rgb(all_nan)
        np.testing.assert_array_equal(rgb_nan, np.zeros((2, 3, 3), dtype=np.float32))

    def test_embeddings_to_rgb_validation(self) -> None:
        with pytest.raises(ValueError, match=r'\(H, W, D\)'):
            embeddings_to_rgb(np.zeros((4, 4), dtype=np.float32))
        with pytest.raises(ValueError, match=r'\(H, W, D\)'):
            embeddings_to_rgb(np.zeros((2, 2, 0), dtype=np.float32))
        with pytest.raises(ValueError, match='3 non-negative indices'):
            embeddings_to_rgb(np.zeros((2, 2, 4)), components=(0, 1))
        with pytest.raises(ValueError, match='3 non-negative indices'):
            embeddings_to_rgb(np.zeros((2, 2, 4)), components=(-1, 0, 1))
        with pytest.raises(ValueError, match='percentile'):
            embeddings_to_rgb(np.zeros((2, 2, 4)), percentile=(90.0, 10.0))
        with pytest.raises(ValueError, match='percentile'):
            embeddings_to_rgb(np.zeros((2, 2, 4)), percentile=(-1.0, 99.0))

    def test_plot_embedding_map_and_missing_matplotlib(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        figs = _install_fake_matplotlib(monkeypatch)
        lon, lat, emb = encode_grid('xyz', width=5, height=4)
        out_png = tmp_path / 'map.png'

        fig, ax = plot_embedding_map(
            lon, lat, emb, title='XYZ Map', figsize=(8.0, 4.0), save_path=out_png
        )
        assert fig is figs[0]
        assert fig.figsize == (8.0, 4.0)
        assert ax.title == 'XYZ Map'
        assert ax.xlabel == 'Longitude (°)'
        assert ax.ylabel == 'Latitude (°)'
        assert len(ax.imshow_calls) == 1
        img, kwargs = ax.imshow_calls[0]
        assert img.shape == (4, 5, 3)
        assert kwargs['origin'] == 'upper'
        assert kwargs['extent'] == pytest.approx((-180.0, 180.0, -90.0, 90.0))
        assert fig.saved[0][0] == str(out_png)

        # Reuse existing Axes + 1x1 grid edge case
        lon1, lat1, emb1 = encode_grid('xyz', width=1, height=1)
        fig2, ax2 = plot_embedding_map(lon1, lat1, emb1, ax=ax)
        assert fig2 is fig
        assert ax2 is ax
        assert len(ax.imshow_calls) == 2

        # Shape validation
        with pytest.raises(ValueError, match='matching 2D grids'):
            plot_embedding_map(lon[0], lat, emb)
        with pytest.raises(ValueError, match='does not match'):
            plot_embedding_map(lon, lat, emb[:, :-1, :])

        # Missing matplotlib raises DependencyNotFoundError
        monkeypatch.delitem(sys.modules, 'matplotlib', raising=False)
        monkeypatch.delitem(sys.modules, 'matplotlib.pyplot', raising=False)
        real_import = importlib.import_module

        def _block_mpl(name: str, package: str | None = None) -> Any:
            if name == 'matplotlib' or name.startswith('matplotlib.'):
                raise ModuleNotFoundError("No module named 'matplotlib'")
            return real_import(name, package)

        monkeypatch.setattr(importlib, 'import_module', _block_mpl)
        with pytest.raises(DependencyNotFoundError, match='matplotlib'):
            plot_embedding_map(lon, lat, emb)

    def test_make_land_mask_and_masked_area_weighted_pca(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # St. Louis (land), central Pacific (ocean), South Pole (Antarctica)
        lon = np.array([-90.2, -150.0, 0.0], dtype=np.float64)
        lat = np.array([38.6, 0.0, -82.0], dtype=np.float64)
        m_hab = make_land_mask(lon, lat)
        assert m_hab.tolist() == [True, False, False]
        m_all = make_land_mask(lon, lat, include_antarctica=True)
        assert m_all.tolist() == [True, False, True]

        with pytest.raises(ValueError, match='matching shapes'):
            make_land_mask(lon, lat[:2])

        # Masked + weighted embeddings_to_rgb fills masked pixels with background
        lon_g, lat_g, emb = encode_grid('sincos', step_deg=30.0)
        mask = make_land_mask(lon_g, lat_g)
        weights = np.clip(np.cos(np.deg2rad(lat_g)), 1e-6, None)
        bg = (0.9, 0.95, 1.0)
        rgb = embeddings_to_rgb(emb, mask=mask, weights=weights, background=bg)
        assert rgb.shape == (*lon_g.shape, 3)
        np.testing.assert_allclose(rgb[~mask], np.broadcast_to(bg, rgb[~mask].shape))

        # Validation branches for mask, weights, background
        with pytest.raises(ValueError, match='background'):
            embeddings_to_rgb(emb, background=(0.0, 1.5, 0.0))
        with pytest.raises(ValueError, match='mask shape'):
            embeddings_to_rgb(emb, mask=np.ones((2, 2), dtype=bool))
        with pytest.raises(ValueError, match='weights shape'):
            embeddings_to_rgb(emb, weights=np.ones((2, 2), dtype=np.float64))
        with pytest.raises(ValueError, match='finite non-negative'):
            embeddings_to_rgb(emb, weights=np.full(lon_g.shape, -1.0))
        with pytest.raises(ValueError, match='positive total weight'):
            embeddings_to_rgb(emb, weights=np.zeros(lon_g.shape))

        # plot_embedding_map with land_only=True, custom mask, area_weighted=False
        _install_fake_matplotlib(monkeypatch)
        _, ax = plot_embedding_map(lon_g, lat_g, emb, land_only=True)
        img, _ = ax.imshow_calls[0]
        np.testing.assert_allclose(
            img[~mask], np.broadcast_to((0.96, 0.97, 0.99), img[~mask].shape)
        )

        custom_mask = np.ones_like(mask, dtype=bool)
        custom_mask[0, :] = False
        _, ax2 = plot_embedding_map(
            lon_g, lat_g, emb, mask=custom_mask, land_only=True, area_weighted=False
        )
        img2, _ = ax2.imshow_calls[0]
        combined = mask & custom_mask
        np.testing.assert_allclose(
            img2[~combined], np.broadcast_to((0.96, 0.97, 0.99), img2[~combined].shape)
        )

        with pytest.raises(ValueError, match='mask shape'):
            plot_embedding_map(
                lon_g, lat_g, emb, mask=np.ones((2, 2), dtype=bool), land_only=True
            )

    def test_top_level_viz_exports(self) -> None:
        assert neuralftw.make_coord_grid is make_coord_grid
        assert neuralftw.make_land_mask is make_land_mask
        assert neuralftw.encode_grid is encode_grid
        assert neuralftw.embeddings_to_rgb is embeddings_to_rgb
        assert neuralftw.plot_embedding_map is plot_embedding_map
