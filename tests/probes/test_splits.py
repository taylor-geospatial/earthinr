# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

import io
import os
import pathlib
import urllib.request
import zipfile
from typing import Self

import numpy as np
import pytest

from neuralftw._errors import DatasetNotFoundError
from neuralftw.probes.splits import (
    RANDOM_FOLD_ALGORITHM,
    buffered_exclusion_mask,
    load_geonames_cities,
    random_fold_ids,
    spatial_fold_ids,
    urban_rural_tags,
)


class TestRandomFoldIds:
    def test_deterministic_and_shuffles_seed_zero(self) -> None:
        assert RANDOM_FOLD_ALGORITHM == 'torch-randperm-v1'
        f0 = random_fold_ids(50, folds=5, seed=0)
        f0_repeat = random_fold_ids(50, folds=5, seed=0)
        f1 = random_fold_ids(50, folds=5, seed=1)
        np.testing.assert_array_equal(f0, f0_repeat)
        assert not np.array_equal(f0, np.arange(50) % 5)
        assert not np.array_equal(f0, f1)
        assert f0.dtype == np.int64
        counts = np.bincount(f0, minlength=5)
        np.testing.assert_array_equal(counts, np.full(5, 10))

    def test_empty_input(self) -> None:
        out = random_fold_ids(0, folds=5, seed=0)
        assert out.shape == (0,)
        assert out.dtype == np.int64

    def test_invalid_args(self) -> None:
        with pytest.raises(ValueError, match='non-negative'):
            random_fold_ids(-1)
        with pytest.raises(ValueError, match='at least 2'):
            random_fold_ids(10, folds=1)


class TestSpatialFoldIds:
    def test_same_cell_shares_fold_and_matches_reference(self) -> None:
        lat = np.array([1.0, 2.0, 15.0, 16.0, 35.0, -25.0, -45.0, 55.0])
        lon = np.array([1.0, 2.0, 15.0, 16.0, 35.0, -25.0, -45.0, 55.0])
        folds = spatial_fold_ids(lat, lon, folds=3, cell_deg=10.0, seed=4)
        assert folds[0] == folds[1]
        assert folds[2] == folds[3]
        assert folds.dtype == np.int64
        # Deterministic across calls
        np.testing.assert_array_equal(
            folds, spatial_fold_ids(lat, lon, folds=3, cell_deg=10.0, seed=4)
        )

    def test_empty_input(self) -> None:
        out = spatial_fold_ids(np.array([]), np.array([]), folds=5, cell_deg=10.0)
        assert out.shape == (0,)

    def test_invalid_args(self) -> None:
        with pytest.raises(ValueError, match='at least 2'):
            spatial_fold_ids(np.array([0.0]), np.array([0.0]), folds=1)
        with pytest.raises(ValueError, match='positive'):
            spatial_fold_ids(np.array([0.0]), np.array([0.0]), cell_deg=0.0)
        with pytest.raises(ValueError, match='1D arrays'):
            spatial_fold_ids(np.zeros((2, 1)), np.zeros((2, 1)))
        with pytest.raises(ValueError, match='1D arrays'):
            spatial_fold_ids(np.array([0.0, 1.0]), np.array([0.0]))


class TestLoadGeonamesCities:
    def _sample_lines(self) -> str:
        return (
            '1\tCityA\tCityA\t\t37.7749\t-122.4194\tP\tPPL\tUS\n'
            '\n'
            '2\tCityB\tCityB\t\t51.5074\t-0.1278\tP\tPPL\tGB\n'
        )

    def test_load_existing_file(self, tmp_path: pathlib.Path) -> None:
        txt = tmp_path / 'cities15000.txt'
        txt.write_text(self._sample_lines(), encoding='utf-8')
        lats, lons = load_geonames_cities(txt)
        np.testing.assert_allclose(lats, [37.7749, 51.5074])
        np.testing.assert_allclose(lons, [-122.4194, -0.1278])

    def test_missing_file_raises_without_download(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        missing = tmp_path / 'missing.txt'
        with pytest.raises(DatasetNotFoundError, match=r'cities15000\.txt not found'):
            load_geonames_cities(missing, download=False)

        monkeypatch.setattr(os.path, 'expanduser', lambda _p: str(tmp_path))
        with pytest.raises(DatasetNotFoundError, match=r'cities15000\.txt not found'):
            load_geonames_cities(None, download=False)

    def test_download_zip_offline_via_monkeypatch(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as zf:
            zf.writestr('cities15000.txt', self._sample_lines())
        payload = buf.getvalue()

        class _FakeResp:
            def __enter__(self) -> Self:
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self) -> bytes:
                return payload

        monkeypatch.setattr(
            urllib.request, 'urlopen', lambda _url, timeout=120: _FakeResp()
        )
        nested = tmp_path / 'sub' / 'cities15000.txt'
        lats, lons = load_geonames_cities(nested, download=True)
        assert nested.is_file()
        np.testing.assert_allclose(lats, [37.7749, 51.5074])
        np.testing.assert_allclose(lons, [-122.4194, -0.1278])

        # Also test relative filename with empty dirname
        monkeypatch.chdir(tmp_path)
        lats2, lons2 = load_geonames_cities('local_cities.txt', download=True)
        np.testing.assert_allclose(lats2, [37.7749, 51.5074])
        np.testing.assert_allclose(lons2, [-122.4194, -0.1278])


class TestUrbanRuralTags:
    def test_urban_buffer_rural_classification(self) -> None:
        c_lat = np.array([0.0])
        c_lon = np.array([0.0])
        # 1 deg longitude at equator ~ 111.195 km
        # 0.05 deg ~ 5.56 km (urban <= 10)
        # 0.25 deg ~ 27.8 km (buffer 10..50 -> -1)
        # 1.00 deg ~ 111.2 km (rural > 50)
        lat = np.array([0.0, 0.0, 0.0])
        lon = np.array([0.05, 0.25, 1.0])
        tags = urban_rural_tags(lat, lon, c_lat, c_lon, urban_km=10.0, rural_km=50.0)
        np.testing.assert_array_equal(tags, np.array([0, -1, 1], dtype=np.int64))

    def test_empty_queries(self) -> None:
        tags = urban_rural_tags(
            np.array([]), np.array([]), np.array([0.0]), np.array([0.0])
        )
        assert tags.shape == (0,)

    def test_invalid_args(self) -> None:
        with pytest.raises(ValueError, match='non-negative'):
            urban_rural_tags(
                np.array([0.0]),
                np.array([0.0]),
                np.array([0.0]),
                np.array([0.0]),
                urban_km=-1.0,
            )
        with pytest.raises(ValueError, match='strictly greater'):
            urban_rural_tags(
                np.array([0.0]),
                np.array([0.0]),
                np.array([0.0]),
                np.array([0.0]),
                urban_km=20.0,
                rural_km=20.0,
            )
        with pytest.raises(ValueError, match='1D arrays'):
            urban_rural_tags(
                np.zeros((2, 1)), np.zeros((2, 1)), np.array([0.0]), np.array([0.0])
            )
        with pytest.raises(ValueError, match='1D arrays'):
            urban_rural_tags(
                np.array([0.0, 1.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])
            )
        with pytest.raises(ValueError, match='non-empty 1D arrays'):
            urban_rural_tags(
                np.array([0.0]), np.array([0.0]), np.array([]), np.array([])
            )
        with pytest.raises(ValueError, match='non-empty 1D arrays'):
            urban_rural_tags(
                np.array([0.0]), np.array([0.0]), np.array([0.0, 1.0]), np.array([0.0])
            )
        with pytest.raises(ValueError, match='non-empty 1D arrays'):
            urban_rural_tags(
                np.array([0.0]), np.array([0.0]), np.zeros((1, 1)), np.zeros((1, 1))
            )


class TestBufferedExclusionMask:
    def test_buffer_filtering_and_zero_radius(self) -> None:
        lat = np.array([0.0, 0.0, 0.0, 0.0])
        lon = np.array([0.0, 0.1, 1.0, 2.0])
        train_idx = np.array([1, 2, 3])
        test_idx = np.array([0])
        # lon=0.1 is ~11.1 km, lon=1.0 is ~111 km, lon=2.0 is ~222 km
        mask_50 = buffered_exclusion_mask(lat, lon, train_idx, test_idx, radius_km=50.0)
        np.testing.assert_array_equal(mask_50, np.array([False, True, True]))

        mask_0 = buffered_exclusion_mask(lat, lon, train_idx, test_idx, radius_km=0.0)
        np.testing.assert_array_equal(mask_0, np.array([True, True, True]))

        mask_empty_te = buffered_exclusion_mask(
            lat, lon, train_idx, np.array([], dtype=np.int64), radius_km=50.0
        )
        np.testing.assert_array_equal(mask_empty_te, np.array([True, True, True]))

        mask_empty_tr = buffered_exclusion_mask(
            lat, lon, np.array([], dtype=np.int64), test_idx, radius_km=50.0
        )
        assert mask_empty_tr.shape == (0,)

    def test_invalid_args(self) -> None:
        with pytest.raises(ValueError, match='non-negative'):
            buffered_exclusion_mask(
                np.array([0.0]),
                np.array([0.0]),
                np.array([0]),
                np.array([0]),
                radius_km=-5.0,
            )
        with pytest.raises(ValueError, match='1D arrays'):
            buffered_exclusion_mask(
                np.zeros((2, 1)),
                np.zeros((2, 1)),
                np.array([0]),
                np.array([1]),
                radius_km=10.0,
            )
        with pytest.raises(ValueError, match='1D arrays'):
            buffered_exclusion_mask(
                np.array([0.0, 1.0]),
                np.array([0.0]),
                np.array([0]),
                np.array([0]),
                radius_km=10.0,
            )
