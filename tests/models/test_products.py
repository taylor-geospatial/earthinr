# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import earthinr.models.products as products_mod
from earthinr._errors import DependencyNotFoundError
from earthinr.models.products import (
    AEF_DIM,
    AEF_LAT_ORIGIN,
    AEF_LON_ORIGIN,
    AEF_NODATA,
    AEF_PIXEL_DEG,
    TESSERA_CHUNK_PX,
    TESSERA_DIM,
    AlphaEarthLocationEncoder,
    TESSERALocationEncoder,
    _decode_blosc_chunk,
    _http_session,
    _open_aef_mosaic,
    _open_tessera_zone,
    _parse_shard_index,
    _project_to_utm,
    _range_read,
    _read_inner_chunk,
    aef_lonlat_to_pixel,
    dequantize_aef,
    impute_nan_rows,
    resolve_years,
    utm_zone,
)


class TestProductHelpers:
    def test_impute_nan_rows(self) -> None:
        clean = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        assert np.array_equal(impute_nan_rows(clean), clean)

        with_nan = np.array(
            [[1.0, np.nan, np.nan], [3.0, 4.0, np.nan], [np.nan, np.nan, np.nan]],
            dtype=np.float32,
        )
        imputed = impute_nan_rows(with_nan)
        assert imputed.dtype == np.float32
        assert np.allclose(imputed[0], [1.0, 4.0, 0.0])
        assert np.allclose(imputed[1], [3.0, 4.0, 0.0])
        assert np.allclose(imputed[2], [2.0, 4.0, 0.0])

    def test_dequantize_aef(self) -> None:
        raw = np.zeros((3, AEF_DIM), dtype=np.float32)
        raw[0, 0] = 127.0
        raw[0, 1] = -64.0
        raw[1, :] = 0.0
        raw[2, :] = np.nan
        deq = dequantize_aef(raw)
        assert deq.dtype == np.float32
        assert np.isclose(np.linalg.norm(deq[0]), 1.0, atol=1e-6)
        assert deq[0, 0] > 0.0
        assert deq[0, 1] < 0.0
        assert np.all(deq[1] == 0.0)
        assert np.isnan(deq[2]).all()

    def test_aef_lonlat_to_pixel_and_utm_zone(self) -> None:
        lon = np.array([AEF_LON_ORIGIN + 0.5 * AEF_PIXEL_DEG])
        lat = np.array([AEF_LAT_ORIGIN - 0.5 * AEF_PIXEL_DEG])
        row, col = aef_lonlat_to_pixel(lon, lat)
        assert np.allclose(row, [0.0], atol=1e-9)
        assert np.allclose(col, [0.0], atol=1e-9)

        zones = utm_zone(np.array([-180.0, -122.4, 0.0, 179.9, 180.0]))
        assert zones.tolist() == [1, 10, 31, 60, 60]

    def test_resolve_years(self) -> None:
        assert resolve_years(3, None, default_year=2021).tolist() == [2021, 2021, 2021]
        assert resolve_years(
            3, np.array([2019.0, 2020.0, 2022.0]), fixed_year=2023
        ).tolist() == [2023, 2023, 2023]
        assert resolve_years(
            4, np.array([2010.0, np.nan, 2022.0, 2030.0]), default_year=2021
        ).tolist() == [2017, 2021, 2022, 2025]

    def test_optional_dep_lazy_wrappers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeConfig:
            def set(self, cfg: dict[str, Any]) -> None:
                self.cfg = cfg

        class _FakeGroup:
            def __init__(self) -> None:
                self.attrs = {
                    'spatial:transform': [10.0, 0.0, 1000.0, 0.0, -10.0, 5000.0],
                    'proj:code': 'EPSG:32610',
                }

            def __getitem__(self, key: str) -> Any:
                return types.SimpleNamespace(shape=(9, 128, 100, 200))

        fake_zarr = types.SimpleNamespace(
            config=_FakeConfig(),
            storage=types.SimpleNamespace(
                FsspecStore=types.SimpleNamespace(
                    from_url=lambda url, storage_options: (url, storage_options)
                )
            ),
            open_group=lambda store, mode: _FakeGroup(),
        )
        fake_pyproj = types.SimpleNamespace(
            Transformer=types.SimpleNamespace(
                from_crs=lambda src, dst, always_xy: types.SimpleNamespace(
                    transform=lambda lon, lat: (lon * 100.0, lat * 100.0)
                )
            )
        )

        class _FakeResp:
            def __init__(self, status_code: int, content: bytes) -> None:
                self.status_code = status_code
                self.content = content

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    raise OSError('http error')

        class _FakeSession:
            def get(
                self, url: str, headers: dict[str, str], timeout: float
            ) -> _FakeResp:
                if 'missing' in url:
                    return _FakeResp(404, b'')
                return _FakeResp(200, b'abcd')

        fake_requests = types.SimpleNamespace(Session=_FakeSession)
        fake_numcodecs = types.SimpleNamespace(
            Blosc=lambda: types.SimpleNamespace(decode=lambda raw: raw)
        )
        modules = {
            's3fs': types.SimpleNamespace(),
            'zarr': fake_zarr,
            'pyproj': fake_pyproj,
            'requests': fake_requests,
            'numcodecs': fake_numcodecs,
        }
        monkeypatch.setattr(
            products_mod, 'lazy_import', lambda name, extra=None: modules[name]
        )
        monkeypatch.setattr(products_mod, '_TLS', types.SimpleNamespace())

        arr = _open_aef_mosaic()
        assert arr.shape == (9, 128, 100, 200)

        meta = _open_tessera_zone(10)
        assert meta['epsg'] == 32610
        assert meta['transform'] == (10.0, 1000.0, -10.0, 5000.0)
        assert meta['ny'] == 100
        assert meta['nx'] == 200
        assert meta['emb_url'].startswith('https://s3.us-west-2.amazonaws.com/')

        e, n = _project_to_utm(32610, np.array([1.0]), np.array([2.0]))
        assert e.tolist() == [100.0]
        assert n.tolist() == [200.0]

        s1 = _http_session()
        s2 = _http_session()
        assert s1 is s2
        assert _range_read('https://example.org/ok', '0-3') == b'abcd'
        assert _range_read('https://example.org/missing', '0-3') is None

        decoded = _decode_blosc_chunk(
            np.arange(4, dtype=np.int8).tobytes(), (2, 2), 'int8'
        )
        assert decoded.shape == (2, 2)

    def test_missing_optional_dep_raises_helpful_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _missing(name: str, extra: str | None = None) -> Any:
            raise DependencyNotFoundError(f'pip install earthinr[{extra}]')

        monkeypatch.setattr(products_mod, 'lazy_import', _missing)
        with pytest.raises(DependencyNotFoundError, match=r'earthinr\[products\]'):
            _open_aef_mosaic()

    def test_shard_index_and_inner_chunk_missing_branches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert _parse_shard_index(None) is None
        assert _parse_shard_index(b'short') is None

        idx = np.full(
            (products_mod._TESSERA_CHUNKS_PER_SIDE**2, 2),
            products_mod._TESSERA_MISSING,
            dtype='<u8',
        )
        assert _read_inner_chunk('u', idx, 0, 0, (2,), 'int8') is None

        idx[0] = (0, 4)
        monkeypatch.setattr(products_mod, '_range_read', lambda url, br: None)
        assert _read_inner_chunk('u', idx, 0, 0, (2,), 'int8') is None


class TestAlphaEarthLocationEncoder:
    def test_encode_forward_cache_and_imputation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Synthetic [T=9, B=64, H=4, W=4] int8 mosaic.
        mosaic = np.full((9, AEF_DIM, 4, 4), AEF_NODATA, dtype=np.int8)
        mosaic[4, :, 0, 0] = 64  # year 2021 valid pixel
        mosaic[4, 0, 0, 0] = -120
        mosaic[5, :, 1, 1] = 32  # year 2022 valid pixel

        open_calls = 0

        def _fake_open(url: str, *, anon: bool = True) -> np.ndarray:
            nonlocal open_calls
            open_calls += 1
            return mosaic

        monkeypatch.setattr(products_mod, '_open_aef_mosaic', _fake_open)

        lon = np.array(
            [
                AEF_LON_ORIGIN + 0.5 * AEF_PIXEL_DEG,  # col 0 (valid)
                AEF_LON_ORIGIN + 1.5 * AEF_PIXEL_DEG,  # col 1 (valid for 2022)
                AEF_LON_ORIGIN + 2.5 * AEF_PIXEL_DEG,  # col 2 (nodata)
            ]
        )
        lat = np.array(
            [
                AEF_LAT_ORIGIN - 0.5 * AEF_PIXEL_DEG,  # row 0
                AEF_LAT_ORIGIN - 1.5 * AEF_PIXEL_DEG,  # row 1
                AEF_LAT_ORIGIN - 2.5 * AEF_PIXEL_DEG,  # row 2
            ]
        )
        years = np.array([2021.0, 2022.0, 2021.0])

        enc = AlphaEarthLocationEncoder(impute_nan=False, cache_dir=tmp_path, workers=2)
        assert enc.name == 'aef'
        assert enc.embed_dim == 64

        out = enc.encode(lon, lat, year=years)
        assert out.shape == (3, 64)
        assert np.isclose(np.linalg.norm(out[0]), 1.0, atol=1e-5)
        assert np.isclose(np.linalg.norm(out[1]), 1.0, atol=1e-5)
        assert np.isnan(out[2]).all()
        assert open_calls == 1

        # Second call hits disk cache (and does not reopen mosaic).
        out_cached = enc.encode(lon, lat, year=years)
        assert np.allclose(out_cached[:2], out[:2])
        assert np.isnan(out_cached[2]).all()

        # Re-querying uncached coordinates reuses the cached array handle.
        enc_no_cache = AlphaEarthLocationEncoder(impute_nan=True, workers=2)
        emb_imputed = enc_no_cache.encode(lon, lat, year=years)
        assert np.isfinite(emb_imputed).all()
        assert np.allclose(emb_imputed[2], 0.5 * (emb_imputed[0] + emb_imputed[1]))
        # Second call reuses self._array without reopening.
        _ = enc_no_cache.encode(lon[:1], lat[:1])

        # Empty input returns (0, 0) cleanly even with impute_nan=True.
        empty = enc_no_cache.encode(np.array([]), np.array([]))
        assert empty.shape == (0, 0)

        # Tensor forward path works with and without year.
        coords_t = torch.tensor(np.stack((lon, lat), axis=-1), dtype=torch.float32)
        out_t = enc_no_cache(coords_t, year=torch.tensor(years, dtype=torch.float32))
        assert out_t.shape == (3, 64)
        assert torch.isfinite(out_t).all()
        out_t_noyear = enc_no_cache(coords_t)
        assert out_t_noyear.shape == (3, 64)


class TestTESSERALocationEncoder:
    def test_encode_forward_cache_and_error_handling(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Build one valid chunk (cy=0, cx=0), one missing-offset chunk (cy=0, cx=1),
        # one error-raising chunk (cy=1, cx=0), and one 404 shard (sy=1, sx=0).
        emb_block = np.zeros(
            (TESSERA_DIM, TESSERA_CHUNK_PX, TESSERA_CHUNK_PX), dtype=np.int8
        )
        emb_block[:, 2, 3] = 10
        emb_block[:, 4, 5] = 0  # all-zero vector -> unembedded -> NaN
        scale_block = np.zeros((TESSERA_CHUNK_PX, TESSERA_CHUNK_PX), dtype='<f4')
        scale_block[2, 3] = 0.5
        scale_block[4, 5] = 0.5

        emb_raw = emb_block.tobytes()
        scale_raw = scale_block.tobytes()

        n_entries = products_mod._TESSERA_CHUNKS_PER_SIDE**2
        emb_idx = np.full((n_entries, 2), products_mod._TESSERA_MISSING, dtype='<u8')
        scale_idx = np.full((n_entries, 2), products_mod._TESSERA_MISSING, dtype='<u8')
        emb_idx[0] = (0, len(emb_raw))
        scale_idx[0] = (0, len(scale_raw))
        # Chunk (1, 0) points to a range that will trigger an OSError.
        bad_slot = products_mod._TESSERA_CHUNKS_PER_SIDE
        emb_idx[bad_slot] = (999, 16)
        scale_idx[bad_slot] = (999, 16)

        emb_idx_bytes = emb_idx.tobytes() + b'\x00\x00\x00\x00'
        scale_idx_bytes = scale_idx.tobytes() + b'\x00\x00\x00\x00'

        def _fake_zone(zone: int, *, url: str) -> dict[str, Any]:
            return {
                'epsg': 32600 + zone,
                'transform': (10.0, 0.0, -10.0, 100000.0),
                'ny': 10000,
                'nx': 10000,
                'emb_url': f'https://fake/utm{zone:02d}/embeddings/c',
                'scales_url': f'https://fake/utm{zone:02d}/scales/c',
            }

        # Map lon/lat directly into deterministic (east, north) grid coords.
        def _fake_project(
            epsg: int, lon: np.ndarray, lat: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            # east = lon_offset * 10, north = 100000 - row * 10
            east = (lon - np.floor(lon)) * 1000.0 * 10.0
            north = 100000.0 - lat * 10.0
            return east, north

        def _fake_range_read(url: str, byte_range: str) -> bytes | None:
            if url.endswith('/1/0'):
                return None  # missing shard (404)
            if url.endswith('/2/0'):
                raise OSError('transient shard failure')
            if byte_range.startswith('-'):
                return emb_idx_bytes if '/embeddings/' in url else scale_idx_bytes
            if byte_range.startswith('999-'):
                raise OSError('chunk read failure')
            return emb_raw if '/embeddings/' in url else scale_raw

        monkeypatch.setattr(products_mod, '_open_tessera_zone', _fake_zone)
        monkeypatch.setattr(products_mod, '_project_to_utm', _fake_project)
        monkeypatch.setattr(products_mod, '_range_read', _fake_range_read)
        monkeypatch.setattr(
            products_mod,
            '_decode_blosc_chunk',
            lambda raw, shape, dtype: np.frombuffer(raw, dtype=dtype).reshape(shape),
        )

        # Points covering:
        # 0: valid pixel (row=2, col=3) -> 10 * 0.5 = 5.0
        # 1: duplicate of point 0 (tests deduplication)
        # 2: zero int8 vector (row=4, col=5) -> NaN
        # 3: missing inner chunk (cy=0, cx=1 -> row=2, col=35) -> NaN
        # 4: failing inner chunk (cy=1, cx=0 -> row=33, col=1) -> NaN
        # 5: 404 shard (sy=1, sx=0 -> row=4100, col=1) -> NaN
        # 6: failing shard index (sy=2, sx=0 -> row=8200, col=1) -> NaN
        # 7: off-grid pixel (row=-5) -> NaN
        lon = np.array([0.0035, 0.0035, 0.0055, 0.0355, 0.0015, 0.0015, 0.0015, 0.0015])
        lat = np.array([2.5, 2.5, 4.5, 2.5, 33.5, 4100.5, 8200.5, -5.0])

        enc = TESSERALocationEncoder(impute_nan=False, cache_dir=tmp_path, workers=2)
        assert enc.name == 'tessera'
        assert enc.embed_dim == 128

        out = enc.encode(lon, lat)
        assert out.shape == (8, 128)
        assert np.allclose(out[0], 5.0)
        assert np.allclose(out[1], 5.0)
        assert np.isnan(out[2:]).all()

        # Disk cache hit.
        out_cached = enc.encode(lon, lat)
        assert np.allclose(out_cached[:2], 5.0)
        assert np.isnan(out_cached[2:]).all()

        # Imputation + empty + tensor forward.
        enc_imp = TESSERALocationEncoder(impute_nan=True, workers=2)
        out_imp = enc_imp.encode(lon, lat)
        assert np.allclose(out_imp, 5.0)
        assert enc_imp.encode(np.array([]), np.array([])).shape == (0, 0)

        coords_t = torch.tensor(
            np.stack((lon[:2], lat[:2]), axis=-1), dtype=torch.float32
        )
        out_t = enc_imp(coords_t, year=torch.tensor([2021.0, 2021.0]))
        assert out_t.shape == (2, 128)
        assert torch.allclose(out_t, torch.full((2, 128), 5.0))
        assert enc_imp(coords_t).shape == (2, 128)
