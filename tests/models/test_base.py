# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

from typing import Any, cast

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch import Tensor

import earthinr.models._base as base_mod
from earthinr.models import LocationEncoder, Weights, WeightsEnum


class _DummyWeights(WeightsEnum):
    HF_MODEL = Weights(
        repo_id='org/model',
        filename='model.safetensors',
        revision='main',
        meta={'embed_dim': 4},
    )
    URL_ONLY = Weights(url='https://example.org/model.pt')


class _DummyEncoder(LocationEncoder):
    name = 'dummy'
    embed_dim = 4

    def __init__(
        self, device: str | torch.device = 'cpu', batch_size: int = 8192
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        self.batch_sizes_seen: list[int] = []
        self.years_seen: list[Tensor | None] = []
        self.months_seen: list[Tensor | None] = []

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        self.batch_sizes_seen.append(coords.shape[0])
        self.years_seen.append(year)
        self.months_seen.append(month)
        zeros = torch.zeros(
            (coords.shape[0], 1), dtype=coords.dtype, device=coords.device
        )
        third = zeros if year is None else year.unsqueeze(-1)
        fourth = zeros if month is None else month.unsqueeze(-1)
        return torch.cat((coords, third, fourth), dim=-1)


class TestWeights:
    def test_init_and_repr(self) -> None:
        w = Weights(repo_id='org/repo', filename='w.safetensors', meta={'dim': 64})
        assert w.repo_id == 'org/repo'
        assert w.filename == 'w.safetensors'
        assert w.url is None
        assert w.revision is None
        assert w.meta == {'dim': 64}
        assert 'org/repo' in repr(w)

    def test_requires_repo_id_or_url(self) -> None:
        with pytest.raises(ValueError, match='repo_id or url'):
            Weights()


class TestWeightsEnum:
    def test_properties(self) -> None:
        w = _DummyWeights.HF_MODEL
        assert w.repo_id == 'org/model'
        assert w.filename == 'model.safetensors'
        assert w.url is None
        assert w.revision == 'main'
        assert w.meta == {'embed_dim': 4}

        u = _DummyWeights.URL_ONLY
        assert u.url == 'https://example.org/model.pt'
        assert u.repo_id is None
        assert u.meta == {}

    def test_verify(self) -> None:
        assert _DummyWeights.verify(_DummyWeights.HF_MODEL) is _DummyWeights.HF_MODEL
        assert _DummyWeights.verify('HF_MODEL') is _DummyWeights.HF_MODEL
        with pytest.raises(ValueError, match='Invalid weights'):
            _DummyWeights.verify('UNKNOWN')
        with pytest.raises(ValueError, match='Invalid weights'):
            _DummyWeights.verify(123)

    def test_download_hf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, str, str | None]] = []

        def _fake_download(
            repo_id: str, filename: str, *, revision: str | None = None
        ) -> str:
            calls.append((repo_id, filename, revision))
            return '/tmp/fake.safetensors'

        monkeypatch.setattr(base_mod, 'hf_download', _fake_download)
        path = _DummyWeights.HF_MODEL.download()
        assert path == '/tmp/fake.safetensors'
        assert calls == [('org/model', 'model.safetensors', 'main')]

    def test_download_missing_hf_fields_raises(self) -> None:
        with pytest.raises(ValueError, match='both repo_id and filename'):
            _DummyWeights.URL_ONLY.download()


class TestLocationEncoder:
    def test_invalid_batch_size(self) -> None:
        for bad_val in (0, -4):
            with pytest.raises(ValueError, match='batch_size'):
                _DummyEncoder(batch_size=bad_val)
        for bad_type in (True, 1.5):
            with pytest.raises(TypeError, match='batch_size'):
                _DummyEncoder(batch_size=cast(Any, bad_type))

    def test_encode_batching_year_and_month(self) -> None:
        enc = _DummyEncoder(device='cpu', batch_size=2)
        lon = np.array([-120.0, 0.0, 45.0, 90.0, 180.0], dtype=np.float32)
        lat = np.array([30.0, 0.0, -45.0, 60.0, -10.0], dtype=np.float32)

        out = enc.encode(lon, lat)
        assert out.shape == (5, 4)
        assert out.dtype == np.float32
        assert enc.batch_sizes_seen == [2, 2, 1]
        assert all(y is None for y in enc.years_seen)
        assert all(m is None for m in enc.months_seen)
        assert np.allclose(out[:, 0], lon)
        assert np.allclose(out[:, 1], lat)
        assert np.allclose(out[:, 2], 0.0)
        assert np.allclose(out[:, 3], 0.0)

        enc.batch_sizes_seen.clear()
        enc.years_seen.clear()
        enc.months_seen.clear()
        years = np.array([2020.0, 2021.0, 2022.0, 2023.0, 2024.0], dtype=np.float64)
        months = np.array([3.0, 6.0, 9.0, 12.0, 1.0], dtype=np.float64)
        out_ym = enc.encode(lon, lat, year=years, month=months)
        assert out_ym.shape == (5, 4)
        assert out_ym.dtype == np.float32
        assert enc.batch_sizes_seen == [2, 2, 1]
        assert np.allclose(out_ym[:, 2], years)
        assert np.allclose(out_ym[:, 3], months)

    def test_encode_empty(self) -> None:
        enc = _DummyEncoder()
        empty = np.array([], dtype=np.float64)
        out = enc.encode(empty, empty)
        assert out.shape == (0, 0)
        assert out.dtype == np.float32

    def test_encode_validation_errors(self) -> None:
        enc = _DummyEncoder()
        with pytest.raises(ValueError, match='1D arrays'):
            enc.encode(np.zeros((2, 1)), np.zeros((2, 1)))
        with pytest.raises(ValueError, match='1D arrays'):
            enc.encode(np.zeros(2), np.zeros((2, 1)))
        with pytest.raises(ValueError, match='same length'):
            enc.encode(np.zeros(2), np.zeros(3))
        with pytest.raises(ValueError, match='year must be a 1D array'):
            enc.encode(np.zeros(2), np.zeros(2), year=np.zeros((2, 1)))
        with pytest.raises(ValueError, match='year must be a 1D array'):
            enc.encode(np.zeros(2), np.zeros(2), year=np.zeros(3))
        with pytest.raises(ValueError, match='month must be a 1D array'):
            enc.encode(np.zeros(2), np.zeros(2), month=np.zeros((2, 1)))
        with pytest.raises(ValueError, match='month must be a 1D array'):
            enc.encode(np.zeros(2), np.zeros(2), month=np.zeros(3))

    def test_custom_encode_override(self) -> None:
        class _ArrayOnlyEncoder(LocationEncoder):
            embed_dim = 2

            def forward(
                self,
                coords: Tensor,
                year: Tensor | None = None,
                month: Tensor | None = None,
            ) -> Tensor:
                return coords

            def _encode(
                self,
                lon: NDArray[np.floating[Any]],
                lat: NDArray[np.floating[Any]],
                year: NDArray[np.floating[Any]] | None,
                month: NDArray[np.floating[Any]] | None = None,
            ) -> NDArray[np.float32]:
                assert lon.dtype == np.float64
                assert lat.dtype == np.float64
                return cast(Any, np.stack((lon, lat), axis=-1).astype(np.float64))

        enc = _ArrayOnlyEncoder(batch_size=3)
        out = enc.encode(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        assert out.dtype == np.float32
        assert np.allclose(out, [[1.0, 3.0], [2.0, 4.0]])
