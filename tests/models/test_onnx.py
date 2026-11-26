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

import neuralftw.models.onnx as onnx_mod
from neuralftw import DependencyNotFoundError
from neuralftw.models import (
    ONNX_PRESET_PRECISIONS,
    ONNX_PRESET_SPECS,
    ONNXLocationEncoder,
    list_models,
    onnx_encoder,
)


class _FakeNodeArg:
    def __init__(self, name: str, shape: list[Any]) -> None:
        self.name = name
        self.shape = shape


class _FakeInferenceSession:
    def __init__(
        self,
        model_path: str,
        *,
        providers: list[str] | None = None,
        embed_dim: int = 8,
        accepts_year: bool = False,
    ) -> None:
        self.model_path = model_path
        self.providers = list(providers) if providers is not None else []
        self.embed_dim = embed_dim
        self.accepts_year = accepts_year
        self.calls: list[tuple[list[str] | None, dict[str, np.ndarray]]] = []

    def get_inputs(self) -> list[_FakeNodeArg]:
        inputs = [_FakeNodeArg('coords', ['batch', 2])]
        if self.accepts_year:
            inputs.append(_FakeNodeArg('year', ['batch']))
        return inputs

    def get_outputs(self) -> list[_FakeNodeArg]:
        return [_FakeNodeArg('embedding', ['batch', self.embed_dim])]

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        self.calls.append((output_names, {k: v.copy() for k, v in input_feed.items()}))
        coords = input_feed['coords']
        batch = coords.shape[0]
        base = np.arange(self.embed_dim, dtype=np.float32)[None, :]
        out = coords[:, :1] + 0.1 * coords[:, 1:2] + base
        if 'year' in input_feed:
            out = out + input_feed['year'][:, None] * 0.001
        assert out.shape == (batch, self.embed_dim)
        return [out.astype(np.float32)]


def _install_fake_ort(
    monkeypatch: pytest.MonkeyPatch, *, embed_dim: int = 8, accepts_year: bool = False
) -> list[_FakeInferenceSession]:
    sessions: list[_FakeInferenceSession] = []

    def _make_session(
        path: str, providers: list[str] | None = None
    ) -> _FakeInferenceSession:
        sess = _FakeInferenceSession(
            path, providers=providers, embed_dim=embed_dim, accepts_year=accepts_year
        )
        sessions.append(sess)
        return sess

    fake_ort: Any = ModuleType('onnxruntime')
    fake_ort.InferenceSession = _make_session
    monkeypatch.setitem(sys.modules, 'onnxruntime', fake_ort)
    return sessions


def test_onnx_preset_specs_match_zoo() -> None:
    assert len(ONNX_PRESET_SPECS) == 28
    assert set(ONNX_PRESET_PRECISIONS) == set(ONNX_PRESET_SPECS)
    registered = set(list_models())
    for preset, spec in ONNX_PRESET_SPECS.items():
        repo_id, filename, embed_dim, accepts_year, default_year = spec
        assert preset in registered
        assert repo_id == 'taylor-geospatial/neuralftw-zoo'
        assert filename.endswith('.onnx')
        assert embed_dim > 0
        assert accepts_year is (preset == 'terranova-st')
        assert default_year in (2020, 2021)
    assert ONNX_PRESET_PRECISIONS['mind'] == ('int8', 'int4')
    assert ONNX_PRESET_PRECISIONS['terranova'] == ('fp32',)
    assert ONNX_PRESET_PRECISIONS['geoclip'] == ('fp32', 'fp16', 'int8', 'int4')


def test_missing_onnxruntime_raises_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, 'onnxruntime', raising=False)
    real_import = importlib.import_module

    def _blocked_import(name: str, package: str | None = None) -> Any:
        if name == 'onnxruntime':
            raise ModuleNotFoundError("No module named 'onnxruntime'")
        return real_import(name, package)

    monkeypatch.setattr(importlib, 'import_module', _blocked_import)
    with pytest.raises(DependencyNotFoundError, match='onnxruntime'):
        ONNXLocationEncoder(model_path='fake.onnx')


def test_invalid_arguments_raise_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_ort(monkeypatch)
    with pytest.raises(ValueError, match='model_path'):
        ONNXLocationEncoder()
    with pytest.raises(ValueError, match='model_path'):
        ONNXLocationEncoder(repo_id='taylor-geospatial/neuralftw-zoo')
    with pytest.raises(ValueError, match='slice_dim must be a positive int'):
        ONNXLocationEncoder(model_path='fake.onnx', slice_dim=0)
    with pytest.raises(ValueError, match='slice_dim must be a positive int'):
        ONNXLocationEncoder(model_path='fake.onnx', slice_dim=True)


def test_local_path_cpu_and_dynamic_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions = _install_fake_ort(monkeypatch, embed_dim=12)
    model_file = tmp_path / 'model.onnx'
    model_file.write_bytes(b'fake-onnx')

    enc = ONNXLocationEncoder(model_path=model_file, batch_size=4)
    assert enc.embed_dim == 12
    assert enc.model_path == str(model_file)
    assert enc.providers == ('CPUExecutionProvider',)
    assert sessions[0].providers == ['CPUExecutionProvider']

    for batch_size in (1, 7):
        lon = np.linspace(-120.0, 120.0, batch_size, dtype=np.float64)
        lat = np.linspace(-45.0, 45.0, batch_size, dtype=np.float64)
        np_out = enc.encode(lon, lat)
        assert np_out.shape == (batch_size, 12)
        assert np_out.dtype == np.float32

        coords_t = torch.tensor(np.stack((lon, lat), axis=-1), dtype=torch.float32)
        pt_out = enc(coords_t)
        assert pt_out.shape == (batch_size, 12)
        assert pt_out.dtype == torch.float32
        np.testing.assert_allclose(pt_out.numpy(), np_out, atol=1e-6)


def test_hf_download_cuda_and_explicit_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions = _install_fake_ort(monkeypatch, embed_dim=16)
    downloaded = tmp_path / 'downloaded.onnx'
    downloaded.write_bytes(b'fake-onnx')

    download_calls: list[tuple[str, str]] = []

    def _fake_download(repo_id: str, filename: str) -> str:
        download_calls.append((repo_id, filename))
        return str(downloaded)

    monkeypatch.setattr(onnx_mod, 'hf_download', _fake_download)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)

    enc_cuda = ONNXLocationEncoder(
        repo_id='taylor-geospatial/neuralftw-zoo',
        filename='sinr/model.onnx',
        embed_dim=256,
        device='cuda',
    )
    assert download_calls == [('taylor-geospatial/neuralftw-zoo', 'sinr/model.onnx')]
    assert enc_cuda.embed_dim == 256
    assert enc_cuda.providers == ('CUDAExecutionProvider', 'CPUExecutionProvider')
    assert sessions[0].providers == ['CUDAExecutionProvider', 'CPUExecutionProvider']

    enc_explicit = ONNXLocationEncoder(
        model_path=downloaded, providers=['CoreMLExecutionProvider']
    )
    assert enc_explicit.providers == ('CoreMLExecutionProvider',)
    assert sessions[1].providers == ['CoreMLExecutionProvider']


def test_accepts_year_default_and_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions = _install_fake_ort(monkeypatch, embed_dim=6, accepts_year=True)
    model_file = tmp_path / 'st.onnx'
    model_file.write_bytes(b'fake-onnx')

    enc = ONNXLocationEncoder(
        model_path=model_file, accepts_year=True, default_year=2021
    )
    lon = np.array([-122.4, 8.5, 139.7], dtype=np.float64)
    lat = np.array([37.8, 47.4, 35.7], dtype=np.float64)

    out_default = enc.encode(lon, lat)
    assert out_default.shape == (3, 6)
    last_feed = sessions[0].calls[-1][1]
    np.testing.assert_allclose(last_feed['year'], np.array([2021.0, 2021.0, 2021.0]))

    years = np.array([2018.0, 2020.0, 2024.0], dtype=np.float64)
    out_explicit = enc.encode(lon, lat, year=years)
    assert not np.allclose(out_default, out_explicit)

    coords_t = torch.tensor(np.stack((lon, lat), axis=-1), dtype=torch.float32)
    years_t = torch.tensor(years, dtype=torch.float32)
    pt_out = enc(coords_t, year=years_t)
    np.testing.assert_allclose(pt_out.numpy(), out_explicit, atol=1e-6)


def test_onnx_encoder_factory_preset_and_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_ort(monkeypatch, embed_dim=8, accepts_year=True)
    downloaded = tmp_path / 'preset.onnx'
    downloaded.write_bytes(b'fake-onnx')
    monkeypatch.setattr(onnx_mod, 'hf_download', lambda repo, fn: str(downloaded))

    enc_preset = onnx_encoder('terranova-st')
    assert enc_preset.name == 'terranova-st'
    assert enc_preset.embed_dim == 256
    assert enc_preset.accepts_year is True
    assert enc_preset.default_year == 2020

    enc_override = onnx_encoder(
        'terranova-st', embed_dim=64, accepts_year=False, default_year=2022
    )
    assert enc_override.embed_dim == 64
    assert enc_override.accepts_year is False
    assert enc_override.default_year == 2022

    enc_local_default = onnx_encoder(downloaded)
    assert enc_local_default.model_path == str(downloaded)
    assert enc_local_default.embed_dim == 8
    assert enc_local_default.accepts_year is False
    assert enc_local_default.default_year == 2020

    enc_local_custom = onnx_encoder(
        str(downloaded), embed_dim=16, accepts_year=True, default_year=2023
    )
    assert enc_local_custom.embed_dim == 16
    assert enc_local_custom.accepts_year is True
    assert enc_local_custom.default_year == 2023


def test_onnx_quantized_precisions_and_mind_slicing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_ort(monkeypatch, embed_dim=128)
    downloaded = tmp_path / 'quant.onnx'
    downloaded.write_bytes(b'fake-onnx')
    calls: list[tuple[str, str]] = []

    def _fake_download(repo_id: str, filename: str) -> str:
        calls.append((repo_id, filename))
        return str(downloaded)

    monkeypatch.setattr(onnx_mod, 'hf_download', _fake_download)

    enc_geo_int4 = onnx_encoder('geoclip', precision='int4')
    assert enc_geo_int4.embed_dim == 512
    assert calls[-1] == (
        'taylor-geospatial/neuralftw-zoo',
        'geoclip/model_int4_bs32.onnx',
    )

    enc_mind = onnx_encoder('mind', precision='int4')
    assert enc_mind.embed_dim == 64
    assert enc_mind.slice_dim == 64
    assert calls[-1] == ('taylor-geospatial/neuralftw-zoo', 'mind/model_int4_bs32.onnx')
    out_mind = enc_mind.encode(np.array([0.0, 10.0]), np.array([0.0, 20.0]))
    assert out_mind.shape == (2, 64)

    enc_slice_inferred = ONNXLocationEncoder(model_path=downloaded, slice_dim=32)
    assert enc_slice_inferred.embed_dim == 32
    assert enc_slice_inferred(torch.zeros((3, 2))).shape == (3, 32)

    with pytest.raises(ValueError, match='not available for ONNX preset'):
        onnx_encoder('mind', precision='fp32')
    with pytest.raises(ValueError, match='not available for ONNX preset'):
        onnx_encoder('terranova', precision='int4')
